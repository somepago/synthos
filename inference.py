#!/usr/bin/env python3
"""
Unified inference script for Z-Image text2image and img2img.

Usage:
    # Single t2i
    python inference.py --prompt "a cat on a chair"

    # Single i2i
    python inference.py --image cat.png

    # Both for same input
    python inference.py --prompt "a cat" --image cat.png

    # Batch i2i: folder of images
    python inference.py --image eval_unified/

    # Batch t2i: text file of prompts (one per line)
    python inference.py --prompt prompts.txt

    # Batch both: folder + text file (paired by index)
    python inference.py --image eval_unified/ --prompt captions.txt

    # Generate + compute metrics
    python inference.py --image eval_unified/ --prompt captions.txt --metrics

    # Just metrics on existing outputs (no generation)
    python inference.py --metrics --skip_generation --output_dir outputs/some_run/
"""

from src import env_setup  # noqa: F401

import argparse
import json
import gc
import signal
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

TIMEOUT_PER_IMAGE = 180  # seconds — kill hung generation


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout("Generation timed out")

from src.model_utils import load_pipeline, get_defaults
from src.diffusion import (
    get_latent_shape, generate_noise,
    encode_image_vl, encode_text_vl, _denoise_step, _decode_final,
    run_full_diffusion,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def round_to_16(w, h, max_size=768):
    """Resize preserving AR, round to 16px multiples, cap at max_size."""
    scale = min(max_size / max(w, h), 1.0)
    w, h = round(w * scale), round(h * scale)
    return max(w // 16 * 16, 16), max(h // 16 * 16, 16)


def load_checkpoint(pipe, checkpoint_path: str):
    """Load checkpoint: supports LoRA+siglip (GRPO) or siglip-only (stage1)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    lora_state = checkpoint.get("lora_state_dict", {})
    if lora_state:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(pipe.dit, lora_state)
        print(f"Loaded LoRA weights ({len(lora_state)} params)")

    siglip_state = checkpoint.get("siglip_state_dict", {})
    if siglip_state:
        for name, param in pipe.dit.named_parameters():
            if name in siglip_state:
                param.data.copy_(siglip_state[name].to(param.device, param.dtype))
        print(f"Loaded siglip projection weights ({len(siglip_state)} params)")

    print(f"Loaded checkpoint from {checkpoint_path}")


# =============================================================================
# Core generation from pre-computed embeddings
# =============================================================================

def denoise_loop(pipe, prompt_embeds, height, width, seed, num_steps=8,
                 cfg_scale=1.0):
    """Run denoising loop with pre-computed prompt embeddings."""
    shape = get_latent_shape(height, width)
    noise = generate_noise(seed, shape, pipe.device, pipe.torch_dtype)

    pipe.scheduler.set_timesteps(num_steps, denoising_strength=1.0, shift=None)
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    shared = {"latents": noise}
    posi = {"prompt_embeds": prompt_embeds}
    nega = {}

    if cfg_scale > 1.0:
        neg_embeds = encode_text_vl(pipe, "", pipe.device)
        nega = {"prompt_embeds": neg_embeds}

    with torch.no_grad():
        for pid, ts in enumerate(pipe.scheduler.timesteps):
            _denoise_step(pipe, cfg_scale, shared, posi, nega, models, ts, pid)
        return _decode_final(pipe, shared)


# =============================================================================
# Mode detection and input parsing
# =============================================================================

def parse_prompts(prompt_arg):
    """Parse --prompt: string literal or path to .txt file (one per line).
    Returns list of strings, or None if no prompt."""
    if prompt_arg is None:
        return None
    p = Path(prompt_arg)
    if p.suffix == ".txt" and p.exists():
        lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        print(f"Read {len(lines)} prompts from {p}")
        return lines
    return [prompt_arg]


def parse_images(image_arg):
    """Parse --image: single image path or directory of images.
    Returns list of Path, or None if no image."""
    if image_arg is None:
        return None
    p = Path(image_arg)
    if p.is_dir():
        imgs = sorted([f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS])
        print(f"Found {len(imgs)} images in {p}")
        return imgs
    if p.is_file():
        return [p]
    raise FileNotFoundError(f"Image path not found: {image_arg}")


def resolve_resolution(image_path, default_h, default_w, max_size):
    """If image provided, match its AR (rounded to 16px, capped). Otherwise use defaults."""
    if image_path is None:
        return default_h, default_w
    img = Image.open(image_path)
    w, h = img.size
    img.close()
    rw, rh = round_to_16(w, h, max_size)
    return rh, rw


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Z-Image inference (t2i + i2i)")

    # Input
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt string OR path to .txt file (one per line)")
    parser.add_argument("--image", type=str, default=None,
                        help="Image path OR directory of images for i2i")

    # Output
    parser.add_argument("--output", type=str, default="output.png",
                        help="Output path for single-image mode (default: output.png)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for batch mode (default: auto-timestamped)")

    # Model
    parser.add_argument("--model", type=str, default=None,
                        choices=["z-image-base", "z-image-turbo"])
    parser.add_argument("--text_encoder", type=str, default="qwen3vl",
                        choices=["qwen3", "qwen3vl"])
    parser.add_argument("--lora_path", type=str, default=None)

    # Generation params
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--max_pixels", type=int, default=768*768,
                        help="Max pixels for VL image encoding (controls token count / variation strength)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Process only first N images/prompts (default: all)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")

    # Metrics
    parser.add_argument("--metrics", action="store_true",
                        help="Compute HPSv2.1, CLIP score, DINOv2 sim after generation")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip generation, only run metrics on existing outputs")

    args = parser.parse_args()

    if args.prompt is None and args.image is None and not args.skip_generation:
        parser.error("At least one of --prompt or --image is required (or use --skip_generation)")

    # Parse inputs
    prompts = parse_prompts(args.prompt)
    images = parse_images(args.image)

    # Truncate to --n_samples if specified
    if args.n_samples is not None:
        if prompts:
            prompts = prompts[:args.n_samples]
        if images:
            images = images[:args.n_samples]

    # Determine batch vs single mode
    n_prompts = len(prompts) if prompts else 0
    n_images = len(images) if images else 0
    is_batch = n_prompts > 1 or n_images > 1

    if is_batch and n_prompts > 1 and n_images > 1 and n_prompts != n_images:
        parser.error(f"Prompt count ({n_prompts}) != image count ({n_images}). Must match for paired batch.")

    n = max(n_prompts, n_images, 1)

    # Pad single prompt/image to match batch size
    if prompts and len(prompts) == 1 and n > 1:
        prompts = prompts * n
    if images and len(images) == 1 and n > 1:
        images = images * n

    # Resolve output directory
    ts = datetime.now().strftime("%m%d_%H%M")
    if is_batch or args.skip_generation:
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            parts = ["gen"]
            if prompts:
                parts.append("t2i")
            if images:
                parts.append("i2i")
            parts.append(ts)
            output_dir = Path(f"outputs/{'_'.join(parts)}")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

    # Model setup
    model_key = args.model or "z-image-turbo"
    defaults = get_defaults(model_key)
    num_steps = args.num_steps or defaults["num_inference_steps"]
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else defaults["cfg_scale"]
    max_size = max(args.height, args.width)

    # --- Skip generation mode ---
    if args.skip_generation:
        if not args.metrics:
            print("Nothing to do: --skip_generation without --metrics")
            return
        # Load metadata if available
        meta_path = output_dir / "meta.json"
        prompt_list = None
        has_t2i, has_i2i, has_input = False, False, False
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            n = meta["n"]
            prompt_list = meta.get("prompts")
            has_t2i = meta.get("has_t2i", False)
            has_i2i = meta.get("has_i2i", False)
            has_input = meta.get("has_input", False)
        else:
            # Detect from files
            n = 0
            while (output_dir / f"i2i_{n:03d}.png").exists() or (output_dir / f"t2i_{n:03d}.png").exists():
                n += 1
            if n == 0:
                print(f"No generated images found in {output_dir}")
                return
            has_t2i = (output_dir / "t2i_000.png").exists()
            has_i2i = (output_dir / "i2i_000.png").exists()
            has_input = (output_dir / "input_000.png").exists()
            print(f"Detected {n} images (t2i={has_t2i}, i2i={has_i2i}, input={has_input})")

        from src.metrics import run_metrics, print_results
        results = run_metrics(output_dir, prompts=prompt_list,
                              has_t2i=has_t2i, has_i2i=has_i2i,
                              has_input=has_input, n=n, device=args.device)
        if results:
            print_results(results)
            results_path = output_dir / "results.json"
            results_path.write_text(json.dumps(results, indent=2))
            print(f"\nResults saved to {results_path}")
        return

    # --- Load pipeline ---
    # Determine text encoder: need qwen3vl for i2i, qwen3 for text-only t2i
    text_encoder = args.text_encoder
    if images and text_encoder == "qwen3":
        print("NOTE: i2i requires qwen3vl, overriding --text_encoder to qwen3vl")
        text_encoder = "qwen3vl"

    pipe = load_pipeline(model_key, device=args.device, torch_dtype=args.dtype,
                         text_encoder=text_encoder)

    if args.lora_path:
        load_checkpoint(pipe, args.lora_path)

    # --- Single mode ---
    if not is_batch:
        prompt = prompts[0] if prompts else None
        image_path = images[0] if images else None

        # Build output path (auto-timestamp only if using default output name)
        out_path = Path(args.output)
        if args.output == "output.png":
            out_stem = f"{out_path.stem}_{ts}"
        else:
            out_stem = out_path.stem
        out_parent = out_path.parent
        out_parent.mkdir(parents=True, exist_ok=True)

        h, w = resolve_resolution(image_path, args.height, args.width, max_size)

        if image_path and text_encoder == "qwen3vl":
            # i2i
            edit_img = Image.open(image_path).convert("RGB")
            prompt_embeds = encode_image_vl(pipe, edit_img, pipe.device,
                                            max_pixels=args.max_pixels)
            image = denoise_loop(pipe, prompt_embeds, h, w, args.seed,
                                 num_steps, cfg_scale)
            print(f"i2i: {h}x{w}, steps={num_steps}, cfg={cfg_scale}, max_pixels={args.max_pixels}")

            # Also do t2i if prompt provided
            if prompt:
                t2i_embeds = encode_text_vl(pipe, prompt, pipe.device)
                t2i_image = denoise_loop(pipe, t2i_embeds, h, w, args.seed,
                                         num_steps, cfg_scale)
                image.save(out_parent / f"{out_stem}_i2i.png")
                t2i_image.save(out_parent / f"{out_stem}_t2i.png")
                print(f"t2i: {h}x{w}, steps={num_steps}, cfg={cfg_scale}")
                print(f"Saved to {out_parent / f'{out_stem}_i2i.png'} and {out_parent / f'{out_stem}_t2i.png'}")
                return
        elif prompt and not image_path:
            # t2i only
            if text_encoder == "qwen3vl":
                prompt_embeds = encode_text_vl(pipe, prompt, pipe.device)
                image = denoise_loop(pipe, prompt_embeds, h, w, args.seed,
                                     num_steps, cfg_scale)
            else:
                shape = get_latent_shape(h, w)
                noise = generate_noise(args.seed, shape, args.device,
                                       getattr(torch, args.dtype))
                image = run_full_diffusion(pipe, prompt, noise, num_steps, cfg_scale)
            print(f"t2i: {h}x{w}, steps={num_steps}, cfg={cfg_scale}")
        elif prompt and image_path and text_encoder != "qwen3vl":
            # t2i with qwen3 (image ignored since qwen3 can't do i2i)
            shape = get_latent_shape(h, w)
            noise = generate_noise(args.seed, shape, args.device,
                                   getattr(torch, args.dtype))
            image = run_full_diffusion(pipe, prompt, noise, num_steps, cfg_scale)
            print(f"t2i (qwen3, image ignored): {h}x{w}, steps={num_steps}, cfg={cfg_scale}")
        else:
            print("ERROR: --image without qwen3vl text encoder, and no --prompt")
            return

        save_path = out_parent / f"{out_stem}.png"
        image.save(save_path)
        print(f"Saved to {save_path}")
        return

    # --- Batch mode ---
    has_t2i = prompts is not None
    has_i2i = images is not None
    has_input = images is not None

    signal.signal(signal.SIGALRM, _timeout_handler)
    t0 = time.time()
    skipped = 0
    for i in tqdm(range(n), desc="generating"):
        image_path = images[i] if images else None
        prompt = prompts[i] if prompts else None

        h, w = resolve_resolution(image_path, args.height, args.width, max_size)

        try:
            signal.alarm(TIMEOUT_PER_IMAGE)

            # i2i
            if image_path and text_encoder == "qwen3vl":
                input_img = Image.open(image_path).convert("RGB")
                input_resized = input_img.resize((w, h), Image.LANCZOS)
                input_resized.save(output_dir / f"input_{i:03d}.png")

                prompt_embeds = encode_image_vl(pipe, input_img, pipe.device,
                                                max_pixels=args.max_pixels)
                image = denoise_loop(pipe, prompt_embeds, h, w, args.seed,
                                     num_steps, cfg_scale)
                image.save(output_dir / f"i2i_{i:03d}.png")

            # t2i
            if prompt:
                if text_encoder == "qwen3vl":
                    t2i_embeds = encode_text_vl(pipe, prompt, pipe.device)
                    image = denoise_loop(pipe, t2i_embeds, h, w, args.seed,
                                         num_steps, cfg_scale)
                else:
                    shape = get_latent_shape(h, w)
                    noise = generate_noise(args.seed, shape, args.device,
                                           getattr(torch, args.dtype))
                    image = run_full_diffusion(pipe, prompt, noise, num_steps, cfg_scale)
                image.save(output_dir / f"t2i_{i:03d}.png")

            signal.alarm(0)
        except _Timeout:
            print(f"\n  TIMEOUT [{i:03d}] — skipped after {TIMEOUT_PER_IMAGE}s")
            skipped += 1
            continue
        except Exception as e:
            signal.alarm(0)
            print(f"\n  ERROR [{i:03d}]: {e}")
            skipped += 1
            continue

    signal.alarm(0)
    elapsed = time.time() - t0
    print(f"Generation done: {n} images ({skipped} skipped) in {elapsed:.1f}s")

    # Save metadata for reproducibility
    meta = {
        "n": n,
        "model": model_key,
        "text_encoder": text_encoder,
        "num_steps": num_steps,
        "cfg_scale": cfg_scale,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "max_pixels": args.max_pixels,
        "has_t2i": has_t2i,
        "has_i2i": has_i2i,
        "has_input": has_input,
    }
    if prompts:
        meta["prompts"] = prompts
    if images:
        meta["image_paths"] = [str(p) for p in images]
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    del pipe
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Metrics ---
    if args.metrics:
        from src.metrics import run_metrics, print_results
        t0 = time.time()
        results = run_metrics(output_dir, prompts=prompts,
                              has_t2i=has_t2i, has_i2i=has_i2i,
                              has_input=has_input, n=n, device=args.device)
        print(f"Metrics took {time.time() - t0:.1f}s")
        if results:
            print_results(results)
            results_path = output_dir / "results.json"
            results_path.write_text(json.dumps(results, indent=2))
            print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
