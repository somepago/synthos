#!/usr/bin/env python3
"""
Multi-image inference script for Z-Image.

Reads a JSONL file where each line is a JSON array of interleaved image/text
segments, encodes them through the spliced VL model, and generates one image
per line.

Format (each line is a JSON array):
    [{"img": "path/to/img.jpg"}, {"txt": "in the style of"}, {"img": "path/to/other.jpg"}]
    [{"img": "path/to/img.jpg"}, {"txt": "a detailed description"}]
    [{"img": "path/to/a.jpg"}, {"img": "path/to/b.jpg"}]

Blend modes:
    --blend_mode concat    (default) Full cross-attention, interleaved encoding
    --blend_mode avg       B5: Encode images separately, concatenate with alpha scaling (no cross-attention)
    --blend_mode scale     B6: Encode together (cross-attention), then scale visual tokens by image-of-origin

Usage:
    python inference_multi_image.py --input prompts.jsonl
    python inference_multi_image.py --input prompts.jsonl --blend_mode avg --alpha 0.7
    python inference_multi_image.py --input prompts.jsonl --blend_mode scale --alpha 0.8
    python inference_multi_image.py --input prompts.jsonl --output_dir outputs/my_run/
"""

from src import env_setup  # noqa: F401

import argparse
import json
import gc
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.model_utils import load_pipeline, get_defaults
from src.diffusion import (
    get_latent_shape, generate_noise, encode_text_vl,
    encode_interleaved_vl, encode_weighted_avg_vl, encode_weighted_concat_vl,
    _denoise_step, _decode_final,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def round_to_16(w, h, max_size=768):
    scale = min(max_size / max(w, h), 1.0)
    w, h = round(w * scale), round(h * scale)
    return max(w // 16 * 16, 16), max(h // 16 * 16, 16)


def denoise_loop(pipe, prompt_embeds, height, width, seed, num_steps=8,
                 cfg_scale=1.0):
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


def parse_jsonl(path):
    """Parse JSONL file. Each line is a JSON array of {"img": path} / {"txt": str} dicts."""
    entries = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num}: invalid JSON: {e}")
            if not isinstance(items, list):
                raise ValueError(f"Line {line_num}: expected JSON array, got {type(items).__name__}")
            entries.append(items)
    return entries


def resolve_resolution_from_entry(entry, default_h, default_w, max_size):
    """Use first image in entry to determine resolution, or fall back to defaults."""
    for item in entry:
        if "img" in item:
            img = Image.open(item["img"])
            w, h = img.size
            img.close()
            rw, rh = round_to_16(w, h, max_size)
            return rh, rw
    return default_h, default_w


def load_entry_images(entry):
    """Open all image paths in an entry, return content list with PIL images."""
    content = []
    for item in entry:
        if "img" in item:
            img = Image.open(item["img"]).convert("RGB")
            content.append({"img": img})
        elif "txt" in item:
            content.append({"txt": item["txt"]})
    return content


def describe_entry(entry):
    """Short description of an entry for logging."""
    parts = []
    for item in entry:
        if "img" in item:
            parts.append(f"[{Path(item['img']).name}]")
        elif "txt" in item:
            txt = item["txt"]
            if len(txt) > 40:
                txt = txt[:37] + "..."
            parts.append(f'"{txt}"')
    return " + ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Z-Image multi-image inference")

    parser.add_argument("--input", type=str, required=True,
                        help="Path to JSONL file with interleaved image/text entries")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto-timestamped)")

    parser.add_argument("--model", type=str, default=None,
                        choices=["z-image-base", "z-image-turbo"])
    parser.add_argument("--lora_path", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")

    # Blend modes
    parser.add_argument("--blend_mode", type=str, default="concat",
                        choices=["concat", "avg", "scale"],
                        help="concat: interleaved cross-attention (default), "
                             "avg: B5 weighted averaging (no cross-attn), "
                             "scale: B6 cross-attn + per-image token scaling")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for first image (avg/scale modes). "
                             "Second image gets 1-alpha. Default: 0.5")

    args = parser.parse_args()

    entries = parse_jsonl(args.input)
    n = len(entries)
    print(f"Loaded {n} entries from {args.input}")

    # Output dir (auto-timestamp only when no explicit dir given)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%m%d_%H%M")
        output_dir = Path(f"outputs/multi_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Model setup
    model_key = args.model or "z-image-turbo"
    defaults = get_defaults(model_key)
    num_steps = args.num_steps or defaults["num_inference_steps"]
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else defaults["cfg_scale"]
    max_size = max(args.height, args.width)

    # Load pipeline (always qwen3vl for multi-image)
    pipe = load_pipeline(model_key, device=args.device, torch_dtype=args.dtype,
                         text_encoder="qwen3vl")

    if args.lora_path:
        from inference import load_checkpoint
        load_checkpoint(pipe, args.lora_path)

    # Generate
    t0 = time.time()
    for i, entry in enumerate(tqdm(entries, desc="generating")):
        desc = describe_entry(entry)
        h, w = resolve_resolution_from_entry(entry, args.height, args.width, max_size)

        # Load images and build content
        content = load_entry_images(entry)
        pil_images = [item["img"] for item in content if "img" in item]

        # Encode based on blend mode
        if args.blend_mode == "avg" and len(pil_images) >= 2:
            prompt_embeds = encode_weighted_avg_vl(
                pipe, pil_images, pipe.device, alpha=args.alpha)
        elif args.blend_mode == "scale" and len(pil_images) >= 2:
            prompt_embeds = encode_weighted_concat_vl(
                pipe, content, pipe.device, alpha=args.alpha)
        else:
            # Default: full interleaved cross-attention
            # Also fallback for single-image entries in avg/scale mode
            prompt_embeds = encode_interleaved_vl(pipe, content, pipe.device)
        n_tokens = prompt_embeds.shape[0]

        # Generate
        image = denoise_loop(pipe, prompt_embeds, h, w, args.seed,
                             num_steps, cfg_scale)
        image.save(output_dir / f"{i:03d}.png")

        # Save input images as reference
        img_idx = 0
        for item in entry:
            if "img" in item:
                ref = Image.open(item["img"]).convert("RGB")
                ref.save(output_dir / f"{i:03d}_ref{img_idx}.png")
                img_idx += 1

        mode_str = f" [{args.blend_mode}]" if args.blend_mode != "concat" else ""
        alpha_str = f" α={args.alpha}" if args.blend_mode != "concat" else ""
        print(f"  [{i:03d}] {desc} → {h}x{w}, {n_tokens} tokens{mode_str}{alpha_str}")

    elapsed = time.time() - t0
    print(f"\nDone: {n} images in {elapsed:.1f}s")
    print(f"Outputs: {output_dir}/")

    # Save metadata
    meta = {
        "n": n,
        "model": model_key,
        "num_steps": num_steps,
        "cfg_scale": cfg_scale,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "blend_mode": args.blend_mode,
        "alpha": args.alpha,
        "entries": entries,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    del pipe
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
