#!/usr/bin/env python3
"""
Layer Tap Ablation v2 — on the main eval set.

Runs:
  1. Main: default settings, 4 seeds per layer (seed diversity)
  2. CFG ablation: cfg=[1,2,4], default steps, 1 seed
  3. Steps ablation: steps=[20,30,40], default cfg, 1 seed

Usage:
    # Turbo (8 steps, cfg=1)
    python experiment_layer_tap_v2.py

    # Base (50 steps, cfg=4) — subset of 20 images
    python experiment_layer_tap_v2.py --model z-image-base --n_real 10 --n_synth 10

    # Just ablations on base
    python experiment_layer_tap_v2.py --model z-image-base --run ablations --n_real 10 --n_synth 10
"""

from src import env_setup  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from src.model_utils import load_pipeline, get_defaults
from src.diffusion import (
    _cap_resolution, get_latent_shape, generate_noise,
    encode_text_vl, _denoise_step, _decode_final,
)

# hidden_states index = LLM layer + 1 (index 0 is embedding output)
# Default: emb + every 4th layer + final
DEFAULT_LAYERS = "emb,4,8,12,20,24,30,34,35"


def parse_layer_taps(layers_str):
    """Parse comma-separated layer spec into [(hidden_states_idx, name), ...]"""
    taps = []
    for tok in layers_str.split(","):
        tok = tok.strip()
        if tok == "emb":
            taps.append((0, "emb"))
        else:
            llm_layer = int(tok)
            taps.append((llm_layer + 1, f"layer{llm_layer:02d}"))
    return taps

SEEDS = [42, 123, 777, 2024]
ABLATION_SEED = 42

# Ablation grids (values that differ from default)
CFG_ABLATION = [1, 2, 4]
STEPS_ABLATION = [20, 30, 40]

SPLITS_FILE = Path("eval_unified/eval_splits.json")


def load_eval_images(n_real=None, n_synth=None):
    """Load eval set, optionally limiting to first N of each type."""
    splits = json.loads(SPLITS_FILE.read_text())

    images = []
    real_dir = Path("eval_unified/images")
    real_ids = splits["main_29_real"]
    if n_real is not None:
        real_ids = real_ids[:n_real]
    for rid in real_ids:
        for ext in [".jpg", ".jpeg", ".png"]:
            p = real_dir / f"{rid:03d}{ext}"
            if p.exists():
                images.append(("real", rid, p))
                break

    synth_dir = Path("eval_unified/images_synth_zim")
    synth_ids = splits["main_12_synth"]
    if n_synth is not None:
        synth_ids = synth_ids[:n_synth]
    for sid in synth_ids:
        p = synth_dir / f"t2i_{sid:03d}.png"
        if p.exists():
            images.append(("synth", sid, p))

    return images


@torch.no_grad()
def encode_image_vl_layers(pipe, image, device, layer_taps, max_pixels=768*768):
    """Single VL forward pass, extract all target layers."""
    image = _cap_resolution(image, max_pixels)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": ""},
    ]}]
    text = pipe.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = pipe.vl_processor(text=[text], images=[image], return_tensors="pt").to(device)

    out = pipe.vl_model.model(**inputs, output_hidden_states=True)
    mask = inputs["attention_mask"][0].bool()

    results = {}
    for idx, name in layer_taps:
        h = out.hidden_states[idx][0]
        results[name] = h[mask]

    return results


def denoise_with_embeds(pipe, prompt_embeds, height, width, seed,
                        num_steps=8, cfg_scale=1.0, neg_embeds=None):
    shape = get_latent_shape(height, width)
    noise = generate_noise(seed, shape, pipe.device, pipe.torch_dtype)

    pipe.scheduler.set_timesteps(num_steps, denoising_strength=1.0, shift=None)
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    shared = {"latents": noise}
    posi = {"prompt_embeds": prompt_embeds}
    nega = {"prompt_embeds": neg_embeds} if neg_embeds is not None else {}

    with torch.no_grad():
        for pid, ts in enumerate(pipe.scheduler.timesteps):
            _denoise_step(pipe, cfg_scale, shared, posi, nega, models, ts, pid)
        return _decode_final(pipe, shared)


def build_jobs(default_steps, default_cfg, run_mode):
    """Build list of (suffix, num_steps, cfg_scale, seeds) jobs.

    File naming:
      main:          {tag}_{layer}_s{seed}.png
      cfg ablation:  {tag}_{layer}_cfg{cfg}_s42.png
      steps ablation:{tag}_{layer}_st{steps}_s42.png
    """
    jobs = []

    if run_mode in ("main", "all"):
        # Main run: default settings, all seeds
        jobs.append({
            "name": "main",
            "suffix": "",  # {tag}_{layer}_s{seed}.png
            "num_steps": default_steps,
            "cfg_scale": default_cfg,
            "seeds": SEEDS,
        })

    if run_mode in ("ablations", "all"):
        # CFG ablation: vary cfg, default steps, 1 seed
        for cfg in CFG_ABLATION:
            if cfg == default_cfg:
                continue  # covered by main run
            jobs.append({
                "name": f"cfg={cfg}",
                "suffix": f"_cfg{cfg}",
                "num_steps": default_steps,
                "cfg_scale": cfg,
                "seeds": [ABLATION_SEED],
            })

        # Steps ablation: vary steps, default cfg, 1 seed
        for steps in STEPS_ABLATION:
            if steps == default_steps:
                continue  # covered by main run
            jobs.append({
                "name": f"steps={steps}",
                "suffix": f"_st{steps}",
                "num_steps": steps,
                "cfg_scale": default_cfg,
                "seeds": [ABLATION_SEED],
            })

    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="z-image-turbo",
                        choices=["z-image-turbo", "z-image-base"])
    parser.add_argument("--run", type=str, default="all",
                        choices=["main", "ablations", "all"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_pixels", type=int, default=768 * 768)
    parser.add_argument("--n_real", type=int, default=None,
                        help="Limit to first N real images")
    parser.add_argument("--n_synth", type=int, default=None,
                        help="Limit to first N synth images")
    parser.add_argument("--layers", type=str, default=DEFAULT_LAYERS,
                        help="Comma-separated layers to tap (e.g. 'emb,4,12,24,34,35')")
    args = parser.parse_args()

    LAYER_TAPS = parse_layer_taps(args.layers)

    device = "cuda"
    model_key = args.model
    defaults = get_defaults(model_key)
    default_steps = defaults["num_inference_steps"]
    default_cfg = defaults["cfg_scale"]

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif "turbo" in model_key:
        output_dir = Path("outputs/layer_tap_v2")
    else:
        output_dir = Path("outputs/layer_tap_v2_base")
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(model_key, device=device, torch_dtype="bfloat16",
                         text_encoder="qwen3vl")

    # Pre-compute negative embeddings (needed for any cfg > 1)
    neg_embeds = encode_text_vl(pipe, "", device)
    print(f"Neg embeds: {neg_embeds.shape}")

    images = load_eval_images(n_real=args.n_real, n_synth=args.n_synth)
    jobs = build_jobs(default_steps, default_cfg, args.run)

    # Count total
    n_layers = len(LAYER_TAPS)
    total = sum(len(images) * n_layers * len(j["seeds"]) for j in jobs)

    print(f"\n{len(images)} images × {n_layers} layers")
    print(f"Model: {model_key}, default: steps={default_steps}, cfg={default_cfg}")
    print(f"Jobs:")
    for j in jobs:
        n = len(images) * n_layers * len(j["seeds"])
        print(f"  {j['name']}: steps={j['num_steps']}, cfg={j['cfg_scale']}, "
              f"seeds={j['seeds']} → {n} gens")
    print(f"Total: {total} generations")
    print(f"Output: {output_dir}\n")

    done = 0
    t0 = time.time()

    for img_i, (src_type, src_id, img_path) in enumerate(images):
        tag = f"{src_type}_{src_id:03d}"

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        scale = min(1024 / max(w, h), 1.0)
        out_w = max(round(w * scale) // 16 * 16, 16)
        out_h = max(round(h * scale) // 16 * 16, 16)

        input_path = output_dir / f"{tag}_input.png"
        if not input_path.exists():
            img_resized = img.resize((out_w, out_h), Image.LANCZOS)
            img_resized.save(input_path)

        # Single VL forward pass — embeddings shared across all jobs
        layer_embeds = None

        for job in jobs:
            suffix = job["suffix"]
            num_steps = job["num_steps"]
            cfg_scale = job["cfg_scale"]
            seeds = job["seeds"]

            # Check if all done for this image × job
            all_done = all(
                (output_dir / f"{tag}_{name}{suffix}_s{seed}.png").exists()
                for _, name in LAYER_TAPS
                for seed in seeds
            )
            if all_done:
                done += n_layers * len(seeds)
                continue

            # Lazy VL encode (only if needed)
            if layer_embeds is None:
                layer_embeds = encode_image_vl_layers(
                    pipe, img, device, LAYER_TAPS, max_pixels=args.max_pixels
                )

            ne = neg_embeds if cfg_scale > 1.0 else None

            for _, layer_name in LAYER_TAPS:
                embeds = layer_embeds[layer_name]
                for seed in seeds:
                    out_path = output_dir / f"{tag}_{layer_name}{suffix}_s{seed}.png"
                    if out_path.exists():
                        done += 1
                        continue

                    result = denoise_with_embeds(pipe, embeds, out_h, out_w, seed,
                                                 num_steps=num_steps,
                                                 cfg_scale=cfg_scale,
                                                 neg_embeds=ne)
                    result.save(out_path)
                    done += 1

                    elapsed = time.time() - t0
                    eta = elapsed / done * (total - done) if done > 0 else 0
                    print(f"  [{tag}] {layer_name}{suffix} s{seed} "
                          f"({done}/{total}, ~{eta/60:.0f}m left)")

    meta = {
        "n_images": len(images),
        "model": model_key,
        "default_steps": default_steps,
        "default_cfg": default_cfg,
        "jobs": [{k: v for k, v in j.items()} for j in jobs],
        "max_pixels": args.max_pixels,
        "layer_taps": [{"name": name, "hidden_states_idx": idx, "llm_layer": idx - 1}
                        for idx, name in LAYER_TAPS],
        "images": [{"type": t, "id": i, "path": str(p)} for t, i, p in images],
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone! {total} images in {(time.time() - t0)/60:.1f} min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
