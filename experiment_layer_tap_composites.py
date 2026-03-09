#!/usr/bin/env python3
"""
Experiment: Layer tap i2i on rough composites (object stitch).

Feed rough cut-paste composites through VL encoder at different layers,
then generate with DiT. Tests whether different VL layers can "clean up"
rough compositions zero-shot.

Usage:
    python experiment_layer_tap_composites.py --seed 42
"""

from src import env_setup  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from src.model_utils import load_pipeline
from src.diffusion import (
    _cap_resolution, get_latent_shape, generate_noise,
    _denoise_step, _decode_final,
)

LAYER_TAPS = [
    (0,  "post_merger"),
    (13, "layer12"),
    (19, "layer18"),
    (25, "layer24"),
    (31, "layer30"),
    (35, "layer34"),   # baseline (hidden_states[-2])
    (36, "layer35"),   # final
]


@torch.no_grad()
def encode_image_all_layers(pipe, image, device, max_pixels=768*768):
    """Single VL forward pass, return embeddings at all tapped layers."""
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
    for idx, name in LAYER_TAPS:
        h = out.hidden_states[idx][0]
        results[idx] = h[mask]
    return results


def denoise_with_embeds(pipe, prompt_embeds, height, width, seed, num_steps=8):
    """Run denoising with pre-computed embeddings."""
    shape = get_latent_shape(height, width)
    noise = generate_noise(seed, shape, pipe.device, pipe.torch_dtype)

    pipe.scheduler.set_timesteps(num_steps, denoising_strength=1.0, shift=None)
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    shared = {"latents": noise}
    posi = {"prompt_embeds": prompt_embeds}
    nega = {}

    with torch.no_grad():
        for pid, ts in enumerate(pipe.scheduler.timesteps):
            _denoise_step(pipe, 1.0, shared, posi, nega, models, ts, pid)
        return _decode_final(pipe, shared)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/layer_tap_composites")
    parser.add_argument("--max_pixels", type=int, default=768 * 768)
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline("z-image-turbo", device=device, torch_dtype="bfloat16",
                         text_encoder="qwen3vl")

    # Find composite images
    composites_dir = Path("eval_obj_stitch/composites")
    composites = sorted(composites_dir.glob("*_composite.png"))

    n_layers = len(LAYER_TAPS)
    total = len(composites) * n_layers
    print(f"\n{len(composites)} composites × {n_layers} layers = {total} generations")
    print(f"Layers: {[name for _, name in LAYER_TAPS]}")
    print(f"Output: {output_dir}\n")

    done = 0
    t0 = time.time()

    for img_idx, comp_path in enumerate(composites):
        name = comp_path.stem.replace("_composite", "")
        img = Image.open(comp_path).convert("RGB")
        w, h = img.size
        scale = min(1024 / max(w, h), 1.0)
        out_w = max(round(w * scale) // 16 * 16, 16)
        out_h = max(round(h * scale) // 16 * 16, 16)

        # Save input composite
        img_resized = img.resize((out_w, out_h), Image.LANCZOS)
        img_resized.save(output_dir / f"{img_idx:03d}_{name}_input.png")

        # Also save background if available
        bg_path = composites_dir / f"{name}_background.png"
        if bg_path.exists():
            bg = Image.open(bg_path).convert("RGB").resize((out_w, out_h), Image.LANCZOS)
            bg.save(output_dir / f"{img_idx:03d}_{name}_background.png")

        # Single forward pass, all layers
        layer_embeds = encode_image_all_layers(
            pipe, img, device, max_pixels=args.max_pixels
        )

        for layer_idx, layer_name in LAYER_TAPS:
            embeds = layer_embeds[layer_idx]
            result = denoise_with_embeds(pipe, embeds, out_h, out_w, args.seed)
            result.save(output_dir / f"{img_idx:03d}_{name}_{layer_name}.png")
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"  [{name}] {layer_name} done "
                  f"(tokens={embeds.shape[0]}, {done}/{total}, ~{eta/60:.0f}m left)")

    meta = {
        "n_composites": len(composites),
        "seed": args.seed,
        "max_pixels": args.max_pixels,
        "composites": [str(p) for p in composites],
        "names": [p.stem.replace("_composite", "") for p in composites],
        "layer_taps": {name: idx for idx, name in LAYER_TAPS},
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone! {total} images in {(time.time() - t0)/60:.1f} min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
