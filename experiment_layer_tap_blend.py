#!/usr/bin/env python3
"""
Experiment: Multi-image blending with different internal VL layers.

Combines the layer-tap idea with multi-image blending (avg and scale modes).
For each image pair, generates outputs using embeddings from layers 12-35.

Blend modes:
  - avg α=0.3: Encode each image separately at layer N, scale + concat
  - scale α=0.3: Encode both together (cross-attention) at layer N, scale visual tokens

Layers tested: 12, 18, 24, 30, 34 (baseline), 35 (final)

Uses same image pairs as existing multi-image experiments (from meta.json).

Usage:
    python experiment_layer_tap_blend.py --n_pairs 15 --seed 42
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
    _cap_resolution, _find_image_token_ranges,
    get_latent_shape, generate_noise,
    _denoise_step, _decode_final,
)

LAYER_TAPS = [
    (13, "layer12"),
    (19, "layer18"),
    (25, "layer24"),
    (31, "layer30"),
    (35, "layer34"),   # baseline (hidden_states[-2])
    (36, "layer35"),   # final
]

ALPHA = 0.3


@torch.no_grad()
def encode_avg_at_layer(pipe, images, device, layer_idx, alpha=0.3, max_pixels=512*512):
    """Encode images separately, extract at layer_idx, scale + concat."""
    weights = [alpha, 1.0 - alpha] if len(images) == 2 else [1.0/len(images)] * len(images)

    scaled_parts = []
    for img, w in zip(images, weights):
        img = _cap_resolution(img, max_pixels)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": ""},
        ]}]
        text = pipe.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = pipe.vl_processor(text=[text], images=[img], return_tensors="pt").to(device)

        out = pipe.vl_model.model(**inputs, output_hidden_states=True)
        h = out.hidden_states[layer_idx][0]
        mask = inputs["attention_mask"][0].bool()
        scaled_parts.append(h[mask] * w)

    return torch.cat(scaled_parts, dim=0)


@torch.no_grad()
def encode_scale_at_layer(pipe, images, device, layer_idx, alpha=0.3, max_pixels=512*512, text=""):
    """Encode images together (cross-attention), extract at layer_idx, scale visual tokens."""
    weights = [alpha, 1.0 - alpha] if len(images) == 2 else [1.0/len(images)] * len(images)

    chat_content = []
    pil_images = []
    for img in images:
        img = _cap_resolution(img, max_pixels)
        chat_content.append({"type": "image", "image": img})
        pil_images.append(img)
    chat_content.append({"type": "text", "text": text})

    messages = [{"role": "user", "content": chat_content}]
    text = pipe.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = pipe.vl_processor(
        text=[text], images=pil_images, return_tensors="pt"
    ).to(device)

    image_ranges = _find_image_token_ranges(inputs["input_ids"])

    out = pipe.vl_model.model(**inputs, output_hidden_states=True)
    h = out.hidden_states[layer_idx][0].clone()
    mask = inputs["attention_mask"][0].bool()

    for (start, end), w in zip(image_ranges, weights):
        h[start:end] *= w

    return h[mask]


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
    parser.add_argument("--n_pairs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/layer_tap_blend")
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    parser.add_argument("--entries_file", type=str, default=None,
                        help="JSONL file with entries (e.g. composition_light.jsonl). "
                             "Falls back to multi_avg_a0.3 meta.json")
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    pipe = load_pipeline("z-image-turbo", device=device, torch_dtype="bfloat16",
                         text_encoder="qwen3vl")

    # Load entries
    if args.entries_file:
        entries_path = Path(args.entries_file)
        entries = [json.loads(line) for line in entries_path.read_text().strip().split("\n")]
        entries = entries[:args.n_pairs]
    else:
        meta_path = Path("outputs/baselines_feb25/multi_avg_a0.3/meta.json")
        if not meta_path.exists():
            print("ERROR: No existing multi-image meta.json found. Need image pairs.")
            return
        meta = json.loads(meta_path.read_text())
        entries = meta["entries"][:args.n_pairs]

    n_layers = len(LAYER_TAPS)
    n_modes = 2  # avg, scale
    total = len(entries) * n_layers * n_modes
    print(f"\n{len(entries)} pairs × {n_layers} layers × {n_modes} modes = {total} generations")
    print(f"Layers: {[name for _, name in LAYER_TAPS]}")
    print(f"Modes: avg (α={ALPHA}), scale (α={ALPHA})")
    print(f"Output: {output_dir}\n")

    done = 0
    t0 = time.time()

    for pair_idx, entry in enumerate(entries):
        # Load pair images
        img_items = [item for item in entry if "img" in item]
        if len(img_items) < 2:
            print(f"  [{pair_idx:03d}] skipping — fewer than 2 images")
            continue

        images = [Image.open(item["img"]).convert("RGB") for item in img_items[:2]]

        # Get text prompt if present
        txt_items = [item for item in entry if "txt" in item]
        prompt_text = txt_items[0]["txt"] if txt_items else ""

        # Save refs
        for ri, img in enumerate(images):
            ref_path = output_dir / f"{pair_idx:03d}_ref{ri}.png"
            if not ref_path.exists():
                img.save(ref_path)

        # Output size
        out_h, out_w = 1024, 1024

        for layer_idx, layer_name in LAYER_TAPS:
            # avg mode (no text — encodes images separately)
            embeds_avg = encode_avg_at_layer(
                pipe, images, device, layer_idx, alpha=ALPHA, max_pixels=args.max_pixels
            )
            result = denoise_with_embeds(pipe, embeds_avg, out_h, out_w, args.seed)
            result.save(output_dir / f"{pair_idx:03d}_avg_{layer_name}.png")
            done += 1

            # scale mode (with text if available — encodes together)
            embeds_scale = encode_scale_at_layer(
                pipe, images, device, layer_idx, alpha=ALPHA, max_pixels=args.max_pixels,
                text=prompt_text,
            )
            result = denoise_with_embeds(pipe, embeds_scale, out_h, out_w, args.seed)
            result.save(output_dir / f"{pair_idx:03d}_scale_{layer_name}.png")
            done += 1

            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"  [{pair_idx:03d}] {layer_name} avg+scale done "
                  f"({done}/{total}, ~{eta/60:.0f}m left)")

    # Save meta
    out_meta = {
        "n_pairs": len(entries),
        "seed": args.seed,
        "alpha": ALPHA,
        "max_pixels": args.max_pixels,
        "layer_taps": {name: idx for idx, name in LAYER_TAPS},
        "modes": ["avg", "scale"],
        "entries": entries[:args.n_pairs],
    }
    (output_dir / "meta.json").write_text(json.dumps(out_meta, indent=2))

    print(f"\nDone! {total} images in {(time.time() - t0)/60:.1f} min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
