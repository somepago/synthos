#!/usr/bin/env python3
"""
Experiment: Tap different internal layers of Qwen3-VL for DiT conditioning.

Normal i2i: Image → ViT → PatchMerger → LLM (36 layers) → hidden_states[-2] → DiT
This experiment: Same pipeline, but extract hidden_states from different points:

  - Layer 0:  Post-PatchMerger / embedding output (before any LLM layer)
  - Layer 4:  After LLM layer 3 (early)
  - Layer 8:  After LLM layer 7 (early-mid)
  - Layer 12: After LLM layer 11 (mid)
  - Layer 18: After LLM layer 17 (mid-late)
  - Layer 24: After LLM layer 23 (late)
  - Layer 30: After LLM layer 29 (very late)
  - Layer 34: After LLM layer 33 (penultimate — CURRENT DEFAULT)
  - Layer 36: After LLM layer 35 (final)

hidden_states tuple has 37 entries: [embedding, layer0, layer1, ..., layer35]
Index 0 = embedding output, index N = after LLM layer N-1.
Current default = hidden_states[-2] = index 35 = after LLM layer 34.

Usage:
    python experiment_layer_tap.py --n_images 20 --seed 42
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

# Layers to tap: (hidden_states index, human-readable name)
LAYER_TAPS = [
    (0,  "post_merger"),    # Embedding output = post-PatchMerger, pre-LLM
    (5,  "llm_layer04"),    # After LLM layer 4
    (9,  "llm_layer08"),    # After LLM layer 8
    (13, "llm_layer12"),    # After LLM layer 12
    (19, "llm_layer18"),    # After LLM layer 18
    (25, "llm_layer24"),    # After LLM layer 24
    (31, "llm_layer30"),    # After LLM layer 30
    (35, "llm_layer34"),    # After LLM layer 34 = hidden_states[-2] = BASELINE
    (36, "llm_layer35"),    # After LLM layer 35 = final layer
]


@torch.no_grad()
def encode_image_vl_all_layers(pipe, image, device, max_pixels=768*768):
    """Encode image via VL model, return ALL hidden states + mask.

    Returns dict mapping hidden_states index → (L, 2560) tensor.
    Single forward pass, extract all layers at once.
    """
    image = _cap_resolution(image, max_pixels)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": ""},
    ]}]
    text = pipe.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = pipe.vl_processor(text=[text], images=[image], return_tensors="pt").to(device)

    out = pipe.vl_model.model(**inputs, output_hidden_states=True)
    mask = inputs["attention_mask"][0].bool()

    # Extract only the layers we care about
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
    parser.add_argument("--n_images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/layer_tap_exp")
    parser.add_argument("--max_pixels", type=int, default=768 * 768)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    pipe = load_pipeline("z-image-turbo", device=device, torch_dtype="bfloat16",
                         text_encoder="qwen3vl")

    # Load eval images
    eval_dir = Path("eval_unified/images")
    all_images = sorted(eval_dir.glob("*"))[:args.n_images]

    n_layers = len(LAYER_TAPS)
    total = len(all_images) * n_layers
    print(f"\n{len(all_images)} images × {n_layers} layer taps = {total} generations")
    print(f"Layers: {[name for _, name in LAYER_TAPS]}")
    print(f"Output: {output_dir}\n")

    done = 0
    t0 = time.time()

    for img_idx, img_path in enumerate(all_images):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        scale = min(1024 / max(w, h), 1.0)
        out_w = max(round(w * scale) // 16 * 16, 16)
        out_h = max(round(h * scale) // 16 * 16, 16)

        # Save input
        img_resized = img.resize((out_w, out_h), Image.LANCZOS)
        img_resized.save(output_dir / f"{img_idx:03d}_input.png")

        # Single VL forward pass — get all layer outputs
        layer_embeds = encode_image_vl_all_layers(
            pipe, img, device, max_pixels=args.max_pixels
        )

        # Generate from each layer's embeddings
        for layer_idx, layer_name in LAYER_TAPS:
            embeds = layer_embeds[layer_idx]
            result = denoise_with_embeds(pipe, embeds, out_h, out_w, args.seed)
            result.save(output_dir / f"{img_idx:03d}_{layer_name}.png")
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"  [{img_idx:03d}] {layer_name} done "
                  f"(tokens={embeds.shape[0]}, {done}/{total}, ~{eta/60:.0f}m left)")

    meta = {
        "n_images": len(all_images),
        "seed": args.seed,
        "max_pixels": args.max_pixels,
        "image_paths": [str(p) for p in all_images],
        "layer_taps": {name: {"hidden_states_idx": idx, "description": f"After LLM layer {idx-1}" if idx > 0 else "Post-PatchMerger (pre-LLM)"}
                       for idx, name in LAYER_TAPS},
        "baseline_layer": "llm_layer34 (hidden_states[-2])",
        "total_hidden_states": 37,
        "llm_layers": 36,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone! {total} images in {(time.time() - t0)/60:.1f} min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
