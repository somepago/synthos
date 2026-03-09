#!/usr/bin/env python3
"""
Experiment: Text-guided image variations with different internal VL layers.

For each image+text combo, encode via interleaved VL (image + text prompt together),
but extract hidden states from different layers instead of the default [-2].

Layers tested: 12, 18, 24, 30, 34 (baseline), 35 (final)

Text prompts: baseline (image only), watercolor, pencil, cyberpunk, oil_paint,
              sunset, winter, underwater

Uses same 12 curated images as existing text variation experiment.

Usage:
    python experiment_layer_tap_text.py --seed 42
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
    (13, "layer12"),
    (19, "layer18"),
    (25, "layer24"),
    (31, "layer30"),
    (35, "layer34"),   # baseline (hidden_states[-2])
    (36, "layer35"),   # final
]

TEXT_PROMPTS = {
    "baseline": "",
    "watercolor": "in watercolor style",
    "pencil": "as a pencil sketch",
    "cyberpunk": "in cyberpunk neon style",
    "oil_paint": "as an oil painting",
    "sunset": "at sunset",
    "winter": "in winter with snow",
    "underwater": "underwater",
}

MAX_PIXELS_TEXT = 147456  # 384x384, matches existing text variation experiment


@torch.no_grad()
def encode_interleaved_at_layer(pipe, image, text, device, layer_idx, max_pixels):
    """Encode image+text interleaved, extract at layer_idx."""
    image = _cap_resolution(image, max_pixels)

    chat_content = [{"type": "image", "image": image}]
    if text:
        chat_content.append({"type": "text", "text": text})
    else:
        chat_content.append({"type": "text", "text": ""})

    messages = [{"role": "user", "content": chat_content}]
    text_str = pipe.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = pipe.vl_processor(
        text=[text_str], images=[image], return_tensors="pt"
    ).to(device)

    out = pipe.vl_model.model(**inputs, output_hidden_states=True)
    h = out.hidden_states[layer_idx][0]
    mask = inputs["attention_mask"][0].bool()
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/layer_tap_text")
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    pipe = load_pipeline("z-image-turbo", device=device, torch_dtype="bfloat16",
                         text_encoder="qwen3vl")

    # Load same images as existing text variation experiment
    text_meta_path = Path("outputs/baselines_feb25/vary_text/meta.json")
    if not text_meta_path.exists():
        print("ERROR: No existing text variation meta.json found.")
        return

    text_meta = json.loads(text_meta_path.read_text())
    eval_dir = Path("eval_unified/images")

    image_infos = text_meta["images"]
    n_layers = len(LAYER_TAPS)
    n_prompts = len(TEXT_PROMPTS)
    total = len(image_infos) * n_layers * n_prompts
    print(f"\n{len(image_infos)} images × {n_layers} layers × {n_prompts} prompts = {total} generations")
    print(f"Layers: {[name for _, name in LAYER_TAPS]}")
    print(f"Prompts: {list(TEXT_PROMPTS.keys())}")
    print(f"Output: {output_dir}\n")

    done = 0
    t0 = time.time()

    for img_info in image_infos:
        img_idx = img_info["idx"]
        img_file = img_info["file"]
        img_path = eval_dir / img_file
        if not img_path.exists():
            print(f"  [{img_idx:03d}] skipping — {img_path} not found")
            continue

        img = Image.open(img_path).convert("RGB")

        # Save input
        input_path = output_dir / f"{img_idx:03d}_input.png"
        if not input_path.exists():
            w, h = img.size
            scale = min(1024 / max(w, h), 1.0)
            out_w = max(round(w * scale) // 16 * 16, 16)
            out_h = max(round(h * scale) // 16 * 16, 16)
            img.resize((out_w, out_h), Image.LANCZOS).save(input_path)

        for layer_idx, layer_name in LAYER_TAPS:
            for prompt_key, prompt_text in TEXT_PROMPTS.items():
                embeds = encode_interleaved_at_layer(
                    pipe, img, prompt_text, device, layer_idx,
                    max_pixels=MAX_PIXELS_TEXT,
                )
                result = denoise_with_embeds(pipe, embeds, 1024, 1024, args.seed)
                result.save(output_dir / f"{img_idx:03d}_{layer_name}_{prompt_key}.png")
                done += 1

                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"  [{img_idx:03d}] {layer_name}/{prompt_key} done "
                      f"({done}/{total}, ~{eta/60:.0f}m left)")

    out_meta = {
        "seed": args.seed,
        "max_pixels": MAX_PIXELS_TEXT,
        "layer_taps": {name: idx for idx, name in LAYER_TAPS},
        "prompts": TEXT_PROMPTS,
        "images": image_infos,
    }
    (output_dir / "meta.json").write_text(json.dumps(out_meta, indent=2))

    print(f"\nDone! {total} images in {(time.time() - t0)/60:.1f} min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
