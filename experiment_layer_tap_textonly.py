#!/usr/bin/env python3
"""
Experiment: Text-only conditioning from vision-aware LLM layers.

Images + text go through full VL forward pass (cross-attention in LLM),
but we STRIP the vision tokens from the output and only keep text tokens
as conditioning for DiT.

The text tokens have attended to vision tokens via LLM self-attention,
so they carry image-aware semantic info — but in a purely textual form.

This tests whether the text channel alone (after seeing images) carries
enough signal to guide generation.

Layers tested: 12, 18, 24, 30, 34 (baseline), 35 (final)

Usage:
    python experiment_layer_tap_textonly.py --n_pairs 15 --seed 42
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


@torch.no_grad()
def encode_textonly_at_layer(pipe, images, text, device, layer_idx, max_pixels=512*512):
    """Full VL forward with images+text, but return ONLY non-vision tokens.

    The LLM sees everything (images + text) via self-attention, so text tokens
    are vision-aware. We then strip out all vision tokens and return only the
    text/template tokens.
    """
    chat_content = []
    pil_images = []
    for img in images:
        img = _cap_resolution(img, max_pixels)
        chat_content.append({"type": "image", "image": img})
        pil_images.append(img)
    chat_content.append({"type": "text", "text": text if text else ""})

    messages = [{"role": "user", "content": chat_content}]
    text_str = pipe.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = pipe.vl_processor(
        text=[text_str], images=pil_images, return_tensors="pt"
    ).to(device)

    # Find vision token ranges
    image_ranges = _find_image_token_ranges(inputs["input_ids"])

    out = pipe.vl_model.model(**inputs, output_hidden_states=True)
    h = out.hidden_states[layer_idx][0]  # (seq_len, 2560)
    mask = inputs["attention_mask"][0].bool()

    # Build a mask that excludes vision tokens
    seq_len = h.shape[0]
    keep_mask = mask.clone()
    for (start, end) in image_ranges:
        keep_mask[start:end] = False

    return h[keep_mask]


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
    parser.add_argument("--output_dir", type=str, default="outputs/layer_tap_textonly")
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    parser.add_argument("--entries_file", type=str,
                        default="eval_unified/composition_light.jsonl")
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline("z-image-turbo", device=device, torch_dtype="bfloat16",
                         text_encoder="qwen3vl")

    # Load entries
    entries_path = Path(args.entries_file)
    entries = [json.loads(line) for line in entries_path.read_text().strip().split("\n")]
    entries = entries[:args.n_pairs]

    n_layers = len(LAYER_TAPS)
    total = len(entries) * n_layers
    print(f"\n{len(entries)} pairs × {n_layers} layers = {total} generations")
    print(f"Layers: {[name for _, name in LAYER_TAPS]}")
    print(f"Mode: text-only (vision tokens stripped after forward pass)")
    print(f"Output: {output_dir}\n")

    done = 0
    t0 = time.time()

    for pair_idx, entry in enumerate(entries):
        img_items = [item for item in entry if "img" in item]
        if len(img_items) < 2:
            print(f"  [{pair_idx:03d}] skipping — fewer than 2 images")
            continue

        images = [Image.open(item["img"]).convert("RGB") for item in img_items[:2]]
        txt_items = [item for item in entry if "txt" in item]
        prompt_text = txt_items[0]["txt"] if txt_items else ""

        # Save refs
        for ri, img in enumerate(images):
            ref_path = output_dir / f"{pair_idx:03d}_ref{ri}.png"
            if not ref_path.exists():
                img.save(ref_path)

        out_h, out_w = 1024, 1024

        for layer_idx, layer_name in LAYER_TAPS:
            embeds = encode_textonly_at_layer(
                pipe, images, prompt_text, device, layer_idx,
                max_pixels=args.max_pixels,
            )
            n_tokens = embeds.shape[0]
            result = denoise_with_embeds(pipe, embeds, out_h, out_w, args.seed)
            result.save(output_dir / f"{pair_idx:03d}_textonly_{layer_name}.png")
            done += 1

            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"  [{pair_idx:03d}] {layer_name} textonly done "
                  f"(text_tokens={n_tokens}, {done}/{total}, ~{eta/60:.0f}m left)")

    out_meta = {
        "n_pairs": len(entries),
        "seed": args.seed,
        "max_pixels": args.max_pixels,
        "layer_taps": {name: idx for idx, name in LAYER_TAPS},
        "mode": "textonly (vision tokens stripped)",
        "entries": entries[:args.n_pairs],
    }
    (output_dir / "meta.json").write_text(json.dumps(out_meta, indent=2))

    print(f"\nDone! {total} images in {(time.time() - t0)/60:.1f} min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
