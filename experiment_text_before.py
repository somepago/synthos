#!/usr/bin/env python3
"""
Experiment: Text-before vs text-after image embeddings.

In causal LLM attention, token order matters:
  - text-after  (current): [img_A] [img_B] [text] — images processed blind to text
  - text-before (new):     [text] [img_A] [img_B] — images attend to text, visual
    representations are "steered" by the text context

Tests whether putting the text prompt BEFORE images makes text guidance stronger,
since image tokens would process in the context of the preceding text instruction.

Uses light prompts + image pairs from composition_light.jsonl.

Usage:
    python experiment_text_before.py --n_pairs 15 --seed 42
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

ALPHA = 0.3


@torch.no_grad()
def encode_scale_text_order(pipe, images, text, device, layer_idx,
                             alpha=0.3, max_pixels=512*512, text_before=False):
    """Encode images + text with controllable text position.

    text_before=False (default): [img_A] [img_B] [text]  — current behavior
    text_before=True:            [text] [img_A] [img_B]  — text primes image processing
    """
    weights = [alpha, 1.0 - alpha] if len(images) == 2 else [1.0/len(images)] * len(images)

    chat_content = []
    pil_images = []

    if text_before and text:
        chat_content.append({"type": "text", "text": text})

    for img in images:
        img = _cap_resolution(img, max_pixels)
        chat_content.append({"type": "image", "image": img})
        pil_images.append(img)

    if not text_before:
        chat_content.append({"type": "text", "text": text if text else ""})

    messages = [{"role": "user", "content": chat_content}]
    text_str = pipe.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = pipe.vl_processor(
        text=[text_str], images=pil_images, return_tensors="pt"
    ).to(device)

    image_ranges = _find_image_token_ranges(inputs["input_ids"])

    out = pipe.vl_model.model(**inputs, output_hidden_states=True)
    h = out.hidden_states[layer_idx][0].clone()
    mask = inputs["attention_mask"][0].bool()

    for (start, end), w in zip(image_ranges, weights):
        h[start:end] *= w

    return h[mask]


def denoise_with_embeds(pipe, prompt_embeds, height, width, seed, num_steps=8):
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
    parser.add_argument("--output_dir", type=str, default="outputs/text_before")
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    parser.add_argument("--entries_file", type=str,
                        default="eval_unified/composition_light.jsonl")
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline("z-image-turbo", device=device, torch_dtype="bfloat16",
                         text_encoder="qwen3vl")

    entries_path = Path(args.entries_file)
    entries = [json.loads(line) for line in entries_path.read_text().strip().split("\n")]
    entries = entries[:args.n_pairs]

    # layer 34 only (baseline) — focused comparison
    layer_idx = 35  # hidden_states index for layer 34
    layer_name = "layer34"

    total = len(entries) * 2  # text-after + text-before
    print(f"\n{len(entries)} pairs × 2 modes (text-after, text-before) = {total} generations")
    print(f"Layer: {layer_name}, alpha: {ALPHA}")
    print(f"Output: {output_dir}\n")

    done = 0
    t0 = time.time()

    for pair_idx, entry in enumerate(entries):
        img_items = [item for item in entry if "img" in item]
        if len(img_items) < 2:
            continue

        images = [Image.open(item["img"]).convert("RGB") for item in img_items[:2]]
        txt_items = [item for item in entry if "txt" in item]
        prompt_text = txt_items[0]["txt"] if txt_items else ""

        if not prompt_text:
            print(f"  [{pair_idx:03d}] skipping — no text prompt")
            continue

        # Save refs
        for ri, img in enumerate(images):
            ref_path = output_dir / f"{pair_idx:03d}_ref{ri}.png"
            if not ref_path.exists():
                img.save(ref_path)

        out_h, out_w = 1024, 1024

        # text-after (current behavior)
        embeds_after = encode_scale_text_order(
            pipe, images, prompt_text, device, layer_idx,
            alpha=ALPHA, max_pixels=args.max_pixels, text_before=False,
        )
        result = denoise_with_embeds(pipe, embeds_after, out_h, out_w, args.seed)
        result.save(output_dir / f"{pair_idx:03d}_text_after.png")
        done += 1

        # text-before (new)
        embeds_before = encode_scale_text_order(
            pipe, images, prompt_text, device, layer_idx,
            alpha=ALPHA, max_pixels=args.max_pixels, text_before=True,
        )
        result = denoise_with_embeds(pipe, embeds_before, out_h, out_w, args.seed)
        result.save(output_dir / f"{pair_idx:03d}_text_before.png")
        done += 1

        elapsed = time.time() - t0
        eta = elapsed / done * (total - done)
        print(f"  [{pair_idx:03d}] \"{prompt_text[:50]}\" done "
              f"({done}/{total}, ~{eta/60:.0f}m left)")

    out_meta = {
        "n_pairs": len(entries),
        "seed": args.seed,
        "alpha": ALPHA,
        "max_pixels": args.max_pixels,
        "layer": layer_name,
        "modes": ["text_after ([img][img][text])", "text_before ([text][img][img])"],
        "entries": entries[:args.n_pairs],
    }
    (output_dir / "meta.json").write_text(json.dumps(out_meta, indent=2))

    print(f"\nDone! {total} images in {(time.time() - t0)/60:.1f} min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
