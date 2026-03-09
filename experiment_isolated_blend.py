#!/usr/bin/env python3
"""
Experiment: Cross-image attention blocking for multi-image blending.

Normal scale mode: both images go through VL LLM together, all tokens attend
to all prior tokens (causal). Image B tokens attend to Image A tokens → leakage.

Isolated mode: custom 4D attention mask blocks Image B from attending to Image A
(and vice versa). Text tokens still attend to both images. This prevents
object leakage while keeping text-image alignment.

Compares: scale (normal) vs isolated (cross-image blocked) at layer 34 (baseline).

Usage:
    python experiment_isolated_blend.py --n_pairs 15 --seed 42
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
def encode_isolated_at_layer(pipe, images, text, device, layer_idx,
                              alpha=0.3, max_pixels=512*512):
    """Encode images together but with cross-image attention BLOCKED.

    Creates a custom 4D attention mask:
    - Causal (standard)
    - Image A tokens CANNOT attend to Image B tokens (and vice versa)
    - Text tokens CAN attend to both images
    - Both images CAN attend to text tokens before them

    After encoding, visual tokens are scaled by alpha per image.
    """
    weights = [alpha, 1.0 - alpha] if len(images) == 2 else [1.0/len(images)] * len(images)

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

    # Find image token ranges
    image_ranges = _find_image_token_ranges(inputs["input_ids"])

    # Save original 1D attention mask for RoPE position computation
    orig_attention_mask = inputs["attention_mask"].clone()

    # Pre-compute position_ids with original 1D mask (before we replace it)
    position_ids, rope_deltas = pipe.vl_model.model.get_rope_index(
        inputs["input_ids"],
        inputs.get("image_grid_thw", None),
        inputs.get("video_grid_thw", None),
        attention_mask=orig_attention_mask,
    )

    # Build custom 4D attention mask with cross-image blocking
    seq_len = inputs["input_ids"].shape[1]
    dtype = pipe.vl_model.dtype

    # Start with causal mask (lower triangular)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    # Block cross-image attention
    if len(image_ranges) >= 2:
        for i in range(len(image_ranges)):
            for j in range(len(image_ranges)):
                if i != j:
                    qi_start, qi_end = image_ranges[i]
                    kj_start, kj_end = image_ranges[j]
                    # Image i's queries cannot attend to Image j's keys
                    causal[qi_start:qi_end, kj_start:kj_end] = False

    # Convert to additive float mask: 0.0 = attend, -inf = block
    min_val = torch.finfo(dtype).min
    attn_mask = torch.where(causal, 0.0, min_val).to(dtype=dtype)
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    # Replace the attention mask with our 4D mask, pass pre-computed position_ids
    inputs["attention_mask"] = attn_mask
    inputs["position_ids"] = position_ids

    out = pipe.vl_model.model(**inputs, output_hidden_states=True)
    h = out.hidden_states[layer_idx][0].clone()

    valid = orig_attention_mask[0].bool()

    # Scale visual tokens by weight
    for (start, end), w in zip(image_ranges, weights):
        h[start:end] *= w

    return h[valid]


@torch.no_grad()
def encode_scale_at_layer(pipe, images, text, device, layer_idx,
                           alpha=0.3, max_pixels=512*512):
    """Normal scale mode (cross-attention enabled) for comparison."""
    weights = [alpha, 1.0 - alpha] if len(images) == 2 else [1.0/len(images)] * len(images)

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
    parser.add_argument("--output_dir", type=str, default="outputs/isolated_blend")
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

    # Use baseline layer (34) + layer 24 (showed interesting results)
    layer_taps = [(35, "layer34"), (25, "layer24")]

    total = len(entries) * len(layer_taps) * 2  # scale + isolated per layer
    print(f"\n{len(entries)} pairs × {len(layer_taps)} layers × 2 modes = {total} generations")
    print(f"Modes: scale (normal cross-attention) vs isolated (cross-image blocked)")
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

        # Save refs
        for ri, img in enumerate(images):
            ref_path = output_dir / f"{pair_idx:03d}_ref{ri}.png"
            if not ref_path.exists():
                img.save(ref_path)

        out_h, out_w = 1024, 1024

        for layer_idx, layer_name in layer_taps:
            # Normal scale mode
            embeds_scale = encode_scale_at_layer(
                pipe, images, prompt_text, device, layer_idx,
                alpha=ALPHA, max_pixels=args.max_pixels,
            )
            result = denoise_with_embeds(pipe, embeds_scale, out_h, out_w, args.seed)
            result.save(output_dir / f"{pair_idx:03d}_scale_{layer_name}.png")
            done += 1

            # Isolated mode (cross-image blocked)
            embeds_iso = encode_isolated_at_layer(
                pipe, images, prompt_text, device, layer_idx,
                alpha=ALPHA, max_pixels=args.max_pixels,
            )
            result = denoise_with_embeds(pipe, embeds_iso, out_h, out_w, args.seed)
            result.save(output_dir / f"{pair_idx:03d}_isolated_{layer_name}.png")
            done += 1

            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"  [{pair_idx:03d}] {layer_name} scale+isolated done "
                  f"({done}/{total}, ~{eta/60:.0f}m left)")

    out_meta = {
        "n_pairs": len(entries),
        "seed": args.seed,
        "alpha": ALPHA,
        "max_pixels": args.max_pixels,
        "layer_taps": {name: idx for idx, name in layer_taps},
        "modes": ["scale (normal cross-attention)", "isolated (cross-image blocked)"],
        "entries": entries[:args.n_pairs],
    }
    (output_dir / "meta.json").write_text(json.dumps(out_meta, indent=2))

    print(f"\nDone! {total} images in {(time.time() - t0)/60:.1f} min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
