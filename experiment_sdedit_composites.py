#!/usr/bin/env python3
"""
Experiment: SDEdit-style composite cleanup.

Instead of starting from pure noise (current i2i), encode the composite
through VAE to get clean latents, add noise to an intermediate timestep,
then denoise with VL conditioning. This preserves spatial layout while
letting the model clean up seams and rough edges.

Ablation over denoising_strength:
  - 0.2: Very subtle cleanup (mostly preserves composite)
  - 0.4: Moderate cleanup
  - 0.6: Stronger reinterpretation
  - 0.8: Heavy reinterpretation (close to pure i2i)
  - 1.0: Pure noise (current i2i baseline)

Also sweeps max_pixels for VL encoding at each strength.

Usage:
    python experiment_sdedit_composites.py --seed 42
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
    encode_image_vl, encode_text_vl, encode_image_vae,
    _cap_resolution, get_latent_shape, generate_noise,
    _denoise_step, _decode_final,
)


@torch.no_grad()
def encode_image_vl_at_layer(pipe, image, device, layer_idx, max_pixels=768*768):
    """Like encode_image_vl but taps a specific hidden_states layer."""
    image = _cap_resolution(image, max_pixels)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": ""},
    ]}]
    text = pipe.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = pipe.vl_processor(text=[text], images=[image], return_tensors="pt").to(device)
    out = pipe.vl_model.model(**inputs, output_hidden_states=True)
    h = out.hidden_states[layer_idx][0]
    mask = inputs["attention_mask"][0].bool()
    return h[mask]

DENOISE_STRENGTHS = [0.2, 0.4, 0.6, 0.8, 1.0]
MAX_PIXELS_LEVELS = [
    (147456, "medium"),    # 384x384
    (589824, "default"),   # 768x768
]


def sdedit_generate(pipe, prompt_embeds, image, height, width, seed,
                    denoising_strength, num_steps=8):
    """SDEdit: encode image via VAE, add noise, denoise with VL conditioning."""
    shape = get_latent_shape(height, width)
    noise = generate_noise(seed, shape, pipe.device, pipe.torch_dtype)

    if denoising_strength >= 1.0:
        # Pure noise — standard i2i
        latents = noise
    else:
        # Encode image through VAE
        resized = image.resize((width, height), Image.LANCZOS)
        z_0 = encode_image_vae(pipe, resized)

        # Set timesteps with denoising strength to get the right schedule
        pipe.scheduler.set_timesteps(num_steps, denoising_strength=denoising_strength, shift=None)
        # Add noise at the first timestep of the truncated schedule
        t_start = pipe.scheduler.timesteps[0]
        latents = pipe.scheduler.add_noise(z_0, noise, t_start)

    pipe.scheduler.set_timesteps(num_steps, denoising_strength=denoising_strength, shift=None)
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    shared = {"latents": latents}
    posi = {"prompt_embeds": prompt_embeds}
    nega = {}

    with torch.no_grad():
        for pid, ts in enumerate(pipe.scheduler.timesteps):
            _denoise_step(pipe, 1.0, shared, posi, nega, models, ts, pid)
        return _decode_final(pipe, shared)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/sdedit_composites")
    parser.add_argument("--layer", type=int, default=None,
                        help="hidden_states index to tap (e.g. 25 for layer24). "
                             "Default: None (uses encode_image_vl which taps hidden_states[-2]=layer34)")
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline("z-image-turbo", device=device, torch_dtype="bfloat16",
                         text_encoder="qwen3vl")

    composites_dir = Path("eval_obj_stitch/composites")
    composites = sorted(composites_dir.glob("*_composite.png"))

    total = len(composites) * len(DENOISE_STRENGTHS) * (len(MAX_PIXELS_LEVELS) + 1)  # +1 for nocond
    print(f"\n{len(composites)} composites × {len(DENOISE_STRENGTHS)} strengths × {len(MAX_PIXELS_LEVELS)+1} cond levels = {total} generations")
    print(f"Strengths: {DENOISE_STRENGTHS}")
    print(f"Max pixels: {[(mp, name) for mp, name in MAX_PIXELS_LEVELS]}")
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

        # Save input
        img_resized = img.resize((out_w, out_h), Image.LANCZOS)
        input_path = output_dir / f"{img_idx:03d}_{name}_input.png"
        if not input_path.exists():
            img_resized.save(input_path)

        # No-conditioning row: empty text, pure SDEdit spatial preservation
        nocond_embeds = encode_text_vl(pipe, "", device)
        for strength in DENOISE_STRENGTHS:
            s_str = f"{strength:.1f}".replace(".", "")
            out_path = output_dir / f"{img_idx:03d}_{name}_nocond_s{s_str}.png"
            if not out_path.exists():
                result = sdedit_generate(
                    pipe, nocond_embeds, img, out_h, out_w, args.seed,
                    denoising_strength=strength,
                )
                result.save(out_path)
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"  [{name}] nocond strength={strength:.1f} done "
                  f"({done}/{total}, ~{eta/60:.0f}m left)")

        for max_pixels, mp_name in MAX_PIXELS_LEVELS:
            # Encode once per max_pixels level
            if args.layer is not None:
                vl_embeds = encode_image_vl_at_layer(pipe, img, device, args.layer, max_pixels=max_pixels)
            else:
                vl_embeds = encode_image_vl(pipe, img, device, max_pixels=max_pixels)

            for strength in DENOISE_STRENGTHS:
                s_str = f"{strength:.1f}".replace(".", "")
                out_path = output_dir / f"{img_idx:03d}_{name}_{mp_name}_s{s_str}.png"
                if out_path.exists():
                    done += 1
                    continue
                result = sdedit_generate(
                    pipe, vl_embeds, img, out_h, out_w, args.seed,
                    denoising_strength=strength,
                )
                result.save(out_path)
                done += 1

                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"  [{name}] {mp_name} strength={strength:.1f} done "
                      f"({done}/{total}, ~{eta/60:.0f}m left)")

    meta = {
        "n_composites": len(composites),
        "seed": args.seed,
        "denoise_strengths": DENOISE_STRENGTHS,
        "max_pixels_levels": {name: mp for mp, name in MAX_PIXELS_LEVELS},
        "composites": [str(p) for p in composites],
        "names": [p.stem.replace("_composite", "") for p in composites],
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone! {total} images in {(time.time() - t0)/60:.1f} min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
