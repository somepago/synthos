#!/usr/bin/env python3
"""
Experiment: conditioning schedule for multi-image blending.

Instead of conditioning on both images from step 0, introduce the second
image's conditioning partway through denoising:
  - Steps 0..switch-1: condition on embeds_early
  - Steps switch..7:   condition on embeds_late

Three experiments per pair:
  E1 (a_to_ab): A only → A+B   (introduce B gradually)
  E2 (b_to_ab): B only → A+B   (introduce A gradually)
  E3 (a_to_b):  A only → B only (pure switch, no joint encoding)

Switch points: 0, 2, 4, 6, 8  (5 key points per experiment)

Usage:
    python experiment_cond_schedule.py \
        --dataset eval_unified/composition_light_notext.jsonl \
        --n_pairs 20 --seed 42
"""

from src import env_setup  # noqa: F401

import argparse
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image

from src.model_utils import load_pipeline
from src.diffusion import (
    encode_image_vl,
    encode_interleaved_vl,
    get_latent_shape,
    generate_noise,
    decode_latent,
)


def denoise_with_schedule(pipe, noise, embeds_early, embeds_late, switch_step,
                          num_steps=8, device="cuda", dtype=torch.bfloat16):
    """Denoise with conditioning schedule.

    Steps 0..switch_step-1: use embeds_early
    Steps switch_step..num_steps-1: use embeds_late
    """
    pipe.scheduler.set_timesteps(num_steps, denoising_strength=1.0, shift=None)
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    latents = noise.clone().to(device=device, dtype=dtype)
    inputs_shared = {"latents": latents}

    with torch.no_grad():
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            if progress_id < switch_step:
                embeds = embeds_early
            else:
                embeds = embeds_late

            inputs_posi = {"prompt_embeds": embeds}
            inputs_nega = {"prompt_embeds": embeds}
            timestep_t = timestep.unsqueeze(0).to(dtype=dtype, device=device)

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, 1.0,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep_t, progress_id=progress_id,
            )
            inputs_shared["latents"] = pipe.step(
                pipe.scheduler, progress_id=progress_id,
                noise_pred=noise_pred, **inputs_shared,
            )

        image = decode_latent(pipe, inputs_shared["latents"])
    return image


def load_pairs(jsonl_path, n_pairs, seed):
    """Load image pairs from JSONL."""
    base_dir = Path(jsonl_path).parent
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    random.seed(seed)
    if n_pairs < len(entries):
        entries = random.sample(entries, n_pairs)

    pairs = []
    for entry in entries:
        imgs = []
        for item in entry:
            if "img" in item:
                img_name = item["img"]
                for candidate in [Path(img_name), base_dir / img_name]:
                    if candidate.exists():
                        imgs.append(candidate)
                        break
        if len(imgs) >= 2:
            pairs.append((imgs[0], imgs[1]))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="eval_unified/composition_light_notext.jsonl")
    parser.add_argument("--n_pairs", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="outputs/cond_schedule")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_pixels", type=int, default=384 * 384)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16

    switch_points = [0, 2, 4, 6, 8]

    # Load pairs
    pairs = load_pairs(args.dataset, args.n_pairs, args.seed)
    print(f"Loaded {len(pairs)} pairs")

    # Load pipeline
    print("Loading pipeline...")
    pipe = load_pipeline("z-image-turbo", device=args.device, torch_dtype="bfloat16",
                         text_encoder="qwen3vl")

    shape = get_latent_shape(args.height, args.width)
    noise = generate_noise(args.seed, shape, args.device, dtype)

    all_results = {}
    total_imgs = len(pairs) * 3 * len(switch_points)
    img_count = 0
    t_start = time.time()

    for pair_idx, (path_a, path_b) in enumerate(pairs):
        pair_dir = out_dir / f"pair_{pair_idx:03d}"
        pair_dir.mkdir(exist_ok=True)

        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        img_a.save(pair_dir / "input_a.png")
        img_b.save(pair_dir / "input_b.png")

        print(f"\n[Pair {pair_idx}/{len(pairs)}] {path_a.name} + {path_b.name}")

        # Encode all three conditioning variants
        with torch.no_grad():
            embeds_a = encode_image_vl(pipe, img_a, args.device,
                                       max_pixels=args.max_pixels)
            embeds_b = encode_image_vl(pipe, img_b, args.device,
                                       max_pixels=args.max_pixels)
            content_ab = [{"img": img_a}, {"img": img_b}]
            embeds_ab = encode_interleaved_vl(pipe, content_ab, args.device,
                                              max_pixels=args.max_pixels)

        print(f"  tokens: A={embeds_a.shape[0]}, B={embeds_b.shape[0]}, AB={embeds_ab.shape[0]}")

        experiments = [
            ("a_to_ab", embeds_a, embeds_ab, "A→A+B"),
            ("b_to_ab", embeds_b, embeds_ab, "B→A+B"),
            ("a_to_b",  embeds_a, embeds_b,  "A→B"),
        ]

        for exp_name, e_early, e_late, exp_desc in experiments:
            for switch in switch_points:
                label = f"{exp_name}_s{switch}"
                img = denoise_with_schedule(pipe, noise, e_early, e_late, switch,
                                            num_steps=args.num_steps,
                                            device=args.device, dtype=dtype)
                img.save(pair_dir / f"{label}.png")
                img_count += 1

                elapsed = time.time() - t_start
                rate = elapsed / img_count
                remaining = rate * (total_imgs - img_count)
                print(f"  {label} done ({img_count}/{total_imgs}, ~{remaining/60:.0f}m left)")

        all_results[f"pair_{pair_idx:03d}"] = {
            "img_a": str(path_a), "img_b": str(path_b),
        }

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nDone. {img_count} images in {elapsed/60:.1f}m")
    print(f"Output: {out_dir}/")
    print(f"\nPer pair, compare across switch points:")
    print(f"  a_to_ab_s0 = A+B all steps (baseline multi-image)")
    print(f"  a_to_ab_s4 = A for 4 steps, then A+B for 4 steps")
    print(f"  a_to_ab_s8 = A only (single i2i)")
    print(f"  a_to_b_s4  = A for 4 steps, B for 4 steps (pure switch)")


if __name__ == "__main__":
    main()
