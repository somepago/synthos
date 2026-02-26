#!/usr/bin/env python3
"""
Text-guided variation ablation.

For a curated subset of eval images, generates:
1. Image-only baseline (encode_image_vl)
2. Image + text prompt variations (encode_interleaved_vl)

All at medium token level (max_pixels=384*384) to isolate the text effect.

Usage:
    python run_text_variations.py
    python run_text_variations.py --dry_run   # print plan without generating

Output: outputs/baselines_feb25/vary_text/
"""

from src import env_setup  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.model_utils import load_pipeline, get_defaults
from src.diffusion import encode_image_vl, encode_interleaved_vl
from inference import denoise_loop

# --- Configuration ---

MAX_PIXELS = 384 * 384  # medium token level
OUTPUT_DIR = Path("outputs/baselines_feb25/vary_text")
IMAGE_DIR = Path("eval_unified/images")

# Curated images — diverse subjects for text-guided variation
# (index, filename, short description for reference)
CURATED_IMAGES = [
    (0,  "000.jpg",  "plumber illustration"),
    (1,  "001.jpg",  "fashion runway model"),
    (3,  "003.jpg",  "naval officer"),
    (4,  "004.jpg",  "yellow robot"),
    (5,  "005.jpg",  "scuba diver"),
    (7,  "007.jpg",  "surreal interior painting"),
    (11, "011.jpg",  "rhino pencil sketch"),
    (12, "012.jpeg", "pastry chef portrait"),
    (15, "015.jpeg", "orange fashion photo"),
    (16, "016.jpeg", "harbor red ship"),
    (17, "017.jpeg", "face closeup"),
    (20, "020.jpeg", "MJ art"),
]

# Text prompts for variation — short, directive phrases
TEXT_VARIATIONS = {
    # Style transfer
    "watercolor":  "in watercolor style",
    "pencil":      "as a pencil sketch",
    "cyberpunk":   "in cyberpunk neon style",
    "oil_paint":   "as an oil painting",
    # Scene modification
    "sunset":      "at sunset",
    "winter":      "in winter with snow",
    "underwater":  "underwater",
    # Subject alteration
    "as_cat":      "but as a cat",
    "top_hat":     "wearing a top hat",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Print plan only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Validate images exist
    valid_images = []
    for idx, fname, desc in CURATED_IMAGES:
        path = IMAGE_DIR / fname
        if not path.exists():
            # Try other extensions
            found = list(IMAGE_DIR.glob(f"{idx:03d}.*"))
            if found:
                path = found[0]
                fname = path.name
            else:
                print(f"  SKIP: {fname} not found")
                continue
        valid_images.append((idx, fname, desc, path))

    n_images = len(valid_images)
    n_variants = 1 + len(TEXT_VARIATIONS)  # baseline + text variants
    total = n_images * n_variants

    print(f"Text-guided variation ablation")
    print(f"  Images: {n_images}")
    print(f"  Variants per image: {n_variants} (1 baseline + {len(TEXT_VARIATIONS)} text)")
    print(f"  Total generations: {total}")
    print(f"  max_pixels: {MAX_PIXELS}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    for idx, fname, desc, path in valid_images:
        print(f"  [{idx:03d}] {fname} — {desc}")
    print()

    for key, text in TEXT_VARIATIONS.items():
        print(f"  {key}: \"{text}\"")
    print()

    if args.dry_run:
        print("(dry run — no generation)")
        return

    # Load pipeline
    defaults = get_defaults("z-image-turbo")
    num_steps = defaults["num_inference_steps"]
    cfg_scale = defaults["cfg_scale"]

    pipe = load_pipeline("z-image-turbo", device=args.device, torch_dtype="bfloat16",
                         text_encoder="qwen3vl")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing completions
    meta_path = OUTPUT_DIR / "meta.json"
    if meta_path.exists():
        print(f"WARNING: {meta_path} already exists. Resuming — skipping existing files.")

    t0 = time.time()
    generated = 0
    skipped = 0

    for idx, fname, desc, path in tqdm(valid_images, desc="images"):
        img = Image.open(path).convert("RGB")
        prefix = f"{idx:03d}"

        # Save input reference
        input_path = OUTPUT_DIR / f"{prefix}_input.png"
        if not input_path.exists():
            # Resize for display (match output res)
            from inference import resolve_resolution, round_to_16
            max_size = max(args.height, args.width)
            w, h = img.size
            rw, rh = round_to_16(w, h, max_size)
            img.resize((rw, rh), Image.LANCZOS).save(input_path)

        # 1. Baseline: image only
        out_path = OUTPUT_DIR / f"{prefix}_baseline.png"
        if out_path.exists():
            skipped += 1
        else:
            prompt_embeds = encode_image_vl(pipe, img, pipe.device,
                                            max_pixels=MAX_PIXELS)
            result = denoise_loop(pipe, prompt_embeds, args.height, args.width,
                                  args.seed, num_steps, cfg_scale)
            result.save(out_path)
            generated += 1

        # 2. Text-guided variants
        for key, text in TEXT_VARIATIONS.items():
            out_path = OUTPUT_DIR / f"{prefix}_{key}.png"
            if out_path.exists():
                skipped += 1
                continue

            content = [{"img": img}, {"txt": text}]
            prompt_embeds = encode_interleaved_vl(pipe, content, pipe.device,
                                                   max_pixels=MAX_PIXELS)
            result = denoise_loop(pipe, prompt_embeds, args.height, args.width,
                                  args.seed, num_steps, cfg_scale)
            result.save(out_path)
            generated += 1

    elapsed = time.time() - t0
    print(f"\nDone: {generated} generated, {skipped} skipped in {elapsed:.1f}s")

    # Save metadata
    meta = {
        "n": len(valid_images),
        "model": "z-image-turbo",
        "text_encoder": "qwen3vl",
        "num_steps": num_steps,
        "cfg_scale": cfg_scale,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "max_pixels": MAX_PIXELS,
        "type": "text_variation",
        "images": [
            {"idx": idx, "file": fname, "desc": desc}
            for idx, fname, desc, _ in valid_images
        ],
        "text_variants": TEXT_VARIATIONS,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
