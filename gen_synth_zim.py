#!/usr/bin/env python3
"""Generate t2i images from eval_prompts.txt with varied aspect ratios."""

from src import env_setup  # noqa: F401

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.model_utils import load_pipeline
from src.diffusion import encode_text_vl
from inference import denoise_loop

PROMPTS_FILE = "/home/gnan/projects/diffscapes/data/eval_prompts.txt"
OUTPUT_DIR = Path("/home/gnan/projects/synthos/eval_unified/images_synth_zim")
SEED = 42

# Resolutions: (width, height) — all multiples of 16
SQ = (1024, 1024)   # square
PT = (768, 1024)     # portrait 3:4
LD = (1024, 768)     # landscape 4:3
WD = (1024, 576)     # wide cinematic 16:9

# Per-prompt aspect ratio assignment based on content
# 0-indexed to match line numbers
ASPECT_MAP = {
    0: PT,   # poster/album cover — vertical sections
    1: LD,   # garden scene with dogs
    2: PT,   # close-up portrait
    3: PT,   # movie poster portrait
    4: LD,   # two playing cards side by side
    5: LD,   # man in car
    6: PT,   # silhouette in profile
    7: LD,   # cinematic stop-motion scene
    8: SQ,   # 3D art toy character
    9: PT,   # character on stool
    10: SQ,  # repeating pattern
    11: LD,  # horse + leopard running
    12: SQ,  # close-up beauty portrait
    13: PT,  # military portrait
    14: PT,  # glamorous studio portrait
    15: LD,  # oil painting cityscape
    16: SQ,  # zebra face
    17: SQ,  # two cats illustration
    18: SQ,  # logo design
    19: LD,  # close-up peonies
    20: SQ,  # frog illustration
    21: WD,  # cinematic still frame, train
    22: SQ,  # white horse
    23: LD,  # cat with mountains
    24: PT,  # steampunk poker card
    25: PT,  # travel poster
    26: LD,  # man in armchair
    27: PT,  # knitted doll
    28: SQ,  # hand with spoon
    29: LD,  # macbook in forest
    30: LD,  # typography composition
    31: PT,  # korean couple full body
    32: SQ,  # close-up perfume
    33: SQ,  # cat on stage
    34: WD,  # tiny people on plant — wide scene
    35: PT,  # fashion editorial (prompt says vertical 4:5)
    36: PT,  # studio portrait
    37: PT,  # fashion photoshoot
    38: PT,  # studio portrait with flowers
    39: PT,  # fashion portrait
    40: PT,  # fashion portrait
    41: SQ,  # vector heart
    42: SQ,  # retro graphic design
    43: PT,  # t-shirt design
    44: LD,  # wildlife illustration landscape
    45: PT,  # fantasy portrait with wings
    46: PT,  # frontal shot flower hybrid
    47: SQ,  # ethereal woman with orb
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prompts = [l.strip() for l in Path(PROMPTS_FILE).read_text().splitlines() if l.strip()]
    print(f"{len(prompts)} prompts")

    # Count AR distribution
    ar_counts = {}
    for i in range(len(prompts)):
        w, h = ASPECT_MAP.get(i, SQ)
        key = f"{w}x{h}"
        ar_counts[key] = ar_counts.get(key, 0) + 1
    print(f"AR distribution: {ar_counts}")

    device = "cuda"
    pipe = load_pipeline("z-image-turbo", device=device, torch_dtype="bfloat16",
                         text_encoder="qwen3vl")

    t0 = time.time()
    for i, prompt in enumerate(tqdm(prompts, desc="generating")):
        out_path = OUTPUT_DIR / f"t2i_{i:03d}.png"
        if out_path.exists():
            continue

        w, h = ASPECT_MAP.get(i, SQ)
        embeds = encode_text_vl(pipe, prompt, device)
        image = denoise_loop(pipe, embeds, h, w, SEED, num_steps=8, cfg_scale=1.0)
        image.save(out_path)

    elapsed = time.time() - t0
    print(f"\nDone! {len(prompts)} images in {elapsed:.1f}s")

    meta = {
        "n": len(prompts),
        "seed": SEED,
        "model": "z-image-turbo",
        "text_encoder": "qwen3vl",
        "num_steps": 8,
        "cfg_scale": 1.0,
        "prompts": prompts,
        "resolutions": {f"{i:03d}": list(ASPECT_MAP.get(i, SQ)) for i in range(len(prompts))},
    }
    (OUTPUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
