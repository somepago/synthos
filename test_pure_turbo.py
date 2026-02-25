#!/usr/bin/env python3
"""Native pipeline t2i on all eval prompts — compare against VL gens in t2i_all."""

from src import env_setup  # noqa: F401

import json
import signal
import shutil
from pathlib import Path
from PIL import Image
from src.model_utils import load_pipeline

TIMEOUT_PER_IMAGE = 120  # seconds — kill if single image takes longer


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Image generation timed out")


def round_to_16(w, h, max_size=768):
    """Resize preserving AR, round to 16px multiples, cap at max_size."""
    scale = min(max_size / max(w, h), 1.0)
    w, h = round(w * scale), round(h * scale)
    return max(w // 16 * 16, 16), max(h // 16 * 16, 16)


with open("outputs/baselines_feb25/t2i_all_512/meta.json") as f:
    meta = json.load(f)

prompts = meta["prompts"]
image_paths = meta.get("image_paths", [])
n = len(prompts)

out_dir = Path("outputs/baselines_feb25/native_t2i")
out_dir.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_SIZE = 1024
STEPS = 8

print(f"=== Native pipeline (qwen3) — {n} prompts, AR-matched ===")
pipe = load_pipeline("z-image-turbo", device="cuda", torch_dtype="bfloat16",
                     text_encoder="qwen3")

signal.signal(signal.SIGALRM, timeout_handler)

for i, prompt in enumerate(prompts):
    out_path = out_dir / f"t2i_{i:03d}.png"
    if out_path.exists():
        print(f"  SKIP [{i:03d}]")
        continue

    # Match AR from original input image
    if i < len(image_paths) and image_paths[i]:
        img = Image.open(image_paths[i])
        w, h = img.size
        img.close()
        W, H = round_to_16(w, h, MAX_SIZE)
    else:
        H, W = MAX_SIZE, MAX_SIZE

    print(f"  [{i:03d}] {H}x{W} — {prompt[:60]}...")
    try:
        signal.alarm(TIMEOUT_PER_IMAGE)
        img = pipe(
            prompt=prompt,
            height=H, width=W,
            seed=SEED,
            num_inference_steps=STEPS,
            cfg_scale=1.0,
        )
        signal.alarm(0)
        img.save(out_path)
    except TimeoutError:
        print(f"  TIMEOUT [{i:03d}] — skipped after {TIMEOUT_PER_IMAGE}s")
        continue
    except Exception as e:
        signal.alarm(0)
        print(f"  ERROR [{i:03d}]: {e}")
        continue

signal.alarm(0)

# Copy input images for reference
for i in range(n):
    src = Path(f"outputs/baselines_feb25/t2i_all_512/input_{i:03d}.png")
    dst = out_dir / f"input_{i:03d}.png"
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)

# Save meta
out_meta = {
    "n": n,
    "model": "z-image-turbo",
    "text_encoder": "qwen3",
    "num_steps": STEPS,
    "cfg_scale": 1.0,
    "seed": SEED,
    "max_size": MAX_SIZE,
    "has_t2i": True,
    "has_i2i": False,
    "has_input": True,
    "prompts": prompts,
    "image_paths": image_paths,
}
(out_dir / "meta.json").write_text(json.dumps(out_meta, indent=2))

print(f"\nDone. Outputs: {out_dir}/")
