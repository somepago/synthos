#!/usr/bin/env python3
"""
Experiment: Swap Qwen3-VL-4B's ViT blocks with Qwen3-VL-Embedding-2B's ViT blocks.

Both have identical ViT architecture (1024-dim, 24 layers, 16 heads).
We keep our PatchMerger + LLM (Z-Image spliced weights) and only swap
the ViT blocks + patch_embed. The Embedding-2B ViT was trained for
retrieval — it may encode images differently, producing different
generation styles.

For each eval image, generates:
  - baseline: normal VL i2i (original ViT)
  - swapped: same pipeline but with Embedding-2B ViT blocks

Usage:
    python experiment_vit_swap.py --n_images 10 --seed 42
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
    encode_image_vl, get_latent_shape, generate_noise,
    _denoise_step, _decode_final,
)


def swap_vit_blocks(pipe, device, dtype):
    """Replace VL model's ViT blocks + patch_embed with Embedding-2B's.

    Keeps: PatchMerger, LLM (spliced Z-Image weights)
    Swaps: patch_embed, blocks (ViT transformer layers)
    """
    from transformers import Qwen3VLForConditionalGeneration

    print("Loading Qwen3-VL-Embedding-2B for ViT swap...")
    emb_model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-Embedding-2B", torch_dtype=dtype,
    )

    emb_vit = emb_model.model.visual
    our_vit = pipe.vl_model.model.visual

    # Verify compatibility
    assert len(emb_vit.blocks) == len(our_vit.blocks), \
        f"Block count mismatch: {len(emb_vit.blocks)} vs {len(our_vit.blocks)}"

    # Swap patch_embed
    our_vit.patch_embed.load_state_dict(emb_vit.patch_embed.state_dict())
    print(f"  Swapped patch_embed")

    # Swap all ViT blocks
    for i in range(len(our_vit.blocks)):
        our_vit.blocks[i].load_state_dict(emb_vit.blocks[i].state_dict())
    print(f"  Swapped {len(our_vit.blocks)} ViT blocks")

    # Do NOT swap merger (different output dim: 2048 vs 2560)
    # Do NOT swap LLM (we keep Z-Image's spliced weights)

    del emb_model
    torch.cuda.empty_cache()
    print("  ViT swap complete (PatchMerger + LLM unchanged)")


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
    parser.add_argument("--n_images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/vit_swap_exp")
    parser.add_argument("--max_pixels", type=int, default=768 * 768)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load normal pipeline
    pipe = load_pipeline("z-image-turbo", device=device, torch_dtype="bfloat16",
                         text_encoder="qwen3vl")

    # Load eval images
    eval_dir = Path("eval_unified/images")
    all_images = sorted(eval_dir.glob("*"))[:args.n_images]

    total = len(all_images) * 2
    print(f"\n{len(all_images)} images × 2 (baseline + swapped) = {total} generations")
    print(f"Output: {output_dir}\n")

    done = 0
    t0 = time.time()

    # Phase 1: Generate baselines with original ViT
    print("=== Phase 1: Baseline (original ViT) ===")
    for img_idx, img_path in enumerate(all_images):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        scale = min(1024 / max(w, h), 1.0)
        out_w = max(round(w * scale) // 16 * 16, 16)
        out_h = max(round(h * scale) // 16 * 16, 16)

        # Save input
        img_resized = img.resize((out_w, out_h), Image.LANCZOS)
        img_resized.save(output_dir / f"{img_idx:03d}_input.png")

        # Baseline i2i
        vl_embeds = encode_image_vl(pipe, img, device, max_pixels=args.max_pixels)
        result = denoise_with_embeds(pipe, vl_embeds, out_h, out_w, args.seed)
        result.save(output_dir / f"{img_idx:03d}_baseline.png")
        done += 1
        elapsed = time.time() - t0
        eta = elapsed / done * (total - done)
        print(f"  [{img_idx:03d}] baseline done ({done}/{total}, ~{eta / 60:.0f}m left)")

    # Phase 2: Swap ViT and generate
    print("\n=== Phase 2: Swapped ViT (Embedding-2B) ===")
    swap_vit_blocks(pipe, device, dtype)

    for img_idx, img_path in enumerate(all_images):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        scale = min(1024 / max(w, h), 1.0)
        out_w = max(round(w * scale) // 16 * 16, 16)
        out_h = max(round(h * scale) // 16 * 16, 16)

        # Swapped i2i
        vl_embeds = encode_image_vl(pipe, img, device, max_pixels=args.max_pixels)
        result = denoise_with_embeds(pipe, vl_embeds, out_h, out_w, args.seed)
        result.save(output_dir / f"{img_idx:03d}_swapped.png")
        done += 1
        elapsed = time.time() - t0
        eta = elapsed / done * (total - done)
        print(f"  [{img_idx:03d}] swapped done ({done}/{total}, ~{eta / 60:.0f}m left)")

    meta = {
        "n_images": len(all_images),
        "seed": args.seed,
        "max_pixels": args.max_pixels,
        "image_paths": [str(p) for p in all_images],
        "swap_source": "Qwen/Qwen3-VL-Embedding-2B",
        "swapped_components": ["patch_embed", "blocks (24 ViT layers)"],
        "kept_components": ["PatchMerger", "LLM (Z-Image spliced weights)"],
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone! {total} images in {(time.time() - t0) / 60:.1f} min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
