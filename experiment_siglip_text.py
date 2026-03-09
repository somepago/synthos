#!/usr/bin/env python3
"""
Experiment: t2i via SigLip text encoder (channel misalignment).

Normal t2i: text → Qwen3-VL LLM → (L, 2560) → DiT
This experiment: text → SigLip2 text encoder → (L, 1152) → Linear(1152→2560) → DiT

The Linear projection is randomly initialized (xavier). SigLip text features
are contrastively aligned with image features in 1152-dim space, so even a
random projection might carry some semantic signal.

Generates side-by-side: normal VL t2i vs SigLip t2i for the same prompts.

Usage:
    python experiment_siglip_text.py --seed 42
"""

from src import env_setup  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from src.model_utils import load_pipeline
from src.diffusion import (
    encode_text_vl, get_latent_shape, generate_noise,
    _denoise_step, _decode_final,
)


PROMPTS = [
    "a cat sitting on a chair",
    "a beautiful sunset over the ocean",
    "cyberpunk cityscape at night",
    "watercolor painting of flowers",
    "portrait of an old man",
    "abstract geometric shapes",
    "a cozy cabin in the snow",
    "astronaut riding a horse",
    "japanese garden with cherry blossoms",
    "dark gothic castle on a cliff",
]


def load_siglip_text_encoder(device, dtype):
    """Load SigLip2 text model + tokenizer."""
    from transformers import Siglip2Model, AutoTokenizer

    print("Loading SigLip2 text encoder...")
    model = Siglip2Model.from_pretrained("google/siglip2-so400m-patch16-naflex")
    text_model = model.text_model.to(device=device, dtype=dtype).eval()
    text_model.requires_grad_(False)
    del model.vision_model
    del model
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained("google/siglip2-so400m-patch16-naflex")
    params = sum(p.numel() for p in text_model.parameters()) / 1e6
    print(f"  SigLip2 text encoder: {params:.0f}M params")
    return text_model, tokenizer


@torch.no_grad()
def encode_text_siglip(text_model, tokenizer, text, device, dtype):
    """Encode text through SigLip2's text tower. Returns (L, 1152)."""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                       max_length=64).to(device)
    out = text_model(**tokens)
    h = out.last_hidden_state[0].to(dtype)
    mask = tokens["attention_mask"][0].bool()
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
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="outputs/siglip_text_exp")
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load VL splice pipeline
    pipe = load_pipeline("z-image-turbo", device=device, torch_dtype="bfloat16",
                         text_encoder="qwen3vl")

    # Load SigLip text encoder
    text_model, tokenizer = load_siglip_text_encoder(device, dtype)

    # Random projection 1152 → 2560 (xavier init)
    torch.manual_seed(args.seed)
    siglip_proj = nn.Linear(1152, 2560, bias=False).to(device=device, dtype=dtype)
    nn.init.xavier_uniform_(siglip_proj.weight)
    siglip_proj.eval()
    siglip_proj.requires_grad_(False)

    total = len(PROMPTS) * 2  # VL + SigLip per prompt
    print(f"\n{len(PROMPTS)} prompts × 2 variants = {total} generations")
    print(f"Output: {output_dir}\n")

    done = 0
    t0 = time.time()

    for i, prompt in enumerate(PROMPTS):
        safe_name = prompt.replace(" ", "_")[:40]

        # Normal t2i: text → VL → DiT
        vl_embeds = encode_text_vl(pipe, prompt, device)
        result_vl = denoise_with_embeds(pipe, vl_embeds, args.height, args.width, args.seed)
        result_vl.save(output_dir / f"{i:02d}_vl_{safe_name}.png")
        done += 1

        # SigLip t2i: text → SigLip text encoder → project → DiT
        siglip_feats = encode_text_siglip(text_model, tokenizer, prompt, device, dtype)
        siglip_embeds = siglip_proj(siglip_feats)
        result_siglip = denoise_with_embeds(pipe, siglip_embeds, args.height, args.width, args.seed)
        result_siglip.save(output_dir / f"{i:02d}_siglip_{safe_name}.png")
        done += 1

        elapsed = time.time() - t0
        eta = elapsed / done * (total - done)
        print(f"  [{i:02d}] \"{prompt}\" done ({done}/{total}, ~{eta/60:.0f}m left)")
        print(f"       VL tokens: {vl_embeds.shape[0]}, SigLip tokens: {siglip_feats.shape[0]}")

    meta = {
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "prompts": PROMPTS,
        "projection": "random xavier_uniform, Linear(1152→2560, no bias)",
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone! {total} images in {(time.time() - t0)/60:.1f} min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
