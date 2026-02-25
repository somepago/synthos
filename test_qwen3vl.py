#!/usr/bin/env python3
"""Test: Replace Z-Image's Qwen3 text encoder with Qwen3-VL-4B.

The idea: Qwen3-VL-4B has the same LLM backbone (Qwen3-4B, hidden_size=2560,
36 layers) but also has a vision tower. By splicing Z-Image's trained LLM
weights into the VL model, we get a text encoder that can ALSO process images
natively — the VL vision tokens pass through the same transformer layers and
produce embeddings in the same 2560-dim space the DiT expects.

Approach (sequential to stay memory-safe on 128GB unified):
  Phase 1: Load VL model → splice Z-Image weights → encode all inputs → save embeddings → free VL
  Phase 2: Load DiT + VAE pipeline (no text encoder) → denoise from cached embeddings → save images

Results: outputs/qwen3vl_test/
"""

from src import env_setup  # noqa: F401

import gc
import sys
import torch
from PIL import Image
from pathlib import Path
from safetensors.torch import load_file

OUT_DIR = Path("outputs/qwen3vl_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_DIR = Path("eval")


def log(msg):
    print(msg, flush=True)


# ── Phase 1: Encode with VL model ──────────────────────────────────────

def load_zimage_llm_weights():
    model_dir = Path("models/Tongyi-MAI/Z-Image-Turbo/text_encoder")
    state_dict = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        state_dict.update(load_file(str(f), device="cpu"))
    return {k.removeprefix("model."): v for k, v in state_dict.items()}


def load_vl_model(device):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    log("Loading Qwen3-VL-4B-Instruct...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct", torch_dtype=torch.bfloat16,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

    log("Splicing Z-Image LLM weights...")
    z_sd = load_zimage_llm_weights()
    result = model.model.language_model.load_state_dict(z_sd, strict=False)
    del z_sd; gc.collect()
    log(f"  missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")

    return model, processor


@torch.no_grad()
def encode_text(model, processor, prompt, device):
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(device)
    out = model.model(**inputs, output_hidden_states=True)
    h = out.hidden_states[-2][0]
    mask = inputs["attention_mask"][0].bool()
    return h[mask].cpu()


@torch.no_grad()
def encode_image(model, processor, image, device):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": ""},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    out = model.model(**inputs, output_hidden_states=True)
    h = out.hidden_states[-2][0]
    mask = inputs["attention_mask"][0].bool()
    return h[mask].cpu()


def phase1_encode(device):
    model, processor = load_vl_model(device)
    embeddings = {}

    # --- t2i: encode all prompts from eval_prompts.txt ---
    prompts_file = EVAL_DIR / "eval_prompts.txt"
    prompts = [line.strip() for line in prompts_file.read_text().splitlines() if line.strip()]
    log(f"\nEncoding {len(prompts)} text prompts...")
    for i, prompt in enumerate(prompts):
        label = f"t2i_{i:03d}"
        embeds = encode_text(model, processor, prompt, device)
        embeddings[label] = (embeds, prompt[:80])
        log(f"  [{i+1}/{len(prompts)}] {label}: {embeds.shape} — {prompt[:60]}...")

    # --- i2i: encode all images from eval/ ---
    image_files = sorted(
        [f for f in EVAL_DIR.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.webp', '.png')]
    )
    log(f"\nEncoding {len(image_files)} images...")
    for i, fpath in enumerate(image_files):
        stem = fpath.stem
        label = f"i2i_{stem}"
        img = Image.open(fpath).convert("RGB").resize((512, 512))
        img.save(OUT_DIR / f"input_{stem}.png")
        embeds = encode_image(model, processor, img, device)
        embeddings[label] = (embeds, fpath.name)
        log(f"  [{i+1}/{len(image_files)}] {label}: {embeds.shape}")

    del model, processor
    gc.collect(); torch.cuda.empty_cache()
    log("\nVL model freed.")
    return embeddings


# ── Phase 2: Generate with DiT + VAE ───────────────────────────────────

def load_pipeline_no_textenc(device="cuda"):
    from diffsynth.core.loader.config import ModelConfig
    from diffsynth.pipelines.z_image import ZImagePipeline

    model_configs = [
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ]
    tokenizer_config = ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/")

    log("Loading DiT + VAE (no text encoder)...")
    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
        model_configs=model_configs, tokenizer_config=tokenizer_config,
    )
    return pipe


@torch.no_grad()
def generate_from_embeds(pipe, prompt_embeds, seed=42, h=512, w=512):
    from src.diffusion import get_latent_shape, generate_noise, _denoise_step, _decode_final

    noise = generate_noise(seed, get_latent_shape(h, w), pipe.device, pipe.torch_dtype)
    prompt_embeds = prompt_embeds.to(device=pipe.device, dtype=pipe.torch_dtype)

    pipe.scheduler.set_timesteps(8, denoising_strength=1.0, shift=None)
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    shared = {"latents": noise}
    posi = {"prompt_embeds": prompt_embeds}

    for pid, ts in enumerate(pipe.scheduler.timesteps):
        _denoise_step(pipe, 1.0, shared, posi, {}, models, ts, pid)

    return _decode_final(pipe, shared)


def phase2_generate(embeddings, device):
    pipe = load_pipeline_no_textenc(device)

    total = len(embeddings)
    for idx, (label, (embeds, source)) in enumerate(embeddings.items()):
        log(f"\n[{idx+1}/{total}] Generating: {label} (embeds={embeds.shape}) — {source}")
        img = generate_from_embeds(pipe, embeds)
        out_path = OUT_DIR / f"{label}.png"
        img.save(out_path)
        log(f"  → {out_path}")

    del pipe; gc.collect(); torch.cuda.empty_cache()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    device = "cuda"

    log("=" * 60)
    log("PHASE 1: Encoding with Qwen3-VL (Z-Image weights)")
    log("=" * 60)
    embeddings = phase1_encode(device)

    log("\n" + "=" * 60)
    log(f"PHASE 2: Generating {len(embeddings)} images with DiT + VAE")
    log("=" * 60)
    phase2_generate(embeddings, device)

    log(f"\nDone. All results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
