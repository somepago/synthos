# Synthos

Image-conditioned generation with Z-Image-Turbo via SigLip2 projection training.

## Overview

Two-stage training pipeline:
1. **Stage 1** — Train SigLip projection layers (embedder + refiner + pad_token, ~358M params) with flow matching velocity loss. Everything else frozen (DiT 6.5B, SigLip2 encoder, text encoder, VAE).
2. **Stage 2** — GRPO reinforcement learning (TODO)

## Stage 1: Projection Training

Teaches the randomly initialized SigLip projection layers to pass image features into the frozen DiT for image-conditioned generation.

```bash
bash run_stg1.sh
```

Key details:
- **Loss**: Flow matching velocity MSE — `v_target = noise - z_0` (matches `model_fn` negation convention)
- **Omni mode**: DiT sees `[reference_image_latents, noisy_latents]` with `image_noise_mask=[0, 1]`
- **Prompts**: Empty text (`""`) — pure image conditioning for Stage 1
- **LR schedule**: Linear warmup + cosine decay
- **Eval**: 12 curated images (6 real + 6 midjourney), HPSv2.1 scoring
- **Logging**: wandb project `synthos-train-stg1`

## Project Structure

```
train_stage1_projection.py  # Stage 1 training script
inference.py                # Text2img + img2img inference
run_stg1.sh                 # Training launch script
src/
  model_utils.py            # Pipeline loading + SigLip setup
  diffusion.py              # VAE/SigLip encoding, img2img helpers
  constants.py              # Shared constants (timesteps, scheduler params)
  env_setup.py              # Environment setup
eval/
  eval_set.txt              # 12 curated eval images
  eval_prompts.txt          # Text prompts for future use
  *.jpg/jpeg/webp           # Eval image files
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Models download automatically from ModelScope on first run. Set `DIFFSYNTH_MODEL_BASE_PATH` to control model cache location.
