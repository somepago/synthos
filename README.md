# Synthos

Image-conditioned generation with Z-Image-Turbo via Qwen3-VL splice encoding.

## Overview

Z-Image-Turbo text-to-image model repurposed for image-conditioned generation by splicing Z-Image's trained LLM weights into Qwen3-VL, enabling visual tokens to condition the frozen DiT through the standard text conditioning path.

## Inference

### Single image (t2i / i2i)

```bash
# Text-to-image
python inference.py --prompt "a cat on a chair"

# Image-to-image
python inference.py --image photo.jpg

# Both (same seed/resolution)
python inference.py --prompt "a cat" --image photo.jpg
```

### Batch

```bash
# i2i on folder of images
python inference.py --image eval_unified/images/

# t2i from text file (one prompt per line)
python inference.py --prompt captions.txt

# Paired batch + metrics
python inference.py --image eval_unified/images/ --prompt captions.txt --metrics
```

### Multi-image composition

JSONL format — one JSON array per line with interleaved `{"img": path}` and `{"txt": str}`:

```bash
# Default: full cross-attention (B3)
python inference_multi_image.py --input prompts.jsonl

# B5: weighted averaging, no cross-attention
python inference_multi_image.py --input prompts.jsonl --blend_mode avg --alpha 0.7

# B6: cross-attention + per-image token scaling
python inference_multi_image.py --input prompts.jsonl --blend_mode scale --alpha 0.7
```

Blend modes:
- `concat` — interleaved encoding, full cross-image attention in LLM (default)
- `avg` — encode images separately, concatenate with alpha scaling (no cross-attention)
- `scale` — encode together, then scale visual tokens by image-of-origin

### Flags

| Flag | Description |
|------|-------------|
| `--model` | `z-image-turbo` (default) or `z-image-base` |
| `--text_encoder` | `qwen3vl` (default) or `qwen3` |
| `--seed` | Random seed (default: 42) |
| `--height/--width` | Output resolution (default: 1024) |
| `--num_steps` | Denoising steps (default: model-specific) |
| `--cfg_scale` | Classifier-free guidance scale |
| `--max_pixels` | Max pixels for VL image encoding (default: 768*768). Controls token count / variation strength |
| `--n_samples` | Process only first N images/prompts (default: all) |
| `--metrics` | Compute HPSv2.1, CLIP score, DINOv2 sim |
| `--blend_mode` | `concat`, `avg`, `scale` (multi-image only) |
| `--alpha` | First image weight for avg/scale (default: 0.5) |

## Baselines

```bash
# Run all baselines (resumable — skips completed runs)
screen -S baselines
bash run_baselines.sh 2>&1 | tee outputs/baselines.log
```

See `run_baselines.sh` for full list. Includes i2i/t2i on 84 eval images, composition with dense/light prompts, caption-drop ablation, and alpha sweeps for B5/B6.

### Variation strength (max_pixels sweep)

```bash
# Run all 6 levels on first 20 eval images
bash run_variations.sh

# Single image at specific token count
python inference.py --image photo.jpg --max_pixels 65536  # strong variation
python inference.py --image photo.jpg --max_pixels 589824 # subtle variation
```

### Text-guided variations

```bash
python run_text_variations.py            # 12 images x 9 text prompts
python run_text_variations.py --dry_run  # preview plan
```

## Project Structure

```
inference.py                    # Unified t2i + i2i inference (single + batch)
inference_multi_image.py        # Multi-image composition inference
run_baselines.sh                # Overnight baseline runs (resumable)
run_variations.sh               # Variation strength ablation (max_pixels sweep)
run_text_variations.py          # Text-guided variation ablation
train_stage1_projection.py      # Stage 1 SigLip projection training
src/
  diffusion.py                  # Encoding functions (VL, interleaved, weighted avg/scale)
  model_utils.py                # Pipeline loading + VL splice
  metrics.py                    # HPSv2.1, CLIP score, DINOv2 similarity
  constants.py                  # Shared constants
  env_setup.py                  # Environment setup
eval_unified/
  eval.csv                      # Image manifest (84 images)
  images/                       # Eval images
  composition_prompts.jsonl     # 50 dense multi-image prompts
  composition_light.jsonl       # 50 light multi-image prompts
  composition_light_notext.jsonl # Caption-drop ablation (images only)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Models download automatically from ModelScope on first run.
