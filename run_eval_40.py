#!/usr/bin/env python3
"""
Eval: t2i vs i2i quality comparison on 40 images.

For each image from relaion-art-lowres:
- t2i: generate from its text caption (qwen3 text encoder)
- i2i: generate from the image itself (qwen3vl VL encoder)
Both use same seed, same resolution (matched to input AR).

Metrics:
- HPSv2.1: aesthetic quality (higher = better)
- CLIP score: prompt adherence (higher = better)
- DINOv2 cosine sim (i2i only): structural similarity to input (higher = better)

Outputs saved to outputs/eval_40/
"""

from src import env_setup  # noqa: F401

import argparse
import json
import gc
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.model_utils import load_pipeline, get_defaults
from src.diffusion import (
    get_latent_shape, generate_noise, run_full_diffusion,
    encode_image_vl, _denoise_step, _decode_final,
)


DATA_DIR = Path("/home/gnan/projects/data/datasets/relaion-art-lowres")
OUTPUT_DIR = Path("outputs/eval_40")
SEED = 42
N_IMAGES = 40
MAX_SIZE = 768  # cap resolution


def round_to_16(w, h, max_size=MAX_SIZE):
    """Resize preserving AR, round to 16px multiples, cap at max_size."""
    scale = min(max_size / max(w, h), 1.0)
    w, h = round(w * scale), round(h * scale)
    return max(w // 16 * 16, 16), max(h // 16 * 16, 16)


def select_images(n=N_IMAGES, seed=SEED):
    """Select n images from the dataset. Returns list of dicts with file_name, text, w, h."""
    df = pd.read_parquet(DATA_DIR / "downloaded.parquet")
    df = df[df["download_status"] == "ok"].reset_index(drop=True)

    # All have aesthetic > 8, so just sample directly
    selected = df.sample(n=n, random_state=seed)

    samples = []
    for _, row in selected.iterrows():
        img_path = DATA_DIR / "images" / row["file_name"]
        if not img_path.exists():
            continue
        w, h = round_to_16(row["width"], row["height"])
        samples.append({
            "file_name": row["file_name"],
            "text": row["text"],
            "width": w,
            "height": h,
        })
    print(f"Selected {len(samples)} images")
    return samples


# =============================================================================
# Generation
# =============================================================================

def generate_t2i(samples, device="cuda", dtype="bfloat16"):
    """Pass 1: Generate all t2i images using qwen3 text encoder."""
    print("\n=== Pass 1: Text-to-Image (qwen3) ===")
    pipe = load_pipeline("z-image-turbo", device=device, torch_dtype=dtype,
                         text_encoder="qwen3")
    defaults = get_defaults("z-image-turbo")
    num_steps = defaults["num_inference_steps"]
    cfg_scale = defaults["cfg_scale"]

    for i, s in enumerate(tqdm(samples, desc="t2i")):
        shape = get_latent_shape(s["height"], s["width"])
        noise = generate_noise(SEED, shape, device, getattr(torch, dtype))
        image = run_full_diffusion(pipe, s["text"], noise, num_steps, cfg_scale)
        image.save(OUTPUT_DIR / f"t2i_{i:03d}.png")

    # Free pipeline
    del pipe
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"Saved {len(samples)} t2i images")


def generate_i2i(samples, device="cuda", dtype="bfloat16"):
    """Pass 2: Generate all i2i images using qwen3vl VL encoder."""
    print("\n=== Pass 2: Image-to-Image (qwen3vl) ===")
    pipe = load_pipeline("z-image-turbo", device=device, torch_dtype=dtype,
                         text_encoder="qwen3vl")
    num_steps = 8
    cfg_scale = 1.0

    for i, s in enumerate(tqdm(samples, desc="i2i")):
        input_img = Image.open(DATA_DIR / "images" / s["file_name"]).convert("RGB")

        # Save input (resized to target resolution for fair comparison)
        input_resized = input_img.resize((s["width"], s["height"]), Image.LANCZOS)
        input_resized.save(OUTPUT_DIR / f"input_{i:03d}.png")

        # Encode via VL
        prompt_embeds = encode_image_vl(pipe, input_img, device)

        shape = get_latent_shape(s["height"], s["width"])
        noise = generate_noise(SEED, shape, device, getattr(torch, dtype))

        pipe.scheduler.set_timesteps(num_steps, denoising_strength=1.0, shift=None)
        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

        shared = {"latents": noise}
        posi = {"prompt_embeds": prompt_embeds}
        nega = {}

        with torch.no_grad():
            for pid, ts in enumerate(pipe.scheduler.timesteps):
                _denoise_step(pipe, cfg_scale, shared, posi, nega, models, ts, pid)
            image = _decode_final(pipe, shared)

        image.save(OUTPUT_DIR / f"i2i_{i:03d}.png")

    del pipe
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"Saved {len(samples)} i2i images")


# =============================================================================
# Metrics
# =============================================================================

def compute_hpsv2(image_paths, prompts):
    """Compute HPSv2.1 scores. Returns list of floats."""
    import hpsv2
    scores = []
    for img_path, prompt in tqdm(zip(image_paths, prompts), total=len(image_paths), desc="HPSv2.1"):
        result = hpsv2.score(str(img_path), prompt, hps_version="v2.1")
        score = result[0] if isinstance(result, (list, np.ndarray)) else float(result)
        scores.append(float(score))
    return scores


def compute_clip_score(image_paths, prompts, device="cuda"):
    """Compute CLIP similarity (image, text) via open_clip. Returns list of floats."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    scores = []
    with torch.no_grad():
        for img_path, prompt in tqdm(zip(image_paths, prompts), total=len(image_paths), desc="CLIP score"):
            img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            text = tokenizer([prompt]).to(device)
            img_feat = model.encode_image(img)
            txt_feat = model.encode_text(text)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ txt_feat.T).item()
            scores.append(sim)

    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return scores


def compute_dino_similarity(gen_paths, ref_paths, device="cuda"):
    """Compute DINOv2 cosine similarity between generated and reference images."""
    from torchvision import transforms

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    scores = []
    with torch.no_grad():
        for gen_path, ref_path in tqdm(zip(gen_paths, ref_paths), total=len(gen_paths), desc="DINOv2 sim"):
            gen_img = transform(Image.open(gen_path).convert("RGB")).unsqueeze(0).to(device)
            ref_img = transform(Image.open(ref_path).convert("RGB")).unsqueeze(0).to(device)
            gen_feat = model(gen_img)
            ref_feat = model(ref_img)
            gen_feat = gen_feat / gen_feat.norm(dim=-1, keepdim=True)
            ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
            sim = (gen_feat @ ref_feat.T).item()
            scores.append(sim)

    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return scores


def run_metrics(samples, device="cuda"):
    """Compute all metrics on generated images."""
    print("\n=== Computing Metrics ===")
    n = len(samples)
    prompts = [s["text"] for s in samples]
    t2i_paths = [OUTPUT_DIR / f"t2i_{i:03d}.png" for i in range(n)]
    i2i_paths = [OUTPUT_DIR / f"i2i_{i:03d}.png" for i in range(n)]
    input_paths = [OUTPUT_DIR / f"input_{i:03d}.png" for i in range(n)]

    # Verify all files exist
    for paths, label in [(t2i_paths, "t2i"), (i2i_paths, "i2i"), (input_paths, "input")]:
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"WARNING: {len(missing)} missing {label} images, skipping metrics")
            return None

    results = {"samples": [], "summary": {}}

    # HPSv2.1
    print("\n--- HPSv2.1 ---")
    hps_t2i = compute_hpsv2(t2i_paths, prompts)
    hps_i2i = compute_hpsv2(i2i_paths, prompts)

    # CLIP score
    print("\n--- CLIP Score ---")
    clip_t2i = compute_clip_score(t2i_paths, prompts, device)
    clip_i2i = compute_clip_score(i2i_paths, prompts, device)

    # DINOv2 similarity (i2i vs input, and t2i vs input)
    print("\n--- DINOv2 Similarity ---")
    dino_i2i = compute_dino_similarity(i2i_paths, input_paths, device)
    dino_t2i = compute_dino_similarity(t2i_paths, input_paths, device)

    # Build per-image results
    for i in range(n):
        results["samples"].append({
            "idx": i,
            "file_name": samples[i]["file_name"],
            "text": samples[i]["text"],
            "resolution": f"{samples[i]['width']}x{samples[i]['height']}",
            "t2i": {
                "hpsv2": round(hps_t2i[i], 4),
                "clip_score": round(clip_t2i[i], 4),
                "dino_sim": round(dino_t2i[i], 4),
            },
            "i2i": {
                "hpsv2": round(hps_i2i[i], 4),
                "clip_score": round(clip_i2i[i], 4),
                "dino_sim": round(dino_i2i[i], 4),
            },
        })

    # Summary
    results["summary"] = {
        "n_images": n,
        "t2i": {
            "hpsv2_mean": round(float(np.mean(hps_t2i)), 4),
            "hpsv2_std": round(float(np.std(hps_t2i)), 4),
            "clip_score_mean": round(float(np.mean(clip_t2i)), 4),
            "clip_score_std": round(float(np.std(clip_t2i)), 4),
            "dino_sim_mean": round(float(np.mean(dino_t2i)), 4),
            "dino_sim_std": round(float(np.std(dino_t2i)), 4),
        },
        "i2i": {
            "hpsv2_mean": round(float(np.mean(hps_i2i)), 4),
            "hpsv2_std": round(float(np.std(hps_i2i)), 4),
            "clip_score_mean": round(float(np.mean(clip_i2i)), 4),
            "clip_score_std": round(float(np.std(clip_i2i)), 4),
            "dino_sim_mean": round(float(np.mean(dino_i2i)), 4),
            "dino_sim_std": round(float(np.std(dino_i2i)), 4),
        },
    }

    return results


def print_results(results):
    """Print formatted results table."""
    if results is None:
        return

    print("\n" + "=" * 90)
    print("PER-IMAGE RESULTS")
    print("=" * 90)
    print(f"{'Idx':>3} {'Res':>9} {'t2i HPS':>8} {'i2i HPS':>8} {'t2i CLIP':>9} {'i2i CLIP':>9} {'t2i DINO':>9} {'i2i DINO':>9}")
    print("-" * 90)
    for s in results["samples"]:
        t, ii = s["t2i"], s["i2i"]
        print(f"{s['idx']:>3} {s['resolution']:>9} {t['hpsv2']:>8.4f} {ii['hpsv2']:>8.4f} "
              f"{t['clip_score']:>9.4f} {ii['clip_score']:>9.4f} "
              f"{t['dino_sim']:>9.4f} {ii['dino_sim']:>9.4f}")

    print("\n" + "=" * 90)
    print("SUMMARY (mean +/- std)")
    print("=" * 90)
    summ = results["summary"]
    for metric in ["hpsv2", "clip_score", "dino_sim"]:
        t_mean = summ["t2i"][f"{metric}_mean"]
        t_std = summ["t2i"][f"{metric}_std"]
        i_mean = summ["i2i"][f"{metric}_mean"]
        i_std = summ["i2i"][f"{metric}_std"]
        diff = i_mean - t_mean
        print(f"  {metric:>12}: t2i = {t_mean:.4f} +/- {t_std:.4f}  |  i2i = {i_mean:.4f} +/- {i_std:.4f}  |  diff = {diff:+.4f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="t2i vs i2i quality evaluation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip generation, only compute metrics on existing images")
    parser.add_argument("--skip_t2i", action="store_true", help="Skip t2i generation")
    parser.add_argument("--skip_i2i", action="store_true", help="Skip i2i generation")
    parser.add_argument("--skip_metrics", action="store_true", help="Skip metrics computation")
    parser.add_argument("--n_images", type=int, default=N_IMAGES)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Select images
    samples = select_images(n=args.n_images)

    # Save sample list for reproducibility
    with open(OUTPUT_DIR / "samples.json", "w") as f:
        json.dump(samples, f, indent=2)

    if not args.skip_generation:
        # Step 2a: t2i pass
        if not args.skip_t2i:
            t0 = time.time()
            generate_t2i(samples, device=args.device, dtype=args.dtype)
            print(f"t2i generation took {time.time() - t0:.1f}s")

        # Step 2b: i2i pass
        if not args.skip_i2i:
            t0 = time.time()
            generate_i2i(samples, device=args.device, dtype=args.dtype)
            print(f"i2i generation took {time.time() - t0:.1f}s")

    if not args.skip_metrics:
        # Step 3: Metrics
        t0 = time.time()
        results = run_metrics(samples, device=args.device)
        print(f"Metrics computation took {time.time() - t0:.1f}s")

        if results:
            # Step 4: Report
            print_results(results)

            with open(OUTPUT_DIR / "results.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
