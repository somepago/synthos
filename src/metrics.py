"""
Metric functions for evaluating generated images.

- HPSv2.1: aesthetic quality
- CLIP score: text-image similarity
- DINOv2: structural similarity between image pairs
"""

import gc
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


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


def run_metrics(output_dir, prompts=None, has_t2i=False, has_i2i=False,
                has_input=False, n=0, device="cuda"):
    """Compute all applicable metrics on generated images in output_dir.

    Args:
        output_dir: Path to directory with generated images
        prompts: list of prompt strings (needed for HPSv2/CLIP)
        has_t2i: whether t2i_NNN.png files exist
        has_i2i: whether i2i_NNN.png files exist
        has_input: whether input_NNN.png files exist
        n: number of images
        device: compute device

    Returns:
        dict with per-image results and summary stats
    """
    output_dir = Path(output_dir)
    results = {"samples": [], "summary": {}}

    t2i_paths = [output_dir / f"t2i_{i:03d}.png" for i in range(n)] if has_t2i else []
    i2i_paths = [output_dir / f"i2i_{i:03d}.png" for i in range(n)] if has_i2i else []
    input_paths = [output_dir / f"input_{i:03d}.png" for i in range(n)] if has_input else []

    # Verify files exist
    for paths, label in [(t2i_paths, "t2i"), (i2i_paths, "i2i"), (input_paths, "input")]:
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"WARNING: {len(missing)} missing {label} images, skipping {label} metrics")
            if label == "t2i":
                t2i_paths, has_t2i = [], False
            elif label == "i2i":
                i2i_paths, has_i2i = [], False
            else:
                input_paths, has_input = [], False

    if not has_t2i and not has_i2i:
        print("No generated images found, skipping metrics")
        return None

    # HPSv2.1 (needs prompts)
    hps_t2i, hps_i2i = [], []
    if prompts:
        print("\n--- HPSv2.1 ---")
        if has_t2i:
            hps_t2i = compute_hpsv2(t2i_paths, prompts)
        if has_i2i:
            hps_i2i = compute_hpsv2(i2i_paths, prompts)

    # CLIP score (needs prompts)
    clip_t2i, clip_i2i = [], []
    if prompts:
        print("\n--- CLIP Score ---")
        if has_t2i:
            clip_t2i = compute_clip_score(t2i_paths, prompts, device)
        if has_i2i:
            clip_i2i = compute_clip_score(i2i_paths, prompts, device)

    # DINOv2 similarity (needs input references)
    dino_t2i, dino_i2i = [], []
    if has_input:
        print("\n--- DINOv2 Similarity ---")
        if has_i2i:
            dino_i2i = compute_dino_similarity(i2i_paths, input_paths, device)
        if has_t2i:
            dino_t2i = compute_dino_similarity(t2i_paths, input_paths, device)

    # Build per-image results
    for i in range(n):
        entry = {"idx": i}
        if has_t2i:
            entry["t2i"] = {}
            if hps_t2i:
                entry["t2i"]["hpsv2"] = round(hps_t2i[i], 4)
            if clip_t2i:
                entry["t2i"]["clip_score"] = round(clip_t2i[i], 4)
            if dino_t2i:
                entry["t2i"]["dino_sim"] = round(dino_t2i[i], 4)
        if has_i2i:
            entry["i2i"] = {}
            if hps_i2i:
                entry["i2i"]["hpsv2"] = round(hps_i2i[i], 4)
            if clip_i2i:
                entry["i2i"]["clip_score"] = round(clip_i2i[i], 4)
            if dino_i2i:
                entry["i2i"]["dino_sim"] = round(dino_i2i[i], 4)
        results["samples"].append(entry)

    # Summary stats
    def _stats(vals, prefix):
        return {
            f"{prefix}_mean": round(float(np.mean(vals)), 4),
            f"{prefix}_std": round(float(np.std(vals)), 4),
        }

    for mode, hps, clip, dino in [("t2i", hps_t2i, clip_t2i, dino_t2i),
                                    ("i2i", hps_i2i, clip_i2i, dino_i2i)]:
        if not (hps or clip or dino):
            continue
        s = {}
        if hps:
            s.update(_stats(hps, "hpsv2"))
        if clip:
            s.update(_stats(clip, "clip_score"))
        if dino:
            s.update(_stats(dino, "dino_sim"))
        results["summary"][mode] = s

    return results


def print_results(results):
    """Print formatted results table."""
    if results is None:
        return

    summ = results["summary"]
    modes = list(summ.keys())
    metrics = set()
    for m in modes:
        metrics.update(k.replace("_mean", "").replace("_std", "") for k in summ[m])

    print("\n" + "=" * 70)
    print("SUMMARY (mean +/- std)")
    print("=" * 70)
    for metric in sorted(metrics):
        parts = []
        for mode in modes:
            m_key = f"{metric}_mean"
            s_key = f"{metric}_std"
            if m_key in summ[mode]:
                parts.append(f"{mode} = {summ[mode][m_key]:.4f} +/- {summ[mode][s_key]:.4f}")
        print(f"  {metric:>12}: {'  |  '.join(parts)}")
    print("=" * 70)
