#!/usr/bin/env python3
"""
Compute metrics for layer tap v2 experiment.

For each image × layer × seed:
  - DINO distance to input (semantic similarity)
  - CLIP distance to input (semantic similarity)
  - SSIM to input (structural similarity)

For each image × layer (across seeds):
  - DINO pairwise mean (diversity across seeds)

Outputs: metrics CSV + per-layer summary plots.

Usage:
    python compute_layer_tap_metrics.py
"""

import sys
sys.path.insert(0, "/home/gnan/projects/diffscapes")

import json
import csv
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.metrics import MetricComputer

OUTPUT_DIR = Path("outputs/layer_tap_v2")
RESULTS_DIR = Path("outputs/layer_tap_v2_metrics")


def discover_layers_and_seeds(output_dir):
    """Discover available layers and seeds from filenames on disk."""
    import re
    layers = set()
    seeds = set()
    for p in output_dir.glob("*_s*.png"):
        name = p.stem
        if "_input" in name:
            continue
        # pattern: {tag}_{layer}_s{seed} or {tag}_{layer}_{suffix}_s{seed}
        m = re.search(r'_(emb|layer\d+)(?:_[^_]+)?_s(\d+)$', name)
        if m:
            layers.add(m.group(1))
            seeds.add(int(m.group(2)))
    # Sort layers: emb first, then by layer number
    def layer_sort_key(l):
        if l == "emb":
            return -1
        return int(l.replace("layer", ""))
    return sorted(layers, key=layer_sort_key), sorted(seeds)


def make_layer_label(name):
    if name == "emb":
        return "emb\n(post-proj)"
    num = name.replace("layer", "")
    if num == "34":
        return "34\n(default)"
    if num == "35":
        return "35\n(final)"
    return num


def get_image_tags():
    tags = []
    for p in sorted(OUTPUT_DIR.glob("*_input.png")):
        tag = p.stem.replace("_input", "")
        tags.append(tag)
    return tags


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mc = MetricComputer(device="cuda", hps_version="v2")

    tags = get_image_tags()
    LAYER_NAMES, SEEDS = discover_layers_and_seeds(OUTPUT_DIR)
    print(f"{len(tags)} images, {len(LAYER_NAMES)} layers ({', '.join(LAYER_NAMES)}), {len(SEEDS)} seeds ({SEEDS})")

    rows = []

    for tag in tqdm(tags, desc="images"):
        input_path = OUTPUT_DIR / f"{tag}_input.png"
        if not input_path.exists():
            continue
        input_img = Image.open(input_path).convert("RGB")

        # Pre-compute input embeddings
        input_dino = mc.dino_embed(input_img)
        input_clip = mc.clip_embed_batch([input_img])[0:1]

        for layer in LAYER_NAMES:
            gen_imgs = []
            gen_dino_embeds = []

            for seed in SEEDS:
                gen_path = OUTPUT_DIR / f"{tag}_{layer}_s{seed}.png"
                if not gen_path.exists():
                    continue

                gen_img = Image.open(gen_path).convert("RGB")
                gen_imgs.append(gen_img)

                # DINO distance
                gen_dino = mc.dino_embed(gen_img)
                gen_dino_embeds.append(gen_dino)
                dino_dist = 1.0 - float(torch.nn.functional.cosine_similarity(
                    input_dino, gen_dino, dim=1).item())

                # CLIP distance
                gen_clip = mc.clip_embed_batch([gen_img])
                clip_dist = 1.0 - float(torch.nn.functional.cosine_similarity(
                    input_clip, gen_clip, dim=1).item())

                # SSIM
                ssim_val = mc.ssim(input_img, gen_img)

                rows.append({
                    "tag": tag,
                    "layer": layer,
                    "seed": seed,
                    "dino_dist": round(dino_dist, 4),
                    "clip_dist": round(clip_dist, 4),
                    "ssim": round(ssim_val, 4),
                })

            # Diversity: DINO pairwise across seeds for this layer
            if len(gen_dino_embeds) >= 2:
                embeds = torch.cat(gen_dino_embeds, dim=0)
                sim_matrix = torch.mm(embeds, embeds.T)
                n = sim_matrix.shape[0]
                # Mean of off-diagonal (pairwise similarity)
                mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
                pairwise_sim = sim_matrix[mask].mean().item()
                pairwise_dist = 1.0 - pairwise_sim
            else:
                pairwise_dist = 0.0

            # Update rows for this layer with diversity
            for r in rows:
                if r["tag"] == tag and r["layer"] == layer and "seed_diversity" not in r:
                    r["seed_diversity"] = round(pairwise_dist, 4)

    # Save CSV
    csv_path = RESULTS_DIR / "metrics.csv"
    fieldnames = ["tag", "layer", "seed", "dino_dist", "clip_dist", "ssim", "seed_diversity"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {csv_path}")

    # Compute per-layer summaries
    import pandas as pd
    df = pd.DataFrame(rows)

    summary = df.groupby("layer").agg({
        "dino_dist": ["mean", "std"],
        "clip_dist": ["mean", "std"],
        "ssim": ["mean", "std"],
        "seed_diversity": "first",  # same for all seeds within a layer×image
    }).reset_index()

    # Flatten column names
    summary.columns = ["layer", "dino_dist_mean", "dino_dist_std",
                        "clip_dist_mean", "clip_dist_std",
                        "ssim_mean", "ssim_std", "seed_diversity_mean"]

    # Recompute seed_diversity as mean across images per layer
    div_df = df.drop_duplicates(subset=["tag", "layer"])[["tag", "layer", "seed_diversity"]]
    div_summary = div_df.groupby("layer")["seed_diversity"].agg(["mean", "std"]).reset_index()
    div_summary.columns = ["layer", "seed_diversity_mean", "seed_diversity_std"]

    # Merge
    summary = summary.drop(columns=["seed_diversity_mean"])
    summary = summary.merge(div_summary, on="layer")

    # Sort by layer order (re-discover to be safe)
    all_layers, _ = discover_layers_and_seeds(OUTPUT_DIR)
    layer_order = {name: i for i, name in enumerate(all_layers)}
    summary["order"] = summary["layer"].map(layer_order)
    summary = summary.sort_values("order").drop(columns=["order"])

    summary_path = RESULTS_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    print(summary.to_string(index=False))

    # Generate plots
    plot_metrics(summary)


def plot_metrics(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = summary["layer"].tolist()
    x = range(len(layers))
    labels = [make_layer_label(l) for l in layers]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Layer Tap Ablation — Metrics Summary", fontsize=14, fontweight="bold")

    # DINO distance (lower = more similar to input)
    ax = axes[0, 0]
    ax.errorbar(x, summary["dino_dist_mean"], yerr=summary["dino_dist_std"],
                fmt="o-", capsize=3, color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("DINO Distance to Input")
    ax.set_title("Semantic Similarity (lower = closer to input)")
    ax.grid(True, alpha=0.3)

    # CLIP distance
    ax = axes[0, 1]
    ax.errorbar(x, summary["clip_dist_mean"], yerr=summary["clip_dist_std"],
                fmt="o-", capsize=3, color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("CLIP Distance to Input")
    ax.set_title("CLIP Similarity (lower = closer to input)")
    ax.grid(True, alpha=0.3)

    # SSIM (higher = more structurally similar)
    ax = axes[1, 0]
    ax.errorbar(x, summary["ssim_mean"], yerr=summary["ssim_std"],
                fmt="o-", capsize=3, color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("SSIM")
    ax.set_title("Structural Similarity (higher = closer to input)")
    ax.grid(True, alpha=0.3)

    # Seed diversity (higher = more diverse across seeds)
    ax = axes[1, 1]
    ax.errorbar(x, summary["seed_diversity_mean"], yerr=summary["seed_diversity_std"],
                fmt="o-", capsize=3, color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("DINO Pairwise Distance (across seeds)")
    ax.set_title("Seed Diversity (higher = more diverse outputs)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "layer_tap_metrics.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
