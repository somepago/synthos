#!/usr/bin/env python3
"""
Streamlit viewer for experiments on the main eval set.

Usage: streamlit run viz/eval30_viewer.py --server.port 8506
"""

import io
import json
from pathlib import Path

from PIL import Image
import streamlit as st

THUMB_SIZE = 384
THUMB_QUALITY = 60

SPLITS_FILE = Path("eval_unified/eval_splits.json")
LAYER_TAP_DIR = Path("outputs/layer_tap_v2")

MAX_COLS = 4


def make_layer_label(name):
    if name == "emb":
        return "emb (post-projection)"
    num = name.replace("layer", "")
    if num == "34":
        return "34 (default)"
    if num == "35":
        return "35 (final)"
    return num


def discover_outputs(output_dir_str):
    """Discover layers, seeds, and ablation variants from filenames on disk.

    Returns: (layer_names, main_seeds, ablations)
      - layer_names: sorted list of layer names
      - main_seeds: sorted list of seeds used in main run (no suffix)
      - ablations: list of (label, suffix, seed) for each ablation variant
    """
    import re
    output_dir = Path(output_dir_str)
    layers = set()
    main_seeds = set()
    ablation_suffixes = set()  # (suffix_str, seed)

    for p in output_dir.glob("*_s*.png"):
        name = p.stem
        if "_input" in name:
            continue
        # Match: {tag}_{layer}_s{seed} or {tag}_{layer}_{suffix}_s{seed}
        m = re.search(r'_(emb|layer\d+)(?:_(cfg\d+|st\d+))?_s(\d+)$', name)
        if m:
            layers.add(m.group(1))
            suffix = m.group(2)
            seed = int(m.group(3))
            if suffix is None:
                main_seeds.add(seed)
            else:
                ablation_suffixes.add((suffix, seed))

    def layer_sort_key(l):
        if l == "emb":
            return -1
        return int(l.replace("layer", ""))

    # Build ablation list with human-readable labels
    ablations = []
    for suffix, seed in sorted(ablation_suffixes):
        if suffix.startswith("cfg"):
            label = f"CFG={suffix[3:]}"
        elif suffix.startswith("st"):
            label = f"Steps={suffix[2:]}"
        else:
            label = suffix
        ablations.append((label, suffix, seed))

    return sorted(layers, key=layer_sort_key), sorted(main_seeds), ablations


@st.cache_data
def load_splits():
    return json.loads(SPLITS_FILE.read_text())


@st.cache_data
def load_thumb(path_str):
    """Load image as low-res JPEG thumbnail for fast display."""
    try:
        img = Image.open(path_str)
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=THUMB_QUALITY)
        return buf.getvalue()
    except Exception:
        return None


def get_image_tags(output_dir):
    """Get list of (tag, input_path) for images that have outputs."""
    tags = []
    if not output_dir.exists():
        return tags
    for p in sorted(output_dir.glob("*_input.png")):
        tag = p.stem.replace("_input", "")
        tags.append((tag, p))
    return tags


def get_category(splits, tag):
    """Get human-readable category for an image tag."""
    parts = tag.split("_")
    src_type = parts[0]
    src_id = int(parts[1])

    if src_type == "real":
        # Search all real category keys
        for key in splits:
            if key.endswith("_categories") and "synth" not in key:
                cat = splits[key].get(str(src_id), "")
                if cat:
                    return cat
    else:
        for key in splits:
            if "synth_categories" in key:
                cat = splits[key].get(str(src_id), "")
                if cat:
                    return cat
    return ""


def count_done(output_dir, tag, layer_names, seeds):
    """Count how many of the expected outputs exist."""
    total = len(layer_names) * len(seeds)
    done = sum(1 for l in layer_names for s in seeds
               if (output_dir / f"{tag}_{l}_s{s}.png").exists())
    return done, total


def image_nav(tags, splits, key_prefix):
    """Sidebar image navigation. Returns (tag, input_path) or None."""
    if not tags:
        st.sidebar.warning("No images found yet")
        return None

    filter_type = st.sidebar.radio("Filter", ["All", "Real", "Synth"],
                                    horizontal=True, key=f"{key_prefix}_filter")
    if filter_type == "Real":
        tags = [(t, p) for t, p in tags if t.startswith("real_")]
    elif filter_type == "Synth":
        tags = [(t, p) for t, p in tags if t.startswith("synth_")]

    if not tags:
        st.sidebar.warning("No images match filter")
        return None

    idx_key = f"{key_prefix}_idx"
    cur = st.session_state.get(idx_key, 0)
    cur = min(cur, len(tags) - 1)

    col_prev, col_next = st.sidebar.columns(2)
    with col_prev:
        if st.button("< Prev", key=f"{key_prefix}_prev", use_container_width=True) and cur > 0:
            st.session_state[idx_key] = cur - 1
            st.rerun()
    with col_next:
        if st.button("Next >", key=f"{key_prefix}_next", use_container_width=True) and cur < len(tags) - 1:
            st.session_state[idx_key] = cur + 1
            st.rerun()

    # Show clickable list of images
    for i, (tag, _) in enumerate(tags):
        cat = get_category(splits, tag)
        label = f"**{tag}** — {cat}" if i == cur else f"{tag} — {cat}"
        if st.sidebar.button(label, key=f"{key_prefix}_img_{i}", use_container_width=True):
            st.session_state[idx_key] = i
            st.rerun()

    return tags[cur]


def show_img(path, caption=None, width=None):
    """Display image as thumbnail for fast loading."""
    thumb = load_thumb(str(path))
    if thumb:
        st.image(thumb, caption=caption, use_container_width=width is None, width=width)
    else:
        st.caption(f"{caption}: failed to load")


def render_grid(dir_path, tag, layers, seeds, show_seed=True):
    """Render images in rows of MAX_COLS. Each image gets layer + seed label."""
    items = []
    for layer in layers:
        for seed in seeds:
            p = dir_path / f"{tag}_{layer}_s{seed}.png"
            label = make_layer_label(layer)
            if show_seed:
                label = f"{label}  |  s{seed}"
            items.append((p, label))

    for row_start in range(0, len(items), MAX_COLS):
        row = items[row_start:row_start + MAX_COLS]
        cols = st.columns(MAX_COLS)
        for ci, (p, label) in enumerate(row):
            with cols[ci]:
                if p.exists():
                    show_img(p, caption=label)
                else:
                    st.caption(f"{label}: pending")


def tab_layer_tap(output_dir, key_prefix, title):
    """Layer Tap Ablation viewer for a given output directory."""
    splits = load_splits()
    tags = get_image_tags(output_dir)
    layer_names, main_seeds, ablations = discover_outputs(str(output_dir))

    if not layer_names:
        st.warning(f"No outputs found yet in `{output_dir}`")
        return

    result = image_nav(tags, splits, key_prefix)
    if result is None:
        return
    tag, input_path = result

    # Header: input image + info
    col_input, col_info = st.columns([1, 3])
    with col_input:
        show_img(input_path, caption="Input", width=300)
    with col_info:
        cat = get_category(splits, tag)
        done, total = count_done(output_dir, tag, layer_names, main_seeds)
        info = f"{len(layer_names)} layers, {len(main_seeds)} seeds"
        if ablations:
            info += f", {len(ablations)} ablations"
        st.markdown(f"### {tag} — {cat}")
        st.caption(f"{done}/{total} main done | {info}")

    st.divider()

    # Main run: each layer = one row of seeds
    st.markdown("## Main")
    for layer_name in layer_names:
        st.subheader(make_layer_label(layer_name))
        cols = st.columns(MAX_COLS)
        for si, seed in enumerate(main_seeds):
            p = output_dir / f"{tag}_{layer_name}_s{seed}.png"
            with cols[si]:
                if p.exists():
                    show_img(p, caption=f"seed={seed}")
                else:
                    st.caption(f"s{seed}: pending")

    # Ablations: each ablation variant gets its own section
    if ablations:
        st.divider()
        st.markdown("## Ablations")
        for label, suffix, seed in ablations:
            st.markdown(f"### {label}")
            # Show all layers in rows of MAX_COLS for this ablation
            items = []
            for layer_name in layer_names:
                p = output_dir / f"{tag}_{layer_name}_{suffix}_s{seed}.png"
                items.append((p, make_layer_label(layer_name)))
            for row_start in range(0, len(items), MAX_COLS):
                row = items[row_start:row_start + MAX_COLS]
                cols = st.columns(MAX_COLS)
                for ci, (p, lbl) in enumerate(row):
                    with cols[ci]:
                        if p.exists():
                            show_img(p, caption=lbl)
                        else:
                            st.caption(f"{lbl}: pending")


LAYER_TAP_BASE_DIR = Path("outputs/layer_tap_v2_base")


def main():
    st.set_page_config(page_title="Eval Experiments", layout="wide")
    st.title("Eval Experiment Viewer")

    tabs = st.tabs([
        "Layer Tap — Turbo",
        "Layer Tap — Base",
    ])

    with tabs[0]:
        tab_layer_tap(LAYER_TAP_DIR, "lt_turbo", "Layer Tap — Turbo (8 steps, cfg=1)")

    with tabs[1]:
        tab_layer_tap(LAYER_TAP_BASE_DIR, "lt_base", "Layer Tap — Base (50 steps, cfg=4)")


if __name__ == "__main__":
    main()
