"""
Streamlit viewer for baseline generation results.

Usage:
    streamlit run viz/baselines_viewer.py

Expects outputs in outputs/baselines_feb25/ (or pass --base_dir).
"""

import base64
import json
from pathlib import Path

import streamlit as st

st.set_page_config(layout="wide", page_title="Baselines Viewer")

# Inject CSS/JS for click-to-zoom lightbox on all images
st.markdown("""
<style>
/* Make all streamlit images show pointer cursor */
div[data-testid="stImage"] img {
    cursor: zoom-in;
    transition: opacity 0.1s;
}
div[data-testid="stImage"] img:hover {
    opacity: 0.85;
}
/* Lightbox overlay */
#lightbox-overlay {
    display: none;
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    background: rgba(0,0,0,0.85);
    z-index: 999999;
    cursor: zoom-out;
    justify-content: center;
    align-items: center;
}
#lightbox-overlay img {
    max-width: 90vw;
    max-height: 90vh;
    object-fit: contain;
    border-radius: 4px;
}
#lightbox-overlay.active {
    display: flex;
}
</style>
<div id="lightbox-overlay" onclick="this.classList.remove('active')">
    <img id="lightbox-img" src="" />
</div>
<script>
// Attach click listeners to all streamlit images
const observer = new MutationObserver(() => {
    document.querySelectorAll('div[data-testid="stImage"] img').forEach(img => {
        if (!img.dataset.lbBound) {
            img.dataset.lbBound = "1";
            img.addEventListener('click', () => {
                document.getElementById('lightbox-img').src = img.src;
                document.getElementById('lightbox-overlay').classList.add('active');
            });
        }
    });
});
observer.observe(document.body, {childList: true, subtree: true});
</script>
""", unsafe_allow_html=True)

BASE_DIR = Path("outputs/baselines_feb25")


def load_meta(run_dir):
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return None


def get_completed_runs(base_dir):
    runs = {}
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and (d / "meta.json").exists():
            runs[d.name] = d
    return runs


# =============================================================================
# Main
# =============================================================================

st.title("Baseline Generations Viewer")

if not BASE_DIR.exists():
    st.error(f"Base directory not found: {BASE_DIR}")
    st.stop()

runs = get_completed_runs(BASE_DIR)
if not runs:
    st.warning("No completed runs found.")
    st.stop()

# Group runs
groups = {
    "Single Image (i2i / t2i)": [],
    "Variations (token count)": [],
    "Variations (text-guided)": [],
    "Dense Composition (50 prompts)": [],
    "Light Composition (50 prompts)": [],
    "Caption-Drop Ablation (images only)": [],
}

for name, path in runs.items():
    if name.startswith(("i2i_all", "t2i_all", "native_t2i")):
        groups["Single Image (i2i / t2i)"].append((name, path))
    elif name == "vary_text":
        groups["Variations (text-guided)"].append((name, path))
    elif name.startswith("vary_"):
        groups["Variations (token count)"].append((name, path))
    elif name.startswith("multi_"):
        groups["Dense Composition (50 prompts)"].append((name, path))
    elif name.startswith("light_"):
        groups["Light Composition (50 prompts)"].append((name, path))
    elif name.startswith("notext_"):
        groups["Caption-Drop Ablation (images only)"].append((name, path))

# Sidebar: group + run selection
st.sidebar.header("Run Selection")

group_name = st.sidebar.selectbox("Group", [g for g in groups if groups[g]])
available = groups[group_name]

run_name = st.sidebar.selectbox(
    "Run",
    [name for name, _ in available],
    format_func=lambda x: x,
)

run_dir = dict(available)[run_name]
meta = load_meta(run_dir)

# Show meta info
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model**: {meta.get('model', '?')}")
if meta.get("text_encoder"):
    st.sidebar.markdown(f"**Text Enc**: {meta['text_encoder']}")
st.sidebar.markdown(f"**Steps**: {meta.get('num_steps', '?')}")
st.sidebar.markdown(f"**CFG**: {meta.get('cfg_scale', '?')}")
st.sidebar.markdown(f"**Seed**: {meta.get('seed', '?')}")
if meta.get("max_pixels"):
    st.sidebar.markdown(f"**Max Pixels**: {meta['max_pixels']:,}")
if "blend_mode" in meta:
    st.sidebar.markdown(f"**Blend**: {meta['blend_mode']}")
    st.sidebar.markdown(f"**Alpha**: {meta['alpha']}")
st.sidebar.markdown(f"**N**: {meta.get('n', '?')}")

# Sample range filter
n_total = meta.get("n", 0)
sample_range = st.sidebar.slider(
    "Sample range",
    min_value=0,
    max_value=max(n_total - 1, 0),
    value=(0, min(9, max(n_total - 1, 0))),
)

# Compare mode: show same sample across multiple runs in one group
st.sidebar.markdown("---")
compare_mode = st.sidebar.checkbox("Compare runs side-by-side")

if compare_mode and len(available) > 1:
    compare_runs = st.sidebar.multiselect(
        "Runs to compare",
        [name for name, _ in available],
        default=[name for name, _ in available[:3]],
    )

    sample_idx = st.sidebar.number_input(
        "Sample index", min_value=0, max_value=n_total - 1, value=0
    )

    st.header(f"Comparison — Sample #{sample_idx}")

    # Show refs from first run
    first_dir = dict(available)[compare_runs[0]]
    idx = f"{sample_idx:03d}"

    # Show prompt if available
    first_meta = load_meta(first_dir)
    if "entries" in first_meta and sample_idx < len(first_meta["entries"]):
        entry = first_meta["entries"][sample_idx]
        txt_parts = [item["txt"] for item in entry if "txt" in item]
        if txt_parts:
            st.caption(" ".join(txt_parts))
    elif "prompts" in first_meta and sample_idx < len(first_meta.get("prompts", [])):
        p = first_meta["prompts"][sample_idx]
        if p:
            st.caption(p)

    # Ref images
    refs = []
    j = 0
    while True:
        ref_path = first_dir / f"{idx}_ref{j}.png"
        if ref_path.exists():
            refs.append(ref_path)
            j += 1
        else:
            break

    # Also check for input images (for single-image runs)
    input_path = first_dir / f"input_{idx}.png"
    if input_path.exists() and not refs:
        refs.append(input_path)

    if refs:
        ref_cols = st.columns(len(refs))
        for j, ref_path in enumerate(refs):
            label = f"Ref {j}" if len(refs) > 1 else "Input"
            ref_cols[j].image(str(ref_path), caption=label, use_container_width=True)
        st.markdown("---")

    # Generated images side by side
    cols = st.columns(len(compare_runs))
    for col, rname in zip(cols, compare_runs):
        rdir = dict(available)[rname]
        # Try multi-image format first, then single-image formats
        gen_path = rdir / f"{idx}.png"
        t2i_path = rdir / f"t2i_{idx}.png"
        i2i_path = rdir / f"i2i_{idx}.png"
        if gen_path.exists():
            col.image(str(gen_path), caption=rname, use_container_width=True)
        elif t2i_path.exists():
            col.image(str(t2i_path), caption=rname, use_container_width=True)
        elif i2i_path.exists():
            col.image(str(i2i_path), caption=rname, use_container_width=True)
        else:
            col.warning(f"No image for {idx}")

else:
    # Normal single-run view
    st.header(f"{run_name}")

    # Determine run type and render
    is_text_variation = meta.get("type") == "text_variation"
    is_multi = "entries" in meta

    if is_text_variation:
        # Text-guided variation view: input + baseline + text variants per image
        images_meta = meta.get("images", [])
        text_variants = meta.get("text_variants", {})
        variant_keys = list(text_variants.keys())

        # Load/init picks (persisted to JSON alongside the run)
        picks_path = run_dir / "picks.json"
        if "text_vary_picks" not in st.session_state:
            if picks_path.exists():
                st.session_state.text_vary_picks = json.loads(picks_path.read_text())
            else:
                st.session_state.text_vary_picks = {}
        picks = st.session_state.text_vary_picks

        # Pick mode toggle in sidebar
        st.sidebar.markdown("---")
        pick_mode = st.sidebar.checkbox("Label mode (mark what worked)")
        n_picked = sum(1 for v in picks.values() if v)
        if n_picked:
            st.sidebar.markdown(f"**Picked**: {n_picked} variants")
        if picks and st.sidebar.button("Save picks"):
            picks_path.write_text(json.dumps(picks, indent=2))
            st.sidebar.success(f"Saved to {picks_path}")
        if picks and st.sidebar.button("Show only picked"):
            st.session_state.text_vary_filter_picked = not st.session_state.get("text_vary_filter_picked", False)
        filter_picked = st.session_state.get("text_vary_filter_picked", False)

        for list_idx, img_info in enumerate(images_meta):
            if list_idx < sample_range[0] or list_idx > sample_range[1]:
                continue
            idx_num = img_info["idx"]
            prefix = f"{idx_num:03d}"
            desc = img_info.get("desc", "")

            input_path = run_dir / f"{prefix}_input.png"
            baseline_path = run_dir / f"{prefix}_baseline.png"
            if not baseline_path.exists():
                continue

            # Collect variants for this image
            variant_cols_data = []
            for key in variant_keys:
                vpath = run_dir / f"{prefix}_{key}.png"
                if vpath.exists():
                    pick_key = f"{prefix}_{key}"
                    label = f"{key}: \"{text_variants[key]}\""
                    variant_cols_data.append((key, label, vpath, pick_key))

            # If filtering, skip images with no picks
            if filter_picked:
                has_any_pick = any(picks.get(pk, False) for _, _, _, pk in variant_cols_data)
                if not has_any_pick:
                    continue

            st.markdown(f"**#{idx_num}** — {desc}")

            # Row 1: input + baseline
            col1, col2 = st.columns(2)
            if input_path.exists():
                col1.image(str(input_path), caption="Input", use_container_width=True)
            col2.image(str(baseline_path), caption="Baseline (image only)", use_container_width=True)

            # Row 2: text variants in groups of 3, with optional checkboxes
            display_variants = variant_cols_data
            if filter_picked:
                display_variants = [(k, l, v, pk) for k, l, v, pk in variant_cols_data if picks.get(pk, False)]

            if display_variants:
                for chunk_start in range(0, len(display_variants), 3):
                    chunk = display_variants[chunk_start:chunk_start + 3]
                    cols = st.columns(len(chunk))
                    for col, (key, label, vpath, pick_key) in zip(cols, chunk):
                        is_picked = picks.get(pick_key, False)
                        border = "3px solid #4CAF50" if is_picked else "none"
                        col.markdown(
                            f'<div style="border: {border}; border-radius: 4px; padding: 2px;">',
                            unsafe_allow_html=True,
                        )
                        col.image(str(vpath), caption=label, use_container_width=True)
                        col.markdown('</div>', unsafe_allow_html=True)
                        if pick_mode:
                            if col.checkbox("worked", value=is_picked, key=pick_key):
                                picks[pick_key] = True
                            else:
                                picks[pick_key] = False

            st.divider()

        # Auto-save picks on any change
        if picks:
            picks_path.write_text(json.dumps(picks, indent=2))

    elif is_multi:
        # Render with original indices
        n = meta["n"]
        entries = meta.get("entries", [])

        for i in range(sample_range[0], min(sample_range[1] + 1, n)):
            idx = f"{i:03d}"
            gen_path = run_dir / f"{idx}.png"
            if not gen_path.exists():
                continue

            refs = []
            j = 0
            while True:
                ref_path = run_dir / f"{idx}_ref{j}.png"
                if ref_path.exists():
                    refs.append(ref_path)
                    j += 1
                else:
                    break

            prompt = ""
            img_names = []
            if i < len(entries):
                txt_parts = [item["txt"] for item in entries[i] if "txt" in item]
                prompt = " ".join(txt_parts)
                img_names = [Path(item["img"]).name for item in entries[i] if "img" in item]

            caption_parts = [f"**#{i}**"]
            if img_names:
                caption_parts.append(" + ".join(f"`{n}`" for n in img_names))
            st.markdown(" — ".join(caption_parts))

            if prompt:
                st.caption(prompt)

            n_cols = len(refs) + 1
            cols = st.columns(n_cols)
            for j, ref_path in enumerate(refs):
                cols[j].image(str(ref_path), caption=f"Ref {j}", use_container_width=True)
            cols[-1].image(str(gen_path), caption="Generated", use_container_width=True)
            st.divider()

    else:
        # inference.py batch run
        n = meta["n"]
        has_t2i = meta.get("has_t2i", False)
        has_i2i = meta.get("has_i2i", False)
        prompts = meta.get("prompts", [])
        image_paths = meta.get("image_paths", [])

        for i in range(sample_range[0], min(sample_range[1] + 1, n)):
            idx = f"{i:03d}"
            cols_data = []

            input_path = run_dir / f"input_{idx}.png"
            if input_path.exists():
                cols_data.append(("Input", input_path))

            if has_i2i:
                i2i_path = run_dir / f"i2i_{idx}.png"
                if i2i_path.exists():
                    cols_data.append(("i2i", i2i_path))

            if has_t2i:
                t2i_path = run_dir / f"t2i_{idx}.png"
                if t2i_path.exists():
                    cols_data.append(("t2i (VL)", t2i_path))

            # Show native turbo t2i alongside if available
            native_t2i_path = BASE_DIR / "native_t2i" / f"t2i_{idx}.png"
            if has_t2i and native_t2i_path.exists():
                cols_data.append(("t2i (native)", native_t2i_path))

            if not cols_data:
                continue

            caption_parts = [f"**#{i}**"]
            if i < len(image_paths) and image_paths[i]:
                caption_parts.append(f"`{Path(image_paths[i]).name}`")

            st.markdown(" — ".join(caption_parts))

            if i < len(prompts) and prompts[i]:
                st.caption(prompts[i])

            cols = st.columns(len(cols_data))
            for col, (label, path) in zip(cols, cols_data):
                col.image(str(path), caption=label, use_container_width=True)
            st.divider()
