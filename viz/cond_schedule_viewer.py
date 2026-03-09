#!/usr/bin/env python3
"""Streamlit viewer for all experiments: conditioning schedule, blend modes, variations."""

import json
import streamlit as st
from pathlib import Path

st.set_page_config(layout="wide", page_title="Experiment Viewer")

BASELINES = Path("outputs/baselines_feb25")
COND_SCHED = Path("outputs/cond_schedule")

tab_sched, tab_blend, tab_vary, tab_text_vary, tab_layer, tab_layer_blend, tab_layer_blend_light, tab_layer_textonly, tab_composites, tab_sdedit, tab_sdedit_l24, tab_isolated, tab_text_before, tab_layer_text = st.tabs([
    "Cond Schedule", "Blend Modes", "Variation Strength", "Text-Guided Variations",
    "Layer Tap", "Layer Tap Blend", "Layer Tap Blend (Light)", "Text-Only (Vision-Aware)",
    "Composites", "SDEdit (Layer 34)", "SDEdit (Layer 24)", "Isolated Blend", "Text Before vs After", "Layer Tap Text",
])

# ── Tab 1: Conditioning Schedule ──
with tab_sched:
    st.header("Conditioning Schedule Experiment")
    pairs = sorted([p for p in COND_SCHED.iterdir() if p.is_dir() and p.name.startswith("pair_")])

    if not pairs:
        st.error(f"No pair directories found in {COND_SCHED}")
    else:
        pair_idx = st.slider("Pair", 0, len(pairs) - 1, 0, key="sched_pair")
        pair_dir = pairs[pair_idx]

        st.subheader(f"{pair_dir.name} — Input Images")
        col_a, col_b = st.columns(2)
        col_a.image(str(pair_dir / "input_a.png"), caption="Image A", use_container_width=True)
        col_b.image(str(pair_dir / "input_b.png"), caption="Image B", use_container_width=True)

        switch_points = [0, 2, 4, 6, 8]
        experiments = [
            ("a_to_ab", "A → A+B", "Start with A only, introduce B at switch point"),
            ("b_to_ab", "B → A+B", "Start with B only, introduce A at switch point"),
            ("a_to_b", "A → B", "Start with A only, switch to B only at switch point"),
        ]

        for exp_name, exp_title, exp_desc in experiments:
            st.subheader(exp_title)
            st.caption(exp_desc)
            cols = st.columns(len(switch_points))
            for i, sp in enumerate(switch_points):
                img_path = pair_dir / f"{exp_name}_s{sp}.png"
                if img_path.exists():
                    cols[i].image(str(img_path), caption=f"switch={sp}", use_container_width=True)

        with st.expander("How to read"):
            st.markdown("""
- **switch=0**: Use late conditioning for ALL steps
- **switch=8**: Use early conditioning for ALL steps (never switch)
- **switch=4**: Early for 4 steps, late for 4 steps
- **A→A+B**: s0=both images baseline, s8=A only
- **B→A+B**: s0=both images baseline, s8=B only
- **A→B**: s0=B only, s8=A only
""")

# ── Tab 2: Blend Modes ──
with tab_blend:
    st.header("Multi-Image Blend Modes")

    # Group available runs
    blend_groups = {
        "Dense prompts": {
            "concat": BASELINES / "multi_concat",
            "avg α=0.3": BASELINES / "multi_avg_a0.3",
            "avg α=0.5": BASELINES / "multi_avg_a0.5",
            "avg α=0.7": BASELINES / "multi_avg_a0.7",
            "avg α=0.9": BASELINES / "multi_avg_a0.9",
            "scale α=0.3": BASELINES / "multi_scale_a0.3",
            "scale α=0.5": BASELINES / "multi_scale_a0.5",
            "scale α=0.7": BASELINES / "multi_scale_a0.7",
            "scale α=0.9": BASELINES / "multi_scale_a0.9",
        },
        "Light prompts": {
            "concat": BASELINES / "light_concat",
            "avg α=0.3": BASELINES / "light_avg_a0.3",
            "scale α=0.3": BASELINES / "light_scale_a0.3",
        },
        "No text (images only)": {
            "concat": BASELINES / "notext_concat",
            "avg α=0.3": BASELINES / "notext_avg_a0.3",
            "scale α=0.3": BASELINES / "notext_scale_a0.3",
        },
    }

    group_name = st.selectbox("Prompt type", list(blend_groups.keys()))
    group = blend_groups[group_name]

    # Find valid modes
    valid_modes = {k: v for k, v in group.items() if v.exists()}
    if not valid_modes:
        st.warning("No outputs found for this group")
    else:
        # Count entries from first valid dir
        first_dir = list(valid_modes.values())[0]
        n_entries = len([f for f in first_dir.glob("*.png") if not f.name.startswith("meta") and "_ref" not in f.name])
        n_entries = max(n_entries, 1)

        entry_idx = st.slider("Entry", 0, n_entries - 1, 0, key="blend_entry")

        # Show refs from first valid dir
        ref0 = first_dir / f"{entry_idx:03d}_ref0.png"
        ref1 = first_dir / f"{entry_idx:03d}_ref1.png"
        if ref0.exists() and ref1.exists():
            st.subheader(f"Entry {entry_idx:03d} — Reference Images")
            c0, c1 = st.columns(2)
            c0.image(str(ref0), caption="Ref 0", use_container_width=True)
            c1.image(str(ref1), caption="Ref 1", use_container_width=True)

        # Show prompt if available
        meta_path = first_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            entries = meta.get("entries", [])
            if entry_idx < len(entries):
                entry_data = entries[entry_idx]
                if isinstance(entry_data, list):
                    texts = [item.get("txt", "") for item in entry_data if isinstance(item, dict) and "txt" in item]
                    if texts:
                        st.caption(f"Prompt: {texts[0][:200]}")

        # Show outputs for each blend mode
        st.subheader("Outputs by Blend Mode")
        mode_names = list(valid_modes.keys())
        cols = st.columns(min(len(mode_names), 5))
        for i, mode_name in enumerate(mode_names):
            col = cols[i % len(cols)]
            img_path = valid_modes[mode_name] / f"{entry_idx:03d}.png"
            if img_path.exists():
                col.image(str(img_path), caption=mode_name, use_container_width=True)

# ── Tab 3: Variation Strength ──
with tab_vary:
    st.header("Variation Strength (max_pixels sweep)")

    levels = ["very_strong", "strong", "medium", "default", "subtle", "very_subtle"]
    level_info = {
        "very_strong": "128x128, 82 tokens",
        "strong": "256x256, 82 tokens",
        "medium": "384x384, 153 tokens",
        "default": "512x512, 280 tokens",
        "subtle": "768x768, 582 tokens",
        "very_subtle": "896x896, 760 tokens",
    }

    # Find how many images
    vary_dirs = {l: BASELINES / f"vary_{l}" for l in levels}
    valid_levels = {l: d for l, d in vary_dirs.items() if d.exists()}

    if not valid_levels:
        st.warning("No variation outputs found")
    else:
        first_dir = list(valid_levels.values())[0]
        n_images = max(len(list(first_dir.glob("i2i_*.png"))), 1)

        img_idx = st.slider("Image", 0, n_images - 1, 0, key="vary_img")

        # Show input from eval set
        input_path = Path("eval_unified/images") / f"{img_idx:03d}.jpg"
        if not input_path.exists():
            # try other extensions
            for ext in [".jpeg", ".png", ".webp"]:
                input_path = Path("eval_unified/images") / f"{img_idx:03d}{ext}"
                if input_path.exists():
                    break

        if input_path.exists():
            st.subheader(f"Input image {img_idx:03d}")
            st.image(str(input_path), width=400)

        st.subheader("Outputs by variation level")
        cols = st.columns(len(valid_levels))
        for i, (level, d) in enumerate(valid_levels.items()):
            img_path = d / f"i2i_{img_idx:03d}.png"
            if img_path.exists():
                cols[i].image(str(img_path), caption=f"{level}\n({level_info.get(level, '')})", use_container_width=True)

# ── Tab 4: Text-Guided Variations ──
with tab_text_vary:
    st.header("Text-Guided Variations")

    vary_text_dir = BASELINES / "vary_text"
    if not vary_text_dir.exists():
        st.warning("No text variation outputs found")
    else:
        prompts = ["baseline", "watercolor", "pencil", "cyberpunk", "oil_paint",
                    "sunset", "winter", "underwater", "as_cat", "top_hat"]

        # Count images by looking at baseline files
        n_images = max(len(list(vary_text_dir.glob("*_baseline.png"))), 1)

        img_idx = st.slider("Image", 0, n_images - 1, 0, key="text_vary_img")

        # Show input
        input_path = vary_text_dir / f"{img_idx:03d}_input.png"
        if input_path.exists():
            st.subheader(f"Input image {img_idx:03d}")
            st.image(str(input_path), width=400)

        st.subheader("Text-guided outputs")
        cols = st.columns(5)
        for i, prompt in enumerate(prompts):
            col = cols[i % 5]
            img_path = vary_text_dir / f"{img_idx:03d}_{prompt}.png"
            if img_path.exists():
                col.image(str(img_path), caption=prompt.replace("_", " "), use_container_width=True)

# ── Tab 5: Layer Tap ──
with tab_layer:
    st.header("Layer Tap Ablation")
    st.caption("Same image encoded through Qwen3-VL, but DiT is conditioned from different internal layers")

    layer_dir = Path("outputs/layer_tap_exp")
    if not layer_dir.exists():
        st.warning("No layer tap outputs found")
    else:
        layer_names = [
            ("post_merger", "Post-Merger (pre-LLM)"),
            ("llm_layer04", "LLM Layer 4"),
            ("llm_layer08", "LLM Layer 8"),
            ("llm_layer12", "LLM Layer 12"),
            ("llm_layer18", "LLM Layer 18"),
            ("llm_layer24", "LLM Layer 24"),
            ("llm_layer30", "LLM Layer 30"),
            ("llm_layer34", "LLM Layer 34 (baseline)"),
            ("llm_layer35", "LLM Layer 35 (final)"),
        ]

        # Count images from input files
        n_images = len(list(layer_dir.glob("*_input.png")))
        if n_images == 0:
            st.warning("Experiment still running — no complete images yet")
        else:
            img_idx = st.slider("Image", 0, n_images - 1, 0, key="layer_img")

            # Show input
            input_path = layer_dir / f"{img_idx:03d}_input.png"
            if input_path.exists():
                st.subheader(f"Input image {img_idx:03d}")
                st.image(str(input_path), width=400)

            # Show all layers in a grid
            st.subheader("Outputs by layer")
            cols = st.columns(5)
            for i, (layer_key, layer_label) in enumerate(layer_names):
                col = cols[i % 5]
                img_path = layer_dir / f"{img_idx:03d}_{layer_key}.png"
                if img_path.exists():
                    col.image(str(img_path), caption=layer_label, use_container_width=True)

            with st.expander("How to read"):
                st.markdown("""
- **Post-Merger**: Raw visual features after ViT + PatchMerger, before any LLM processing
- **LLM Layer 4-30**: Progressively more "language-processed" representations
- **LLM Layer 34**: Current default (`hidden_states[-2]`) — what the DiT was trained on
- **LLM Layer 35**: Final LLM layer output

Earlier layers = more raw perception, later layers = more semantic/reasoned.
""")

# ── Tab 6: Layer Tap Blend ──
with tab_layer_blend:
    st.header("Layer Tap + Multi-Image Blending")
    st.caption("Two images blended (avg/scale, alpha=0.3) with embeddings from different VL layers")

    blend_layer_dir = Path("outputs/layer_tap_blend")
    if not blend_layer_dir.exists():
        st.warning("No layer tap blend outputs found")
    else:
        blend_layer_names = [
            ("layer12", "Layer 12"),
            ("layer18", "Layer 18"),
            ("layer24", "Layer 24"),
            ("layer30", "Layer 30"),
            ("layer34", "Layer 34 (baseline)"),
            ("layer35", "Layer 35 (final)"),
        ]

        n_pairs_lb = len(list(blend_layer_dir.glob("*_ref0.png")))
        if n_pairs_lb == 0:
            st.warning("Experiment still running")
        else:
            pair_idx_lb = st.slider("Pair", 0, n_pairs_lb - 1, 0, key="layer_blend_pair")

            # Show refs
            ref0 = blend_layer_dir / f"{pair_idx_lb:03d}_ref0.png"
            ref1 = blend_layer_dir / f"{pair_idx_lb:03d}_ref1.png"
            if ref0.exists() and ref1.exists():
                st.subheader(f"Pair {pair_idx_lb:03d} — Reference Images")
                c0, c1 = st.columns(2)
                c0.image(str(ref0), caption="Image A (alpha=0.3)", use_container_width=True)
                c1.image(str(ref1), caption="Image B (alpha=0.7)", use_container_width=True)

            # Show prompt from meta
            meta_path_lb = blend_layer_dir / "meta.json"
            if meta_path_lb.exists():
                meta_lb = json.loads(meta_path_lb.read_text())
                lb_entries = meta_lb.get("entries", [])
                if pair_idx_lb < len(lb_entries):
                    txt_items = [item for item in lb_entries[pair_idx_lb] if "txt" in item]
                    if txt_items:
                        st.info(f"**Prompt:** {txt_items[0]['txt']}")

            # Row 1: avg mode (encode separately, no cross-attention)
            st.subheader("Avg mode (encode separately, scale + concat)")
            cols = st.columns(len(blend_layer_names))
            for i, (lkey, llabel) in enumerate(blend_layer_names):
                img_path = blend_layer_dir / f"{pair_idx_lb:03d}_avg_{lkey}.png"
                if img_path.exists():
                    cols[i].image(str(img_path), caption=llabel, use_container_width=True)

            # Row 2: scale mode (encode together, cross-attention)
            st.subheader("Scale mode (encode together with cross-attention, then scale)")
            cols = st.columns(len(blend_layer_names))
            for i, (lkey, llabel) in enumerate(blend_layer_names):
                img_path = blend_layer_dir / f"{pair_idx_lb:03d}_scale_{lkey}.png"
                if img_path.exists():
                    cols[i].image(str(img_path), caption=llabel, use_container_width=True)

# ── Tab 7: Layer Tap Blend (Light Prompts) ──
with tab_layer_blend_light:
    st.header("Layer Tap + Multi-Image Blending (Light Prompts)")
    st.caption("Same as Layer Tap Blend but using light text prompts alongside image pairs")

    blend_light_dir = Path("outputs/layer_tap_blend_light")
    if not blend_light_dir.exists():
        st.warning("No layer tap blend (light) outputs found")
    else:
        blend_light_layer_names = [
            ("layer12", "Layer 12"),
            ("layer18", "Layer 18"),
            ("layer24", "Layer 24"),
            ("layer30", "Layer 30"),
            ("layer34", "Layer 34 (baseline)"),
            ("layer35", "Layer 35 (final)"),
        ]

        n_pairs_bll = len(list(blend_light_dir.glob("*_ref0.png")))
        if n_pairs_bll == 0:
            st.warning("Experiment still running")
        else:
            pair_idx_bll = st.slider("Pair", 0, n_pairs_bll - 1, 0, key="layer_blend_light_pair")

            # Show refs
            ref0 = blend_light_dir / f"{pair_idx_bll:03d}_ref0.png"
            ref1 = blend_light_dir / f"{pair_idx_bll:03d}_ref1.png"
            if ref0.exists() and ref1.exists():
                st.subheader(f"Pair {pair_idx_bll:03d} — Reference Images")
                c0, c1 = st.columns(2)
                c0.image(str(ref0), caption="Image A (alpha=0.3)", use_container_width=True)
                c1.image(str(ref1), caption="Image B (alpha=0.7)", use_container_width=True)

            # Show prompt — read from source entries file directly
            light_entries_path = Path("eval_unified/composition_light.jsonl")
            if light_entries_path.exists():
                light_entries = [json.loads(line) for line in light_entries_path.read_text().strip().split("\n")]
                if pair_idx_bll < len(light_entries):
                    txt_items = [item for item in light_entries[pair_idx_bll] if "txt" in item]
                    if txt_items:
                        st.info(f"**Prompt:** {txt_items[0]['txt']}")

            # Row 1: avg mode
            st.subheader("Avg mode (encode separately, scale + concat)")
            cols = st.columns(len(blend_light_layer_names))
            for i, (lkey, llabel) in enumerate(blend_light_layer_names):
                img_path = blend_light_dir / f"{pair_idx_bll:03d}_avg_{lkey}.png"
                if img_path.exists():
                    cols[i].image(str(img_path), caption=llabel, use_container_width=True)

            # Row 2: scale mode
            st.subheader("Scale mode (encode together with cross-attention + text, then scale)")
            cols = st.columns(len(blend_light_layer_names))
            for i, (lkey, llabel) in enumerate(blend_light_layer_names):
                img_path = blend_light_dir / f"{pair_idx_bll:03d}_scale_{lkey}.png"
                if img_path.exists():
                    cols[i].image(str(img_path), caption=llabel, use_container_width=True)

# ── Tab 8: Text-Only (Vision-Aware) ──
with tab_layer_textonly:
    st.header("Text-Only Conditioning (Vision-Aware)")
    st.caption("Images+text go through full VL forward pass, but vision tokens are stripped — only text tokens (which attended to images) condition the DiT")

    textonly_dir = Path("outputs/layer_tap_textonly")
    if not textonly_dir.exists():
        st.warning("No text-only outputs found")
    else:
        textonly_layer_names = [
            ("layer12", "Layer 12"),
            ("layer18", "Layer 18"),
            ("layer24", "Layer 24"),
            ("layer30", "Layer 30"),
            ("layer34", "Layer 34 (baseline)"),
            ("layer35", "Layer 35 (final)"),
        ]

        n_pairs_to = len(list(textonly_dir.glob("*_ref0.png")))
        if n_pairs_to == 0:
            st.warning("Experiment still running")
        else:
            pair_idx_to = st.slider("Pair", 0, n_pairs_to - 1, 0, key="textonly_pair")

            # Show refs
            ref0 = textonly_dir / f"{pair_idx_to:03d}_ref0.png"
            ref1 = textonly_dir / f"{pair_idx_to:03d}_ref1.png"
            if ref0.exists() and ref1.exists():
                st.subheader(f"Pair {pair_idx_to:03d} — Reference Images")
                c0, c1 = st.columns(2)
                c0.image(str(ref0), caption="Image A", use_container_width=True)
                c1.image(str(ref1), caption="Image B", use_container_width=True)

            # Show prompt — read from source entries file
            light_entries_path_to = Path("eval_unified/composition_light.jsonl")
            if light_entries_path_to.exists():
                light_entries_to = [json.loads(line) for line in light_entries_path_to.read_text().strip().split("\n")]
                if pair_idx_to < len(light_entries_to):
                    txt_items = [item for item in light_entries_to[pair_idx_to] if "txt" in item]
                    if txt_items:
                        st.info(f"**Prompt:** {txt_items[0]['txt']}")

            st.subheader("Outputs by layer (text tokens only, vision stripped)")
            cols = st.columns(len(textonly_layer_names))
            for i, (lkey, llabel) in enumerate(textonly_layer_names):
                img_path = textonly_dir / f"{pair_idx_to:03d}_textonly_{lkey}.png"
                if img_path.exists():
                    cols[i].image(str(img_path), caption=llabel, use_container_width=True)

# ── Tab 9: Composites Layer Tap ──
with tab_composites:
    st.header("Composites — Layer Tap i2i")
    st.caption("Rough cut-paste composites fed through VL at different layers — can it clean up compositions zero-shot?")

    comp_dir = Path("outputs/layer_tap_composites")
    if not comp_dir.exists():
        st.warning("No composite outputs found")
    else:
        comp_layer_names = [
            ("post_merger", "Post-Merger"),
            ("layer12", "Layer 12"),
            ("layer18", "Layer 18"),
            ("layer24", "Layer 24"),
            ("layer30", "Layer 30"),
            ("layer34", "Layer 34 (baseline)"),
            ("layer35", "Layer 35 (final)"),
        ]

        # Find composites by input files
        input_files = sorted(comp_dir.glob("*_input.png"))
        if not input_files:
            st.warning("Experiment still running")
        else:
            # Extract names from filenames: 000_bears_wall_input.png -> bears_wall
            comp_names = []
            for f in input_files:
                parts = f.stem.split("_")
                idx = parts[0]
                name = "_".join(parts[1:-1])  # everything between idx and "input"
                comp_names.append((idx, name))

            sel_idx = st.slider("Composite", 0, len(comp_names) - 1, 0, key="comp_idx")
            idx_str, name = comp_names[sel_idx]

            # Show input composite + background side by side
            st.subheader(f"{name.replace('_', ' ').title()}")
            input_path = comp_dir / f"{idx_str}_{name}_input.png"
            bg_path = comp_dir / f"{idx_str}_{name}_background.png"
            c0, c1 = st.columns(2)
            if input_path.exists():
                c0.image(str(input_path), caption="Composite (input)", use_container_width=True)
            if bg_path.exists():
                c1.image(str(bg_path), caption="Background", use_container_width=True)

            # Show outputs across layers × max_pixels
            comp_resolutions = [
                ("outputs/layer_tap_composites_strong", "Strong (128x128)"),
                ("outputs/layer_tap_composites_medium", "Medium (384x384)"),
                ("outputs/layer_tap_composites", "Default (768x768)"),
                ("outputs/layer_tap_composites_subtle", "Subtle (1024x1024)"),
            ]

            for res_dir_str, res_label in comp_resolutions:
                res_dir = Path(res_dir_str)
                if not res_dir.exists():
                    continue
                st.subheader(f"{res_label}")
                cols = st.columns(len(comp_layer_names))
                for i, (lkey, llabel) in enumerate(comp_layer_names):
                    img_path = res_dir / f"{idx_str}_{name}_{lkey}.png"
                    if img_path.exists():
                        cols[i].image(str(img_path), caption=llabel, use_container_width=True)

# ── Tab 10: SDEdit Composites (Layer 34) ──
with tab_sdedit:
    st.header("SDEdit Composites — Layer 34 (baseline)")
    st.caption("VL conditioning from layer 34 (default). Late-layer semantic summary may lose secondary subjects.")

    sdedit_dir = Path("outputs/sdedit_composites")
    if not sdedit_dir.exists():
        st.warning("No SDEdit outputs found")
    else:
        input_files = sorted(sdedit_dir.glob("*_input.png"))
        if not input_files:
            st.warning("Experiment still running")
        else:
            comp_names_sd = []
            for f in input_files:
                parts = f.stem.split("_")
                idx = parts[0]
                name = "_".join(parts[1:-1])
                comp_names_sd.append((idx, name))

            sel_sd = st.slider("Composite", 0, len(comp_names_sd) - 1, 0, key="sdedit_idx")
            idx_str_sd, name_sd = comp_names_sd[sel_sd]

            # Show input
            input_path = sdedit_dir / f"{idx_str_sd}_{name_sd}_input.png"
            if input_path.exists():
                st.subheader(f"{name_sd.replace('_', ' ').title()}")
                st.image(str(input_path), width=400, caption="Composite (input)")

            strengths = ["02", "04", "06", "08", "10"]
            strength_labels = ["0.2", "0.4", "0.6", "0.8", "1.0 (pure noise)"]
            mp_levels = [
                ("nocond", "No VL conditioning (empty text only)"),
                ("medium", "VL Medium (384x384, ~400 tokens)"),
                ("default", "VL Default (768x768, ~1000 tokens)"),
            ]

            for mp_key, mp_label in mp_levels:
                st.subheader(mp_label)
                cols = st.columns(len(strengths))
                for i, (s_str, s_label) in enumerate(zip(strengths, strength_labels)):
                    img_path = sdedit_dir / f"{idx_str_sd}_{name_sd}_{mp_key}_s{s_str}.png"
                    if img_path.exists():
                        cols[i].image(str(img_path), caption=f"strength={s_label}", use_container_width=True)

# ── Tab 10b: SDEdit Composites (Layer 24) ──
with tab_sdedit_l24:
    st.header("SDEdit Composites — Layer 24")
    st.caption("VL conditioning from layer 24 — more perceptual/spatial, preserves secondary subjects better than layer 34.")

    sdedit_l24_dir = Path("outputs/sdedit_composites_layer24")
    if not sdedit_l24_dir.exists():
        st.warning("No SDEdit Layer 24 outputs found")
    else:
        input_files_l24 = sorted(sdedit_l24_dir.glob("*_input.png"))
        if not input_files_l24:
            st.warning("Experiment still running")
        else:
            comp_names_l24 = []
            for f in input_files_l24:
                parts = f.stem.split("_")
                idx = parts[0]
                name = "_".join(parts[1:-1])
                comp_names_l24.append((idx, name))

            sel_l24 = st.slider("Composite", 0, len(comp_names_l24) - 1, 0, key="sdedit_l24_idx")
            idx_str_l24, name_l24 = comp_names_l24[sel_l24]

            input_path = sdedit_l24_dir / f"{idx_str_l24}_{name_l24}_input.png"
            if input_path.exists():
                st.subheader(f"{name_l24.replace('_', ' ').title()}")
                st.image(str(input_path), width=400, caption="Composite (input)")

            strengths = ["02", "04", "06", "08", "10"]
            strength_labels = ["0.2", "0.4", "0.6", "0.8", "1.0 (pure noise)"]
            mp_levels_l24 = [
                ("nocond", "No VL conditioning (empty text only)"),
                ("medium", "VL Medium (384x384)"),
                ("default", "VL Default (768x768)"),
            ]

            for mp_key, mp_label in mp_levels_l24:
                st.subheader(mp_label)
                cols = st.columns(len(strengths))
                for i, (s_str, s_label) in enumerate(zip(strengths, strength_labels)):
                    img_path = sdedit_l24_dir / f"{idx_str_l24}_{name_l24}_{mp_key}_s{s_str}.png"
                    if img_path.exists():
                        cols[i].image(str(img_path), caption=f"strength={s_label}", use_container_width=True)

# ── Tab 11: Isolated Blend ──
with tab_isolated:
    st.header("Isolated Blend (Cross-Image Attention Blocked)")
    st.caption("Scale mode vs Isolated mode: in isolated, image tokens cannot attend to each other — only text attends to both")

    iso_dir = Path("outputs/isolated_blend")
    if not iso_dir.exists():
        st.warning("No isolated blend outputs found")
    else:
        n_pairs_iso = len(list(iso_dir.glob("*_ref0.png")))
        if n_pairs_iso == 0:
            st.warning("Experiment still running")
        else:
            pair_idx_iso = st.slider("Pair", 0, n_pairs_iso - 1, 0, key="iso_pair")

            # Show refs
            ref0 = iso_dir / f"{pair_idx_iso:03d}_ref0.png"
            ref1 = iso_dir / f"{pair_idx_iso:03d}_ref1.png"
            if ref0.exists() and ref1.exists():
                st.subheader(f"Pair {pair_idx_iso:03d} — Reference Images")
                c0, c1 = st.columns(2)
                c0.image(str(ref0), caption="Image A (alpha=0.3)", use_container_width=True)
                c1.image(str(ref1), caption="Image B (alpha=0.7)", use_container_width=True)

            # Show prompt
            light_entries_iso = Path("eval_unified/composition_light.jsonl")
            if light_entries_iso.exists():
                iso_entries = [json.loads(line) for line in light_entries_iso.read_text().strip().split("\n")]
                if pair_idx_iso < len(iso_entries):
                    txt_items = [item for item in iso_entries[pair_idx_iso] if "txt" in item]
                    if txt_items:
                        st.info(f"**Prompt:** {txt_items[0]['txt']}")

            iso_layers = [("layer24", "Layer 24"), ("layer34", "Layer 34 (baseline)")]

            # One row per layer: scale + isolated side by side
            for lkey, llabel in iso_layers:
                st.subheader(f"{llabel}")
                c_scale, c_iso = st.columns(2)
                scale_path = iso_dir / f"{pair_idx_iso:03d}_scale_{lkey}.png"
                iso_path = iso_dir / f"{pair_idx_iso:03d}_isolated_{lkey}.png"
                if scale_path.exists():
                    c_scale.image(str(scale_path), caption="Scale (normal cross-attn)", use_container_width=True)
                if iso_path.exists():
                    c_iso.image(str(iso_path), caption="Isolated (cross-image blocked)", use_container_width=True)

# ── Tab 12: Text Before vs After ──
with tab_text_before:
    st.header("Text Before vs After Image Embeddings")
    st.caption("In causal attention, token order matters. Text-before: [text][img_A][img_B] — images attend to text, visual features are steered. Text-after: [img_A][img_B][text] — images processed blind to text.")

    tb_dir = Path("outputs/text_before")
    if not tb_dir.exists():
        st.warning("No text-before outputs found")
    else:
        n_pairs_tb = len(list(tb_dir.glob("*_ref0.png")))
        if n_pairs_tb == 0:
            st.warning("Experiment still running")
        else:
            pair_idx_tb = st.slider("Pair", 0, n_pairs_tb - 1, 0, key="tb_pair")

            # Show refs
            ref0 = tb_dir / f"{pair_idx_tb:03d}_ref0.png"
            ref1 = tb_dir / f"{pair_idx_tb:03d}_ref1.png"
            if ref0.exists() and ref1.exists():
                st.subheader(f"Pair {pair_idx_tb:03d} — Reference Images")
                c0, c1 = st.columns(2)
                c0.image(str(ref0), caption="Image A (alpha=0.3)", use_container_width=True)
                c1.image(str(ref1), caption="Image B (alpha=0.7)", use_container_width=True)

            # Show prompt
            light_entries_tb = Path("eval_unified/composition_light.jsonl")
            if light_entries_tb.exists():
                tb_entries = [json.loads(line) for line in light_entries_tb.read_text().strip().split("\n")]
                if pair_idx_tb < len(tb_entries):
                    txt_items = [item for item in tb_entries[pair_idx_tb] if "txt" in item]
                    if txt_items:
                        st.info(f"**Prompt:** {txt_items[0]['txt']}")

            # Side by side: text-after vs text-before
            st.subheader("Comparison")
            c_after, c_before = st.columns(2)
            after_path = tb_dir / f"{pair_idx_tb:03d}_text_after.png"
            before_path = tb_dir / f"{pair_idx_tb:03d}_text_before.png"
            if after_path.exists():
                c_after.image(str(after_path), caption="Text AFTER images\n[img_A][img_B][text]", use_container_width=True)
            if before_path.exists():
                c_before.image(str(before_path), caption="Text BEFORE images\n[text][img_A][img_B]", use_container_width=True)

            with st.expander("How to read"):
                st.markdown("""
**Text-after** (current default): `[img_A] [img_B] [text]`
- Image tokens are processed first, blind to text
- Text tokens attend to images but can't influence how images are encoded
- Text has weak influence (~3% of tokens)

**Text-before** (this experiment): `[text] [img_A] [img_B]`
- Text is processed first as context
- Image tokens attend to preceding text via causal self-attention
- Visual representations are "steered" by the text instruction
- Could make text guidance stronger since images are processed in text's context
""")

# ── Tab 13: Layer Tap Text ──
with tab_layer_text:
    st.header("Layer Tap + Text-Guided Variations")
    st.caption("Image+text encoded together, with embeddings from different VL layers")

    text_layer_dir = Path("outputs/layer_tap_text")
    if not text_layer_dir.exists():
        st.warning("No layer tap text outputs found")
    else:
        text_layer_names = [
            ("layer12", "Layer 12"),
            ("layer18", "Layer 18"),
            ("layer24", "Layer 24"),
            ("layer30", "Layer 30"),
            ("layer34", "Layer 34 (baseline)"),
            ("layer35", "Layer 35 (final)"),
        ]
        text_prompts_list = ["baseline", "watercolor", "pencil", "cyberpunk",
                             "oil_paint", "sunset", "winter", "underwater"]

        # Find available images by looking at input files
        input_files = sorted(text_layer_dir.glob("*_input.png"))
        if not input_files:
            st.warning("Experiment still running")
        else:
            # Extract actual indices from filenames
            img_indices = [int(f.stem.split("_")[0]) for f in input_files]
            sel_pos = st.slider("Image", 0, len(img_indices) - 1, 0, key="layer_text_img")
            img_idx_lt = img_indices[sel_pos]

            # Show input
            input_path = text_layer_dir / f"{img_idx_lt:03d}_input.png"
            if input_path.exists():
                st.subheader(f"Input image {img_idx_lt:03d}")
                st.image(str(input_path), width=400)

            prompt_lt = st.selectbox("Text prompt", text_prompts_list, key="layer_text_prompt")

            st.subheader(f"Outputs — '{prompt_lt}' across layers")
            cols = st.columns(len(text_layer_names))
            for i, (lkey, llabel) in enumerate(text_layer_names):
                img_path = text_layer_dir / f"{img_idx_lt:03d}_{lkey}_{prompt_lt}.png"
                if img_path.exists():
                    cols[i].image(str(img_path), caption=llabel, use_container_width=True)

            # Also show all prompts for selected layer
            layer_lt = st.selectbox("Or pick a layer to see all prompts",
                                    [l[0] for l in text_layer_names],
                                    index=4, key="layer_text_layer")
            st.subheader(f"All prompts — {layer_lt}")
            cols = st.columns(4)
            for i, p in enumerate(text_prompts_list):
                col = cols[i % 4]
                img_path = text_layer_dir / f"{img_idx_lt:03d}_{layer_lt}_{p}.png"
                if img_path.exists():
                    col.image(str(img_path), caption=p, use_container_width=True)
