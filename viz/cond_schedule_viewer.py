#!/usr/bin/env python3
"""Streamlit viewer for conditioning schedule experiment results."""

import streamlit as st
from pathlib import Path
from PIL import Image

st.set_page_config(layout="wide", page_title="Conditioning Schedule Viewer")
st.title("Conditioning Schedule Experiment")

OUT_DIR = Path("outputs/cond_schedule")
pairs = sorted([p for p in OUT_DIR.iterdir() if p.is_dir() and p.name.startswith("pair_")])

if not pairs:
    st.error(f"No pair directories found in {OUT_DIR}")
    st.stop()

pair_idx = st.sidebar.slider("Pair", 0, len(pairs) - 1, 0)
pair_dir = pairs[pair_idx]

# Show inputs
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
    st.subheader(f"{exp_title}")
    st.caption(exp_desc)
    cols = st.columns(len(switch_points))
    for i, sp in enumerate(switch_points):
        img_path = pair_dir / f"{exp_name}_s{sp}.png"
        if img_path.exists():
            cols[i].image(str(img_path), caption=f"switch={sp}", use_container_width=True)
        else:
            cols[i].write(f"Missing: {img_path.name}")

st.sidebar.markdown("""
### How to read
- **switch=0**: Use late conditioning for ALL steps (s0 means switch at step 0)
- **switch=8**: Use early conditioning for ALL steps (never switch)
- **switch=4**: Early for 4 steps, late for 4 steps

### Experiments
- **A→A+B**: s0=both images, s8=A only
- **B→A+B**: s0=both images, s8=B only
- **A→B**: s0=B only, s8=A only
""")
