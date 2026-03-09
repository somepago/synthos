#!/usr/bin/env python3
"""Generate a static HTML gallery for conditioning schedule experiment results."""

import base64
import json
from pathlib import Path

OUT_DIR = Path("outputs/cond_schedule")
HTML_OUT = OUT_DIR / "gallery.html"

pairs = sorted([p for p in OUT_DIR.iterdir() if p.is_dir() and p.name.startswith("pair_")])
results = json.loads((OUT_DIR / "results.json").read_text())

switch_points = [0, 2, 4, 6, 8]
experiments = [
    ("a_to_ab", "A → A+B", "Start with A only, introduce A+B at switch point"),
    ("b_to_ab", "B → A+B", "Start with B only, introduce A+B at switch point"),
    ("a_to_b", "A → B", "Start with A only, switch to B only at switch point"),
]


def img_to_b64(path):
    data = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


html_parts = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Conditioning Schedule Experiment</title>
<style>
body { font-family: system-ui, sans-serif; background: #1a1a1a; color: #e0e0e0; margin: 20px; }
h1 { text-align: center; }
h2 { border-bottom: 1px solid #444; padding-bottom: 8px; margin-top: 40px; }
h3 { color: #aaa; margin-top: 24px; }
.pair-nav { text-align: center; margin: 20px 0; }
.pair-nav a { color: #6af; margin: 0 6px; text-decoration: none; }
.pair-nav a:hover { text-decoration: underline; }
.inputs { display: flex; gap: 20px; justify-content: center; margin: 16px 0; }
.inputs img { max-width: 300px; max-height: 300px; border: 2px solid #555; border-radius: 4px; }
.inputs .label { text-align: center; color: #aaa; font-size: 14px; margin-top: 4px; }
.row { display: flex; gap: 8px; align-items: flex-start; margin: 8px 0; }
.row img { width: 18%; border: 1px solid #333; border-radius: 3px; }
.row .caption { text-align: center; font-size: 12px; color: #888; }
.exp-desc { color: #888; font-size: 13px; margin-bottom: 8px; }
.legend { background: #222; padding: 12px 16px; border-radius: 6px; margin: 16px 0; font-size: 13px; line-height: 1.6; }
</style></head><body>
<h1>Conditioning Schedule Experiment</h1>
<div class="legend">
<b>How to read:</b><br>
<b>switch=0</b>: Use late conditioning for ALL steps &nbsp;|&nbsp;
<b>switch=8</b>: Use early conditioning for ALL steps (never switch) &nbsp;|&nbsp;
<b>switch=4</b>: Early for 4 steps, late for 4 steps<br>
<b>A→A+B</b>: s0 = both images baseline, s8 = A only &nbsp;|&nbsp;
<b>B→A+B</b>: s0 = both images baseline, s8 = B only &nbsp;|&nbsp;
<b>A→B</b>: s0 = B only, s8 = A only
</div>
"""]

# Navigation
html_parts.append('<div class="pair-nav">')
for pair_dir in pairs:
    name = pair_dir.name
    html_parts.append(f'<a href="#{name}">{name.replace("pair_", "P")}</a>')
html_parts.append('</div>')

for pair_dir in pairs:
    pair_name = pair_dir.name
    info = results.get(pair_name, {})
    img_a_src = info.get("img_a", "?")
    img_b_src = info.get("img_b", "?")

    html_parts.append(f'<h2 id="{pair_name}">{pair_name}</h2>')
    html_parts.append(f'<div style="color:#888;font-size:13px">A: {img_a_src} &nbsp;|&nbsp; B: {img_b_src}</div>')

    # Input images
    html_parts.append('<div class="inputs">')
    for label, fname in [("Image A", "input_a.png"), ("Image B", "input_b.png")]:
        img_path = pair_dir / fname
        if img_path.exists():
            b64 = img_to_b64(img_path)
            html_parts.append(f'<div><img src="{b64}"><div class="label">{label}</div></div>')
    html_parts.append('</div>')

    # Experiments
    for exp_name, exp_title, exp_desc in experiments:
        html_parts.append(f'<h3>{exp_title}</h3>')
        html_parts.append(f'<div class="exp-desc">{exp_desc}</div>')
        html_parts.append('<div class="row">')
        for sp in switch_points:
            img_path = pair_dir / f"{exp_name}_s{sp}.png"
            if img_path.exists():
                b64 = img_to_b64(img_path)
                html_parts.append(f'<div><img src="{b64}"><div class="caption">switch={sp}</div></div>')
        html_parts.append('</div>')

html_parts.append('<div style="text-align:center;color:#555;margin:40px 0">End of gallery</div>')
html_parts.append('</body></html>')

html = "\n".join(html_parts)
HTML_OUT.write_text(html)
print(f"Gallery written to {HTML_OUT} ({len(html) / 1e6:.1f} MB)")
