#!/bin/bash
# =============================================================================
# Overnight baseline runs (resumable)
#
# 1) i2i on all eval images (84 images)
# 2) t2i on all eval captions (extracted from CSV, skipped if empty)
# 3) Multi-image composition with blend_mode x alpha variations
#
# Resume: just re-run the script. Completed runs (meta.json present) are skipped.
#
# Usage:
#   screen -S baselines
#   bash run_baselines.sh 2>&1 | tee outputs/baselines.log
# =============================================================================

set -e

EVAL_CSV="eval_unified/eval.csv"
MULTI_JSONL="eval_unified/composition_prompts.jsonl"
BASE_OUT="outputs/baselines_feb25"

# Helper: skip run if meta.json already exists in output dir
run_if_needed() {
    local out_dir="$1"
    shift
    if [ -f "${out_dir}/meta.json" ]; then
        echo "  SKIP (already complete): ${out_dir}"
        return 0
    fi
    "$@"
}

echo "============================================="
echo "Baseline runs — $(date)"
echo "============================================="

# -----------------------------------------------------------------
# 1) i2i on all eval images
# -----------------------------------------------------------------
echo ""
echo ">>> [1/3] i2i on all eval images"
run_if_needed ${BASE_OUT}/i2i_all \
    python3 inference.py \
        --image eval_unified/images/ \
        --output_dir ${BASE_OUT}/i2i_all \
        --seed 42

# -----------------------------------------------------------------
# 2) t2i on all eval captions (skip if captions not ready)
# -----------------------------------------------------------------
echo ""
echo ">>> [2/3] t2i on eval captions"

CAPTIONS_TXT="${BASE_OUT}/captions_tmp.txt"
mkdir -p ${BASE_OUT}

# Extract captions from CSV (exit 1 if none found — caught by ||)
python3 -c "
import csv, sys
with open('${EVAL_CSV}') as f:
    rows = list(csv.DictReader(f))
caps = [row['caption'].strip() for row in rows]
n_filled = sum(1 for c in caps if c)
if n_filled == 0:
    print('No captions found in CSV — skipping t2i run')
    sys.exit(1)
if n_filled < len(caps):
    print(f'WARNING: only {n_filled}/{len(caps)} captions filled')
print(f'Extracted {n_filled} captions')
with open('${CAPTIONS_TXT}', 'w') as f:
    for c in caps:
        f.write((c if c else '') + '\n')
" && {
    run_if_needed ${BASE_OUT}/t2i_all \
        python3 inference.py \
            --prompt ${CAPTIONS_TXT} \
            --image eval_unified/images/ \
            --output_dir ${BASE_OUT}/t2i_all \
            --seed 42 \
            --metrics
    rm -f ${CAPTIONS_TXT}
} || echo "  (skipped — captions not populated yet)"

# -----------------------------------------------------------------
# 3) Multi-image composition baselines
# -----------------------------------------------------------------
echo ""
echo ">>> [3/3] Multi-image composition baselines"

# B3: concat (default — full cross-attention)
echo ""
echo "--- B3: concat (cross-attention) ---"
run_if_needed ${BASE_OUT}/multi_concat \
    python3 inference_multi_image.py \
        --input ${MULTI_JSONL} \
        --output_dir ${BASE_OUT}/multi_concat \
        --seed 42

# B5: avg (no cross-attention) — alpha sweep
for ALPHA in 0.3 0.5 0.7 0.9; do
    echo ""
    echo "--- B5: avg, alpha=${ALPHA} ---"
    run_if_needed ${BASE_OUT}/multi_avg_a${ALPHA} \
        python3 inference_multi_image.py \
            --input ${MULTI_JSONL} \
            --blend_mode avg \
            --alpha ${ALPHA} \
            --output_dir ${BASE_OUT}/multi_avg_a${ALPHA} \
            --seed 42
done

# B6: scale (cross-attention + per-image token scaling) — alpha sweep
for ALPHA in 0.3 0.5 0.7 0.9; do
    echo ""
    echo "--- B6: scale, alpha=${ALPHA} ---"
    run_if_needed ${BASE_OUT}/multi_scale_a${ALPHA} \
        python3 inference_multi_image.py \
            --input ${MULTI_JSONL} \
            --blend_mode scale \
            --alpha ${ALPHA} \
            --output_dir ${BASE_OUT}/multi_scale_a${ALPHA} \
            --seed 42
done

# -----------------------------------------------------------------
# 4) Light composition prompts (less dense text)
# -----------------------------------------------------------------
LIGHT_JSONL="eval_unified/composition_light.jsonl"
NOTEXT_JSONL="eval_unified/composition_light_notext.jsonl"

echo ""
echo ">>> [4/5] Light composition prompts"

# B3: concat with text
echo ""
echo "--- Light B3: concat (with text) ---"
run_if_needed ${BASE_OUT}/light_concat \
    python3 inference_multi_image.py \
        --input ${LIGHT_JSONL} \
        --output_dir ${BASE_OUT}/light_concat \
        --seed 42

# B5: avg alpha=0.3
echo ""
echo "--- Light B5: avg, alpha=0.3 ---"
run_if_needed ${BASE_OUT}/light_avg_a0.3 \
    python3 inference_multi_image.py \
        --input ${LIGHT_JSONL} \
        --blend_mode avg \
        --alpha 0.3 \
        --output_dir ${BASE_OUT}/light_avg_a0.3 \
        --seed 42

# B6: scale alpha=0.3
echo ""
echo "--- Light B6: scale, alpha=0.3 ---"
run_if_needed ${BASE_OUT}/light_scale_a0.3 \
    python3 inference_multi_image.py \
        --input ${LIGHT_JSONL} \
        --blend_mode scale \
        --alpha 0.3 \
        --output_dir ${BASE_OUT}/light_scale_a0.3 \
        --seed 42

# -----------------------------------------------------------------
# 5) Caption-drop ablation (images only, no text)
# -----------------------------------------------------------------
echo ""
echo ">>> [5/5] Caption-drop ablation (images only)"

# B3: concat, no text
echo ""
echo "--- NoText B3: concat (images only) ---"
run_if_needed ${BASE_OUT}/notext_concat \
    python3 inference_multi_image.py \
        --input ${NOTEXT_JSONL} \
        --output_dir ${BASE_OUT}/notext_concat \
        --seed 42

# B5: avg alpha=0.3, no text
echo ""
echo "--- NoText B5: avg, alpha=0.3 ---"
run_if_needed ${BASE_OUT}/notext_avg_a0.3 \
    python3 inference_multi_image.py \
        --input ${NOTEXT_JSONL} \
        --blend_mode avg \
        --alpha 0.3 \
        --output_dir ${BASE_OUT}/notext_avg_a0.3 \
        --seed 42

# B6: scale alpha=0.3, no text
echo ""
echo "--- NoText B6: scale, alpha=0.3 ---"
run_if_needed ${BASE_OUT}/notext_scale_a0.3 \
    python3 inference_multi_image.py \
        --input ${NOTEXT_JSONL} \
        --blend_mode scale \
        --alpha 0.3 \
        --output_dir ${BASE_OUT}/notext_scale_a0.3 \
        --seed 42

echo ""
echo "============================================="
echo "All baselines done — $(date)"
echo "============================================="
echo ""
echo "Output structure:"
echo "  ${BASE_OUT}/i2i_all/              — 84 i2i reconstructions"
echo "  ${BASE_OUT}/t2i_all/              — 84 t2i from captions (if available)"
echo "  ${BASE_OUT}/multi_concat/         — B3: interleaved cross-attention (dense prompts)"
echo "  ${BASE_OUT}/multi_avg_a{N}/       — B5: no cross-attn, alpha sweep"
echo "  ${BASE_OUT}/multi_scale_a{N}/     — B6: cross-attn + token scaling, alpha sweep"
echo "  ${BASE_OUT}/light_concat/         — B3: light prompts"
echo "  ${BASE_OUT}/light_avg_a0.3/       — B5: light prompts, alpha=0.3"
echo "  ${BASE_OUT}/light_scale_a0.3/     — B6: light prompts, alpha=0.3"
echo "  ${BASE_OUT}/notext_concat/        — B3: images only (caption drop ablation)"
echo "  ${BASE_OUT}/notext_avg_a0.3/      — B5: images only, alpha=0.3"
echo "  ${BASE_OUT}/notext_scale_a0.3/    — B6: images only, alpha=0.3"
