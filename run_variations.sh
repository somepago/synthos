#!/bin/bash
# =============================================================================
# Variation Strength Ablation: max_pixels sweep on first 20 eval images
#
# Tests how VL token count (controlled via max_pixels) affects i2i fidelity.
# Lower max_pixels = fewer tokens = coarser representation = stronger variation.
#
# Resume: re-run the script. Completed runs (meta.json present) are skipped.
#
# Usage:
#   screen -S variations
#   bash run_variations.sh 2>&1 | tee outputs/variations.log
# =============================================================================

set -e

BASE_OUT="outputs/baselines_feb25"
IMAGE_DIR="eval_unified/images/"
N_SAMPLES=20

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
echo "Variation strength ablation (${N_SAMPLES} images) — $(date)"
echo "============================================="

# Level -> max_pixels mapping
declare -A LEVELS
LEVELS[very_strong]=16384    # 128x128
LEVELS[strong]=65536         # 256x256
LEVELS[medium]=147456        # 384x384
LEVELS[default]=262144       # 512x512
LEVELS[subtle]=589824        # 768x768
LEVELS[very_subtle]=802816   # 896x896 (conservative — avoid CUBLAS OOM)

# Run order: from strongest variation to most subtle
for LEVEL in very_strong strong medium default subtle very_subtle; do
    MP=${LEVELS[$LEVEL]}
    OUT_DIR="${BASE_OUT}/vary_${LEVEL}"
    echo ""
    echo "--- vary_${LEVEL} (max_pixels=${MP}) ---"
    run_if_needed "${OUT_DIR}" \
        python3 inference.py \
            --image "${IMAGE_DIR}" \
            --output_dir "${OUT_DIR}" \
            --max_pixels ${MP} \
            --n_samples ${N_SAMPLES} \
            --height 1024 --width 1024 \
            --seed 42
done

echo ""
echo "============================================="
echo "All variation runs complete — $(date)"
echo "============================================="
