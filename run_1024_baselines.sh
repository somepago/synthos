#!/bin/bash
# 1024px baseline runs: native t2i, VL t2i, VL i2i
# Runs sequentially — resume safe (checks meta.json).

set -e
source venv/bin/activate

EVAL_CSV="eval_unified/eval.csv"
BASE_OUT="outputs/baselines_feb25"

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
echo "1024px baseline runs — $(date)"
echo "============================================="

# -----------------------------------------------------------------
# 1) Native t2i (Qwen3 text encoder, no VL) at 1024
# -----------------------------------------------------------------
echo ""
echo ">>> [1/3] Native t2i (1024px)"
run_if_needed ${BASE_OUT}/native_t2i \
    python3 test_pure_turbo.py

# -----------------------------------------------------------------
# 2) VL i2i on all eval images at 1024
# -----------------------------------------------------------------
echo ""
echo ">>> [2/3] VL i2i (1024px)"
run_if_needed ${BASE_OUT}/i2i_all \
    python3 inference.py \
        --image eval_unified/images/ \
        --output_dir ${BASE_OUT}/i2i_all \
        --height 1024 --width 1024 \
        --seed 42

# -----------------------------------------------------------------
# 3) VL t2i on all eval captions at 1024
# -----------------------------------------------------------------
echo ""
echo ">>> [3/3] VL t2i (1024px)"

CAPTIONS_TXT="${BASE_OUT}/captions_tmp.txt"
mkdir -p ${BASE_OUT}

python3 -c "
import csv, sys
with open('${EVAL_CSV}') as f:
    rows = list(csv.DictReader(f))
caps = [row['caption'].strip() for row in rows]
n_filled = sum(1 for c in caps if c)
if n_filled == 0:
    print('No captions found in CSV — skipping t2i run')
    sys.exit(1)
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
            --height 1024 --width 1024 \
            --seed 42
    rm -f ${CAPTIONS_TXT}
} || echo "  (skipped — captions not populated yet)"

echo ""
echo "============================================="
echo "Done — $(date)"
echo "============================================="
