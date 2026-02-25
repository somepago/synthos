#!/bin/bash
# Base i2i test (50 steps, cfg=4.0, full VL splice)
set -e
cd /home/gnan/projects/synthos

OUT=outputs/splice_test3
mkdir -p $OUT

echo "=== Base i2i (full VL splice, 50 steps, cfg=4.0) ==="

echo "[base 1/3] worker 512x512"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl --model z-image-base \
  --edit_image eval/000371083.jpg --output $OUT/base_worker_512.png \
  --height 512 --width 512 --seed 42

echo "[base 2/3] fashion 512x512"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl --model z-image-base \
  --edit_image eval/000481713.jpg --output $OUT/base_fashion_512.png \
  --height 512 --width 512 --seed 42

echo "[base 3/3] captain 512x512"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl --model z-image-base \
  --edit_image eval/001574009.jpg --output $OUT/base_captain_512.png \
  --height 512 --width 512 --seed 42

echo ""
echo "=== ALL DONE ==="
ls -la $OUT/base_*.png
