#!/bin/bash
# Test inference.py: t2i (qwen3) + i2i (qwen3vl splice) at multiple resolutions
set -e
cd /home/gnan/projects/synthos

OUT=outputs/splice_test
mkdir -p $OUT

echo "=== T2I tests (qwen3, default text encoder) ==="

echo "[t2i 1/3] 512x512"
venv/bin/python3 inference.py \
  --prompt "a delightful scene of four Cavalier King Charles Spaniels in a garden setting" \
  --output $OUT/t2i_dogs_512x512.png --height 512 --width 512 --seed 42

echo "[t2i 2/3] 768x512"
venv/bin/python3 inference.py \
  --prompt "Silhouette of a black woman in profile, wearing a vibrant yellow dress, surrounded by flowers" \
  --output $OUT/t2i_silhouette_768x512.png --height 768 --width 512 --seed 7

echo "[t2i 3/3] 512x768"
venv/bin/python3 inference.py \
  --prompt "A man sitting inside a car, natural candid moment, film-frame stillness, high-end fashion editorial" \
  --output $OUT/t2i_car_512x768.png --height 512 --width 768 --seed 99

echo ""
echo "=== I2I tests (qwen3vl splice) ==="

echo "[i2i 1/4] 000371083.jpg @ 512x512"
venv/bin/python3 inference.py \
  --prompt "" --text_encoder qwen3vl \
  --edit_image eval/000371083.jpg \
  --output $OUT/i2i_worker_512x512.png --height 512 --width 512 --seed 42

echo "[i2i 2/4] 000481713.jpg @ 512x512"
venv/bin/python3 inference.py \
  --prompt "" --text_encoder qwen3vl \
  --edit_image eval/000481713.jpg \
  --output $OUT/i2i_481713_512x512.png --height 512 --width 512 --seed 42

echo "[i2i 3/4] 000371083.jpg @ 768x512"
venv/bin/python3 inference.py \
  --prompt "" --text_encoder qwen3vl \
  --edit_image eval/000371083.jpg \
  --output $OUT/i2i_worker_768x512.png --height 768 --width 512 --seed 42

echo "[i2i 4/4] 001574009.jpg @ 512x768"
venv/bin/python3 inference.py \
  --prompt "" --text_encoder qwen3vl \
  --edit_image eval/001574009.jpg \
  --output $OUT/i2i_1574009_512x768.png --height 512 --width 768 --seed 42

echo ""
echo "=== ALL DONE ==="
ls -la $OUT/*.png
