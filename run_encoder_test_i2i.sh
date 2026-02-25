#!/bin/bash
# qwen3vl-native tests: t2i + i2i with/without M-RoPE
# Memory killswitch at 80%
set -e

cd /home/gnan/projects/synthos
OUT=outputs/encoder_test
IMG=eval/000371083.jpg
MEM_LIMIT=80

(
    while true; do
        MEM_USED=$(free | awk '/^Mem:/ {printf "%.0f", $3/$2 * 100}')
        if [ "$MEM_USED" -ge "$MEM_LIMIT" ]; then
            echo ""
            echo "!!! MEMORY KILLSWITCH: ${MEM_USED}% >= ${MEM_LIMIT}% !!!"
            pkill -P $$ -f "inference.py" 2>/dev/null
            kill $$ 2>/dev/null
            exit 1
        fi
        sleep 2
    done
) &
WATCHDOG_PID=$!
trap "kill $WATCHDOG_PID 2>/dev/null" EXIT

echo "Memory watchdog (limit=${MEM_LIMIT}%)"

echo ""
echo "--- t2i: qwen3vl-native ---"
venv/bin/python3 inference.py --prompt "a worker holding a wrench, flat illustration" --text_encoder qwen3vl-native --output $OUT/t2i_qwen3vl_native.png
echo ""

echo "--- i2i: qwen3vl-native (with M-RoPE) ---"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl-native --edit_image $IMG --output $OUT/i2i_qwen3vl_native.png
echo ""

echo "--- i2i: qwen3vl-native (WITHOUT M-RoPE) ---"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl-native --disable_mrope --edit_image $IMG --output $OUT/i2i_qwen3vl_native_no_mrope.png
echo ""

echo "ALL DONE."
ls -la $OUT/*native*.png 2>/dev/null
