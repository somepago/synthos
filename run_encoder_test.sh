#!/bin/bash
# Test text encoders: t2i + i2i
# qwen3vl now loads only visual encoder (~1.5GB) + reuses Z-Image text encoder
# qwen3vl-native loads full VL model (~9GB, replaces text encoder)
#
# Memory killswitch: monitor thread kills the test if memory > 80%
set -e

cd /home/gnan/projects/synthos
OUT=outputs/encoder_test
IMG=eval/000371083.jpg
PROMPT="a cat sitting on a chair"
MEM_LIMIT=80

# Start memory watchdog in background
(
    while true; do
        MEM_USED=$(free | awk '/^Mem:/ {printf "%.0f", $3/$2 * 100}')
        if [ "$MEM_USED" -ge "$MEM_LIMIT" ]; then
            echo ""
            echo "!!! MEMORY KILLSWITCH: ${MEM_USED}% >= ${MEM_LIMIT}% — killing all test processes !!!"
            # Kill any python inference processes spawned by this script
            pkill -P $$ -f "inference.py" 2>/dev/null
            kill $$ 2>/dev/null
            exit 1
        fi
        sleep 2
    done
) &
WATCHDOG_PID=$!
trap "kill $WATCHDOG_PID 2>/dev/null" EXIT

echo "Memory watchdog started (PID=$WATCHDOG_PID, limit=${MEM_LIMIT}%)"
echo ""

echo "=========================================="
echo "TEXT-TO-IMAGE TESTS"
echo "=========================================="

echo ""
echo "--- t2i: qwen3 (default) ---"
venv/bin/python3 inference.py --prompt "$PROMPT" --text_encoder qwen3 --output $OUT/t2i_qwen3.png
echo ""

echo "--- t2i: qwen3vl (visual-only + Z-Image LLM) ---"
venv/bin/python3 inference.py --prompt "$PROMPT" --text_encoder qwen3vl --output $OUT/t2i_qwen3vl.png
echo ""

echo "--- t2i: qwen3vl-native (full VL, native weights) ---"
venv/bin/python3 inference.py --prompt "$PROMPT" --text_encoder qwen3vl-native --output $OUT/t2i_qwen3vl_native.png
echo ""

echo "=========================================="
echo "IMAGE-TO-IMAGE TESTS"
echo "=========================================="

echo ""
echo "--- i2i: qwen3vl (visual-only, ~1.5GB extra) ---"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl --edit_image $IMG --output $OUT/i2i_qwen3vl.png
echo ""

echo "--- i2i: qwen3vl-native (full VL model) ---"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl-native --edit_image $IMG --output $OUT/i2i_qwen3vl_native.png
echo ""

echo "=========================================="
echo "ALL DONE. Results in $OUT/"
echo "=========================================="
ls -la $OUT/
