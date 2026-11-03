#!/bin/bash

ROOT_DIR=${1:-${MTK_BOARD_MODELZOO_ROOT:-/opt/mtk/MTK_G720_DLA}}
COUNT=${COUNT:-1000}
RESULT="benchmark_g720.csv"
LOG_DIR="benchmark_logs"

mkdir -p "$LOG_DIR"

echo "model,path,status,latency_ms,fps" > "$RESULT"

find "$ROOT_DIR" -type f -name "*.dla" | while read -r MODEL
do
    NAME=$(basename "$MODEL")

    # 暂时跳过 LLM / VLM / ASR
    if echo "$MODEL" | grep -Eqi "llm|vlm|asr|whisper|qwen|moonshine"; then
        echo "[SKIP] $MODEL"
        continue
    fi

    LOG="$LOG_DIR/${NAME}.log"

    echo ""
    echo "========================================"
    echo "[RUN] $MODEL"
    echo "========================================"

    neuronrt \
        -m hw \
        -a "$MODEL" \
        --use-random-inputs \
        -c "$COUNT" \
        > "$LOG" 2>&1

    RET=$?

    if [ $RET -eq 0 ]; then

        LATENCY=$(grep "ms/inf" "$LOG" \
            | tail -1 \
            | sed -n 's/.*(\([0-9.]*\) ms\/inf).*/\1/p')

        FPS=$(grep "Avg. FPS" "$LOG" \
            | tail -1 \
            | awk -F ':' '{print $2}' \
            | xargs)

        echo "\"$NAME\",\"$MODEL\",\"OK\",\"$LATENCY\",\"$FPS\"" \
            >> "$RESULT"

        echo "[OK] latency=${LATENCY} ms, fps=${FPS}"

    else

        echo "\"$NAME\",\"$MODEL\",\"FAIL\",\"\",\"\"" \
            >> "$RESULT"

        echo "[FAIL] $MODEL"
    fi

done

echo ""
echo "========================================"
echo "Done"
echo "Result: $RESULT"
echo "Logs  : $LOG_DIR/"
echo "========================================"
