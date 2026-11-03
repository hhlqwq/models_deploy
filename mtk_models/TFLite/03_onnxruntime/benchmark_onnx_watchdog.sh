#!/bin/bash
set -u

ROOT="${MTK_ONNX_ROOT:-/opt/mtk/ort}"
BENCH="/usr/share/onnxruntime_benchmark/onnxruntime_benchmark.py"

RUNS="${RUNS:-10}"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"
POLL_SEC="${POLL_SEC:-1}"
AUTO_REBOOT_ON_OOPS="${AUTO_REBOOT_ON_OOPS:-1}"

STATE_DIR="$ROOT/.benchmark_state"
DONE_FILE="$STATE_DIR/done_models.txt"
SKIP_FILE="$STATE_DIR/kernel_oops_models.txt"
CURRENT_FILE="$STATE_DIR/current_model.txt"

SUMMARY="$ROOT/benchmark_onnx_summary.csv"
ALL_LOG="$ROOT/benchmark_onnx_all.log"

mkdir -p "$STATE_DIR"
touch "$DONE_FILE" "$SKIP_FILE" "$ALL_LOG" "$CURRENT_FILE"

if [ ! -f "$SUMMARY" ]; then
    echo "model,status,avg_ms,min_ms,max_ms,log" > "$SUMMARY"
fi

# Known bad models already observed on this board.
KNOWN_BAD=(
"/opt/mtk/ort/Legacy/Classification/DenseNet/densenet121_quant.onnx"
)

for m in "${KNOWN_BAD[@]}"; do
    grep -Fxq "$m" "$SKIP_FILE" || echo "$m" >> "$SKIP_FILE"
done

mark_summary() {
    local model="$1"
    local status="$2"
    local avg="${3:-}"
    local min="${4:-}"
    local max="${5:-}"
    local log="${6:-}"
    echo "\"$model\",$status,$avg,$min,$max,\"$log\"" >> "$SUMMARY"
}

kernel_bad_since() {
    local base="$1"
    dmesg 2>/dev/null | tail -n +"$((base + 1))" | \
        grep -Eqi 'Internal error: Oops|Kernel panic|Unable to handle kernel|BUG:|Oops:'
}

append_new_dmesg() {
    local base="$1"
    local log="$2"
    {
        echo
        echo "===== NEW KERNEL LOG ====="
        dmesg 2>/dev/null | tail -n +"$((base + 1))"
    } >> "$log"
}

safe_kill_group() {
    local pid="$1"
    kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    sleep 1
    kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
}

find "$ROOT" -type f -name "*.onnx" | sort | while IFS= read -r MODEL; do

    if grep -Fxq "$MODEL" "$DONE_FILE"; then
        echo "[DONE] $MODEL"
        continue
    fi

    if grep -Fxq "$MODEL" "$SKIP_FILE"; then
        echo "[SKIP-KERNEL-OOPS] $MODEL"
        continue
    fi

    case "$MODEL" in
        *"/ConvNeXt/"*)
            echo "[SKIP-KNOWN-BAD] $MODEL"
            grep -Fxq "$MODEL" "$SKIP_FILE" || echo "$MODEL" >> "$SKIP_FILE"
            continue
            ;;
    esac

    REL="${MODEL#$ROOT/}"
    SAFE_NAME="$(printf '%s' "${REL%.onnx}" | sed 's#[/ ]#_#g; s#[^A-Za-z0-9._-]#_#g')"
    LOG="$ROOT/benchmark_${SAFE_NAME}.log"

    printf '%s\n' "$MODEL" > "$CURRENT_FILE"
    sync

    echo
    echo "============================================================" | tee -a "$ALL_LOG"
    echo "MODEL: $MODEL" | tee -a "$ALL_LOG"
    echo "LOG  : $LOG" | tee -a "$ALL_LOG"
    echo "============================================================" | tee -a "$ALL_LOG"

    : > "$LOG"

    BASE_DMESG_LINES="$(dmesg 2>/dev/null | wc -l)"
    START_TS="$(date +%s)"

    setsid python3 "$BENCH" \
        --model "$MODEL" \
        --execution_provider NeuronExecutionProvider \
        --num_runs "$RUNS" \
        --num_threads 1 \
        --neuron_flag_use_fp16 1 \
        --neuron_flag_min_group_size 0 \
        > "$LOG" 2>&1 &

    PID=$!
    STATUS=""
    NEED_REBOOT=0

    while kill -0 "$PID" 2>/dev/null; do
        sleep "$POLL_SEC"

        if kernel_bad_since "$BASE_DMESG_LINES"; then
            STATUS="KERNEL_OOPS"
            NEED_REBOOT=1

            append_new_dmesg "$BASE_DMESG_LINES" "$LOG"
            grep -Fxq "$MODEL" "$SKIP_FILE" || echo "$MODEL" >> "$SKIP_FILE"
            mark_summary "$MODEL" "$STATUS" "" "" "" "$LOG"

            echo "[KERNEL OOPS] $MODEL" | tee -a "$ALL_LOG"
            safe_kill_group "$PID"
            break
        fi

        NOW_TS="$(date +%s)"
        ELAPSED=$((NOW_TS - START_TS))

        if [ "$ELAPSED" -ge "$TIMEOUT_SEC" ]; then
            STATUS="TIMEOUT"
            echo "[TIMEOUT ${TIMEOUT_SEC}s] $MODEL" | tee -a "$LOG" "$ALL_LOG"

            append_new_dmesg "$BASE_DMESG_LINES" "$LOG"
            safe_kill_group "$PID"

            if kill -0 "$PID" 2>/dev/null; then
                STATUS="TIMEOUT_D_STATE"
                NEED_REBOOT=1
            fi

            mark_summary "$MODEL" "$STATUS" "" "" "" "$LOG"
            echo "$MODEL" >> "$DONE_FILE"
            break
        fi
    done

    if [ -z "$STATUS" ]; then
        wait "$PID" 2>/dev/null || true
        cat "$LOG" >> "$ALL_LOG"

        if grep -q "Average Inference Time:" "$LOG"; then
            STATUS="PASS"
        else
            STATUS="FAIL"
        fi

        AVG="$(grep "Average Inference Time:" "$LOG" | tail -1 | awk '{print $(NF-1)}')"
        MIN="$(grep "Min Inference Time:" "$LOG" | tail -1 | awk '{print $(NF-1)}')"
        MAX="$(grep "Max Inference Time:" "$LOG" | tail -1 | awk '{print $(NF-1)}')"

        mark_summary "$MODEL" "$STATUS" "${AVG:-}" "${MIN:-}" "${MAX:-}" "$LOG"
        echo "$MODEL" >> "$DONE_FILE"
        echo "[RESULT] $STATUS avg=${AVG:-N/A} ms"
    fi

    : > "$CURRENT_FILE"
    sync

    if [ "$NEED_REBOOT" -eq 1 ]; then
        echo "[REBOOT] $STATUS detected, rebooting in 3 seconds..." | tee -a "$ALL_LOG"
        sync
        sleep 3
        reboot -f
        exit 100
    fi

done

echo
echo "============================================================"
echo "ALL AVAILABLE MODELS FINISHED"
echo "Summary : $SUMMARY"
echo "All log : $ALL_LOG"
echo "Skip    : $SKIP_FILE"
echo "============================================================"
