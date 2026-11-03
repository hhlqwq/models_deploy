#!/usr/bin/env bash
# 独立评价已有模型,主机仅上传评价参考,不重新编译或部署模型.
set -euo pipefail
export MTK_OUTPUT_METRICS=0
export MTK_EVALUATION_WORK=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MTK_EVAL_SCOPE=full

if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    export PYTHONDONTWRITEBYTECODE=1
    exec 9>"${SCRIPT_DIR}/.run.lock"
    flock -n 9 || { echo "[错误] 本模型已有运行任务." >&2; exit 2; }
    echo "[板端 1/2] 全量 ImageNet 推理、必要预热与性能采集."
    python3 "${SCRIPT_DIR}/board/infer.py"
    echo "[板端 2/2] 核对覆盖和任务精度参考,发布唯一 summary.json."
    python3 "${SCRIPT_DIR}/board/evaluate.py"
else
    : "${MTK_MODELS_ROOT:?请先进入已有编译容器}"
    echo "[容器 1/2] 使用已有 FP32 ONNX 计算全量 Top-1."
    python "${SCRIPT_DIR}/host/evaluate_onnx.py"
    echo "[容器 2/2] 将同协议参考上传至现有板端部署包."
    python "${MTK_MODELS_ROOT}/tools/deploy.py" --upload-reference "${SCRIPT_DIR}"
fi
