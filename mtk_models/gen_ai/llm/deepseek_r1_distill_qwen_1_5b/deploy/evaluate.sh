#!/usr/bin/env bash
# 独立评价已有模型的精度、性能与峰值 RSS,不量化、编译或部署.
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
    echo "[全量 1/3] 已有模型性能与 RSS,运行编号: ${EVAL_RUN_ID}."
    python3 "${SCRIPT_DIR}/board/infer.py"
    echo "[全量 2/3] 完整 MMLU Accuracy 和 WikiText2 PPL."
    python3 "${SCRIPT_DIR}/board/evaluate.py"
    echo "[全量 3/3] 保存任务精度、性能与峰值 RSS 到唯一 summary.json."
    python3 "${SCRIPT_DIR}/board/summary.py"
else
    : "${MTK_MODELS_ROOT:?请先进入已有编译容器,仅对已有浮点模型准备评价参考}"
    echo "[全量参考] 仅推理已有浮点模型,不导出、量化或编译,完成后自动上传参考."
    python "${SCRIPT_DIR}/host/evaluate.py"
    python "${MTK_MODELS_ROOT}/tools/deploy.py" --upload-reference "${SCRIPT_DIR}"
fi
