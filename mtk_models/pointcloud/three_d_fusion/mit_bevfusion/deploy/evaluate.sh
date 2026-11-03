#!/usr/bin/env bash
# 评价独立使用已有子图和官方 val 数据,不重新转换或编译.
set -euo pipefail
export MTK_OUTPUT_METRICS=0
export MTK_EVALUATION_WORK=1
export MTK_EVAL_SCOPE=full
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${HOME}/.local/share/mtk_models/bevfusion-evaluation${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    export PYTHONDONTWRITEBYTECODE=1
    export OPENBLAS_NUM_THREADS=4
    export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    exec 9>"${SCRIPT_DIR}/.run.lock"
    flock -n 9 || { echo "[错误] 本模型已有任务运行." >&2; exit 2; }
    python3 "${SCRIPT_DIR}/board/evaluate.py"
else
    : "${MTK_MODELS_ROOT:?请先进入已有编译容器}"
    python "${SCRIPT_DIR}/host/prepare.py" --evaluate
    python "${SCRIPT_DIR}/host/reference.py"
    python "${MTK_MODELS_ROOT}/tools/deploy.py" --upload-reference "${SCRIPT_DIR}"
fi
