#!/usr/bin/env bash
# 使用现有 HBM 独立执行全量精度和性能评价.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    export PYTHONDONTWRITEBYTECODE=1
    exec 9>"${SCRIPT_DIR}/.run.lock"
    flock -n 9
    python3 "${SCRIPT_DIR}/board/evaluate.py" --full
    python3 "${SCRIPT_DIR}/board/evaluate.py" --full --summarize
else
    : "${RDK_MODELS_ROOT:?请先进入已有 S100 编译容器}"
    python3 "${SCRIPT_DIR}/host/deploy.py" --full
fi
