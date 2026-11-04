#!/usr/bin/env bash
# 主机编译双图并发布固定子集,板端验证同一批音频.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    export PYTHONDONTWRITEBYTECODE=1
    exec 9>"${SCRIPT_DIR}/.run.lock"
    flock -n 9
    python3 "${SCRIPT_DIR}/board/evaluate.py"
    python3 "${SCRIPT_DIR}/board/evaluate.py" --summarize
else
    : "${RDK_MODELS_ROOT:?请先进入已有 S100 编译容器}"
    python3 "${SCRIPT_DIR}/host/compile.py"
    python3 "${SCRIPT_DIR}/host/deploy.py"
fi
