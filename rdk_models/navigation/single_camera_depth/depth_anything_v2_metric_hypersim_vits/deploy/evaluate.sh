#!/usr/bin/env bash
# 使用已有产物独立评价,不重新导出、量化或编译.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    echo "[S100 评价] NYU 标准 654 帧、Runtime 调用耗时与进程峰值 RSS."
    python3 "${SCRIPT_DIR}/board/evaluate.py"
else
    : "${RDK_MODELS_ROOT:?请先进入 S100 编译环境}"
    echo "[S100 评价准备] 已有 ONNX 的同协议参考与评价资源上传."
    python3 "${SCRIPT_DIR}/host/evaluate.py"
fi
