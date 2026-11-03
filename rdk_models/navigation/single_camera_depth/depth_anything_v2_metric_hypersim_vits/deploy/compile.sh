#!/usr/bin/env bash
# 主机转换、量化、编译和部署;板端只做通路检查与示例.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    echo "[S100 板端] 少量通路检查与示例."
    python3 "${SCRIPT_DIR}/board/infer.py"
else
    : "${RDK_MODELS_ROOT:?请先进入 docker/s100/enter_container.sh 对应环境}"
    echo "[S100 1/4] 导出室内米制深度模型."
    python3 "${SCRIPT_DIR}/host/export.py"
    echo "[S100 2/4] 使用 NYU train 准备校准数据."
    python3 "${SCRIPT_DIR}/host/calibrate.py"
    echo "[S100 3/4] 量化与编译."
    python3 "${SCRIPT_DIR}/host/compile.py"
    echo "[S100 4/4] 打包并上传,不自动启动板端推理."
    python3 "${SCRIPT_DIR}/host/deploy.py"
fi
