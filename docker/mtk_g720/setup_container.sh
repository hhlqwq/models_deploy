#!/usr/bin/env bash
# 构建阶段核对镜像默认 Python 和工具依赖,进入容器不执行安装.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${ROOT}/verify_environment.py" "${1:-}"
"${NCC_ROOT}/bin/ncc-tflite" --version
