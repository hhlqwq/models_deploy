#!/usr/bin/env bash
# 进入已运行的 S100 容器,不启动、重启或修改其他容器.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RDK_ENVIRONMENT="$(python3 "${PROJECT_ROOT}/tools/rdk_config.py")"
eval "${RDK_ENVIRONMENT}"
echo "[S100] 进入独立编译容器."
exec docker exec -it --user "$(id -u):$(id -g)" \
    --workdir "${PROJECT_ROOT}" "${RDK_CONTAINER_NAME}" bash
