#!/usr/bin/env bash
# 使用本地已有官方工具链镜像创建独立容器,不下载或安装资源.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RDK_ENVIRONMENT="$(python3 "${PROJECT_ROOT}/tools/rdk_config.py")"
eval "${RDK_ENVIRONMENT}"
echo "[S100 1/3] 检查已有镜像与 NAS 挂载."
docker image inspect "${RDK_IMAGE}" >/dev/null
test -d "${RDK_MODEL_STORAGE_ROOT}"
test -d "${RDK_DATASET_ROOT}"
test -d "${RDK_TOOLCHAIN_ROOT}"
findmnt -T "${RDK_MODEL_STORAGE_ROOT}" -n -o FSTYPE | grep -Eq '^(cifs|nfs|nfs4)$'
findmnt -T "${RDK_DATASET_ROOT}" -n -o FSTYPE | grep -Eq '^(cifs|nfs|nfs4)$'
findmnt -T "${RDK_TOOLCHAIN_ROOT}" -n -o FSTYPE | grep -Eq '^(cifs|nfs|nfs4)$'
echo "[S100 2/3] 创建独立容器,同名容器存在时直接报错."
docker run --detach --pull never \
    --name "${RDK_CONTAINER_NAME}" \
    --entrypoint /bin/bash \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    --env HOME=/tmp \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONUNBUFFERED=1 \
    --env PYTHONPATH="${PROJECT_ROOT}" \
    --env RDK_MODELS_ROOT="${PROJECT_ROOT}" \
    --volume "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --volume "${RDK_MODEL_STORAGE_ROOT}:${RDK_MODEL_STORAGE_ROOT}" \
    --volume "${RDK_DATASET_ROOT}:${RDK_DATASET_ROOT}:ro" \
    --volume "${RDK_TOOLCHAIN_ROOT}:${RDK_TOOLCHAIN_ROOT}:ro" \
    --workdir "${PROJECT_ROOT}" \
    "${RDK_IMAGE}" -c 'exec sleep infinity'
echo "[S100 3/3] 容器已创建,使用 docker/s100/enter_container.sh 进入."
