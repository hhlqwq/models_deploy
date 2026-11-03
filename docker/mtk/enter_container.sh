#!/usr/bin/env bash

set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "$(id -u)" == 0 ]]; then
    echo "[错误] 请用宿主机普通用户进入容器,不要使用 sudo bash." >&2
    exit 1
fi
source "${PROJECT_ROOT}/tools/storage.sh"
readonly CONTAINER_NAME="${MTK_CONTAINER_NAME}"

# 复用已运行容器,初始化普通账户即可,不安装依赖或重启容器.
source "${PROJECT_ROOT}/docker/mtk/user.sh"
echo "[进入容器] 编译用户 ${MTK_CONTAINER_UID}:${MTK_CONTAINER_GID},HOME=${MTK_CONTAINER_HOME}."
exec docker exec -it --user "${MTK_CONTAINER_UID}" \
    --env HOME="${MTK_CONTAINER_HOME}" \
    --env XDG_CACHE_HOME="${MTK_CONTAINER_HOME}/.cache" \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --workdir "${PROJECT_ROOT}" "${CONTAINER_NAME}" bash -l
