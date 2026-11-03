#!/usr/bin/env bash
# 初始化当前用户并进入已运行的 S100 容器,不启动或重启容器.
set -euo pipefail
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "$(id -u)" == 0 ]]; then
    echo "[错误] 请用宿主机普通用户进入容器,不要使用 sudo bash." >&2
    exit 1
fi
RDK_ENVIRONMENT="$(python3 "${PROJECT_ROOT}/tools/rdk_config.py")"
eval "${RDK_ENVIRONMENT}"
# 复用已运行容器,初始化普通账户即可,不安装依赖或重启容器.
source "${PROJECT_ROOT}/docker/rdk_s100/user.sh"
echo "[S100] 进入独立编译容器,用户 ${RDK_CONTAINER_UID}:${RDK_CONTAINER_GID},HOME=${RDK_CONTAINER_HOME}."
# 只指定 UID,让 Docker 加载账户的主组及附加组.
exec docker exec -it --user "${RDK_CONTAINER_UID}" \
    --env HOME="${RDK_CONTAINER_HOME}" \
    --env XDG_CACHE_HOME="${RDK_CONTAINER_HOME}/.cache" \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --workdir "${PROJECT_ROOT}" "${RDK_CONTAINER_NAME}" bash -l
