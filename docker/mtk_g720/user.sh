#!/usr/bin/env bash
# 由创建和进入入口加载,将容器账户映射到当前宿主机普通用户.
MTK_CONTAINER_UID="$(id -u)"
MTK_CONTAINER_GID="$(id -g)"
if [[ "${MTK_CONTAINER_UID}" == 0 ]]; then
    echo "[错误] 请用宿主机普通用户运行 Docker 入口,不要使用 sudo bash." >&2
    return 1
fi

MTK_CONTAINER_HOME="$(docker exec --user 0 -i "${CONTAINER_NAME}" bash -s -- \
    "${MTK_CONTAINER_UID}" "${MTK_CONTAINER_GID}" "$(id -G)" <<'BASH'
set -euo pipefail
HOST_UID="$1"
HOST_GID="$2"
HOST_GROUPS="$3"
if ! getent group "${HOST_GID}" >/dev/null; then
    groupadd --gid "${HOST_GID}" "host_${HOST_GID}"
fi
if ! getent passwd "${HOST_UID}" >/dev/null; then
    useradd --uid "${HOST_UID}" --gid "${HOST_GID}" --create-home \
        --home-dir "/home/mtk_${HOST_UID}" --shell /bin/bash "mtk_${HOST_UID}"
fi
CONTAINER_USER="$(getent passwd "${HOST_UID}" | cut -d: -f1)"
CONTAINER_HOME="$(getent passwd "${HOST_UID}" | cut -d: -f6)"
if [[ "${CONTAINER_HOME}" != /home/* || "${CONTAINER_HOME}" == *'/../'* ]]; then
    echo "[错误] 普通用户 HOME 必须位于 /home/ 下: ${CONTAINER_HOME}." >&2
    exit 1
fi
# 保留宿主机共享目录和 GPU 设备所需的附加组,不继承 root 组.
DEVICE_GROUPS="$(
    for DEVICE in /dev/nvidia* /dev/nvidia-caps/* /dev/dri/*; do
        if [[ -c "${DEVICE}" ]]; then
            stat -c '%g' "${DEVICE}"
        fi
    done
)"
for GROUP_ID in ${HOST_GROUPS} ${DEVICE_GROUPS}; do
    if [[ "${GROUP_ID}" == 0 ]]; then
        continue
    fi
    if ! getent group "${GROUP_ID}" >/dev/null; then
        groupadd --gid "${GROUP_ID}" "host_${GROUP_ID}"
    fi
    usermod --append --groups "${GROUP_ID}" "${CONTAINER_USER}"
done
install -d -o "${HOST_UID}" -g "${HOST_GID}" -m 755 \
    "${CONTAINER_HOME}" "${CONTAINER_HOME}/.cache"
# 只开放共享临时根目录和当天目录,不递归修改已有任务数据.
install -d -m 1777 /tmp/hailongcodex "/tmp/hailongcodex/$(date +%F)"
printf '%s\n' "${CONTAINER_HOME}"
BASH
)" || return 1

# 首次初始化复制当前用户的 SSH 配置,已有配置不覆盖,凭据不进入仓库.
if [[ -d "${HOME}/.ssh" ]] && \
        ! docker exec --user 0 "${CONTAINER_NAME}" test -e "${MTK_CONTAINER_HOME}/.ssh"; then
    docker cp "${HOME}/.ssh" "${CONTAINER_NAME}:${MTK_CONTAINER_HOME}/.ssh" || return 1
    docker exec --user 0 "${CONTAINER_NAME}" sh -c '
        chown -R --no-dereference "$1:$2" "$3"
        find -P "$3" -type d -exec chmod 700 {} +
        find -P "$3" -type f -exec chmod 600 {} +
    ' sh "${MTK_CONTAINER_UID}" "${MTK_CONTAINER_GID}" "${MTK_CONTAINER_HOME}/.ssh" || return 1
fi
