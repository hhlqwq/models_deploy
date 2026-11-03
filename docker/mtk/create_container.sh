#!/usr/bin/env bash

set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly HOST_UID="$(id -u)"
readonly HOST_GID="$(id -g)"
if [[ "${HOST_UID}" == 0 ]]; then
    echo "[错误] 请用宿主机普通用户创建容器,不要使用 sudo bash." >&2
    exit 1
fi
source "${PROJECT_ROOT}/tools/storage.sh"
readonly DATASET_ROOT="${MTK_DATASET_ROOT}"
readonly MODEL_STORAGE_ROOT="${MTK_MODEL_STORAGE_ROOT}"
readonly TOOLCHAIN_ROOT="${MTK_TOOLCHAIN_ROOT}"
readonly IMAGE_NAME="${MTK_G720_IMAGE:-mtk/ai_toolchain_ubuntu_22_g720_gpu:np8.0.11}"
readonly CONTAINER_NAME="${MTK_CONTAINER_NAME}"

echo "[1/4] 检查目录和同名容器."
test -d "${PROJECT_ROOT}"
test -d "${DATASET_ROOT}"
test -d "${MODEL_STORAGE_ROOT}"
test -d "${TOOLCHAIN_ROOT}"
# 先确认 NAS 实际挂载,防止失去挂载时将资源写入同名本地目录.
for STORAGE_ROOT in "${MODEL_STORAGE_ROOT}" "${DATASET_ROOT}"; do
    STORAGE_TYPE="$(findmnt -T "${STORAGE_ROOT}" -n -o FSTYPE)"
    case "${STORAGE_TYPE}" in
        cifs|nfs|nfs4) ;;
        *) echo "[ERROR] 模型和数据集目录必须位于已挂载的 NAS." >&2; exit 1 ;;
    esac
done
# 缺少完整 LLM 镜像时先构建,依赖在镜像中预装.
IMAGE_LLM_VERSION="$(docker image inspect --format '{{index .Config.Labels "mtk.llm_sdk"}}' "${IMAGE_NAME}" 2>/dev/null || true)"
if [[ "${IMAGE_LLM_VERSION}" != "2.7.5" ]]; then
    bash "${PROJECT_ROOT}/docker/mtk/build_image.sh"
fi

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    if [[ "$(docker inspect --format '{{.Image}}' "${CONTAINER_NAME}")" != "$(docker image inspect --format '{{.Id}}' "${IMAGE_NAME}")" ]]; then
        echo "[ERROR] 同名容器为旧镜像, 需要先安排迁移." >&2
        exit 1
    fi
    if ! docker inspect --format '{{range .Mounts}}{{println .Source "=>" .Destination}}{{end}}' \
        "${CONTAINER_NAME}" | grep -Fxq "${PROJECT_ROOT} => ${PROJECT_ROOT}"; then
        echo "[ERROR] 同名容器的项目挂载路径不一致,需要重新创建容器." >&2
        exit 1
    fi
    if [[ "$(docker inspect --format '{{.Config.WorkingDir}}' "${CONTAINER_NAME}")" != "${PROJECT_ROOT}" ]]; then
        echo "[ERROR] 同名容器的默认工作目录不一致,需要重新创建容器." >&2
        exit 1
    fi
    if ! docker inspect --format '{{range .Mounts}}{{println .Source "=>" .Destination .RW}}{{end}}' \
        "${CONTAINER_NAME}" | grep -Fxq "${MODEL_STORAGE_ROOT} => ${MODEL_STORAGE_ROOT} true"; then
        echo "[ERROR] 现有容器缺少模型存储读写挂载,请先保存环境并安排重建." >&2
        exit 1
    fi
    if ! docker inspect --format '{{range .Mounts}}{{println .Source "=>" .Destination .RW}}{{end}}' \
        "${CONTAINER_NAME}" | grep -Fxq "${DATASET_ROOT} => ${DATASET_ROOT} false"; then
        echo "[ERROR] 现有容器缺少数据集只读挂载,请安排重建." >&2
        exit 1
    fi
    if ! docker inspect --format '{{range .Mounts}}{{println .Source "=>" .Destination}}{{end}}' \
        "${CONTAINER_NAME}" | grep -Fxq "${TOOLCHAIN_ROOT} => ${TOOLCHAIN_ROOT}"; then
        echo "[ERROR] 现有容器缺少公共工具链挂载,请先保存环境并安排重建." >&2
        exit 1
    fi
    echo "[INFO] 容器 ${CONTAINER_NAME} 已存在,不重复创建."
    docker start "${CONTAINER_NAME}" >/dev/null
else
    echo "[2/4] 使用已经构建好的 Ubuntu MTK 镜像."

    echo "[3/4] 创建 Genio 720 转换容器."
    docker run --detach \
        --name "${CONTAINER_NAME}" \
        --hostname "${CONTAINER_NAME}" \
        --user "${HOST_UID}:${HOST_GID}" \
        --gpus all \
        --network host \
        --shm-size 8g \
        --env NVIDIA_VISIBLE_DEVICES=all \
        --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        --env PYTHONDONTWRITEBYTECODE=1 \
        --env HOME="/home/mtk_${HOST_UID}" \
        --env MTK_MODELS_ROOT="${PROJECT_ROOT}" \
        --env PYTHONPATH="${PROJECT_ROOT}" \
        --env MTK_MODEL_STORAGE_ROOT \
        --env MTK_DATASET_ROOT \
        --env MTK_TOOLCHAIN_ROOT \
        --env MTK_CONTAINER_NAME \
        --env MTK_HOST_SDK_ROOT \
        --env MTK_NP_ROOT \
        --env NCC_ROOT \
        --env MTK_NCC_BIN \
        --env MTK_NCC_LIB \
        --env MTK_NEURON_INCLUDE \
        --env MTK_BOARD_HOST \
        --env MTK_BOARD_ROOT \
        --env MTK_BOARD_OPEN_MODELS_ROOT \
        --env MTK_BOARD_DATASETS_ROOT \
        --env MTK_BOARD_MODELZOO_ROOT \
        --env MTK_GAI_PYTHON \
        --env MTK_LLM_PYTHON \
        --env MTK_YAMNET_PYTHON \
        --volume "${PROJECT_ROOT}:${PROJECT_ROOT}" \
        --volume "${DATASET_ROOT}:${DATASET_ROOT}:ro" \
        --volume "${MODEL_STORAGE_ROOT}:${MODEL_STORAGE_ROOT}" \
        --volume "${TOOLCHAIN_ROOT}:${TOOLCHAIN_ROOT}" \
        --workdir "${PROJECT_ROOT}" \
        "${IMAGE_NAME}" >/dev/null
fi

echo "[4/4] 固定终端环境并设置项目访问权限."
source "${PROJECT_ROOT}/docker/mtk/user.sh"
# 创建时写入普通登录终端的环境,进入容器时无需重新读取配置.
{
    python3 "${PROJECT_ROOT}/tools/storage.py"
    printf 'export MTK_MODELS_ROOT=%q\n' "${PROJECT_ROOT}"
    printf 'export PYTHONPATH=%q\n' "${PROJECT_ROOT}"
    printf 'export PATH=%q:$PATH\n' "${NCC_ROOT}/bin"
} | docker exec --user 0 -i "${CONTAINER_NAME}" sh -c 'cat > /etc/profile.d/mtk-models.sh'
docker exec --user "${MTK_CONTAINER_UID}" \
    --env HOME="${MTK_CONTAINER_HOME}" "${CONTAINER_NAME}" git config --global --add safe.directory \
    "${PROJECT_ROOT}"
echo "[OK] 容器 ${CONTAINER_NAME} 已就绪,编译用户 ${MTK_CONTAINER_UID}:${MTK_CONTAINER_GID}."
