#!/usr/bin/env bash

# 从统一平台配置加载当前容器的 SDK 与板端环境.
export MTK_MODELS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${MTK_MODELS_ROOT}/tools/storage.sh" "$@" || return 1
export PYTHONPATH="${MTK_MODELS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${MTK_NCC_BIN}:${PATH}"

# 清除终端继承的 SDK 库路径,避免旧 libstdc++ 影响 Python,保留其他库目录.
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    IFS=: read -r -a MTK_LIBRARY_DIRS <<< "${LD_LIBRARY_PATH}"
    MTK_RUNTIME_LIBRARIES=""
    for MTK_LIBRARY_DIR in "${MTK_LIBRARY_DIRS[@]}"; do
        if [[ -n "${MTK_LIBRARY_DIR}" && "${MTK_LIBRARY_DIR%/}" != */neuron_sdk/host/lib ]]; then
            MTK_RUNTIME_LIBRARIES="${MTK_RUNTIME_LIBRARIES:+${MTK_RUNTIME_LIBRARIES}:}${MTK_LIBRARY_DIR}"
        fi
    done
    if [[ -n "${MTK_RUNTIME_LIBRARIES}" ]]; then
        export LD_LIBRARY_PATH="${MTK_RUNTIME_LIBRARIES}"
    else
        unset LD_LIBRARY_PATH
    fi
    unset MTK_LIBRARY_DIRS MTK_LIBRARY_DIR MTK_RUNTIME_LIBRARIES
fi

echo "[MTK] 项目目录: ${MTK_MODELS_ROOT}"
echo "[MTK] NeuroPilot SDK: ${MTK_NP_ROOT}"
echo "[MTK] 目标开发板: ${MTK_BOARD_HOST}"
