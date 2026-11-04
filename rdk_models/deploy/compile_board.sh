#!/usr/bin/env bash
# 总入口只顺序调用单模型入口,共用日志和结果回传.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common.sh"
start_log "${SCRIPT_DIR}/compile.log"
MODELS=(
    navigation/single_camera_depth/depth_anything_v2_metric_hypersim_vits
    audio/stt/whisper_tiny
)

INDEX=0
for MODEL in "${MODELS[@]}"; do
    INDEX=$((INDEX + 1))
    echo "[板端 ${INDEX}/${#MODELS[@]}] ${MODEL##*/}"
    bash "${MODELS_ROOT}/${MODEL##*/}/compile.sh"
done

return_results subset
