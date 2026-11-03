#!/usr/bin/env bash
# 编译服务器顺序执行各模型 evaluate,不处理模型专属逻辑.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common.sh"
start_log "${SCRIPT_DIR}/evaluate.log"
MODELS=(
    perception/image_classification/resnet101
    perception/image_classification/vit_base_patch16_224
    perception/object_detection/yolov5s
    perception/object_detection/yolov8n
    interaction/face_recognition/mobilefacenet
    interaction/pose_detection/rtmpose_body2d
    navigation/segmentation/fastsam
    navigation/single_camera_depth/depth_anything_v2_small
    audio/ambient_sound/yamnet
    audio/stt/whisper_tiny
    pointcloud/three_d_fusion/mit_bevfusion
    gen_ai/llm/minicpm5_2b
    gen_ai/llm/deepseek_r1_distill_qwen_1_5b
)

: "${MTK_MODELS_ROOT:?请先进入编译容器}"

INDEX=0
MODEL=总入口
trap 'STATUS=$?; echo "[失败] 容器: ${MODEL},退出码 ${STATUS}." >&2; exit "${STATUS}"' ERR
for MODEL in "${MODELS[@]}"; do
    INDEX=$((INDEX + 1))
    echo "[容器 ${INDEX}/${#MODELS[@]}] ${MODEL##*/}"
    bash "${MODELS_ROOT}/${MODEL}/deploy/evaluate.sh"
done

echo "[完成] 请在板端 deploy/ 下执行 bash evaluate_board.sh."
