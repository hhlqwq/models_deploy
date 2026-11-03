#!/usr/bin/env bash
# 主机编译上传,板端用一帧真实输入检查 CPU + NPU 完整链路.
set -euo pipefail
export MTK_EVAL_SCOPE=subset
export MTK_OUTPUT_METRICS=1
export MTK_EVALUATION_WORK=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${HOME}/.local/share/mtk_models/bevfusion-evaluation${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    export PYTHONDONTWRITEBYTECODE=1
    export OPENBLAS_NUM_THREADS=4
    exec 9>"${SCRIPT_DIR}/.run.lock"
    flock -n 9 || { echo "[错误] 本模型已有任务运行." >&2; exit 2; }
    python3 "${SCRIPT_DIR}/board/compile_accuracy.py"
    echo "[完成] 子集精度对比和示例已完成,子集结果见 results/summary_test.json."
else
    : "${MTK_MODELS_ROOT:?请先进入已有编译容器}"
    echo "[容器 1/3] 导出完整权重对应子图并编译 MDLA 5.3."
    python "${SCRIPT_DIR}/host/compile.py"
    echo "[容器 2/3] 使用已有 nuScenes 数据准备一帧输入."
    python "${SCRIPT_DIR}/host/prepare.py"
    echo "[容器 3/3] 评估浮点子集并上传完整部署包和参考."
    echo "[浮点子集精度] 评估完整选定子集并上传模型、精度及数值参考."
    python "${MTK_MODELS_ROOT}/tools/compile_accuracy.py" "${SCRIPT_DIR}"
    echo "[子集输入] 上传本模型清单对应的相机、LiDAR 和历史帧文件."
    python "${SCRIPT_DIR}/host/upload.py" --inputs
fi
