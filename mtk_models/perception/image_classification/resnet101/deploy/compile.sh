#!/usr/bin/env bash
# 主机编译上传并准备子集参考,板端检查固定 1000 张图像精度并生成示例.
set -euo pipefail
export MTK_OUTPUT_METRICS=1
export MTK_EVALUATION_WORK=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MTK_EVAL_SCOPE=subset

if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    export PYTHONDONTWRITEBYTECODE=1
    exec 9>"${SCRIPT_DIR}/.run.lock"
    flock -n 9 || { echo "[错误] 本模型已有运行任务." >&2; exit 2; }
    echo "[板端 1/2] 固定 1000 张验证图像推理."
    python3 "${SCRIPT_DIR}/board/infer.py"
    echo "[子集精度] 计算任务精度并与同子集浮点参考对比."
    python3 "${SCRIPT_DIR}/board/compile_accuracy.py"
    echo "[板端 2/2] 保存分类展示示例."
    python3 "${SCRIPT_DIR}/board/render_examples.py"
    echo "[完成] 示例已生成,子集精度和损失已保存至 results/summary_test.json,全量 summary.json 未改写."
else
    : "${MTK_MODELS_ROOT:?请先通过 docker/mtk_g720/enter_container.sh 进入已有编译容器}"
    echo "[容器 1/4] 从本地官方权重导出 FP32 ONNX."
    python "${SCRIPT_DIR}/host/export_onnx.py"
    echo "[容器 2/4] 固定样本 INT8 校准量化."
    python "${SCRIPT_DIR}/host/convert_int8.py"
    echo "[容器 3/4] 编译 MDLA 5.3 DLA."
    python "${SCRIPT_DIR}/host/compile_model.py"
    echo "[容器 4/4] 评估浮点子集并上传模型和参考."
    echo "[浮点子集精度] 评估完整选定子集并上传模型、精度及数值参考."
    python "${MTK_MODELS_ROOT}/tools/compile_accuracy.py" "${SCRIPT_DIR}"
    echo "[待板端检查] 请登录开发板,在部署目录执行 bash compile.sh."
fi
