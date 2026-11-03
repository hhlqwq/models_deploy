#!/usr/bin/env bash
# 编译、部署与少量展示示例;正式指标只由 evaluate.sh 生成.
set -euo pipefail
export MTK_OUTPUT_METRICS=1
export MTK_EVALUATION_WORK=0
export MTK_EVAL_SCOPE=subset
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    export PYTHONDONTWRITEBYTECODE=1
    exec 9>"${SCRIPT_DIR}/.run.lock"
    flock -n 9 || { echo "[错误] 本模型已有运行任务." >&2; exit 2; }
    echo "[板端 1/2] 固定子集硬件推理."
    python3 "${SCRIPT_DIR}/board/infer.py"
    echo "[子集精度] 计算任务精度并与同子集浮点参考对比."
    python3 "${SCRIPT_DIR}/board/compile_accuracy.py"
    echo "[板端 2/2] 保存 README 展示示例."
    python3 "${SCRIPT_DIR}/board/render_examples.py"
    echo "[编译完成] 通路和示例已准备,子集精度和损失已保存至 results/summary_test.json,全量 summary.json 未改写."
else
    # 编译容器: 各 Python 文件读取 YAML 并直接执行当前步骤.
    echo "[容器 1/4] 导出 FP32 ONNX."
    python "${SCRIPT_DIR}/host/export_model.py"
    echo "[容器 2/4] INT8 校准与量化."
    python "${SCRIPT_DIR}/host/convert_int8.py"
    echo "[容器 3/4] 编译 DLA."
    python "${SCRIPT_DIR}/host/compile_model.py"
    echo "[容器 4/4] 评估浮点子集并上传模型和参考."
    echo "[浮点子集精度] 评估完整选定子集并上传模型、精度及数值参考."
    python "${MTK_MODELS_ROOT}/tools/compile_accuracy.py" "${SCRIPT_DIR}"
    echo "[编译完成] 模型已上传,请在开发板执行 compile.sh 检查通路并生成展示示例."
fi
