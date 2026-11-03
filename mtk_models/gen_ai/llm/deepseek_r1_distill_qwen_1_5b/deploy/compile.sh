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
    echo "[板端 2/2] 生成文本示例已保存至 examples/output/."
    echo "[编译完成] 通路和示例已准备,子集精度和损失已保存至 results/summary_test.json,全量 summary.json 未改写."
else
    MODEL_BUILD_DIR="$(python "${MTK_MODELS_ROOT}/tools/storage.py" --print-build-dir "${SCRIPT_DIR}/../config.yaml")"
    mkdir -p "${MODEL_BUILD_DIR}"
    exec 9>"${MODEL_BUILD_DIR}/.run.lock"
    flock -n 9 || { echo '[错误] 本模型已有转换任务.' >&2; exit 2; }
    echo '[容器 1/3] 准备模型、校准、量化并导出双静态图.'
    python "${SCRIPT_DIR}/host/convert.py"
    echo '[容器 2/3] 编译 Prefill 和 Decode 双 DLA.'
    python "${SCRIPT_DIR}/host/compile.py"
    echo '[容器 3/3] 打包已有模型,随后评估浮点子集并上传.'
    python "${SCRIPT_DIR}/host/package.py"
    echo "[浮点子集精度] 评估完整选定子集并上传模型、精度及数值参考."
    python "${MTK_MODELS_ROOT}/tools/compile_accuracy.py" "${SCRIPT_DIR}"
    echo "[编译完成] 模型已上传,请在开发板执行 compile.sh 检查通路并生成展示示例."
fi
