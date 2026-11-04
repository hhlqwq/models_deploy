#!/usr/bin/env bash
set -euo pipefail
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/tools/storage.sh"
readonly SDK_DIR="${MTK_HOST_SDK_ROOT}"
readonly GAI_SDK_DIR="${MTK_GAI_SDK_ROOT}"
readonly IMAGE_NAME="${MTK_G720_IMAGE:-mtk/ai_toolchain_ubuntu_22_g720_gpu:np8.0.11}"
readonly RECIPE_SHA256="$(cd "${PROJECT_ROOT}/docker/mtk_g720" && \
    sha256sum Dockerfile requirements.txt yolov5-requirements.txt setup_container.sh verify_environment.py | \
    sha256sum | cut -d ' ' -f1)"
readonly CANDIDATE_IMAGE="${IMAGE_NAME}-candidate"

test -d "${SDK_DIR}/offline_tool"
cd "${PROJECT_ROOT}"
# 总入口新增模型时,构建必须同时补充对应环境验收,不能静默漏掉.
python3 - "${PROJECT_ROOT}" <<'PY'
import ast
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
entry = (root / "mtk_models/deploy/compile.sh").read_text(encoding="utf-8")
models = {name.rsplit("/", 1)[-1] for name in
          re.search(r"MODELS=\(\n(.*?)\n\)", entry, re.S).group(1).split()}
tree = ast.parse((root / "docker/mtk_g720/verify_environment.py").read_text(encoding="utf-8"))
checks = next(ast.literal_eval(node.value) for node in ast.walk(tree)
              if isinstance(node, ast.Assign) and any(
                  isinstance(target, ast.Name) and target.id == "modules" for target in node.targets))
if models != set(checks):
    raise SystemExit(f"模型与环境验收清单不一致: 未覆盖={models - set(checks)}, 额外={set(checks) - models}")
print(f"[验收覆盖] {len(models)} 个模型均已声明依赖检查.")
PY

echo "[BUILD] 构建 Ubuntu 预装镜像, MTK SDK 使用服务器本地文件."
docker buildx build --load --progress=plain --network host \
    --build-context "sdk=${SDK_DIR}" \
    --build-context "gai=${GAI_SDK_DIR}" \
    --build-arg "MTK_SDK_ROOT=${SDK_DIR}" \
    --build-arg "MTK_RECIPE_SHA256=${RECIPE_SHA256}" \
    --tag "${CANDIDATE_IMAGE}" \
    --file docker/mtk_g720/Dockerfile docker/mtk_g720

# 全新容器验证,不挂载模型、数据集或旧容器 HOME;通过后才发布正式标签.
echo "[验收] 全新普通用户容器检查全部模型依赖、评分接口及 GPU."
docker run --rm --pull never --gpus all --user "$(id -u):$(id -g)" \
    --env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
    --env TF_NUM_INTEROP_THREADS=2 --env TF_NUM_INTRAOP_THREADS=2 \
    "${CANDIDATE_IMAGE}" python /opt/mtk-build/verify_environment.py
docker tag "${CANDIDATE_IMAGE}" "${IMAGE_NAME}"
docker image rm "${CANDIDATE_IMAGE}" >/dev/null
echo "[验收通过] 正式镜像已发布,旧容器尚未替换."
