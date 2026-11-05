#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "$root/../../.." && pwd)"
mode="${1:-FILE}"
if [[ "$#" -gt 1 || ( "$mode" != FILE && "$mode" != MIC ) ]]; then
    echo "用法: bash build.sh [FILE|MIC]" >&2
    exit 2
fi
# 真实工具路径从本地配置读取,VOICE_CXX 可显式覆盖目标编译器.
configuration="$(python3 - "$repository" <<'PY'
from datetime import datetime
import os
from pathlib import Path
import shlex
import sys

sys.path.insert(0, sys.argv[1])
from tools.rdk_config import load_config

settings = load_config()["toolchain"].get("voice_control", {})
defaults = {
    "OELLM_SDK_ROOT": settings.get("sdk_root"),
    "S100_BOARD_SYSROOT": settings.get("board_sysroot"),
    "ONNXRUNTIME_ROOT": settings.get("onnxruntime_root"),
    "CXX": settings.get("cxx"),
    "VOICE_BUILD_DIR": str(Path("/tmp/hailongcodex") / datetime.now().strftime("%Y-%m-%d") / "voice_control/build"),
}
for name, value in defaults.items():
    if name == "CXX":
        value = os.environ.get("VOICE_CXX") or value or os.environ.get("CXX")
    else:
        value = os.environ.get(name) or value
    if not value:
        raise SystemExit(f"请填写本地 toolchain.voice_control 或环境变量 {name}.")
    print(f"export {name}={shlex.quote(str(value))}")
PY
)"
eval "$configuration"
: "${OELLM_SDK_ROOT:?请指定现有 S100 LLM SDK 1.0 根目录}"
: "${VOICE_BUILD_DIR:?请指定日期临时构建目录}"
: "${S100_BOARD_SYSROOT:?请指定匹配板端的头文件和库目录}"
: "${CXX:?请指定现有官方 aarch64 交叉编译器}"
if [[ "$("$CXX" -dumpmachine)" != aarch64* ]]; then
    echo "[错误] 编译器目标必须为 aarch64." >&2
    exit 1
fi
mkdir -p "$VOICE_BUILD_DIR"
libraries=(-lxlm -ldnn -lhbucp -lonnxruntime -lpthread -ldl)
if [[ "$mode" == MIC ]]; then
    libraries+=(-lasound)
fi

echo "[构建] voice_control,$mode,SDK 1.0 C++"
"$CXX" -std=c++17 -O2 -Wall -Wextra -Wpedantic \
    "-DVOICE_INPUT_$mode" -I"$root/src" \
    -isystem "$OELLM_SDK_ROOT/include" -isystem "$S100_BOARD_SYSROOT/include" \
    -isystem "$ONNXRUNTIME_ROOT/include" -L"$ONNXRUNTIME_ROOT/lib" \
    "$root/src/main.cpp" -L"$OELLM_SDK_ROOT/lib" -L"$S100_BOARD_SYSROOT/lib" \
    -Wl,--allow-shlib-undefined -Wl,--disable-new-dtags \
    '-Wl,-rpath,$ORIGIN/lib:/usr/hobot/lib' \
    "${libraries[@]}" -o "$VOICE_BUILD_DIR/voice_control_${mode,,}"
echo "[完成] $VOICE_BUILD_DIR/voice_control_${mode,,}"
