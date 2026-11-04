#!/usr/bin/env bash
# 构建前统一校验编译环境和最终 SDK
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v cmake >/dev/null || {
  echo "[ERROR] 缺少 CMake。" >&2
  exit 1
}
command -v c++ >/dev/null || {
  echo "[ERROR] 缺少 C++ 编译器。请安装 build-essential/GCC。" >&2
  exit 1
}

sdk="$root/dep/linux-x86_64"
is_complete_sdk() {
  [[ -f "$1/orbbec_sdk/lib/OrbbecSDKConfig.cmake" ]] &&
    compgen -G "$1/orbbec_sdk/lib/libOrbbecSDK.so*" >/dev/null &&
    [[ -d "$1/orbbec_sdk/lib/extensions" ]] &&
    [[ -f "$1/async_infer_sdk/sdk/include/async_infer.h" ]] &&
    grep -q 'TID_MOBILENETV2' "$1/async_infer_sdk/sdk/include/async_infer.h" &&
    [[ -f "$1/async_infer_sdk/sdk/lib/libasync_infer.so" ]] &&
    [[ -f "$1/async_infer_sdk/example/files/config/mobilenetv2_config.json" ]] &&
    [[ -f "$1/async_infer_sdk/example/files/models/mobilenetv2/mobilenetv2.xml" ]] &&
    [[ -f "$1/async_infer_sdk/example/files/models/mobilenetv2/mobilenetv2.bin" ]]
}

if ! is_complete_sdk "$sdk"; then
  echo "[SDK] 检测到 SDK 缺失或不完整，开始恢复。"
  "$root/scripts/restore_sdk.sh"
fi
is_complete_sdk "$sdk" || {
  echo "[ERROR] SDK 恢复后仍不完整：$sdk" >&2
  exit 1
}
echo "[SDK] SDK 校验通过。"
