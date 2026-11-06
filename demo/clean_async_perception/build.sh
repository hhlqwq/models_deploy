#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode=Normal
if [[ $# -gt 0 && "$1" != --platform ]]; then
  mode="$1"
  shift
fi
platform=linux-amd64
if [[ $# -eq 2 && "$1" == --platform ]]; then
  platform="$2"
elif [[ $# -ne 0 ]]; then
  echo "Usage: ./build.sh [Normal|Test|Release|clean] [--platform linux-amd64|mtk-g720]" >&2
  exit 1
fi

case "$mode" in
  Normal|Test|Release|clean) ;;
  *) echo "Usage: ./build.sh [Normal|Test|Release|clean]" >&2; exit 1 ;;
esac

case "$platform" in
  linux-amd64) target_path=linux/amd64 ;;
  mtk-g720) target_path=linux/mtk-g720 ;;
  *) echo "未知构建平台: $platform" >&2; exit 1 ;;
esac
build_dir="$root/build/$target_path"
output_root="$root/output/$target_path"

# 只清理可重建的中间文件和运行包；SDK、cache、发布包及运行数据均保留。
if [[ "$mode" == clean ]]; then
  rm -rf -- "$build_dir" "$output_root"
  echo "[OK] 已清理 $platform 构建临时文件和输出目录。"
  exit 0
fi

cmake_args=()
if [[ "$platform" == linux-amd64 ]]; then
  # 加载原 Intel 构建环境.
  set +u
  for f in /opt/ros/humble/setup.bash /opt/intel/l_openvino_toolkit_ubuntu22_2024.4.0.16579.c3152d32c9c_x86_64/setupvars.sh; do
    [[ -f "$f" ]] && source "$f"
  done
  set -u
  "$root/scripts/check_sdk.sh"
else
  # G720 C++ 依赖统一放在 Demo 自己的 ARM64 SDK 目录.
  sdk_dir="${G720_SDK_DIR:-$root/dep/linux-arm64}"
  if [[ -z "${G720_TOOLCHAIN_FILE:-}" ]]; then
    G720_TOOLCHAIN_FILE="$root/cmake/mtk-g720-gnu-toolchain.cmake"
    cmake_args+=("-DG720_BOARD_OPENCV=ON")
    build_dir="$build_dir/sdk"
  fi
  cmake_args+=(
    "-DCMAKE_TOOLCHAIN_FILE=$G720_TOOLCHAIN_FILE"
    "-DG720_SDK_DIR=$sdk_dir"
    "-DORBBEC_SDK_ROOT=${G720_ORBBEC_SDK_ROOT:-$sdk_dir/orbbec_sdk}"
    "-DG720_NEURON_INCLUDE_DIR=${G720_NEURON_INCLUDE_DIR:-$sdk_dir/neuron/include}"
    "-DG720_NEURON_LIBRARY=${G720_NEURON_LIBRARY:-$sdk_dir/board-libs/libneuronusdk_runtime.mtk.so.8}"
    "-DG720_MODEL_DIR=${G720_MODEL_DIR:-$root/models/g720}"
    "-DG720_ENABLE_ROS2=${G720_ENABLE_ROS2:-OFF}"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DBUILD_TESTING=OFF"
  )
fi

if [[ "$platform" == linux-amd64 ]]; then
  rm -rf "$build_dir"
fi
echo "[BUILD] 配置 $platform ($mode)"
cmake \
  -S "$root" \
  -B "$build_dir" \
  -DASYNC_PERCEPTION_BUILD_MODE="$mode" \
  -DASYNC_PERCEPTION_PLATFORM="$platform" \
  "${cmake_args[@]}"

cmake --build "$build_dir" --parallel

output_dir="$output_root/async_perception"
echo "[OK] Build complete: $output_dir"

if [[ "$platform" == mtk-g720 ]]; then
  echo "[RUN] 将运行包复制到 G720, 进入运行包目录并执行 ./async_perception"
elif [[ "$mode" == Test ]]; then
  LD_LIBRARY_PATH="$output_dir:${LD_LIBRARY_PATH:-}" "$output_dir/async_perception"
fi
