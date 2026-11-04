#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:-Normal}"

case "$mode" in
  Normal|Test|Release|clean) ;;
  *) echo "Usage: ./build.sh [Normal|Test|Release|clean]" >&2; exit 1 ;;
esac

build_dir="$root/build/linux/amd64"
output_root="$root/output/linux/amd64"

# 只清理可重建的中间文件和运行包；SDK、cache、发布包及运行数据均保留。
if [[ "$mode" == clean ]]; then
  rm -rf -- "$build_dir" "$output_root"
  echo "[OK] 已清理 Linux amd64 构建临时文件和输出目录。"
  exit 0
fi

# source ROS2 和 OpenVINO 环境
set +u
for f in /opt/ros/humble/setup.bash /opt/intel/l_openvino_toolkit_ubuntu22_2024.4.0.16579.c3152d32c9c_x86_64/setupvars.sh; do
  [[ -f "$f" ]] && source "$f"
done
set -u

# 检查并自动恢复 SDK 依赖
"$root/scripts/check_sdk.sh"

rm -rf "$build_dir"
cmake \
  -S "$root" \
  -B "$build_dir" \
  -DASYNC_PERCEPTION_BUILD_MODE="$mode"

cmake --build "$build_dir" --parallel

output_dir="$root/output/linux/amd64/async_perception"
echo "[OK] Build complete: $output_dir"

if [[ "$mode" == Test ]]; then
  LD_LIBRARY_PATH="$output_dir:${LD_LIBRARY_PATH:-}" "$output_dir/async_perception"
fi
