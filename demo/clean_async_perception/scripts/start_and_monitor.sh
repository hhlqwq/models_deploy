#!/usr/bin/env bash
set -euo pipefail

# ─── async_perception 一键启动 + 内存监控 ───
# 启动 async_perception（如果未运行），然后自动开始内存监控。
#
# 用法:
#   ./scripts/start_and_monitor.sh          # 启动程序 + 开始监控
#   ./scripts/start_and_monitor.sh --skip-monitor  # 仅启动程序，不监控
#
# 环境变量可选:
#   INTERVAL=30        监控采样间隔（秒），默认 30
#   DURATION=86400     监控总时长（秒），默认 86400（24h），0 表示无限
#   OUTDIR=./mem_logs   监控输出目录，默认 ./mem_logs

# 在 source 任何环境之前，先固化路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MONITOR_SCRIPT="$SCRIPT_DIR/mem_monitor.sh"
BINARY="$PROJECT_DIR/output/linux/amd64/async_perception/async_perception"
# Linux 进程名最长 15 字符，async_perception 会被截断为 async_perceptio
PROCESS_NAME="async_perception"
TRUNCATED_NAME="${PROCESS_NAME:0:15}"
LOG_FILE="$PROJECT_DIR/async_perception.log"

skip_monitor=false
if [[ "${1:-}" == "--skip-monitor" ]]; then
  skip_monitor=true
fi

# ─── 1. 检查并启动 async_perception ───
existing_pid=$(pgrep -x "$TRUNCATED_NAME" 2>/dev/null | head -1 || true)

if [[ -n "$existing_pid" ]]; then
  echo "[INFO] $PROCESS_NAME 已在运行, PID=$existing_pid, 跳过启动"
else
  if [[ ! -x "$BINARY" ]]; then
    echo "[ERROR] 可执行文件不存在: $BINARY" >&2
    echo "[ERROR] 请先编译项目" >&2
    exit 1
  fi

  echo "[INFO] 正在启动 $PROCESS_NAME ..."

  # 初始化环境
  set +u
  source /opt/ros/humble/setup.bash
  source /opt/intel/l_openvino_toolkit_ubuntu22_2024.4.0.16579.c3152d32c9c_x86_64/setupvars.sh
  set -u

  # 后台启动程序，输出写入日志
  nohup "$BINARY" > "$LOG_FILE" 2>&1 &
  new_pid=$!
  echo "[INFO] 已启动 $PROCESS_NAME, PID=$new_pid"
  echo "[INFO] 程序日志: $LOG_FILE"

  # 等待进程初始化
  echo "[INFO] 等待 3 秒确认进程稳定..."
  sleep 3
  if [[ ! -d "/proc/$new_pid" ]]; then
    echo "[ERROR] $PROCESS_NAME 启动后立即退出，请检查日志: $LOG_FILE" >&2
    exit 1
  fi
  echo "[INFO] $PROCESS_NAME 启动成功"
fi

# ─── 2. 启动内存监控 ───
if [[ "$skip_monitor" == true ]]; then
  echo "[INFO] 已跳过内存监控（--skip-monitor）"
  exit 0
fi

# 检查是否已有监控在运行（排除自身）
monitor_pid=$(pgrep -f "mem_monitor.sh.*--name" 2>/dev/null | while read p; do
  if [[ "$p" != "$$" ]]; then echo "$p"; fi
done | head -1 || true)
if [[ -n "$monitor_pid" ]]; then
  echo "[INFO] 内存监控已在运行, PID=$monitor_pid, 跳过"
  exit 0
fi

echo "[INFO] 启动内存监控..."
# 用 bash 子进程运行，避免 OpenVINO 环境变量污染路径
exec bash "$MONITOR_SCRIPT" --name "$PROCESS_NAME"