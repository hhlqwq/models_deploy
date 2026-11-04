#!/usr/bin/env bash
set -euo pipefail

# ─── async_perception 内存监控脚本 ───
# 每 N 秒采样一次进程内存和 CPU 信息，写入 CSV，
# 用于长时间运行时的内存泄漏检测。
#
# 用法:
#   ./mem_monitor.sh <PID>
#   ./mem_monitor.sh --name async_perception
#
# 可选环境变量:
#   INTERVAL=30       采样间隔，单位秒，默认 30
#   DURATION=86400    总监控时长，单位秒，默认 24 小时
#                     设置为 0 表示无限运行
#   OUTDIR=./mem_logs 输出目录
#   VERBOSE=1         在终端定期打印摘要，默认关闭
#
# CSV 字段:
#   rss_kb       实际驻留物理内存
#   vsz_kb       虚拟地址空间
#   pss_kb       按比例分摊共享页后的物理内存
#   shared_kb    Shared_Clean + Shared_Dirty
#   private_kb   Private_Clean + Private_Dirty
#   cpu_percent  进程 CPU 使用率，以单个 CPU 核心为 100%

interval="${INTERVAL:-30}"
duration="${DURATION:-86400}"
outdir="${OUTDIR:-./mem_logs}"
verbose="${VERBOSE:-0}"

# ─── 参数校验 ───

if ! [[ "$interval" =~ ^[0-9]+$ ]] || [[ "$interval" -le 0 ]]; then
  echo "[ERROR] INTERVAL 必须是大于 0 的整数: $interval" >&2
  exit 1
fi

if ! [[ "$duration" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] DURATION 必须是大于或等于 0 的整数: $duration" >&2
  exit 1
fi

if [[ "$verbose" != "0" && "$verbose" != "1" ]]; then
  echo "[ERROR] VERBOSE 只能设置为 0 或 1: $verbose" >&2
  exit 1
fi

# ─── 解析命令行参数 ───

pid=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --name 后需要提供进程名" >&2
        exit 1
      fi

      shift
      name="$1"

      # Linux /proc/<pid>/comm 的进程名通常最多保留 15 个字符。
      truncated_name="${name:0:15}"

      pid="$(
        pgrep -x "$truncated_name" 2>/dev/null |
          head -n 1 || true
      )"

      if [[ -z "$pid" ]]; then
        echo "[ERROR] 找不到进程: $name" >&2
        exit 1
      fi

      echo "[INFO] 找到进程 $name, PID=$pid"
      ;;

    --verbose|-v)
      verbose=1
      ;;

    --help|-h)
      echo "用法:"
      echo "  $0 <PID>"
      echo "  $0 --name <进程名>"
      echo
      echo "选项:"
      echo "  --verbose, -v    在终端定期打印摘要"
      echo "  --help, -h       显示帮助"
      echo
      echo "环境变量:"
      echo "  INTERVAL=30"
      echo "  DURATION=86400"
      echo "  OUTDIR=./mem_logs"
      echo "  VERBOSE=0"
      exit 0
      ;;

    [0-9]*)
      if [[ -n "$pid" ]]; then
        echo "[ERROR] 只能指定一个 PID" >&2
        exit 1
      fi

      pid="$1"
      ;;

    *)
      echo "[ERROR] 未知参数: $1" >&2
      exit 1
      ;;
  esac

  shift
done

if [[ -z "$pid" ]]; then
  echo "[ERROR] 请提供 PID 或 --name 参数" >&2
  echo "用法: $0 <PID> 或 $0 --name async_perception" >&2
  exit 1
fi

if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] PID 必须是整数: $pid" >&2
  exit 1
fi

if [[ ! -d "/proc/$pid" ]]; then
  echo "[ERROR] PID $pid 不存在" >&2
  exit 1
fi

# ─── 创建输出文件 ───

mkdir -p "$outdir"

timestamp="$(date +%Y%m%d_%H%M%S)"
csv_file="$outdir/mem_${pid}_${timestamp}.csv"

echo \
  "timestamp,elapsed_s,rss_kb,vsz_kb,pss_kb,shared_kb,private_kb,vmdata_kb,vmstk_kb,vmlib_kb,threads,cpu_percent" \
  > "$csv_file"

echo "[INFO] 开始监控 PID=$pid"
echo "[INFO] 采样间隔: ${interval}s"
echo "[INFO] 持续时长: ${duration}s（0 表示无限）"
echo "[INFO] CSV 输出: $csv_file"
echo "[INFO] 按 Ctrl+C 停止"

if [[ "$verbose" -eq 0 ]]; then
  echo "[INFO] 终端静默模式"
  echo "[INFO] 实时查看: tail -f $csv_file"
fi

# ─── 运行状态 ───

start_ts="$(date +%s)"
sample=0
pss_warned=0

# CPU 差值计算所需的上一次采样值。
prev_proc_ticks=""
prev_total_ticks=""

# nproc 会考虑容器或 cpuset 对可用 CPU 数量的限制。
cpu_count="$(nproc 2>/dev/null || echo 1)"

if ! [[ "$cpu_count" =~ ^[0-9]+$ ]] || [[ "$cpu_count" -le 0 ]]; then
  cpu_count=1
fi

cleanup() {
  echo
  echo "[INFO] 监控结束, 共采集 $sample 个样本"
  echo "[INFO] 数据已保存到: $csv_file"
  echo "[INFO] 运行分析:"
  echo "       python3 scripts/mem_analyze.py $csv_file"
  exit 0
}

trap cleanup SIGINT SIGTERM

# ─── 主采样循环 ───

while true; do
  # 检查目标进程是否仍然存在。
  if [[ ! -d "/proc/$pid" ]]; then
    echo "[WARN] 进程 $pid 已退出, 采样 $sample 次后停止"
    break
  fi

  now="$(date +%s)"
  elapsed=$((now - start_ts))

  if [[ "$duration" -gt 0 && "$elapsed" -ge "$duration" ]]; then
    echo "[INFO] 已达到设定时长 ${duration}s, 停止监控"
    break
  fi

  # ─── 从 /proc/<pid>/status 读取进程信息 ───

  rss_kb=0
  vsz_kb=0
  vmdata_kb=0
  vmstk_kb=0
  vmlib_kb=0
  threads=0

  while IFS=':' read -r key val; do
    # 删除空格、制表符、kB 等非数字字符。
    val="${val//[!0-9]/}"
    val="${val:-0}"

    case "$key" in
      VmRSS)
        rss_kb="$val"
        ;;
      VmSize)
        vsz_kb="$val"
        ;;
      VmData)
        vmdata_kb="$val"
        ;;
      VmStk)
        vmstk_kb="$val"
        ;;
      VmLib)
        vmlib_kb="$val"
        ;;
      Threads)
        threads="$val"
        ;;
    esac
  done < "/proc/$pid/status" 2>/dev/null || true

  # ─── 从 smaps_rollup 读取 PSS 和共享/私有内存 ───

  pss_kb=0
  smaps_rss_kb=0

  shared_clean_kb=0
  shared_dirty_kb=0
  private_clean_kb=0
  private_dirty_kb=0

  shared_kb=0
  private_kb=0

  # 使用 fd 3 打开文件。
  # 实际打开成功才说明当前用户有读取权限。
  if exec 3<"/proc/$pid/smaps_rollup" 2>/dev/null; then
    while IFS=':' read -r key val <&3; do
      val="${val//[!0-9]/}"
      val="${val:-0}"

      case "$key" in
        Rss)
          smaps_rss_kb="$val"
          ;;
        Pss)
          pss_kb="$val"
          ;;
        Shared_Clean)
          shared_clean_kb="$val"
          ;;
        Shared_Dirty)
          shared_dirty_kb="$val"
          ;;
        Private_Clean)
          private_clean_kb="$val"
          ;;
        Private_Dirty)
          private_dirty_kb="$val"
          ;;
      esac
    done

    exec 3<&-

    shared_kb=$((shared_clean_kb + shared_dirty_kb))
    private_kb=$((private_clean_kb + private_dirty_kb))

    # 使用与 PSS 相同快照中的 RSS，避免读取时间不同导致 PSS > RSS。
    if [[ "$smaps_rss_kb" -gt 0 ]]; then
      rss_kb="$smaps_rss_kb"
    fi
  else
    if [[ "$pss_warned" -ne 1 ]]; then
      echo \
        "[WARN] 无法读取 /proc/$pid/smaps_rollup（需 root 或与目标进程同用户运行监控），PSS/shared/private 将记 0" \
        >&2
      pss_warned=1
    fi
  fi

  # ─── 计算 CPU 使用率 ───
  #
  # /proc/<pid>/stat:
  #   第 14 列为 utime
  #   第 15 列为 stime
  #
  # /proc/stat:
  #   第一行是所有 CPU 的累计 tick。
  #
  # 这里按两次采样的 tick 差值计算。
  # cpu_count 用于转换为 top 风格：
  #   单个 CPU 核心跑满约等于 100%。

  cpu_pct="0.0"
  proc_ticks=""
  total_ticks=""

  if [[ -r "/proc/$pid/stat" && -r "/proc/stat" ]]; then
    stat_line="$(<"/proc/$pid/stat")"

    # 删除 PID 和括号中的 comm。
    # 使用 ## 匹配最后一个 ") "，避免进程名中存在空格。
    stat_after_comm="${stat_line##*) }"

    read -r -a stat_fields <<< "$stat_after_comm"

    # 删除 PID 和 comm 后：
    #   数组下标 0 对应原始字段 3（state）
    #   数组下标 11 对应原始字段 14（utime）
    #   数组下标 12 对应原始字段 15（stime）
    if [[ "${#stat_fields[@]}" -ge 13 ]]; then
      proc_utime="${stat_fields[11]}"
      proc_stime="${stat_fields[12]}"

      if [[ "$proc_utime" =~ ^[0-9]+$ &&
            "$proc_stime" =~ ^[0-9]+$ ]]; then
        proc_ticks=$((proc_utime + proc_stime))
      fi
    fi

    read -r -a cpu_fields < /proc/stat

    # cpu_fields:
    #   0: cpu
    #   1: user
    #   2: nice
    #   3: system
    #   4: idle
    #   5: iowait
    #   6: irq
    #   7: softirq
    #   8: steal
    #
    # guest 和 guest_nice 已包含在 user/nice 中，不能重复累加。
    if [[ "${cpu_fields[0]:-}" == "cpu" &&
          "${#cpu_fields[@]}" -ge 9 ]]; then
      total_ticks=0

      for ((i = 1; i <= 8; i++)); do
        field_value="${cpu_fields[$i]}"

        if [[ "$field_value" =~ ^[0-9]+$ ]]; then
          total_ticks=$((total_ticks + field_value))
        fi
      done
    fi
  fi

  if [[ -n "$proc_ticks" && -n "$total_ticks" ]]; then
    if [[ -n "$prev_proc_ticks" && -n "$prev_total_ticks" ]]; then
      proc_delta=$((proc_ticks - prev_proc_ticks))
      total_delta=$((total_ticks - prev_total_ticks))

      if [[ "$proc_delta" -ge 0 && "$total_delta" -gt 0 ]]; then
        cpu_pct="$(
          awk \
            -v proc_delta="$proc_delta" \
            -v total_delta="$total_delta" \
            -v cpu_count="$cpu_count" \
            'BEGIN {
              printf "%.1f",
                (proc_delta / total_delta) * cpu_count * 100
            }'
        )"
      fi
    fi

    prev_proc_ticks="$proc_ticks"
    prev_total_ticks="$total_ticks"
  fi

  # ─── 写入 CSV ───

  ts_str="$(date '+%Y-%m-%d %H:%M:%S')"

  echo \
    "$ts_str,$elapsed,$rss_kb,$vsz_kb,$pss_kb,$shared_kb,$private_kb,$vmdata_kb,$vmstk_kb,$vmlib_kb,$threads,$cpu_pct" \
    >> "$csv_file"

  sample=$((sample + 1))

  # ─── 终端摘要 ───

  if [[ "$verbose" -eq 1 ]] && ((sample % 10 == 1)); then
    rss_mb="$(
      awk -v value="$rss_kb" \
        'BEGIN { printf "%.1f", value / 1024 }'
    )"

    pss_mb="$(
      awk -v value="$pss_kb" \
        'BEGIN { printf "%.1f", value / 1024 }'
    )"

    private_mb="$(
      awk -v value="$private_kb" \
        'BEGIN { printf "%.1f", value / 1024 }'
    )"

    printf \
      "[%s] #%04d elapsed=%ds RSS=%sMB PSS=%sMB Private=%sMB threads=%s CPU=%s%%\n" \
      "$ts_str" \
      "$sample" \
      "$elapsed" \
      "$rss_mb" \
      "$pss_mb" \
      "$private_mb" \
      "$threads" \
      "$cpu_pct"
  fi

  sleep "$interval"
done

cleanup