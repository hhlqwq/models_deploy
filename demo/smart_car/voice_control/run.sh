#!/usr/bin/env bash
set -euo pipefail

# 以脚本目录为基准,可从任意工作目录启动.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:-help}"
config="$root/config.json"
long_recording="$root/examples/流式音频.m4a"

if [[ "$#" -gt 1 ]]; then
    echo "用法: $0 {short|replay}" >&2
    exit 2
fi

case "$mode" in
    short)
        binary="$root/voice_control_file"
        shopt -s nullglob nocasematch
        files=()
        # 只枚举顶层音频,按文件名顺序运行,忽略历史归档和指定长录音.
        export LC_ALL=C
        for file in "$root"/examples/*; do
            if [[ ! -f "$file" || "$file" == "$long_recording" ]]; then
                continue
            fi
            case "${file##*.}" in
                wav|m4a|mp3|aac|flac|ogg|opus|webm) files+=("$file") ;;
            esac
        done
        if (( ${#files[@]} == 0 )); then
            echo "[错误] examples/ 顶层没有短音频." >&2
            exit 2
        fi
        args=("$config" --plan "${files[@]}")
        ;;
    replay)
        binary="$root/voice_control_mic"
        if [[ ! -f "$long_recording" ]]; then
            echo "[错误] 找不到 $long_recording" >&2
            exit 2
        fi
        args=("$config" --plan --replay "$long_recording")
        ;;
    help|--help|-h)
        printf '%s\n' "用法: $0 {short|replay}" "  short   验收 examples/ 顶层短音频" "  replay  按实时节奏输入 examples/流式音频.m4a"
        exit 0
        ;;
    *)
        echo "[错误] 未知模式: $mode. 可选 short 或 replay." >&2
        exit 2
        ;;
esac

if [[ ! -x "$binary" || ! -r "$config" ]]; then
    echo "[错误] 请准备可执行程序 $binary 和配置 $config." >&2
    exit 2
fi

# 地瓜运行库可能把诊断写到 stdout,这里仅将应用的 JSON 行留给结果管道.
"$binary" "${args[@]}" | awk '/^\{/ { print; fflush(); next } { print > "/dev/stderr"; fflush("/dev/stderr") }'
