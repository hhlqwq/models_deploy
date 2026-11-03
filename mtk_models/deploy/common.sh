#!/usr/bin/env bash
# 总入口只共用日志和结果传输,不处理模型转换、推理或评分.

start_log() {
    # 覆盖日志并保留终端输出,不重新调用入口脚本.
    local logfile="$1"
    : > "${logfile}"
    exec 3>&1 4>&2
    exec > >(tee -- "${logfile}") 2>&1
    LOG_PID=$!
    export PYTHONUNBUFFERED=1
    trap finish_log EXIT
    echo "[日志] ${logfile}; 本次覆盖旧日志."
}

finish_log() {
    # 等待日志写入完成,保留模型退出码并报告日志写入失败.
    local status=$? log_status=0
    trap - EXIT
    exec 1>&3 2>&4
    wait "${LOG_PID}" || log_status=$?
    if (( status != 0 )); then
        exit "${status}"
    fi
    exit "${log_status}"
}

return_results() {
    # 只回传已完成的摘要和示例,不调用 compile 或 evaluate.
    source "${SCRIPT_DIR}/platform.local.sh"
    : "${MTK_RESULTS_HOST:?缺少本地回传账户}"
    : "${MTK_RESULTS_ROOT:?缺少编译服务器仓库根目录}"
    local SCOPE="$1"
    if [[ "${MTK_RESULTS_ROOT}" != /* || "${MTK_RESULTS_ROOT}" == / ]]; then
        echo "[错误] 回传目标必须为编译服务器仓库根目录的绝对路径." >&2
        exit 2
    fi
    KEY="${MTK_RESULTS_KEY:-${HOME}/.ssh/mtk_results_ed25519}"
    if [[ ! -s "${KEY}" ]]; then
        echo "[错误] 缺少板端专用回传密钥: ${KEY}." >&2
        exit 2
    fi
    printf -v QUOTED_KEY '%q' "${KEY}"
    if ssh -V 2>&1 | grep -qi Dropbear; then
        # Dropbear 不支持 BatchMode;无控制终端并拒绝交互认证,仍检查 known_hosts.
        export SSH_ASKPASS=/bin/false DISPLAY=mtk-upload
        SSH=(setsid --wait ssh -T -i "${KEY}")
        RSYNC_SSH="setsid --wait ssh -T -i ${QUOTED_KEY}"
    else
        SSH=(ssh -i "${KEY}" -o BatchMode=yes -o StrictHostKeyChecking=yes)
        RSYNC_SSH="ssh -i ${QUOTED_KEY} -o BatchMode=yes -o StrictHostKeyChecking=yes"
    fi
    STATUS=0
    INDEX=0
    for MODEL in "${MODELS[@]}"; do
        INDEX=$((INDEX + 1))
        SOURCE="${MODELS_ROOT}/${MODEL##*/}"
        TARGET="${MTK_RESULTS_ROOT%/}/mtk_models/${MODEL}"
        echo "[回传 ${INDEX}/${#MODELS[@]}] ${MODEL##*/}"
        if [[ -L "${SOURCE}/examples/output" || -L "${SOURCE}/results/summary.json" || -L "${SOURCE}/results/summary_test.json" ]]; then
            echo "[错误] 回传源不允许为符号链接: ${MODEL}." >&2
            exit 2
        fi
        printf -v QUOTED_TARGET '%q' "${TARGET}"
        "${SSH[@]}" "${MTK_RESULTS_HOST}" \
            "mkdir -p -- ${QUOTED_TARGET}/examples/output ${QUOTED_TARGET}/results" </dev/null
        if [[ -d "${SOURCE}/examples/output" ]] && \
                [[ -n "$(find "${SOURCE}/examples/output" -type f -size +0c -print -quit)" ]]; then
            rsync -rlt --no-perms --omit-dir-times --no-links --protect-args --info=progress2 \
                -e "${RSYNC_SSH}" "${SOURCE}/examples/output/" \
                "${MTK_RESULTS_HOST}:${TARGET}/examples/output/"
        else
            echo "[缺失] ${MODEL}/examples/output/ 没有非空示例." >&2
            STATUS=1
        fi
        REPORTS=()
        if [[ "${SCOPE}" == all ]]; then
            REPORTS+=(summary.json)
        fi
        if [[ "${SCOPE}" == subset ]] || \
                [[ "${SCOPE}" == all && -s "${SOURCE}/results/summary_test.json" ]]; then
            REPORTS+=(summary_test.json)
        fi
        for REPORT in "${REPORTS[@]}"; do
            if [[ -s "${SOURCE}/results/${REPORT}" ]]; then
                rsync -rlt --no-perms --omit-dir-times --protect-args --info=progress2 \
                    -e "${RSYNC_SSH}" "${SOURCE}/results/${REPORT}" \
                    "${MTK_RESULTS_HOST}:${TARGET}/results/${REPORT}"
            else
                echo "[缺失] ${MODEL}/results/${REPORT} 未生成,不回传空报告." >&2
                STATUS=1
            fi
        done
    done
    if (( STATUS == 0 )); then
        echo "[回传完成] 已同步 ${SCOPE} 范围的结果,未删除服务器其他文件."
    fi
    return "${STATUS}"

}
