#!/usr/bin/env python3
"""
async_perception 内存泄漏分析工具。

读取 mem_monitor.sh 生成的 CSV，分析 RSS、PSS、Private、Shared、
VSZ、线程数和 CPU 使用率，并判断是否存在持续内存增长趋势。

用法:
    python3 mem_analyze.py mem_logs/mem_xxx.csv
    python3 mem_analyze.py mem_logs/mem_xxx.csv --plot
    python3 mem_analyze.py mem_logs/mem_*.csv --plot

参数:
    --plot
        生成趋势图，需要 matplotlib。

    --threshold 5.0
        每小时内存增长百分比阈值，默认 5.0%/h。

    --window 60
        绘图滑动平均窗口，单位为采样点，默认 60。

    --warmup 600
        忽略程序启动后的预热时间，单位秒，默认 600 秒。

    --min-hours 1.0
        进行泄漏判定所需的最短有效监控时间，默认 1 小时。

    --min-samples 30
        进行泄漏判定所需的最少有效样本数，默认 30。
"""

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path


REQUIRED_COLUMNS = {
    "timestamp",
    "elapsed_s",
    "rss_kb",
    "vsz_kb",
    "pss_kb",
    "vmdata_kb",
    "threads",
}

OPTIONAL_COLUMNS = {
    "shared_kb",
    "private_kb",
    "vmstk_kb",
    "vmlib_kb",
    "cpu_percent",
}

COLUMN_TO_KEY = {
    "rss_kb": "rss",
    "vsz_kb": "vsz",
    "pss_kb": "pss",
    "shared_kb": "shared",
    "private_kb": "private",
    "vmdata_kb": "vmdata",
    "vmstk_kb": "vmstk",
    "vmlib_kb": "vmlib",
    "threads": "threads",
    "cpu_percent": "cpu_percent",
}

KEY_TO_COLUMN = {
    value: key for key, value in COLUMN_TO_KEY.items()
}

METRIC_LABELS = {
    "rss": "RSS",
    "pss": "PSS",
    "private": "Private",
    "shared": "Shared",
    "vsz": "VSZ",
    "vmdata": "VmData",
    "vmstk": "VmStk",
    "vmlib": "VmLib",
}


def parse_int(row, column, default=0):
    """从CSV行读取整数，允许数值以浮点字符串形式出现。"""
    value = row.get(column)

    if value is None or value.strip() == "":
        return default

    return int(float(value.strip()))


def parse_float(row, column, default=0.0):
    """从CSV行读取浮点数。"""
    value = row.get(column)

    if value is None or value.strip() == "":
        return default

    result = float(value.strip())

    if not math.isfinite(result):
        return default

    return result


def load_csv(path):
    """加载CSV，跳过不完整或非法数据行。"""

    data = {
        "timestamps": [],
        "elapsed": [],
        "rss": [],
        "pss": [],
        "vsz": [],
        "shared": [],
        "private": [],
        "vmdata": [],
        "vmstk": [],
        "vmlib": [],
        "threads": [],
        "cpu_percent": [],
        "available_columns": set(),
    }

    skipped_rows = 0

    # utf-8-sig 同时兼容普通UTF-8和带BOM的CSV。
    with open(path, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)

        if not reader.fieldnames:
            raise ValueError("CSV没有表头")

        available_columns = {
            column.strip()
            for column in reader.fieldnames
            if column is not None
        }

        missing_columns = REQUIRED_COLUMNS - available_columns

        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV缺少必要列: {missing_text}")

        data["available_columns"] = available_columns

        for line_number, row in enumerate(reader, start=2):
            try:
                elapsed = parse_int(row, "elapsed_s")

                if elapsed < 0:
                    raise ValueError("elapsed_s小于0")

                record = {
                    "timestamp": row.get("timestamp", "").strip(),
                    "elapsed": elapsed,
                    "rss": parse_int(row, "rss_kb"),
                    "pss": parse_int(row, "pss_kb"),
                    "vsz": parse_int(row, "vsz_kb"),
                    "shared": parse_int(row, "shared_kb"),
                    "private": parse_int(row, "private_kb"),
                    "vmdata": parse_int(row, "vmdata_kb"),
                    "vmstk": parse_int(row, "vmstk_kb"),
                    "vmlib": parse_int(row, "vmlib_kb"),
                    "threads": parse_int(row, "threads"),
                    "cpu_percent": parse_float(row, "cpu_percent"),
                }

                for memory_key in (
                    "rss",
                    "pss",
                    "vsz",
                    "shared",
                    "private",
                    "vmdata",
                    "vmstk",
                    "vmlib",
                ):
                    if record[memory_key] < 0:
                        raise ValueError(
                            f"{memory_key}小于0"
                        )

                if record["threads"] < 0:
                    raise ValueError("threads小于0")

            except (TypeError, ValueError) as error:
                skipped_rows += 1
                print(
                    f"[WARN] 跳过第 {line_number} 行: {error}",
                    file=sys.stderr,
                )
                continue

            data["timestamps"].append(record["timestamp"])
            data["elapsed"].append(record["elapsed"])
            data["rss"].append(record["rss"])
            data["pss"].append(record["pss"])
            data["vsz"].append(record["vsz"])
            data["shared"].append(record["shared"])
            data["private"].append(record["private"])
            data["vmdata"].append(record["vmdata"])
            data["vmstk"].append(record["vmstk"])
            data["vmlib"].append(record["vmlib"])
            data["threads"].append(record["threads"])
            data["cpu_percent"].append(record["cpu_percent"])

    if skipped_rows:
        print(f"[WARN] 共跳过 {skipped_rows} 行非法数据")

    if not data["elapsed"]:
        raise ValueError("CSV中没有有效数据")

    # 保证按照elapsed_s排序。
    order = sorted(
        range(len(data["elapsed"])),
        key=lambda index: data["elapsed"][index],
    )

    list_keys = (
        "timestamps",
        "elapsed",
        "rss",
        "pss",
        "vsz",
        "shared",
        "private",
        "vmdata",
        "vmstk",
        "vmlib",
        "threads",
        "cpu_percent",
    )

    for key in list_keys:
        data[key] = [
            data[key][index]
            for index in order
        ]

    return data


def moving_average(values, window):
    """计算长度与原始数据一致的滑动平均。"""

    if not values:
        return []

    window = max(1, int(window))
    result = []
    running_sum = 0.0

    for index, value in enumerate(values):
        running_sum += value

        if index >= window:
            running_sum -= values[index - window]

        count = min(index + 1, window)
        result.append(running_sum / count)

    return result


def linear_regression(x_values, y_values):
    """
    简单线性回归。

    返回:
        slope
        intercept
        r_squared
    """

    count = len(x_values)

    if count < 2 or count != len(y_values):
        return 0.0, 0.0, 0.0

    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_xy = sum(
        x_value * y_value
        for x_value, y_value in zip(x_values, y_values)
    )
    sum_x_squared = sum(
        x_value * x_value
        for x_value in x_values
    )

    denominator = (
        count * sum_x_squared
        - sum_x * sum_x
    )

    if denominator == 0:
        return 0.0, sum_y / count, 0.0

    slope = (
        count * sum_xy
        - sum_x * sum_y
    ) / denominator

    intercept = (
        sum_y
        - slope * sum_x
    ) / count

    predicted = [
        slope * x_value + intercept
        for x_value in x_values
    ]

    mean_y = sum_y / count

    residual_sum = sum(
        (actual - estimate) ** 2
        for actual, estimate in zip(y_values, predicted)
    )

    total_sum = sum(
        (value - mean_y) ** 2
        for value in y_values
    )

    if total_sum > 0:
        r_squared = 1.0 - residual_sum / total_sum
    else:
        r_squared = 0.0

    return slope, intercept, r_squared


def column_available(data, key):
    """
    判断CSV是否具有有效指标。

    对PSS、Private和Shared，不仅要求列存在，还要求至少有一个非零值。
    """

    column = KEY_TO_COLUMN.get(key)

    if column is None:
        return False

    if column not in data["available_columns"]:
        return False

    values = data.get(key, [])

    if key in {"pss", "private", "shared"}:
        return any(value > 0 for value in values)

    return bool(values)


def select_indices(data, warmup_seconds):
    """返回预热阶段结束后的数据索引。"""

    if not data["elapsed"]:
        return []

    first_elapsed = data["elapsed"][0]
    cutoff = first_elapsed + warmup_seconds

    return [
        index
        for index, elapsed in enumerate(data["elapsed"])
        if elapsed >= cutoff
    ]


def select_values(values, indices):
    """根据索引选择数据。"""

    return [
        values[index]
        for index in indices
    ]


def memory_statistics(values_kb):
    """计算内存统计，返回MB单位结果。"""

    values_mb = [
        value / 1024.0
        for value in values_kb
    ]

    return {
        "min": min(values_mb),
        "max": max(values_mb),
        "mean": statistics.mean(values_mb),
        "median": statistics.median(values_mb),
        "first": values_mb[0],
        "last": values_mb[-1],
        "range": max(values_mb) - min(values_mb),
    }


def print_memory_statistics(label, values_kb):
    """打印单项内存统计。"""

    if not values_kb:
        return

    stats = memory_statistics(values_kb)

    print(f"\n📈 内存统计 ({label})")
    print(f"  最小值:       {stats['min']:.1f} MB")
    print(f"  最大值:       {stats['max']:.1f} MB")
    print(f"  平均值:       {stats['mean']:.1f} MB")
    print(f"  中位数:       {stats['median']:.1f} MB")
    print(f"  首次采样:     {stats['first']:.1f} MB")
    print(f"  最后采样:     {stats['last']:.1f} MB")
    print(f"  波动范围:     {stats['range']:.1f} MB")


def calculate_trend(elapsed, values_kb, baseline_window):
    """计算指定内存指标的增长趋势。"""

    slope, intercept, r_squared = linear_regression(
        elapsed,
        values_kb,
    )

    growth_mb_per_hour = (
        slope * 3600.0 / 1024.0
    )

    values_mb = [
        value / 1024.0
        for value in values_kb
    ]

    baseline_count = min(
        len(values_mb),
        max(1, baseline_window),
    )

    baseline_mb = statistics.median(
        values_mb[:baseline_count]
    )

    if baseline_mb > 0:
        growth_pct_per_hour = (
            growth_mb_per_hour
            / baseline_mb
            * 100.0
        )
    else:
        growth_pct_per_hour = 0.0

    return {
        "slope_kb_per_second": slope,
        "intercept_kb": intercept,
        "r_squared": r_squared,
        "growth_mb_per_hour": growth_mb_per_hour,
        "growth_pct_per_hour": growth_pct_per_hour,
        "baseline_mb": baseline_mb,
    }


def analyze(
    data,
    threshold_pct_per_hour,
    window,
    warmup_seconds,
    min_hours,
    min_samples,
):
    """执行趋势分析和泄漏判定。"""

    elapsed_all = data["elapsed"]
    sample_count_all = len(elapsed_all)

    if sample_count_all < 2:
        print("[ERROR] 样本数不足，至少需要2个采样点")
        return None

    total_duration_seconds = (
        elapsed_all[-1]
        - elapsed_all[0]
    )
    total_duration_hours = (
        total_duration_seconds / 3600.0
    )

    analysis_indices = select_indices(
        data,
        warmup_seconds,
    )

    elapsed = select_values(
        data["elapsed"],
        analysis_indices,
    )

    analysis_sample_count = len(elapsed)

    if analysis_sample_count >= 2:
        analysis_duration_seconds = (
            elapsed[-1] - elapsed[0]
        )
    else:
        analysis_duration_seconds = 0

    analysis_duration_hours = (
        analysis_duration_seconds / 3600.0
    )

    print("=" * 72)
    print("  async_perception 内存泄漏分析报告")
    print("=" * 72)

    print("\n📊 基本信息")
    print(f"  总样本数:         {sample_count_all}")
    print(
        f"  总监控时长:       "
        f"{total_duration_seconds}s "
        f"({total_duration_hours:.2f} 小时)"
    )
    print(f"  忽略预热时间:     {warmup_seconds}s")
    print(f"  有效分析样本数:   {analysis_sample_count}")
    print(
        f"  有效分析时长:     "
        f"{analysis_duration_seconds}s "
        f"({analysis_duration_hours:.2f} 小时)"
    )

    if sample_count_all >= 2:
        average_interval = (
            total_duration_seconds
            / (sample_count_all - 1)
        )
        print(
            f"  平均采样间隔:     "
            f"{average_interval:.1f}s"
        )

    # 全时段基础统计。
    print_memory_statistics(
        "RSS",
        data["rss"],
    )

    if column_available(data, "pss"):
        print_memory_statistics(
            "PSS",
            data["pss"],
        )
    else:
        print(
            "\n[WARN] PSS数据不可用或全部为0，"
            "可能没有权限读取smaps_rollup"
        )

    if column_available(data, "private"):
        print_memory_statistics(
            "Private",
            data["private"],
        )
    else:
        print(
            "\n[WARN] Private数据不可用或全部为0，"
            "无法直接分析进程独占内存"
        )

    if column_available(data, "shared"):
        print_memory_statistics(
            "Shared",
            data["shared"],
        )

    # 线程统计。
    if data["threads"]:
        print("\n🧵 线程统计")
        print(f"  首次采样:         {data['threads'][0]}")
        print(f"  最后采样:         {data['threads'][-1]}")
        print(f"  最小值:           {min(data['threads'])}")
        print(f"  最大值:           {max(data['threads'])}")

    # CPU统计。第一条CPU为0是正常现象，因此排除第一条。
    if (
        "cpu_percent" in data["available_columns"]
        and len(data["cpu_percent"]) > 1
    ):
        cpu_values = data["cpu_percent"][1:]

        print("\n⚙️ CPU统计")
        print(
            f"  平均值:           "
            f"{statistics.mean(cpu_values):.1f}%"
        )
        print(
            f"  中位数:           "
            f"{statistics.median(cpu_values):.1f}%"
        )
        print(
            f"  最大值:           "
            f"{max(cpu_values):.1f}%"
        )

    # 预热后样本不足时，不执行回归。
    if analysis_sample_count < 2:
        print(
            "\n⚠️ 预热阶段之后不足2个样本，"
            "无法执行趋势分析"
        )
        return None

    # 选择参与趋势分析的指标。
    trend_metrics = ["rss"]

    if column_available(data, "pss"):
        trend_metrics.append("pss")

    if column_available(data, "private"):
        trend_metrics.append("private")

    trend_results = {}

    # 基线窗口最多使用有效样本的前1/4。
    baseline_window = max(
        1,
        min(
            window,
            max(1, analysis_sample_count // 4),
        ),
    )

    print("\n📉 预热后趋势分析（线性回归）")

    for metric in trend_metrics:
        values = select_values(
            data[metric],
            analysis_indices,
        )

        result = calculate_trend(
            elapsed,
            values,
            baseline_window,
        )

        trend_results[metric] = result
        label = METRIC_LABELS[metric]

        print(
            f"  {label:<8}"
            f"{result['growth_mb_per_hour']:+.3f} MB/h  "
            f"{result['growth_pct_per_hour']:+.3f}%/h  "
            f"(R²={result['r_squared']:.4f})"
        )

    # 优先使用Private，其次PSS，最后RSS作为主要判定指标。
    if "private" in trend_results:
        primary_metric = "private"
    elif "pss" in trend_results:
        primary_metric = "pss"
    else:
        primary_metric = "rss"

    primary_label = METRIC_LABELS[primary_metric]
    primary_values_kb = select_values(
        data[primary_metric],
        analysis_indices,
    )
    primary_values_mb = [
        value / 1024.0
        for value in primary_values_kb
    ]
    primary_result = trend_results[primary_metric]

    # 前1/4与后1/4对比。
    quarter = max(
        1,
        analysis_sample_count // 4,
    )

    early_average = statistics.mean(
        primary_values_mb[:quarter]
    )
    late_average = statistics.mean(
        primary_values_mb[-quarter:]
    )

    average_change = (
        late_average - early_average
    )

    if early_average > 0:
        average_change_pct = (
            average_change
            / early_average
            * 100.0
        )
    else:
        average_change_pct = 0.0

    print(
        f"\n📊 前后区间对比 ({primary_label})"
    )
    print(
        f"  前1/4平均:        "
        f"{early_average:.1f} MB"
    )
    print(
        f"  后1/4平均:        "
        f"{late_average:.1f} MB"
    )
    print(
        f"  变化量:           "
        f"{average_change:+.1f} MB "
        f"({average_change_pct:+.2f}%)"
    )

    # 从当前值向未来24小时外推。
    current_mb = primary_values_mb[-1]
    future_24h_mb = (
        current_mb
        + primary_result["growth_mb_per_hour"] * 24.0
    )

    print(
        f"\n🔮 未来24小时线性外推 ({primary_label})"
    )
    print(f"  当前值:           {current_mb:.1f} MB")
    print(
        f"  24小时后预测:     "
        f"{future_24h_mb:.1f} MB"
    )
    print(
        "  注意: 该结果只是假设当前线性趋势持续，"
        "不能代替实际24小时监控"
    )

    # ─── 泄漏判定 ───

    print(f"\n{'=' * 72}")
    print(
        f"  泄漏判定：主要指标={primary_label}，"
        f"阈值={threshold_pct_per_hour}%/h"
    )
    print(f"{'=' * 72}")

    enough_samples = (
        analysis_sample_count >= min_samples
    )
    enough_duration = (
        analysis_duration_hours >= min_hours
    )

    if not enough_samples or not enough_duration:
        print("\n  ⚠️ 数据量不足，暂不进行泄漏定性")

        if not enough_samples:
            print(
                f"     有效样本数 "
                f"{analysis_sample_count}，"
                f"至少需要 {min_samples}"
            )

        if not enough_duration:
            print(
                f"     有效监控时间 "
                f"{analysis_duration_hours:.2f} 小时，"
                f"至少需要 {min_hours:.2f} 小时"
            )

        print("     当前结果只能用于观察，不能判定泄漏")

        return {
            "primary_metric": primary_metric,
            "qualified": False,
            "trend_results": trend_results,
        }

    growth_pct_per_hour = (
        primary_result["growth_pct_per_hour"]
    )
    growth_mb_per_hour = (
        primary_result["growth_mb_per_hour"]
    )
    r_squared = primary_result["r_squared"]

    # 主指标增长时，检查另一个物理内存指标是否同向。
    confirmation_metric = None
    confirmation_positive = True

    if (
        primary_metric == "private"
        and "pss" in trend_results
    ):
        confirmation_metric = "pss"
    elif (
        primary_metric == "pss"
        and "rss" in trend_results
    ):
        confirmation_metric = "rss"

    if confirmation_metric is not None:
        confirmation_positive = (
            trend_results[confirmation_metric][
                "growth_mb_per_hour"
            ] > 0
        )

    strong_growth = (
        growth_pct_per_hour
        > threshold_pct_per_hour
    )
    moderate_growth = (
        growth_pct_per_hour
        > threshold_pct_per_hour * 0.5
    )
    strong_linearity = r_squared >= 0.8
    moderate_linearity = r_squared >= 0.5
    section_increased = average_change > 0

    if (
        strong_growth
        and strong_linearity
        and section_increased
        and confirmation_positive
    ):
        print("\n  ⚠️ 警告：检测到疑似内存泄漏")
        print(
            f"     {primary_label} 每小时增长 "
            f"{growth_mb_per_hour:.3f} MB"
        )
        print(
            f"     相对增长率 "
            f"{growth_pct_per_hour:.3f}%/h"
        )
        print(
            f"     线性拟合度 R²={r_squared:.4f}"
        )
        print(
            f"     前后区间增加 "
            f"{average_change:.1f} MB"
        )

        if confirmation_metric is not None:
            confirmation_label = (
                METRIC_LABELS[confirmation_metric]
            )
            print(
                f"     {confirmation_label} "
                "也呈同向增长"
            )

        print("\n  建议:")
        print("     1. 将监控时间延长到24小时")
        print("     2. 检查无上限容器、缓存和消息队列")
        print("     3. 检查OpenCV Mat及推理结果生命周期")
        print("     4. 使用ASan、LSan或Valgrind进一步定位")

        verdict = "suspected_leak"

    elif (
        (strong_growth and moderate_linearity)
        or (
            moderate_growth
            and section_increased
        )
    ):
        print("\n  ⚡ 注意：内存存在上升迹象")
        print(
            f"     {primary_label} 每小时增长 "
            f"{growth_mb_per_hour:.3f} MB"
        )
        print(
            f"     相对增长率 "
            f"{growth_pct_per_hour:.3f}%/h"
        )
        print(
            f"     线性拟合度 R²={r_squared:.4f}"
        )
        print("     当前证据不足以直接判定泄漏")
        print("     建议延长监控时间并观察Private/PSS")

        verdict = "needs_attention"

    else:
        print("\n  ✅ 未检测到明确的持续内存泄漏")
        print(
            f"     {primary_label} 每小时变化 "
            f"{growth_mb_per_hour:+.3f} MB"
        )
        print(
            f"     相对变化率 "
            f"{growth_pct_per_hour:+.3f}%/h"
        )
        print(
            f"     线性拟合度 R²={r_squared:.4f}"
        )

        if r_squared < 0.5:
            print(
                "     当前数据不符合明显线性增长趋势，"
                "但这不等同于完全没有波动"
            )

        verdict = "no_clear_leak"

    print()

    return {
        "primary_metric": primary_metric,
        "qualified": True,
        "verdict": verdict,
        "trend_results": trend_results,
        "early_average_mb": early_average,
        "late_average_mb": late_average,
    }


def plot_chart(
    data,
    output_path,
    window,
    warmup_seconds,
):
    """生成内存、线程和CPU趋势图。"""

    try:
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

    except ImportError:
        print("[WARN] matplotlib 未安装，跳过绘图")
        print("       安装命令: pip install matplotlib")
        return

    elapsed_hours = [
        elapsed / 3600.0
        for elapsed in data["elapsed"]
    ]

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(15, 13),
        sharex=True,
    )

    # ─── 图1：RSS、PSS、Private ───

    memory_axis = axes[0]

    memory_series = [
        ("rss", "RSS", "#4e79a7"),
    ]

    if column_available(data, "pss"):
        memory_series.append(
            ("pss", "PSS", "#59a14f")
        )

    if column_available(data, "private"):
        memory_series.append(
            ("private", "Private", "#e15759")
        )

    for key, label, color in memory_series:
        values_mb = [
            value / 1024.0
            for value in data[key]
        ]

        smooth_values = moving_average(
            values_mb,
            min(window, len(values_mb)),
        )

        memory_axis.scatter(
            elapsed_hours,
            values_mb,
            s=3,
            alpha=0.20,
            color=color,
            label=f"{label} raw",
        )

        memory_axis.plot(
            elapsed_hours,
            smooth_values,
            linewidth=1.5,
            color=color,
            label=f"{label} smoothed",
        )

    memory_axis.set_ylabel("Memory (MB)")
    memory_axis.set_title(
        "async_perception Physical Memory"
    )
    memory_axis.grid(True, alpha=0.3)
    memory_axis.legend(
        loc="upper left",
        fontsize=9,
    )

    # ─── 图2：VSZ、Shared、VmData ───

    mapping_axis = axes[1]

    vsz_mb = [
        value / 1024.0
        for value in data["vsz"]
    ]

    vmdata_mb = [
        value / 1024.0
        for value in data["vmdata"]
    ]

    mapping_axis.plot(
        elapsed_hours,
        vsz_mb,
        color="#b07aa1",
        linewidth=1.2,
        label="VSZ",
    )

    mapping_axis.plot(
        elapsed_hours,
        vmdata_mb,
        color="#f28e2b",
        linewidth=1.2,
        label="VmData",
    )

    if column_available(data, "shared"):
        shared_mb = [
            value / 1024.0
            for value in data["shared"]
        ]

        mapping_axis.plot(
            elapsed_hours,
            shared_mb,
            color="#76b7b2",
            linewidth=1.2,
            label="Shared",
        )

    mapping_axis.set_ylabel("Memory (MB)")
    mapping_axis.set_title(
        "Virtual and Shared Memory"
    )
    mapping_axis.grid(True, alpha=0.3)
    mapping_axis.legend(
        loc="upper left",
        fontsize=9,
    )

    # ─── 图3：线程数和CPU ───

    runtime_axis = axes[2]

    runtime_axis.plot(
        elapsed_hours,
        data["threads"],
        color="#edc948",
        linewidth=1.2,
        label="Threads",
    )

    runtime_axis.set_ylabel(
        "Threads",
        color="#9c861c",
    )
    runtime_axis.tick_params(
        axis="y",
        labelcolor="#9c861c",
    )
    runtime_axis.grid(True, alpha=0.3)

    runtime_axis_cpu = runtime_axis.twinx()

    if "cpu_percent" in data["available_columns"]:
        runtime_axis_cpu.plot(
            elapsed_hours,
            data["cpu_percent"],
            color="#e15759",
            linewidth=1.0,
            alpha=0.8,
            label="CPU %",
        )

    runtime_axis_cpu.set_ylabel(
        "CPU (%)",
        color="#e15759",
    )
    runtime_axis_cpu.tick_params(
        axis="y",
        labelcolor="#e15759",
    )

    runtime_axis.set_xlabel("Elapsed time (hours)")
    runtime_axis.set_title(
        "Threads and CPU Usage"
    )

    thread_lines, thread_labels = (
        runtime_axis.get_legend_handles_labels()
    )
    cpu_lines, cpu_labels = (
        runtime_axis_cpu.get_legend_handles_labels()
    )

    if thread_lines or cpu_lines:
        runtime_axis.legend(
            thread_lines + cpu_lines,
            thread_labels + cpu_labels,
            loc="upper left",
            fontsize=9,
        )

    # 标记预热结束位置。
    if data["elapsed"]:
        warmup_end_seconds = (
            data["elapsed"][0]
            + warmup_seconds
        )
        warmup_end_hours = (
            warmup_end_seconds / 3600.0
        )

        if warmup_end_seconds <= data["elapsed"][-1]:
            for axis in axes:
                axis.axvline(
                    warmup_end_hours,
                    linestyle="--",
                    linewidth=1,
                    color="#888888",
                    alpha=0.8,
                )

            memory_axis.text(
                warmup_end_hours,
                memory_axis.get_ylim()[1],
                " warmup end",
                va="top",
                ha="left",
                fontsize=8,
                color="#666666",
            )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=150,
    )
    plt.close(figure)

    print(f"[INFO] 趋势图已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "async_perception 内存泄漏分析"
        )
    )

    parser.add_argument(
        "csv_files",
        nargs="+",
        help="mem_monitor.sh 生成的CSV文件",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="生成趋势图",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help=(
            "每小时增长率阈值，单位%%/h，"
            "默认 %(default)s"
        ),
    )

    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help=(
            "滑动平均窗口，单位采样点，"
            "默认 %(default)s"
        ),
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=600,
        help=(
            "忽略启动后的预热时间，单位秒，"
            "默认 %(default)s"
        ),
    )

    parser.add_argument(
        "--min-hours",
        type=float,
        default=1.0,
        help=(
            "泄漏判定所需最短有效监控时长，"
            "默认 %(default)s 小时"
        ),
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help=(
            "泄漏判定所需最少有效样本数，"
            "默认 %(default)s"
        ),
    )

    args = parser.parse_args()

    if args.threshold < 0:
        parser.error("--threshold 不能小于0")

    if args.window <= 0:
        parser.error("--window 必须大于0")

    if args.warmup < 0:
        parser.error("--warmup 不能小于0")

    if args.min_hours < 0:
        parser.error("--min-hours 不能小于0")

    if args.min_samples < 2:
        parser.error("--min-samples 不能小于2")

    had_error = False

    for csv_path_text in args.csv_files:
        csv_path = Path(csv_path_text)

        if not csv_path.exists():
            print(
                f"[ERROR] 文件不存在: {csv_path}",
                file=sys.stderr,
            )
            had_error = True
            continue

        print(f"\n{'#' * 72}")
        print(f"  分析文件: {csv_path}")
        print(f"{'#' * 72}")

        try:
            data = load_csv(str(csv_path))
        except (
            OSError,
            ValueError,
            csv.Error,
        ) as error:
            print(
                f"[ERROR] 无法读取 {csv_path}: {error}",
                file=sys.stderr,
            )
            had_error = True
            continue

        analyze(
            data=data,
            threshold_pct_per_hour=args.threshold,
            window=args.window,
            warmup_seconds=args.warmup,
            min_hours=args.min_hours,
            min_samples=args.min_samples,
        )

        if args.plot:
            output_path = csv_path.with_name(
                f"{csv_path.stem}_trend.png"
            )

            plot_chart(
                data=data,
                output_path=str(output_path),
                window=args.window,
                warmup_seconds=args.warmup,
            )

    if had_error:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())