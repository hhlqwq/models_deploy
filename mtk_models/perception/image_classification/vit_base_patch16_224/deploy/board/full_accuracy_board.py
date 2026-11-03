"""仅从板端 Python 预测计算 ImageNet val 全量 Top-1/Top-5 指标."""


import os
import argparse
import json
import math
import statistics
from pathlib import Path

from tools.board import configure_parser

from tools.evaluation import (
    is_test, record_selection, result_status, scope_metadata,
    select_samples,
)
from tools.evaluation import accuracy_result



def parse_args() -> argparse.Namespace:
    """解析指标阶段输入参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    configure_parser(parser, __file__, {
        "predictions": lambda c: c.work / "predictions.jsonl",
        "labels": lambda c: c.dataset / "val_labels_0based.txt",
        "report": lambda c: c.work / "report",
        "run_id": lambda c: c.run_id,
    })
    return parser.parse_args()


def read_labels(path: Path) -> list[int]:
    """核对官方图片顺序对应的 50000 个零基类别."""
    labels = []
    for line_number, line in enumerate(path.read_text(
            encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) not in (1, 2):
            raise ValueError(f"标签格式错误: {line_number}.")
        if len(parts) == 2 and int(parts[0]) != line_number:
            raise ValueError(f"标签序号不连续: {line_number}.")
        label = int(parts[-1])
        if not 0 <= label < 1000:
            raise ValueError(f"标签类别越界: {line_number}.")
        labels.append(label)
    if len(labels) != 50000:
        raise ValueError(f"预期 50000 个标签,实际 {len(labels)}.")
    return labels


def evaluate(args: argparse.Namespace) -> None:
    """校验全部 Python 输出并汇总精度与纯 NPU 耗时."""
    labels = read_labels(args.labels)
    selected = select_samples(list(enumerate(labels)), "vit_base_patch16_224",
        key=lambda row: f"ILSVRC2012_val_{row[0] + 1:08d}.JPEG")
    expected = len(selected)
    if is_test():
        record_selection(f"ILSVRC2012_val_{index + 1:08d}.JPEG" for index, _ in selected)
    holdout_count = sum(not 1000 <= index < 1100 for index, _ in selected)
    top1 = top5 = holdout_top1 = holdout_top5 = 0
    timings = []
    with args.predictions.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= expected:
                raise ValueError("预测数量超过 50000.")
            record = json.loads(line)
            source_index, label = selected[index]
            expected_image = f"ILSVRC2012_val_{source_index + 1:08d}.JPEG"
            predictions = record.get("top5")
            if (record.get("backend") != "python_neuron_runtime_hw" or
                    record.get("index") != index or
                    record.get("image") != expected_image or
                    not isinstance(predictions, list) or
                    len(predictions) != 5 or
                    len(set(predictions)) != 5 or
                    any(not isinstance(value, int) or
                        not 0 <= value < 1000 for value in predictions)):
                raise ValueError(f"Python 预测记录不匹配: {index}.")
            timing = None if is_test() else float(record["npu_ms"])
            if timing is not None and (not math.isfinite(timing) or timing < 0):
                raise ValueError(f"NPU 耗时非法: {index}.")
            timings.append(timing)
            top1 += predictions[0] == label
            top5 += label in predictions
            if not 1000 <= source_index < 1100:
                holdout_top1 += predictions[0] == label
                holdout_top5 += label in predictions
            if (index + 1) % 5000 == 0:
                print(f"[METRIC] ViT {index + 1}/50000 条.", flush=True)
    if len(timings) != expected:
        raise ValueError(f"预测数量不完整: {len(timings)}/50000.")
    sorted_times = [] if is_test() else sorted(timings)
    memory_path = args.predictions.parent / "memory.json"
    memory = json.loads(memory_path.read_text(encoding="utf-8")) if memory_path.is_file() else {}
    summary = {
        "peak_rss_kib": memory.get("peak_rss_kib"),
        "status": result_status(), **scope_metadata(),
        "model": "vit_base_patch16_224", "backend": "python_neuron_runtime_hw",
        "run_id": args.run_id,
        "dataset": "imagenet_val2012",
        "samples": expected,
        "holdout_samples": holdout_count,
        "npu_top1": top1 / expected,
        "npu_top5": top5 / expected,
        "holdout_top1": holdout_top1 / holdout_count,
        "holdout_top5": holdout_top5 / holdout_count,
        "timing_scope": "persistent Python Neuron Runtime inference call only",
        "npu_mean_ms": None if is_test() else statistics.fmean(timings),
        "npu_p95_ms": None if is_test() else sorted_times[math.ceil(0.95 * len(sorted_times)) - 1],
    }
    args.report.mkdir(parents=True, exist_ok=True)
    summary["evaluation"] = accuracy_result("vit_base_patch16_224", "npu",
        {"Top-1": summary["npu_top1"]}, expected, "image")
    (args.report / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"[OK] ViT 指标: {args.report}.")


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    evaluate(parse_args())
