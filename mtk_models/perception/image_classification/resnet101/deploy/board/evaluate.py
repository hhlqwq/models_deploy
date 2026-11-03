"""核对 ResNet101 全量预测并发布唯一精度、性能和 RSS 报告."""


import json
import math
import os
import platform
import statistics

from resnet101 import MODEL, PROTOCOL, WEIGHTS, label_signature, read_labels, validate_contract
from summary import publish
from tools.board import context
from tools.evaluation import is_test, select_samples, scope_metadata
from tools.files import read_json, sha256
from tools.evaluation import accuracy_result


def matched_reference(ctx, metadata: dict, labels_digest: str):
    """只接受相同 ONNX、标签映射和预处理的完整参考,缺失项如实记录."""
    path = ctx.root / "build/reference/full.json"
    if not path.is_file():
        return None, ["缺少同协议 FP32 ONNX 参考,精度差值未计算."]
    baseline = read_json(path)
    required = {
        "model": MODEL, "weights": WEIGHTS, "status": "complete",
        "evaluation_scope": "full", "protocol": PROTOCOL, "samples": 50000, "metric": "Top-1",
        "onnx_sha256": metadata["onnx_sha256"], "labels_sha256": labels_digest,
        "pillow": metadata["pillow"],
    }
    if any(baseline.get(key) != value for key, value in required.items()):
        raise ValueError("FP32 参考与本次模型、标签或评价协议不匹配,请重新准备参考.")
    accuracy = baseline["accuracy"]
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("参考 Top-1 无效.")
    return accuracy, []


def main() -> None:
    """评分不补跑推理,完整覆盖后才发布一个报告."""
    ctx = context(__file__)
    if is_test():
        labels = read_labels(ctx.dataset / "val_labels_0based.txt")
        selected = select_samples(list(range(len(labels))), MODEL,
            key=lambda index: f"ILSVRC2012_val_{index + 1:08d}.JPEG")
        rows = [json.loads(line) for line in
                (ctx.work / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        if (len(rows) != len(selected) or any(
                row.get("index") != position or row.get("image") !=
                f"ILSVRC2012_val_{index + 1:08d}.JPEG" or
                row.get("backend") != "python_neuron_runtime_hw"
                for position, (index, row) in enumerate(zip(selected, rows)))):
            raise ValueError("ResNet 子集预测覆盖或来源不匹配.")
        # 样本指纹使用图片名称,与主机及推理阶段完全一致.
        from tools.evaluation import record_selection

        record_selection(row["image"] for row in rows)
        report = {"metric": "Top-1", "accuracy": sum(
            row["top5"][0] == labels[index] for index, row in zip(selected, rows)) / len(rows),
            "samples": len(rows), **scope_metadata()}
        report["evaluation"] = accuracy_result(MODEL, "npu",
            {"Top-1": report["accuracy"]}, len(rows), "image")
        (ctx.work / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return
    metadata = read_json(ctx.root / "build/reference/quantization.json")
    validate_contract(metadata)
    run = read_json(ctx.work / "inference.json")
    labels = read_labels(ctx.dataset / "val_labels_0based.txt")
    labels_digest = label_signature(labels)
    required = {
        "model": MODEL, "run_id": ctx.run_id, "samples": 50000,
        "evaluation_scope": "full", "backend": "python_neuron_runtime_hw",
        "onnx_sha256": metadata["onnx_sha256"], "dla_sha256": metadata["dla_sha256"],
        "labels_sha256": labels_digest,
    }
    if any(run.get(key) != value for key, value in required.items()):
        raise ValueError("推理记录与本次模型、运行编号或完整评价范围不匹配.")
    if sha256(ctx.root / "model_int8.dla") != run["dla_sha256"]:
        raise ValueError("推理后 DLA 已发生变化.")
    baseline, unavailable = matched_reference(ctx, metadata, labels_digest)
    calibration_images = set(metadata["calibration"]["images"])
    overlap = set()
    top1 = 0
    npu_times, end_to_end_times = [], []
    with (ctx.work / "predictions.jsonl").open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= 50000:
                raise ValueError("预测数量超过 50000.")
            row = json.loads(line)
            name = f"ILSVRC2012_val_{index + 1:08d}.JPEG"
            predictions = row["top5"]
            if (row.get("index") != index or row.get("image") != name
                    or row.get("backend") != "python_neuron_runtime_hw"
                    or not isinstance(predictions, list) or len(predictions) != 5
                    or any(type(value) is not int or not 0 <= value < 1000 for value in predictions)
                    or len(set(predictions)) != 5):
                raise ValueError(f"预测记录不匹配: {name}.")
            npu, elapsed = row["npu_ms"], row["end_to_end_ms"]
            if not math.isfinite(npu) or npu <= 0 or not math.isfinite(elapsed) or elapsed < npu:
                raise ValueError(f"性能记录无效: {name}.")
            npu_times.append(npu)
            end_to_end_times.append(elapsed)
            top1 += predictions[0] == labels[index]
            if name in calibration_images:
                overlap.add(name)
            if (index + 1) % 5000 == 0:
                print(f"[评分] {MODEL} {index + 1}/50000.", flush=True)
    if len(npu_times) != 50000:
        raise ValueError(f"预测覆盖不完整: {len(npu_times)}/50000.")
    if type(run["peak_rss_kib"]) is not int or run["peak_rss_kib"] <= 0 or run["warmup_runs"] != 5:
        raise ValueError("峰值 RSS 或预热记录无效.")
    accuracy = top1 / 50000
    reference = baseline
    report = {
        "status": "complete" if baseline is not None else "partial", "model": MODEL,
        "run_id": ctx.run_id, "dataset": PROTOCOL["dataset"],
        "samples": len(npu_times), "metric": "Top-1", "board_accuracy": accuracy,
        "npu_mean_ms": statistics.fmean(npu_times),
        "timing_scope": "常驻模型,50000 张图片的 NeuronRuntime_inference 调用,预热 5 次",
        "peak_rss_mib": run["peak_rss_kib"] / 1024,
        "memory_scope": "Python 推理进程峰值 RSS",
        "reference_accuracy": reference,
        "accuracy_change_percentage_points": None if reference is None else round((accuracy - reference) * 100, 6),
        "reference_source": None if baseline is None else "本次 ONNX FP32 全量实测,使用与板端相同的评测协议",
        "model_size": {"unit": "MB", "compiled_dla": (ctx.root / "model_int8.dla").stat().st_size / 1000000},
        "evaluation_scope": "full", "weights": WEIGHTS, "precision": "W8A8",
        "calibration_samples": metadata["calibration"]["samples"],
        "calibration_overlap_samples": len(overlap),
        "npu_p95_ms": sorted(npu_times)[math.ceil(0.95 * 50000) - 1],
        "end_to_end_mean_ms": statistics.fmean(end_to_end_times),
        "end_to_end_timing_scope": "图片读取至类别排序,不含模型加载及预测写盘",
        "target_backend": "python_neuron_runtime_hw",
        "runtime": {"machine": platform.machine(), "kernel": platform.release()},
        "unavailable_metrics": unavailable,
    }
    from tools.evaluation import record_selection

    record_selection(f"ILSVRC2012_val_{index:08d}.JPEG" for index in range(1, 50001))
    report["evaluation"] = accuracy_result(MODEL, "npu", {"Top-1": accuracy}, 50000, "image")
    publish(ctx, report)


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    main()
