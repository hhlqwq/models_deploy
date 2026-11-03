"""在 Genio 720 上完成 LFW 全量十折人脸验证。"""


import os
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from tools.board import configure_parser, inference_records

from tools.evaluation import is_test, lfw_pairs, result_status, scope_metadata
from tools.evaluation import accuracy_result


EXPECTED_PAIRS = 6000


def parse_args() -> argparse.Namespace:
    """解析固定数据集、模型与运行目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    configure_parser(parser, __file__, {
        "dataset_root": lambda c: c.dataset, "work_dir": lambda c: c.work,
        "run_id": lambda c: c.run_id, "output": lambda c: c.work / "report/summary.json",
    })
    return parser.parse_args()


def load_pairs(dataset_root: Path) -> list[dict[str, str]]:
    """核对完整十折 6000 对与全部已对齐图片。"""
    alignment_path = dataset_root / "alignment_summary.json"
    if not alignment_path.is_file():
        raise FileNotFoundError(f"缺少 LFW 对齐报告: {alignment_path}。")
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    if (alignment.get("status") != "complete" or
            alignment.get("aligned_images") != 7701):
        raise ValueError("LFW 对齐图未覆盖全部 7,701 张验证图。")
    with (dataset_root / "pairs.csv").open(
            newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_PAIRS:
        raise ValueError(f"LFW 验证对数量不符: {len(rows)}。")
    folds = Counter(int(row["fold_id"]) for row in rows)
    labels = Counter(int(row["is_same"]) for row in rows)
    if (folds != {fold: 600 for fold in range(1, 11)} or
            labels != {0: 3000, 1: 3000}):
        raise ValueError("LFW 十折或正负样本数量不符。")
    images_dir = dataset_root / "images"
    for row in rows:
        for field in ("image_a", "image_b"):
            name = row[field]
            if Path(name).name != name or not (images_dir / name).is_file():
                raise ValueError(f"LFW 图片缺失或路径非法: {name}。")
    return lfw_pairs(rows)


def load_feature(path: Path) -> np.ndarray:
    """读取 Python 已反量化并单位化的 128 维特征,不重复执行后处理."""
    feature = np.fromfile(path, dtype=np.float32)
    if (feature.size != 128 or not np.isfinite(feature).all() or
            not np.isclose(np.linalg.norm(feature), 1.0, atol=1e-5)):
        raise ValueError(f"Python 人脸特征无效: {path}.")
    return feature


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """仅用其他九折训练分数选择最高准确率的余弦阈值。"""
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    positive_total = int(sorted_labels.sum())
    best_correct = positive_total
    best_value = float(sorted_scores[0]) - 1e-6
    positive_below = negative_below = 0
    for index, score in enumerate(sorted_scores):
        positive_below += int(sorted_labels[index] == 1)
        negative_below += int(sorted_labels[index] == 0)
        if index + 1 < len(sorted_scores) and score == sorted_scores[index + 1]:
            continue
        correct = negative_below + positive_total - positive_below
        threshold = (float(score + sorted_scores[index + 1]) / 2.0
                     if index + 1 < len(sorted_scores) else float(score) + 1e-6)
        if correct > best_correct:
            best_correct = correct
            best_value = threshold
    return best_value


def evaluate_pairs(rows: list[dict[str, str]],
                   features: dict[str, np.ndarray]) -> tuple[dict, list[dict]]:
    """计算预定义十折验证准确率与逐对判定。"""
    scores = np.array([
        float(np.dot(features[row["image_a"]], features[row["image_b"]]))
        for row in rows], dtype=np.float64)
    labels = np.array([int(row["is_same"]) for row in rows], dtype=np.int8)
    folds = np.array([int(row["fold_id"]) for row in rows], dtype=np.int8)
    details = [dict(row, cosine_similarity=float(score))
               for row, score in zip(rows, scores)]
    fold_reports = []
    for fold in range(1, 11):
        train = folds != fold
        test = folds == fold
        threshold = best_threshold(scores[train], labels[train])
        predictions = scores[test] >= threshold
        correct = int(np.count_nonzero(predictions == labels[test]))
        fold_reports.append({"fold": fold, "pairs": int(test.sum()),
                             "correct": correct, "accuracy": correct / int(test.sum()),
                             "threshold": threshold})
        for detail, prediction in zip(
                (item for item in details if int(item["fold_id"]) == fold),
                predictions):
            detail["predicted_same"] = int(prediction)
    correct_total = sum(item["correct"] for item in fold_reports)
    summary = {
        "metric_protocol": "lfw_10_fold_train_9_test_1_cosine_threshold",
        "pairs": len(rows),
        "positive_pairs": int(labels.sum()),
        "negative_pairs": int(len(labels) - labels.sum()),
        "correct_pairs": correct_total,
        "verification_accuracy": correct_total / len(rows),
        "fold_mean_accuracy": float(np.mean(
            [item["accuracy"] for item in fold_reports])),
        "fold_std_accuracy": float(np.std(
            [item["accuracy"] for item in fold_reports])),
        "folds": fold_reports,
    }
    return summary, details


def evaluate(args: argparse.Namespace) -> None:
    """仅计算 Python 预测的十折人脸验证指标,不执行模型推理."""
    rows = load_pairs(args.dataset_root)
    names = {row[field] for row in rows for field in ("image_a", "image_b")}
    records = inference_records(args.work_dir, names)
    features = {}
    for index, name in enumerate(sorted(names), 1):
        features[name] = load_feature(args.work_dir / records[name]["directory"] / "feature.bin")
        if index % 100 == 0 or index == len(names):
            print(f"[评测] LFW {index}/{len(names)}", flush=True)
    verification, details = evaluate_pairs(rows, features)
    times = [] if is_test() else [row["npu_ms"] for row in records.values()]
    report = {
        "status": result_status(), **scope_metadata(),
        "model": "mobilefacenet", "run_id": args.run_id,
        "dataset": "LFW_upstream_MTCNN_5_point_aligned",
        "input_protocol": "RGB_ImageNet_normalize_aligned_112x112",
        "distinct_images": len(names), **verification,
        "backend": "python_neuron_runtime_hw", "npu_mean_ms": None if is_test() else float(np.mean(times)),
        "npu_p95_ms": None if is_test() else float(np.percentile(times, 95)),
        "peak_rss_kib": None if is_test() else max(row["peak_rss_kib"] for row in records.values()),
        "timing_scope": "常驻 Python 模型,单独统计 NeuronRuntime_inference 调用",
    }
    report["evaluation"] = accuracy_result("mobilefacenet", "npu",
        {"LFW 验证准确率": verification["verification_accuracy"]}, len(rows), "face_pair")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[评测] LFW 验证准确率: {report['verification_accuracy']:.6f}; {args.output}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    evaluate(parse_args())
