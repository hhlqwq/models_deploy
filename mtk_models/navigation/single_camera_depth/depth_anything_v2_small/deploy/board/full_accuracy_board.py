"""在 Genio 720 上评测 DA-2K 全量相对深度点对准确率。"""


import os
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.board import configure_parser, inference_records

from tools.evaluation import (
    is_test, result_status, scope_metadata, select_samples,
)
from tools.evaluation import accuracy_result


EXPECTED_IMAGES = 1033
EXPECTED_PAIRS = 2068


def parse_args() -> argparse.Namespace:
    """解析固定数据集、模型与本次运行目录。"""
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


def load_protocol(dataset_root: Path) -> dict[str, list[dict]]:
    """核对官方压缩包中全部图片与点对标注。"""
    path = dataset_root / "annotations.json"
    annotations = json.loads(path.read_text(encoding="utf-8"))
    if len(annotations) != EXPECTED_IMAGES:
        raise ValueError(f"DA-2K 图片数不符: {len(annotations)}。")
    pair_count = sum(len(pairs) for pairs in annotations.values())
    if pair_count != EXPECTED_PAIRS:
        raise ValueError(f"DA-2K 点对数不符: {pair_count}。")
    for relative, pairs in annotations.items():
        if not relative.startswith("images/") or ".." in Path(relative).parts:
            raise ValueError(f"非法图片路径: {relative}。")
        if not (dataset_root / relative).is_file():
            raise FileNotFoundError(dataset_root / relative)
        if not pairs:
            raise ValueError(f"缺少深度点对: {relative}。")
        if any(pair.get("closer_point") != "point1" for pair in pairs):
            raise ValueError(f"深度标注方向异常: {relative}。")
    chosen = select_samples(sorted(annotations), "depth_anything_v2_small", required=(
        "images/indoor/1009402064_7c74b0b819_k.jpg",
        "images/outdoor/17943623232_5f974cad30_k.jpg",
        "images/transparent_reflective/49126232263_30c5bf5038_k.jpg"))
    return {key: annotations[key] for key in chosen}


def load_output(path: Path) -> np.ndarray:
    """读取原始 518x518 FP32 深度图,不保存放大后的重复数据."""
    values = np.load(path, allow_pickle=False)
    if values.shape != (518, 518) or not np.isfinite(values).all() or values.std() <= 0:
        raise ValueError(f"深度输出无效: {path}.")
    return values


def score_pairs(image: Path, depth: np.ndarray,
                pairs: list[dict], *, restored: bool = False) -> tuple[int, int]:
    """按原图坐标比较点对,数值较大的一点判为更近。"""
    original = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"无法读取图片: {image}。")
    height, width = original.shape[:2]
    if restored:
        if depth.shape != (height, width):
            raise ValueError(f"Python 深度图尺寸与原图不一致: {image}.")
        values = depth
    else:
        # ONNX 评测复用此指标函数,仍需把模型原始输出恢复到原图尺寸.
        values = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    correct = 0
    for pair in pairs:
        points = []
        for key in ("point1", "point2"):
            y, x = pair[key]
            if not (0 <= y < height and 0 <= x < width):
                raise ValueError(f"标注坐标越界: {image}, {key}。")
            points.append(float(values[int(y), int(x)]))
        correct += points[0] > points[1]
    return int(correct), len(pairs)


def evaluate(args: argparse.Namespace) -> None:
    """只读取本次 Python 预测并计算 DA-2K 指标,不启动硬件推理."""
    annotations = load_protocol(args.dataset_root)
    records = inference_records(args.work_dir, annotations)
    correct_pairs = total_pairs = 0
    scenes = {}
    for index, relative in enumerate(sorted(annotations), 1):
        row = records[relative]
        depth = load_output(args.work_dir / row["directory"] / "depth.npy")
        correct, total = score_pairs(args.dataset_root / relative, depth,
                                     annotations[relative])
        correct_pairs += correct
        total_pairs += total
        scene = Path(relative).parts[1]
        current = scenes.setdefault(scene, {"images": 0, "correct_pairs": 0, "total_pairs": 0})
        current["images"] += 1
        current["correct_pairs"] += correct
        current["total_pairs"] += total
        if index % 100 == 0 or index == len(annotations):
            print(f"[评测] DA-2K {index}/{len(annotations)}", flush=True)
    if total_pairs != sum(len(pairs) for pairs in annotations.values()):
        raise ValueError("实际评测点对数不符.")
    for item in scenes.values():
        item["pairwise_accuracy"] = item["correct_pairs"] / item["total_pairs"]
    times = [] if is_test() else [row["npu_ms"] for row in records.values()]
    report = {
        "status": result_status(), **scope_metadata(),
        "model": "depth_anything_v2_small", "run_id": args.run_id, "dataset": "DA-2K",
        "metric_protocol": "fixed_518_square_bilinear_restore_point1_closer",
        "images": len(records), "correct_pairs": correct_pairs, "total_pairs": total_pairs,
        "pairwise_accuracy": correct_pairs / total_pairs, "scenes": scenes,
        "backend": "python_neuron_runtime_hw", "npu_mean_ms": None if is_test() else float(np.mean(times)),
        "npu_p95_ms": None if is_test() else float(np.percentile(times, 95)),
        "peak_rss_kib": None if is_test() else max(row["peak_rss_kib"] for row in records.values()),
        "timing_scope": "常驻 Python 模型,单独统计 NeuronRuntime_inference 调用",
    }
    report["evaluation"] = accuracy_result("depth_anything_v2_small", "npu",
        {"DA-2K 点对准确率": report["pairwise_accuracy"]}, total_pairs, "annotated_point_pair")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[评测] DA-2K 点对准确率: {report['pairwise_accuracy']:.6f}; {args.output}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    evaluate(parse_args())
