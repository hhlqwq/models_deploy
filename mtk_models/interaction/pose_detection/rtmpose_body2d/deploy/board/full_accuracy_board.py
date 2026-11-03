"""按 MMPose 协议计算 COCO-WholeBody 整体 AP."""


import os
import argparse
import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from tools.board import configure_parser

from tools.evaluation import (
    is_test, pose_image_ids, result_status, scope_metadata,
)
from tools.evaluation import accuracy_result, record_selection

SIGMAS = np.asarray([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.068,
    0.066, 0.066, 0.092, 0.094, 0.094, 0.042, 0.043, 0.044, 0.043,
    0.040, 0.035, 0.031, 0.025, 0.020, 0.023, 0.029, 0.032, 0.037,
    0.038, 0.043, 0.041, 0.045, 0.013, 0.012, 0.011, 0.011, 0.012,
    0.012, 0.011, 0.011, 0.013, 0.015, 0.009, 0.007, 0.007, 0.007,
    0.012, 0.009, 0.008, 0.016, 0.010, 0.017, 0.011, 0.009, 0.011,
    0.009, 0.007, 0.013, 0.008, 0.011, 0.012, 0.010, 0.034, 0.008,
    0.008, 0.009, 0.008, 0.008, 0.007, 0.010, 0.008, 0.009, 0.009,
    0.009, 0.007, 0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.010,
    0.008, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
    0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032,
    0.020, 0.019, 0.022, 0.031, 0.029, 0.022, 0.035, 0.037, 0.047,
    0.026, 0.025, 0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017,
    0.021, 0.021, 0.032, 0.020, 0.019, 0.022, 0.031,
], dtype=np.float32)
CUTS = np.cumsum([0, 17, 6, 68, 21, 21])


def load_predictions(path: Path, backend: str) -> dict[int, list[dict]]:
    """读取 JSONL,按 detection_id 去重并按 image_id 分组."""
    predictions = {}
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("backend") != backend:
                raise ValueError(f"第 {line_number} 行姿态预测来源不符,要求 {backend}.")
            detection_id = int(item["detection_id"])
            keypoints = np.asarray(item["keypoints"], dtype=np.float32)
            if keypoints.shape != (133 * 3,):
                raise ValueError(
                    f"第 {line_number} 行关键点数量异常: {keypoints.shape}")
            item["keypoints"] = keypoints.reshape(133, 3)
            predictions[detection_id] = item
            if line_number % 10000 == 0:
                print(f"[LOAD] 已读取 {line_number} 行板端预测.")
    grouped = defaultdict(list)
    for detection_id in sorted(predictions):
        item = predictions[detection_id]
        scores = item["keypoints"][:, 2]
        valid_scores = scores[scores > 0.2]
        mean_score = float(valid_scores.mean()) if valid_scores.size else 0.0
        item["score"] = float(item["bbox_score"]) * mean_score
        grouped[int(item["image_id"])].append(item)
    return grouped


def oks_iou(reference: dict, candidates: list[dict]) -> np.ndarray:
    """计算一个人体实例与其余实例的 WholeBody OKS."""
    reference_kpts = reference["keypoints"]
    variances = (SIGMAS * 2.0) ** 2
    values = np.zeros(len(candidates), dtype=np.float32)
    for index, candidate in enumerate(candidates):
        candidate_kpts = candidate["keypoints"]
        delta_x = candidate_kpts[:, 0] - reference_kpts[:, 0]
        delta_y = candidate_kpts[:, 1] - reference_kpts[:, 1]
        average_area = (float(reference["area"]) +
                        float(candidate["area"])) * 0.5
        error = ((delta_x**2 + delta_y**2) / variances /
                 (average_area + np.spacing(1)) / 2.0)
        values[index] = np.exp(-error).mean()
    return values


def oks_nms(instances: list[dict], threshold: float = 0.9) -> list[dict]:
    """执行与 MMPose 默认配置一致的贪心 OKS-NMS."""
    if not instances:
        return []
    order = np.argsort([item["score"] for item in instances])[::-1]
    keep = []
    while order.size:
        selected = int(order[0])
        keep.append(selected)
        if order.size == 1:
            break
        remaining = order[1:]
        overlaps = oks_iou(
            instances[selected], [instances[int(index)] for index in remaining])
        order = remaining[overlaps <= threshold]
    return [instances[index] for index in keep]


def format_results(grouped: dict[int, list[dict]]) -> list[dict]:
    """完成重评分和 NMS,转换为 xtcocotools WholeBody 格式."""
    results = []
    image_ids = sorted(grouped)
    for index, image_id in enumerate(image_ids, start=1):
        for item in oks_nms(grouped[image_id]):
            flattened = item["keypoints"].reshape(-1)
            result = {
                "image_id": image_id,
                "category_id": 1,
                "keypoints": flattened[CUTS[0] * 3:CUTS[1] * 3].tolist(),
                "foot_kpts": flattened[CUTS[1] * 3:CUTS[2] * 3].tolist(),
                "face_kpts": flattened[CUTS[2] * 3:CUTS[3] * 3].tolist(),
                "lefthand_kpts": flattened[
                    CUTS[3] * 3:CUTS[4] * 3].tolist(),
                "righthand_kpts": flattened[
                    CUTS[4] * 3:CUTS[5] * 3].tolist(),
                "score": float(item["score"]),
            }
            results.append(result)
        if index % 500 == 0 or index == len(image_ids):
            print(f"[NMS] {index}/{len(image_ids)} 张图片.")
    return results


def evaluate_component(coco_gt: COCO, coco_dt: COCO, name: str,
                       iou_type: str, sigmas: np.ndarray, image_ids=None) -> dict[str, float]:
    """执行 WholeBody COCOeval 并返回核心 AP."""
    print(f"[EVAL] {name}: {iou_type}")
    evaluator = COCOeval(coco_gt, coco_dt, iou_type, sigmas, use_area=True)
    evaluator.params.useSegm = None
    evaluator.params.imgIds = sorted(coco_gt.getImgIds() if image_ids is None else image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {"AP": float(evaluator.stats[0])}


def evaluate(args: argparse.Namespace, backend: str = "python_neuron_runtime_hw") -> None:
    """核对调用方的推理来源,按相同协议计算 WholeBody AP."""
    if backend not in ("python_neuron_runtime_hw", "onnxruntime"):
        raise ValueError(f"不支持的姿态预测来源: {backend}.")
    if SIGMAS.shape != (133,):
        raise RuntimeError(f"WholeBody sigma 数量错误: {SIGMAS.shape}")
    grouped = load_predictions(args.predictions, backend)
    image_ids = None
    if is_test():
        path = args.annotations.parent.parent / "person_detection_results/COCO_val2017_detections_AP_H_56_person.json"
        detections = json.loads(path.read_text(encoding="utf-8"))
        image_ids = pose_image_ids(detections)
        selected = set(image_ids)
        expected = {index for index, row in enumerate(detections) if row["image_id"] in selected}
        actual = {int(row["detection_id"]) for rows in grouped.values() for row in rows}
        if set(grouped) != set(image_ids) or actual != expected:
            raise ValueError("姿态子集必须覆盖选定整图的全部人体框.")
    results = format_results(grouped)
    args.formatted.parent.mkdir(parents=True, exist_ok=True)
    args.formatted.write_text(
        json.dumps(results, ensure_ascii=False), encoding="utf-8")

    coco_gt = COCO(str(args.annotations))
    coco_dt = coco_gt.loadRes(str(args.formatted))
    components = (("wholebody", "keypoints_wholebody", SIGMAS),)
    metrics = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for name, iou_type, sigmas in components:
            metrics[name] = evaluate_component(
                coco_gt, coco_dt, name, iou_type, sigmas, image_ids)
    args.metrics.write_text(
        json.dumps({
            "evaluation": accuracy_result("rtmpose_body2d",
                "npu" if backend == "python_neuron_runtime_hw" else "onnx",
                {"WholeBody AP": metrics["wholebody"]["AP"]},
                sum(map(len, grouped.values())), "person_instance"),
            "status": result_status(), **scope_metadata(),
            "protocol": {
                "score_mode": "bbox_keypoint",
                "keypoint_score_threshold": 0.2,
                "nms_mode": "oks_nms",
                "nms_threshold": 0.9,
                "detections_before_nms": sum(map(len, grouped.values())),
                "detections_after_nms": len(results),
                "evaluated_images": len(grouped),
            },
            "metrics": metrics,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] WholeBody AP 已计算,原始预测和精度报告已保留.")
    print(f"[OK] WholeBody 指标: {args.metrics}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--formatted", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--run-id")
    configure_parser(parser, __file__, {
        "annotations": lambda c: c.dataset / "annotations/coco_wholebody_val_v1.0.json",
        "predictions": lambda c: c.work / "predictions.jsonl",
        "formatted": lambda c: c.work / "wholebody_predictions.json",
        "metrics": lambda c: c.work / "coco_wholebody_metrics.json",
    })
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    evaluate(parse_args())
