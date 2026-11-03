"""在板端评测 FastSAM-s 的 COCO val2017 类别无关实例分割 AP."""


import argparse
from tools.board import configure_parser
import contextlib
import io
import json
import os
import statistics
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from tools.evaluation import coco_ids, is_test, result_status, scope_metadata
from tools.evaluation import accuracy_result


def parse_args() -> argparse.Namespace:
    """解析板端全量分割精度参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path,
                        default=Path(__file__).with_name("accuracy_protocol.json"))
    configure_parser(parser, __file__, {
        "images": lambda c: c.dataset / "images",
        "annotations": lambda c: c.dataset / "annotations/instances_val2017.json",
        "work_dir": lambda c: c.work,
    })
    return parser.parse_args()


def make_class_agnostic_ground_truth(annotation_path: Path) -> COCO:
    """将 COCO 80 类 GT 统一映射到一个实例类别,保持原始掩码与图片."""
    source = COCO(str(annotation_path))
    dataset = dict(source.dataset)
    dataset["categories"] = [{"id": 1, "name": "object", "supercategory": "object"}]
    dataset["annotations"] = [
        {**item, "category_id": 1}
        for item in source.dataset["annotations"]
    ]
    ground_truth = COCO()
    ground_truth.dataset = dataset
    ground_truth.createIndex()
    return ground_truth


def save_checkpoint(path: Path, record: dict) -> None:
    """原子保存单图结果,避免中断后读取到不完整的 JSON."""
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(record, output, ensure_ascii=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def evaluate(args: argparse.Namespace) -> None:
    """核对完整 Python 预测并计算 COCO segm AP."""
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    images = sorted(args.images.glob("*.jpg"))
    if len(images) != protocol["images"]:
        raise ValueError(f"COCO val2017 需要 5000 张图片,实际 {len(images)}.")
    ground_truth = make_class_agnostic_ground_truth(args.annotations)
    expected = set(ground_truth.getImgIds())
    if {int(item.stem) for item in images} != expected:
        raise ValueError("COCO 图片与实例分割标注 ID 不一致.")
    expected = set(coco_ids(expected))
    images = [image for image in images if int(image.stem) in expected]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    report_dir = args.work_dir / "report"
    report_dir.mkdir(exist_ok=True)
    checkpoint_dir = args.work_dir / "predictions_by_image"
    prediction_path = args.work_dir / "coco_segm_predictions.json"
    completed_path = args.work_dir / "processed_ids.txt"
    npu_times = []
    peak_rss_values = []
    end_to_end_times = []
    prediction_count = 0

    if not checkpoint_dir.is_dir():
        raise ValueError("缺少 Python 全量推理检查点目录.")
    with (prediction_path.open("w", encoding="utf-8") as predictions,
          completed_path.open("w", encoding="utf-8") as completed):
        predictions.write("[")
        for image in images:
            record = json.loads((checkpoint_dir / f"{image.stem}.json").read_text(
                encoding="utf-8"))
            if record.get("backend") != "python_neuron_runtime_hw":
                raise ValueError("必须使用本次 Python 推理检查点.")
            if record["image_id"] != int(image.stem):
                raise ValueError(f"检查点图片 ID 不一致: {image.stem}")
            if record.get("protocol") != protocol:
                raise ValueError(f"检查点评测协议不一致: {image.stem}")
            if not is_test():
                npu_times.append(record["npu_ms"])
            if record.get("peak_rss_kib") is not None:
                peak_rss_values.append(record["peak_rss_kib"])
            if not is_test():
                end_to_end_times.append(record["end_to_end_ms"])
            completed.write(f"{record['image_id']}\n")
            for prediction in record["predictions"]:
                if prediction_count:
                    predictions.write(",")
                predictions.write(json.dumps(prediction, ensure_ascii=False))
                prediction_count += 1
        predictions.write("]\n")

    detection = ground_truth.loadRes(str(prediction_path))
    evaluator = COCOeval(ground_truth, detection, "segm")
    evaluator.params.imgIds = sorted(expected)
    evaluator.params.catIds = [1]
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    summary = {
        "status": result_status(),
        **scope_metadata(),
        "model": "fastsam", "backend": "python_neuron_runtime_hw",
        "run_id": args.run_id,
        "dataset": "coco_val2017_instances",
        "metric_protocol": "class_agnostic_segm_ap_all_gt_categories_merged",
        "protocol": protocol,
        "images": len(images),
        "prediction_instances": prediction_count,
        "AP_50_95": float(evaluator.stats[0]),
        "AP_50": float(evaluator.stats[1]),
        "AP_75": float(evaluator.stats[2]),
        "AR_10": float(evaluator.stats[7]),
        "AR_100": float(evaluator.stats[8]),
        "npu_mean_ms": None if is_test() else statistics.fmean(npu_times),
        "peak_rss_kib": max(peak_rss_values) if len(peak_rss_values) == len(images) else None,
        "npu_p95_ms": None if is_test() else float(np.percentile(npu_times, 95)),
        "end_to_end_mean_ms": None if is_test() else statistics.fmean(end_to_end_times),
        "timing_scope": "persistent Python Runtime; NPU inference call reported separately",
    }
    summary["evaluation"] = accuracy_result("fastsam", "npu",
        {"segm AR@100": summary["AR_100"]}, len(images), "image")
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"[OK] FastSAM 类别无关分割 AP: {report_dir}.")


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    evaluate(parse_args())
