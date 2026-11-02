"""使用统一协议评测原始 PyTorch 或自导出 ONNX 的 COCO 精度."""


import os
import argparse
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from export_model import load_model, raw_forward

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from yolov8_utils import COCO_IDS, OUTPUT_NAMES, postprocess, preprocess

from tools.evaluation import coco_ids, is_test, result_status, scope_metadata
from tools.deploy import configure_parser, reference_path
from tools.output_metrics import record_reference, finish_reference
from tools.evaluation import accuracy_result


def evaluate(args):
    """直接评测指定后端,保留原始预处理、解码和 COCO 指标协议."""
    if bool(args.onnx) == bool(args.weights):
        raise ValueError("必须且只能选择 ONNX 或 PyTorch 权重.")
    torch.set_num_threads(2)
    annotation = COCO(str(args.dataset_root / "annotations/instances_val2017.json"))
    ids = sorted(annotation.getImgIds())
    if len(ids) != 5000:
        raise ValueError("正式评测需要完整 COCO val2017 5000 张标注.")
    ids = coco_ids(ids)
    if args.onnx:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(args.onnx), options,
                                       providers=["CPUExecutionProvider"])
    else:
        model = load_model(args.weights).cuda()
        model.model[-1].forward = types.MethodType(raw_forward, model.model[-1])
    predictions = []
    for image_id in tqdm(ids, desc="ONNX 精度" if args.onnx else "PyTorch 精度"):
        record = annotation.loadImgs([image_id])[0]
        tensor, geometry = preprocess(cv2.imread(str(
            args.dataset_root / "images" / record["file_name"])))
        if args.onnx:
            outputs = session.run(OUTPUT_NAMES, {"images": tensor})
            record_reference(args.onnx, [tensor], dict(zip(OUTPUT_NAMES, outputs)),
                sample_id=record["file_name"],
                sample_files=[args.dataset_root / "images" / record["file_name"]])
        else:
            with torch.no_grad():
                outputs = [value.cpu().numpy() for value in
                           model(torch.from_numpy(tensor).cuda())]
        boxes, scores, classes = postprocess(outputs, geometry)
        for box, score, category in zip(boxes, scores, classes):
            predictions.append({"image_id": image_id,
                                "category_id": COCO_IDS[int(category)],
                                "bbox": [float(box[0]), float(box[1]),
                                         float(box[2] - box[0]), float(box[3] - box[1])],
                                "score": float(score)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output.with_name("predictions.json")
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    if not predictions:
        raise RuntimeError("全量预测为空,请检查模型输出.")
    evaluator = COCOeval(annotation, annotation.loadRes(str(predictions_path)), "bbox")
    evaluator.params.imgIds = ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    summary = {"status": result_status(), **scope_metadata(), "model": "yolov8n", "backend": "onnx" if args.onnx else "pytorch",
               "samples": len(ids), "map_50_95": float(evaluator.stats[0]),
               "map_50": float(evaluator.stats[1]),
               "protocol": {"confidence": 0.001, "nms_iou": 0.6,
                            "max_det": 300, "class_selection": "single_best",
                            "input": [1, 3, 640, 640]}}
    summary["evaluation"] = accuracy_result("yolov8n", "onnx" if args.onnx else "pytorch",
        {"mAP@0.5:0.95": summary["map_50_95"]}, len(ids), "image")
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"[OK] 精度汇总: {args.output}", flush=True)
    if args.onnx:
        finish_reference()



def main():
    """无后端参数时依次评测 PyTorch 和 ONNX,显式参数支持单后端."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    ctx = configure_parser(parser, __file__, {
        "dataset_root": lambda c: c.dataset,
        "output": reference_path,
    })
    args = parser.parse_args()
    if args.onnx or args.weights:
        evaluate(args)
        return
    for backend in ("pytorch", "onnx"):
        current = argparse.Namespace(**vars(args))
        current.weights = Path(ctx.config["model"]["weights"]) if backend == "pytorch" else None
        current.onnx = ctx.output / "yolov8n.onnx" if backend == "onnx" else None
        if backend == "pytorch":
            current.output = ctx.work / ("pytorch_reference_test.json" if is_test()
                                         else "pytorch_reference.json")
        print(f"[浮点评测] {backend}", flush=True)
        evaluate(current)


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    main()
