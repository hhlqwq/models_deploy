"""读取模型配置并直接执行 YOLOv5 ONNX 精度评测."""

import os
from argparse import Namespace

from tools.accuracy.yolov5s_val_coco import evaluate_onnx_accuracy
from tools.deploy import arguments, reference_path


def main():
    """保持原有 COCO 阈值、尺寸和样本范围,直接调用评测函数."""
    ctx = arguments(__file__)
    evaluate_onnx_accuracy(Namespace(
        onnx=ctx.output / "yolov5s.onnx", images_dir=ctx.dataset / "images",
        ann=ctx.dataset / "annotations/instances_val2017.json", summary=reference_path(ctx),
        image_size=640, confidence=0.001, iou=0.6, max_det=300))


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    main()
