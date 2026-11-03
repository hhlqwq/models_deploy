"""主机与板端共享 ResNet101 V2 的预处理、标签及张量契约."""

import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image


MODEL = "resnet101"
WEIGHTS = "ResNet101_Weights.IMAGENET1K_V2"
TORCHVISION_VERSION = "0.15.1"
PROTOCOL = {
    "dataset": "imagenet_val2012",
    "samples": 50000,
    "resize_short_side": 232,
    "crop_size": 224,
    "interpolation": "Pillow bilinear",
    "color": "RGB",
    "layout": "NCHW",
    "external_range": [0.0, 1.0],
    "normalization": "in_graph",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "label_order": "torchvision_imagenet_1k_0based",
    "topk_ties": "stable_class_index_ascending",
}


def preprocess(path: Path) -> np.ndarray:
    """复现官方 PIL 缩放和中心裁剪,归一化已包含在导出图中.

    Args:
        path: 原始图片路径.

    Returns:
        连续存储的 `[1,3,224,224]` RGB FP32 `[0,1]` 张量.
    """
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    short, long = sorted((width, height))
    resized_long = int(232 * long / short)
    size = (232, resized_long) if width <= height else (resized_long, 232)
    image = image.resize(size, resample=Image.Resampling.BILINEAR)
    left = int(round((size[0] - 224) / 2.0))
    top = int(round((size[1] - 224) / 2.0))
    image = image.crop((left, top, left + 224, top + 224))
    values = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None])


def validation_images(dataset: Path) -> list[Path]:
    """要求 ImageNet 验证集为连续编号的 50000 张原始 JPEG."""
    images = dataset / "val"
    if not images.is_dir():
        images = dataset / "ILSVRC2012_img_val"
    paths = sorted(images.glob("*.JPEG"))
    if len(paths) != 50000 or any(
            path.name != f"ILSVRC2012_val_{index:08d}.JPEG"
            for index, path in enumerate(paths, 1)):
        raise ValueError("ImageNet val 必须包含连续编号的 50000 张 JPEG.")
    return paths


def read_labels(path: Path) -> list[int]:
    """读取按官方图片顺序排列的 TorchVision 零基标签,拒绝缺失或越界."""
    labels = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) not in (1, 2) or (
                len(fields) == 2 and int(fields[0]) != index):
            raise ValueError(f"标签格式或序号错误: 第 {index} 行.")
        value = int(fields[-1])
        if not 0 <= value < 1000:
            raise ValueError(f"标签类别越界: 第 {index} 行.")
        labels.append(value)
    if len(labels) != 50000:
        raise ValueError(f"ImageNet val 需要 50000 个标签,实际 {len(labels)}.")
    return labels


def label_signature(labels: list[int]) -> str:
    """只用标签内容匹配主机和板端参考,避免不同标签映射混用."""
    return hashlib.sha256(np.asarray(labels, dtype="<i4").tobytes()).hexdigest()


def validate_contract(metadata: dict) -> None:
    """要求固定官方权重、预处理和逐张量 INT8 I/O 契约."""
    if (metadata.get("model") != MODEL or metadata.get("weights") != WEIGHTS
            or metadata.get("torchvision") != TORCHVISION_VERSION
            or metadata.get("protocol") != PROTOCOL):
        raise ValueError("ResNet101 权重或预处理契约不匹配.")
    for name, shape in (("input", [1, 3, 224, 224]), ("output", [1, 1000])):
        item = metadata[name]
        if (item["shape"] != shape or item["dtype"] != "int8"
                or not math.isfinite(item["scale"]) or item["scale"] <= 0
                or type(item["zero_point"]) is not int
                or not -128 <= item["zero_point"] <= 127):
            raise ValueError(f"ResNet101 {name} 张量契约无效.")
    for field in ("onnx_sha256", "dla_sha256"):
        value = metadata[field]
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError(f"缺少有效的模型版本绑定: {field}.")
