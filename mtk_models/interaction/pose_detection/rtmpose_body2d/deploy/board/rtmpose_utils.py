"""RTMPose 输入几何变换与 COCO 人体框读取工具."""

import json
from pathlib import Path

import cv2
import numpy as np


INPUT_WIDTH = 192
INPUT_HEIGHT = 256
INPUT_ASPECT_RATIO = INPUT_WIDTH / INPUT_HEIGHT
SCALE_PADDING = 1.25


def compute_center_scale(
        bbox: tuple[float, float, float, float]) -> tuple[np.ndarray,
                                                          np.ndarray]:
    """将 xywh 人体框调整为 RTMPose 输入比例并增加标准边距."""
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        raise ValueError(f"人体框宽高必须大于 0: {bbox}")
    center = np.array([x + width * 0.5, y + height * 0.5],
                      dtype=np.float32)
    if width > height * INPUT_ASPECT_RATIO:
        height = width / INPUT_ASPECT_RATIO
    else:
        width = height * INPUT_ASPECT_RATIO
    scale = np.array([width * SCALE_PADDING, height * SCALE_PADDING],
                     dtype=np.float32)
    return center, scale


def build_affine_transform(center: np.ndarray,
                           scale: np.ndarray) -> np.ndarray:
    """构造原图到 192x256 网络输入的仿射变换矩阵."""
    half_width = scale[0] * 0.5
    half_height = scale[1] * 0.5
    source = np.array([
        [center[0] - half_width, center[1] - half_height],
        [center[0] + half_width, center[1] - half_height],
        [center[0] - half_width, center[1] + half_height],
    ], dtype=np.float32)
    target = np.array([
        [0.0, 0.0],
        [float(INPUT_WIDTH), 0.0],
        [0.0, float(INPUT_HEIGHT)],
    ], dtype=np.float32)
    return cv2.getAffineTransform(source, target)


def preprocess_image(
        image: np.ndarray,
        bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, dict]:
    """裁剪人体框并返回 NCHW RGB FP32 [0,255] 输入和映射元数据."""
    center, scale = compute_center_scale(bbox)
    transform = build_affine_transform(center, scale)
    crop = cv2.warpAffine(
        image,
        transform,
        (INPUT_WIDTH, INPUT_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0))
    # OpenCV 解码结果为 BGR,官方 MMPose 导出模型输入约定为 RGB.
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    nchw = rgb_crop.transpose(2, 0, 1).astype(np.float32)
    metadata = {
        "bbox_xywh": [float(value) for value in bbox],
        "center": center.tolist(),
        "scale": scale.tolist(),
        "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
        "affine_transform": transform.tolist(),
    }
    return nchw[np.newaxis].copy(), metadata


def load_person_samples(annotations_path: Path) -> list[dict]:
    """读取 COCO 标注并返回按 annotation id 排序的有效 person 样本."""
    dataset = json.loads(annotations_path.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in dataset["images"]}
    person_category_ids = {
        int(item["id"]) for item in dataset["categories"]
        if item["name"] == "person"
    }
    samples = []
    for annotation in dataset["annotations"]:
        bbox = annotation.get("bbox", [])
        image = images.get(int(annotation["image_id"]))
        if (int(annotation["category_id"]) not in person_category_ids or
                image is None or len(bbox) != 4 or float(bbox[2]) <= 1 or
                float(bbox[3]) <= 1 or annotation.get("iscrowd", 0)):
            continue
        samples.append({
            "annotation_id": int(annotation["id"]),
            "image_id": int(annotation["image_id"]),
            "file_name": image["file_name"],
            "image_width": int(image["width"]),
            "image_height": int(image["height"]),
            "bbox": tuple(float(value) for value in bbox),
            "area": float(annotation.get("area", bbox[2] * bbox[3])),
        })
    return sorted(samples, key=lambda item: item["annotation_id"])


def decode_prediction(pred_x: np.ndarray, pred_y: np.ndarray,
                      metadata: dict) -> tuple[list[float], float]:
    """按板端相同的 SimCC 规则解码并映射回原图.

    Args:
        pred_x: X 轴 SimCC 输出.
        pred_y: Y 轴 SimCC 输出.
        metadata: 仿射裁剪元数据.

    Returns:
        展平的 133 点结果和裁剪框面积.
    """
    if pred_x.shape != (1, 133, 384):
        raise ValueError(f"pred_x 形状错误: {pred_x.shape}")
    if pred_y.shape != (1, 133, 512):
        raise ValueError(f"pred_y 形状错误: {pred_y.shape}")
    x_indices = pred_x[0].argmax(axis=1)
    y_indices = pred_y[0].argmax(axis=1)
    x_scores = pred_x[0, np.arange(133), x_indices]
    y_scores = pred_y[0, np.arange(133), y_indices]
    scores = np.minimum(x_scores, y_scores)
    center = np.asarray(metadata["center"], dtype=np.float32)
    scale = np.asarray(metadata["scale"], dtype=np.float32)
    source_x = (x_indices / 2.0 / 192.0 * scale[0] +
                center[0] - scale[0] * 0.5)
    source_y = (y_indices / 2.0 / 256.0 * scale[1] +
                center[1] - scale[1] * 0.5)
    source_x = np.where(scores > 0.0, source_x, -1.0)
    source_y = np.where(scores > 0.0, source_y, -1.0)
    keypoints = np.stack((source_x, source_y, scores), axis=1)
    return keypoints.astype(np.float32).reshape(-1).tolist(), float(np.prod(scale))
