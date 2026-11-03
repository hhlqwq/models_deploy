"""MobileFaceNet 的对齐人脸输入处理。"""

from pathlib import Path

import cv2
import numpy as np


def list_aligned_faces(image_dir: Path) -> list[Path]:
    """按文件名排序读取已对齐图片,不要求文件名带特定后缀标记."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"对齐人脸目录不存在: {image_dir}")
    extensions = {".jpg", ".jpeg", ".png"}
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions)


def load_aligned_face(path: Path) -> np.ndarray:
    """按锁定上游的验证协议生成 RGB NCHW 浮点输入。"""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    normalized = (normalized - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32)
    return np.transpose(normalized, (2, 0, 1))[None, ...].copy()
