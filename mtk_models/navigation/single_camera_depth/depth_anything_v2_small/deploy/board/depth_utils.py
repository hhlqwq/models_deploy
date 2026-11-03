"""Depth Anything V2 Small 的固定输入预处理工具。"""

from pathlib import Path

import cv2
import numpy as np


INPUT_SIZE = 518
INPUT_ROW_STRIDE = ((INPUT_SIZE + 15) // 16) * 16
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path: Path) -> np.ndarray:
    """将 BGR 图片变为固定尺寸的 RGB NCHW FP32 输入。"""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE),
                       interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = (rgb - MEAN) / STD
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
