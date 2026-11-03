"""YOLOv8n 的固定输入、原始检测头解码和 COCO 后处理."""

import cv2
import numpy as np

from tools.boxes import box_iou

IMAGE_SIZE = 640
OUTPUT_NAMES = [f"{kind}_{stride}" for stride in (8, 16, 32)
                for kind in ("box", "score")]
OUTPUT_SHAPES = [[1, channels, 640 // stride, 640 // stride]
                 for stride in (8, 16, 32) for channels in (64, 80)]
COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
            74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def preprocess(image):
    """将 BGR 图片居中填充为 RGB NCHW,返回逆变换参数."""
    if image is None or image.size == 0:
        raise ValueError("图片为空.")
    height, width = image.shape[:2]
    ratio = min(640 / height, 640 / width)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    left, top = (640 - resized_width) // 2, (640 - resized_height) // 2
    canvas = np.full((640, 640, 3), 114, np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = cv2.resize(
        image, (resized_width, resized_height))
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    return tensor, {"original_shape": [height, width], "ratio": ratio,
                    "left": left, "top": top}


def decode_heads(outputs):
    """在 CPU 上执行 DFL 和 Sigmoid,返回 xyxy 框及全部类别分数."""
    if len(outputs) != 6:
        raise ValueError("YOLOv8n 必须包含六个原始输出.")
    for values, shape in zip(outputs, OUTPUT_SHAPES):
        if list(values.shape) != shape or not np.isfinite(values).all():
            raise ValueError(f"输出形状或数值无效: {values.shape}.")
    boxes, scores = [], []
    for level, stride in enumerate((8, 16, 32)):
        box, score = outputs[level * 2:level * 2 + 2]
        logits = box.reshape(4, 16, -1)
        probability = np.exp(logits - logits.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
        distance = (probability * np.arange(16)[None, :, None]).sum(axis=1)
        grid_y, grid_x = np.meshgrid(np.arange(box.shape[2]),
                                    np.arange(box.shape[3]), indexing="ij")
        anchors = np.stack((grid_x.ravel(), grid_y.ravel())) + 0.5
        boxes.append(np.concatenate((anchors - distance[:2],
                                     anchors + distance[2:]), axis=0).T * stride)
        scores.append((1 / (1 + np.exp(-np.clip(score.reshape(80, -1),
                                                -80, 80)))).T)
    return np.concatenate(boxes), np.concatenate(scores)


def postprocess(outputs, geometry, confidence=0.001, iou=0.6, max_det=300):
    """采用单最佳类别和类别内 NMS,与本项目 Python 评测协议保持一致."""
    boxes, probabilities = decode_heads(outputs)
    classes = probabilities.argmax(axis=1)
    scores = probabilities[np.arange(len(classes)), classes]
    candidates = np.flatnonzero(scores > confidence)
    order = candidates[np.argsort(-scores[candidates], kind="stable")][:30000]
    kept = []
    while order.size and len(kept) < max_det:
        current = int(order[0])
        kept.append(current)
        rest = order[1:]
        overlap = box_iou(boxes[current], boxes[rest])
        order = rest[(classes[rest] != classes[current]) | (overlap <= iou)]
    boxes = boxes[kept].copy()
    height, width = geometry["original_shape"]
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - geometry["left"]) / geometry["ratio"]).clip(0, width)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - geometry["top"]) / geometry["ratio"]).clip(0, height)
    return boxes, scores[kept], classes[kept]
