"""保留 YOLOv5s 的固定 INT8 契约、原始检测头解码和类别内 NMS."""

import cv2
import numpy as np

from tools.boxes import box_iou


INPUT = {"shape": [1, 3, 640, 640], "scale": 0.003921568859368563, "zero_point": -128}
OUTPUTS = [{"shape": [1, 255, size, size], "scale": scale, "zero_point": zero}
           for size, scale, zero in ((80, 0.08984976261854172, 75),
                                     (40, 0.07972777634859085, 61),
                                     (20, 0.08616162091493607, 55))]
ANCHORS = np.array([[[10, 13], [16, 30], [33, 23]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[116, 90], [156, 198], [373, 326]]], dtype=np.float32)


def preprocess(image):
    """沿用居中 letterbox、RGB NCHW 与正数四舍五入缩放规则."""
    if image is None:
        raise ValueError("无法读取检测图片.")
    height, width = image.shape[:2]
    ratio = np.float32(min(640 / height, 640 / width))
    resized_width = int(np.floor(width * ratio + 0.5))
    resized_height = int(np.floor(height * ratio + 0.5))
    left, top = (640 - resized_width) // 2, (640 - resized_height) // 2
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    values = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    return values, {"ratio": ratio, "left": left, "top": top,
                    "original_shape": (height, width)}


def postprocess(outputs, geometry):
    """解码三个锚框头,按单最佳类别执行 IoU 0.6、最多 300 框 NMS."""
    boxes, probabilities = [], []
    for head, (raw, stride) in enumerate(zip(outputs, (8, 16, 32))):
        size = raw.shape[-1]
        values = 1 / (1 + np.exp(-np.clip(raw.reshape(3, 85, size, size), -80, 80)))
        y, x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        centers = (values[:, :2] * 2 + np.stack((x, y))[None] - 0.5) * stride
        lengths = (values[:, 2:4] * 2) ** 2 * ANCHORS[head, :, :, None, None]
        boxes.append(np.concatenate((centers - lengths / 2, centers + lengths / 2),
                                    axis=1).transpose(0, 2, 3, 1).reshape(-1, 4))
        probabilities.append((values[:, 4:5] * values[:, 5:]).transpose(
            0, 2, 3, 1).reshape(-1, 80))
    boxes, probabilities = np.concatenate(boxes), np.concatenate(probabilities)
    classes = probabilities.argmax(axis=1)
    scores = probabilities[np.arange(len(classes)), classes]
    candidates = np.flatnonzero(scores >= 0.001)
    order = candidates[np.argsort(-scores[candidates], kind="stable")]
    kept = []
    while order.size and len(kept) < 300:
        current = int(order[0])
        kept.append(current)
        rest = order[1:]
        order = rest[(classes[rest] != classes[current]) |
                     (box_iou(boxes[current], boxes[rest]) <= 0.6)]
    boxes = boxes[kept].copy()
    height, width = geometry["original_shape"]
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - geometry["left"]) / geometry["ratio"]).clip(0, width)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - geometry["top"]) / geometry["ratio"]).clip(0, height)
    return boxes, scores[kept], classes[kept]
