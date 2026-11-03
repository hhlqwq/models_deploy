"""FastSAM-s 的固定输入预处理、原始头解码与掩码后处理."""


import cv2
import numpy as np

from tools.boxes import box_iou


IMAGE_SIZE = 640
OUTPUT_NAMES = [
    f"{kind}_{stride}" for stride in (8, 16, 32)
    for kind in ("box", "score", "coefficient")
] + ["prototype"]
OUTPUT_SHAPES = [
    [1, channels, 640 // stride, 640 // stride]
    for stride in (8, 16, 32) for channels in (64, 1, 32)
] + [[1, 32, 160, 160]]


def preprocess(image):
    """将 BGR 图片居中填充为 640 RGB NCHW,返回逆变换参数."""
    if image is None or image.size == 0:
        raise ValueError("图片为空.")
    height, width = image.shape[:2]
    ratio = min(IMAGE_SIZE / height, IMAGE_SIZE / width)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    left = (IMAGE_SIZE - resized_width) // 2
    top = (IMAGE_SIZE - resized_height) // 2
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 114, np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    return tensor, {
        "original_shape": [height, width], "ratio": ratio,
        "left": left, "top": top,
        "resized_shape": [resized_height, resized_width],
    }


def decode_heads(outputs):
    """在 CPU 上执行 DFL 与 Sigmoid,返回 xyxy 框、分数和掩码系数."""
    if len(outputs) != 10:
        raise ValueError("FastSAM-s 必须有 10 个原始输出.")
    for name, values, shape in zip(OUTPUT_NAMES, outputs, OUTPUT_SHAPES):
        if list(values.shape) != shape or not np.isfinite(values).all():
            raise ValueError(f"输出形状或数值错误: {name}, {values.shape}.")
    boxes, scores, coefficients = [], [], []
    for level, stride in enumerate((8, 16, 32)):
        box, score, coefficient = outputs[level * 3:level * 3 + 3]
        logits = box.reshape(4, 16, -1)
        probability = np.exp(logits - logits.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
        distance = (probability * np.arange(16)[None, :, None]).sum(axis=1)
        grid_y, grid_x = np.meshgrid(
            np.arange(box.shape[2]), np.arange(box.shape[3]), indexing="ij")
        anchors = np.stack((grid_x.ravel(), grid_y.ravel())) + 0.5
        boxes.append(np.concatenate((anchors - distance[:2],
                                     anchors + distance[2:]), axis=0).T * stride)
        scores.append(1 / (1 + np.exp(-np.clip(score.ravel(), -80, 80))))
        coefficients.append(coefficient.reshape(32, -1).T)
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(coefficients)


def postprocess(outputs, geometry, confidence=0.4, iou=0.9, max_det=100,
                *, show_progress=True):
    """执行类别无关 NMS 并逐个还原掩码,全量评测可关闭掩码进度."""
    if not 0 <= confidence <= 1 or not 0 <= iou <= 1 or max_det <= 0:
        raise ValueError("阈值必须位于 [0,1], max_det 必须为正数.")
    boxes, scores, coefficients = decode_heads(outputs)
    candidates = np.flatnonzero(scores > confidence)
    order = candidates[np.argsort(-scores[candidates], kind="stable")][:30000]
    kept = []
    while order.size and len(kept) < max_det:
        current = int(order[0])
        kept.append(current)
        order = order[1:][box_iou(boxes[current], boxes[order[1:]]) <= iou]
    boxes, scores, coefficients = boxes[kept], scores[kept], coefficients[kept]
    # 对近似整图框进行官方 FastSAM 全图修正.
    if len(boxes):
        full = np.array([0, 0, IMAGE_SIZE, IMAGE_SIZE], dtype=np.float32)
        boxes[:, :2] = np.where(boxes[:, :2] < 20, 0, boxes[:, :2])
        boxes[:, 2:] = np.where(boxes[:, 2:] > IMAGE_SIZE - 20,
                                IMAGE_SIZE, boxes[:, 2:])
        boxes[box_iou(full, boxes) > 0.9] = full
    height, width = geometry["original_shape"]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - geometry["left"]) / geometry["ratio"]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - geometry["top"]) / geometry["ratio"]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    proto = outputs[-1][0]
    # 对齐官方 process_mask_native 的原型裁边和双线性插值次序.
    gain = min(proto.shape[1] / height, proto.shape[2] / width)
    pad_x = (proto.shape[2] - width * gain) / 2
    pad_y = (proto.shape[1] - height * gain) / 2
    left, top = int(pad_x), int(pad_y)
    right, bottom = int(proto.shape[2] - pad_x), int(proto.shape[1] - pad_y)
    masks = []
    for index, (coefficient, box) in enumerate(zip(coefficients, boxes)):
        logits = (coefficient @ proto.reshape(32, -1)).reshape(proto.shape[1:])
        probability = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        mask = cv2.resize(probability[top:bottom, left:right], (width, height)) > 0.5
        x1, y1, x2, y2 = box
        mask &= ((np.arange(width)[None, :] >= x1)
                 & (np.arange(width)[None, :] < x2)
                 & (np.arange(height)[:, None] >= y1)
                 & (np.arange(height)[:, None] < y2))
        masks.append(mask)
        if show_progress and (index + 1) % 25 == 0:
            print(f"[MASK] {index + 1}/{len(boxes)}", flush=True)
    return boxes, scores, masks


def select_masks(masks, point=None, box=None):
    """点提示保留覆盖该点的掩码,框提示选取与提示框 IoU 最大的掩码."""
    if point is not None and box is not None:
        raise ValueError("一次仅支持一种提示.")
    if not masks:
        return []
    height, width = masks[0].shape
    if point is not None:
        x, y = point
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError("提示点超出原图.")
        return [index for index, mask in enumerate(masks) if mask[y, x]]
    if box is not None:
        x1, y1, x2, y2 = box
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError("提示框必须是原图内有效的 xyxy 框.")
        area = (x2 - x1) * (y2 - y1)
        scores = []
        for mask in masks:
            intersection = int(mask[y1:y2, x1:x2].sum())
            scores.append(intersection / max(area + int(mask.sum()) - intersection, 1))
        return [int(np.argmax(scores))] if max(scores) > 0 else []
    return list(range(len(masks)))
