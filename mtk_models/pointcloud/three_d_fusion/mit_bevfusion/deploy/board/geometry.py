"""相机投影,BEV pooling 和 TransFusion 候选及边界框解码."""

import numpy as np
from PIL import Image


CLASSES = ("car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
           "motorcycle", "bicycle", "pedestrian", "traffic_cone")


def preprocess_image(path):
    """复用官方验证时 0.48 缩放,底部中心裁剪及 RGB 标准化."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        resized_width, resized_height = int(width * 0.48), int(height * 0.48)
        left, top = int(max(0, resized_width - 704) / 2), resized_height - 256
        # PIL 默认重采样与官方 img.resize(resize_dims) 相同.
        image = image.resize((resized_width, resized_height)).crop((left, top, left + 704, top + 256))
        values = np.asarray(image, dtype=np.float32) / 255
    values = (values - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    augmentation = np.eye(4, dtype=np.float32)
    augmentation[0, 0] = augmentation[1, 1] = 0.48
    augmentation[:2, 3] = [-left, -top]
    return values[None].astype(np.float32), augmentation


def projected_depth(points, lidar_to_image, augmentation):
    """按官方点序写入深度,不擅自替换为每像素最近点规则."""
    projected = points[:, :3] @ lidar_to_image[:3, :3].T + lidar_to_image[:3, 3]
    distances = projected[:, 2].copy()
    projected[:, 2] = np.clip(projected[:, 2], 1e-5, 1e5)
    projected[:, :2] /= projected[:, 2:3]
    projected = projected @ augmentation[:3, :3].T + augmentation[:3, 3]
    valid = ((projected[:, 0] >= 0) & (projected[:, 0] < 704)
             & (projected[:, 1] >= 0) & (projected[:, 1] < 256))
    pixels = projected[valid, :2].astype(np.int64)
    # 重复像素显式保留最后一个输入,避免不同 NumPy 后端的重复写入差异.
    keys = pixels[:, 1] * 704 + pixels[:, 0]
    _, reverse_indices = np.unique(keys[::-1], return_index=True)
    indices = len(keys) - 1 - reverse_indices
    depth = np.zeros((1, 256, 704, 1), dtype=np.float32)
    depth[0, pixels[indices, 1], pixels[indices, 0], 0] = distances[valid][indices]
    return depth


def pool_bev(probability, features, camera_to_lidar, intrinsic, augmentation, geometry):
    """按深度分片累加真实视锥特征,避免展开完整六相机五维张量."""
    probability = probability[0].transpose(2, 0, 1)
    features = features[0].reshape(-1, 80)
    frustum = geometry["frustum"]
    output = np.zeros((360 * 360, 80), dtype=np.float32)
    transform = camera_to_lidar[:3, :3] @ np.linalg.inv(intrinsic)
    for start in range(0, len(frustum), 8):
        positions = frustum[start:start + 8].reshape(-1, 3).copy()
        positions = (positions - augmentation[:3, 3]) @ np.linalg.inv(augmentation[:3, :3]).T
        positions[:, :2] *= positions[:, 2:3]
        positions = positions @ transform.T + camera_to_lidar[:3, 3]
        # 官方 Tensor.long 为向零截断,不能用 floor 改变边界附近的体素.
        indices = ((positions - (geometry["bx"] - geometry["dx"] / 2)) / geometry["dx"]).astype(np.int64)
        valid = ((indices >= 0) & (indices < geometry["nx"])).all(axis=1)
        values = (probability[start:start + 8, :, :, None]
                  * features.reshape(1, 32, 88, 80)).reshape(-1, 80)
        indices = indices[valid]
        np.add.at(output, indices[:, 0] * 360 + indices[:, 1], values[valid])
    return output.reshape(1, 360, 360, 80)


def select_queries(features, logits):
    """官方类别内 3x3 局部峰值抑制后选取全类别前 200 个查询."""
    heatmap = 1 / (1 + np.exp(-np.clip(logits[0], -80, 80)))
    maxima = np.zeros_like(heatmap)
    windows = np.lib.stride_tricks.sliding_window_view(heatmap, (3, 3), axis=(0, 1))
    maxima[1:-1, 1:-1] = windows.max(axis=(-1, -2))
    maxima[:, :, 8:] = heatmap[:, :, 8:]
    heatmap *= heatmap == maxima
    scores = heatmap.transpose(2, 0, 1).reshape(-1)
    # 完整排序保留明确的并列规则,不使用会改变并列候选的 argpartition.
    proposals = np.argsort(-scores, kind="stable")[:200]
    labels, positions = proposals // 32400, proposals % 32400
    # 官方 create_2D_grid 依次堆叠 meshgrid(x,y),展平后使用 [x,y].
    x, y = np.meshgrid(np.arange(180), np.arange(180), indexing="ij")
    key_position = np.stack((x, y), -1).reshape(1, 32400, 2).astype(np.float32) + 0.5
    inputs = (features.reshape(1, 32400, 128)[:, positions], key_position[:, positions],
              np.eye(10, dtype=np.float32)[labels][None],
              features.reshape(1, 32400, 128), key_position)
    return inputs, labels, scores[proposals]


def decode(prediction, labels, query_scores):
    """解码 200 个候选,保留官方无 NMS 配置及中心范围筛选."""
    values = prediction[0]
    scores = 1 / (1 + np.exp(-np.clip(values[:, 10:], -80, 80)))
    scores *= np.eye(10, dtype=np.float32)[labels]
    scores *= query_scores[:, None]
    labels = scores.argmax(axis=1)
    scores = scores.max(axis=1)
    dimensions = np.exp(values[:, 3:6])
    centers = np.concatenate((values[:, :2] * 0.6 - 54,
                              values[:, 2:3] - dimensions[:, 2:3] / 2), axis=1)
    yaw = np.arctan2(values[:, 6], values[:, 7])[:, None]
    boxes = np.concatenate((centers, dimensions, yaw, values[:, 8:10]), axis=1)
    valid = ((centers >= [-61.2, -61.2, -10]) & (centers <= [61.2, 61.2, 10])).all(axis=1)
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("检测输出包含非有限数值.")
    return boxes[valid], scores[valid], labels[valid]
