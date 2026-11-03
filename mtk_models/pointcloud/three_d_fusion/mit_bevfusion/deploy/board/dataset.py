"""读取已索引的 nuScenes 帧,复用官方时间差和坐标变换协议."""

from pathlib import Path

import numpy as np

from geometry import preprocess_image


CHANNELS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK",
            "CAM_BACK_LEFT", "CAM_BACK_RIGHT")


def transform(record):
    """将 nuScenes wxyz 四元数和位移转换为齐次矩阵."""
    w, x, y, z = np.asarray(record["rotation"], dtype=np.float64)
    rotation = np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                         [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                         [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, record["translation"]
    return result


def load_points(root, frame):
    """当前帧加最多 9 个历史 sweep,仅无历史时按官方规则补齐."""
    current = np.fromfile(Path(root) / frame["lidar"], dtype=np.float32).reshape(-1, 5)
    current[:, 4] = 0
    pieces = [current]
    if not frame["sweeps"]:
        distant = (np.abs(current[:, 0]) >= 1) | (np.abs(current[:, 1]) >= 1)
        pieces.extend([current[distant]] * 9)
    for sweep in frame["sweeps"]:
        points = np.fromfile(Path(root) / sweep["path"], dtype=np.float32).reshape(-1, 5)
        points = points[(np.abs(points[:, 0]) >= 1) | (np.abs(points[:, 1]) >= 1)]
        matrix = np.asarray(sweep["to_lidar"], dtype=np.float32)
        points[:, :3] = points[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
        points[:, 4] = sweep["time_lag"]
        pieces.append(points)
    points = np.concatenate(pieces)
    valid = ((points[:, :3] > [-54, -54, -5]) & (points[:, :3] < [54, 54, 3])).all(axis=1)
    return points[valid]


def load_frame(root, frame):
    """从相对文件路径装载一帧六相机及点云,失败时不跳过样本."""
    images, augmentations = [], []
    for camera in frame["cameras"]:
        image, augmentation = preprocess_image(Path(root) / camera["path"])
        images.append(image[0])
        augmentations.append(augmentation)
    return {"points": load_points(root, frame), "images": np.stack(images),
            "augmentations": np.stack(augmentations),
            "camera_to_lidar": np.asarray([camera["to_lidar"] for camera in frame["cameras"]], np.float32),
            "lidar_to_image": np.asarray([camera["lidar_to_image"] for camera in frame["cameras"]], np.float32),
            "intrinsics": np.asarray([camera["intrinsic"] for camera in frame["cameras"]], np.float32)}
