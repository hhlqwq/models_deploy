"""室内米制深度的输入、数据集和评分契约,不依赖平台 Runtime."""

import json
import hashlib
from pathlib import Path

import cv2
import numpy as np


def fingerprint(path):
    """计算产物身份,供同一部署包的参考结果和板端报告绑定."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess(image, shape):
    """将 RGB uint8 图像转成固定尺寸的 ImageNet 归一化 NCHW 张量."""
    height, width = shape[-2:]
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    values = resized.astype(np.float32) / 255.0
    values = (values - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32)
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None])


def restore_depth(depth, height, width):
    """按官方 align_corners=True 的双线性坐标将深度恢复到原图."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"深度必须为二维数组: {depth.shape}")
    y = np.linspace(0, depth.shape[0] - 1, height, dtype=np.float32)
    x = np.linspace(0, depth.shape[1] - 1, width, dtype=np.float32)
    y0, x0 = y.astype(np.int64), x.astype(np.int64)
    y1 = np.minimum(y0 + 1, depth.shape[0] - 1)
    x1 = np.minimum(x0 + 1, depth.shape[1] - 1)
    wy, wx = (y - y0)[:, None], (x - x0)[None, :]
    top = depth[y0[:, None], x0] * (1 - wx) + depth[y0[:, None], x1] * wx
    bottom = depth[y1[:, None], x0] * (1 - wx) + depth[y1[:, None], x1] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.float32)


class NYUDataset:
    """直接读取官方 MAT 数据和标准划分,不生成重复的图片与深度副本."""

    def __init__(self, root, config):
        """核对官方样本数量和训练测试互斥关系,按需读取单帧."""
        import h5py
        from scipy.io import loadmat

        root = Path(root)
        splits = loadmat(root / config["split_file"])
        self.train = np.asarray(splits["trainNdxs"]).reshape(-1).astype(int) - 1
        self.test = np.asarray(splits["testNdxs"]).reshape(-1).astype(int) - 1
        if (len(self.train) != 795 or len(self.test) != 654
                or len(set(self.train) | set(self.test)) != 1449
                or set(self.train) & set(self.test)
                or min(*self.train, *self.test) != 0
                or max(*self.train, *self.test) != 1448):
            raise ValueError("NYU 划分必须为互斥的标准 795/654 样本.")
        self.file = h5py.File(root / config["labeled_file"], "r")
        if self.file["images"].shape != (1449, 3, 640, 480):
            self.file.close()
            raise ValueError("NYU MAT 的 images 形状与官方 HDF5 格式不符.")

    def read(self, index):
        """按原始零基索引读取 RGB 和以米为单位的填洞深度."""
        image = np.asarray(self.file["images"][index]).transpose(2, 1, 0)
        depth = np.asarray(self.file["depths"][index], dtype=np.float32).T
        if image.dtype != np.uint8 or depth.shape != image.shape[:2]:
            raise ValueError("NYU 图像格式或深度尺寸不匹配.")
        return image, depth

    def close(self):
        """关闭 HDF5 资源句柄."""
        self.file.close()


def score_depth(prediction, reference, config):
    """按固定裁剪和有效深度计算指标,不进行逐图尺度对齐."""
    top, bottom, left, right = config["crop"]
    truth = reference[top:bottom, left:right]
    pred = prediction[top:bottom, left:right]
    if not np.isfinite(pred).all():
        raise ValueError("预测包含 NaN 或 Inf,停止评分.")
    valid = (np.isfinite(truth) & (truth > config["min_depth"])
             & (truth < config["max_depth"]))
    if not valid.any():
        raise ValueError("当前帧没有有效深度像素.")
    truth = truth[valid].astype(np.float64)
    pred = np.clip(pred[valid], config["min_depth"], config["max_depth"])
    ratio = np.maximum(truth / pred, pred / truth)
    return {
        "abs_rel": float(np.mean(np.abs(truth - pred) / truth)),
        "rmse_m": float(np.sqrt(np.mean((truth - pred) ** 2))),
        "log10": float(np.mean(np.abs(np.log10(truth) - np.log10(pred)))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25 ** 2)),
        "delta3": float(np.mean(ratio < 1.25 ** 3)),
    }


def save_json(path, data):
    """写入唯一目标 JSON,禁止将非有限数值发布为有效结果."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")
