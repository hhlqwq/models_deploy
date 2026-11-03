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
    """读取整理后的 RGB PNG 和米制 FP32 NPY,不依赖 MAT 或 HDF5."""

    def __init__(self, root, config, split="both"):
        """核对样本编号、顺序与文件配对,正式板端评价可只装测试集."""
        if split not in ("both", "test", "subset"):
            raise ValueError("仅支持 both、test 或 subset 范围.")
        self.root = Path(root)
        self.paths = {}
        self.train = self.indices("train", 795) if split == "both" else np.array([], dtype=int)
        if split == "subset":
            self.test = self.indices("test", config["compile_samples"], subset=True)
        else:
            self.test = self.indices("test", 654)
        if split == "both" and (set(self.train) & set(self.test)
                                or set(self.train) | set(self.test) != set(range(1449))):
            raise ValueError("NYU 必须保留标准互斥的 795/654 划分.")

    def indices(self, name, count, *, subset=False):
        """读取从官方划分导出的原始顺序,确认每个 RGB 都有对应深度."""
        folder = self.root / name
        index_file = folder / ("compile_indices.txt" if subset else "indices.txt")
        ids = np.loadtxt(index_file, dtype=int, ndmin=1)
        if len(ids) != count or len(set(ids)) != count or np.any((ids < 1) | (ids > 1449)):
            raise ValueError(f"{name} 样本编号或数量错误.")
        stems = {f"{index:04d}" for index in ids}
        rgb = {p.stem for p in (folder / "rgb").glob("*.png")}
        depth = {p.stem for p in (folder / "depth").glob("*.npy")}
        if (not stems <= rgb or not stems <= depth
                or (not subset and (rgb != stems or depth != stems))):
            raise ValueError(f"{name} 的 RGB 和深度文件没有完整配对.")
        for index in ids:
            self.paths[int(index) - 1] = (
                folder / "rgb" / f"{index:04d}.png", folder / "depth" / f"{index:04d}.npy")
        return ids - 1

    def read(self, index):
        """按原始零基索引读取 RGB 和以米为单位的填洞深度."""
        image_path, depth_path = self.paths[index]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"RGB 图片不可读: {image_path.name}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path, allow_pickle=False)
        if (image.dtype != np.uint8 or depth.dtype != np.float32
                or image.shape != (480, 640, 3) or depth.shape != image.shape[:2]):
            raise ValueError("NYU 图像格式或深度尺寸不匹配.")
        return image, depth

    def close(self):
        """释放路径索引,图片和深度均按次读取而不保留文件句柄."""
        self.paths.clear()

    def test_files(self):
        """列出正式测试实际需要的文件,不上传训练集或原始 MAT."""
        files = [Path("test/indices.txt")]
        for index in self.test:
            files.extend([Path(f"test/rgb/{index + 1:04d}.png"),
                          Path(f"test/depth/{index + 1:04d}.npy")])
        return files

    def identity(self):
        """将测试文件名和内容汇总成单个身份值,不发布逐文件哈希清单."""
        digest = hashlib.sha256()
        for relative in self.test_files():
            digest.update(relative.as_posix().encode() + b"\0")
            digest.update(fingerprint(self.root / relative).encode())
        return digest.hexdigest()


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


def sample_identity(dataset, indices):
    """绑定选定 RGB 和深度内容,两端以同一顺序核对同一子集."""
    digest = hashlib.sha256()
    for index in indices:
        for path in dataset.paths[int(index)]:
            digest.update(path.name.encode())
            digest.update(fingerprint(path).encode())
    return digest.hexdigest()
