"""一次性整理官方 NYU 数据,模型仅使用 RGB PNG 和米制 FP32 NPY."""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.io import loadmat


def main():
    """保留官方划分和原始数值,逐帧核对后归档下载文件."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    root = parser.parse_args().dataset_dir.resolve(strict=True)
    original = root / "original"
    if original.is_symlink():
        raise ValueError("original 必须是数据集内的实际目录,不能指向其他位置.")
    sources = {}
    for name in ("nyu_depth_v2_labeled.mat", "splits.mat"):
        candidates = [p for p in (root / name, original / name) if p.is_file()]
        if len(candidates) != 1:
            raise ValueError(f"原始文件必须有唯一来源,不能覆盖冲突文件: {name}")
        sources[name] = candidates[0]
    split = loadmat(sources["splits.mat"])
    train = np.asarray(split["trainNdxs"]).reshape(-1).astype(int)
    test = np.asarray(split["testNdxs"]).reshape(-1).astype(int)
    if (len(train) != 795 or len(test) != 654 or set(train) & set(test)
            or set(train) | set(test) != set(range(1, 1450))):
        raise ValueError("划分不符合官方互斥的 795/654 样本.")
    with h5py.File(sources["nyu_depth_v2_labeled.mat"], "r") as dataset:
        if (dataset["images"].shape != (1449, 3, 640, 480)
                or dataset["depths"].shape != (1449, 640, 480)
                or dataset["images"].dtype != np.uint8
                or dataset["depths"].dtype != np.float32):
            raise ValueError("官方 MAT 中的 RGB 或深度尺寸不符.")
        for name, indices in (("train", train), ("test", test)):
            rgb, depths = root / name / "rgb", root / name / "depth"
            rgb.mkdir(parents=True, exist_ok=True)
            depths.mkdir(parents=True, exist_ok=True)
            for step, index in enumerate(indices, 1):
                image = np.asarray(dataset["images"][index - 1]).transpose(2, 1, 0)
                depth = np.asarray(dataset["depths"][index - 1], dtype=np.float32).T
                image_path, depth_path = rgb / f"{index:04d}.png", depths / f"{index:04d}.npy"
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if not image_path.exists() and not cv2.imwrite(str(image_path), bgr):
                    raise OSError(f"RGB 保存失败: {image_path.name}")
                if not depth_path.exists():
                    np.save(depth_path, depth)
                restored = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                saved_depth = np.load(depth_path, allow_pickle=False)
                if (not np.array_equal(restored, bgr) or saved_depth.dtype != np.float32
                        or not np.array_equal(saved_depth, depth, equal_nan=True)):
                    raise ValueError(f"现有文件与原始数据不一致,停止以避免覆盖: {index:04d}")
                if step % 25 == 0 or step == len(indices):
                    print(f"[数据整理] {name}: {step}/{len(indices)},RGB/深度均已无损核对.", flush=True)
            index_path = root / name / "indices.txt"
            if index_path.exists():
                if not np.array_equal(np.loadtxt(index_path, dtype=int, ndmin=1), indices):
                    raise ValueError("已有样本索引与官方顺序不一致.")
            else:
                np.savetxt(index_path, indices, fmt="%d")
    original.mkdir(exist_ok=True)
    for name, source in sources.items():
        destination = original / name
        if source != destination:
            if destination.exists():
                raise FileExistsError(f"归档目标已存在: {name}")
            source.rename(destination)
    (original / "source_url.txt").write_text(
        "Dataset: NYU Depth V2 labeled subset\nRGB: PNG uint8\nDepth: NPY float32, meters, no rescaling\n"
        "Split: official train 795 / test 654; filenames retain original one-based IDs\n"
        "Homepage: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html\n"
        "Labeled data: https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat\n"
        "Split: http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat\n", encoding="utf-8")
    print("[完成] 795 张 train、654 张 test;原始 MAT 仅存 original/,模型不再读取 MAT.", flush=True)


if __name__ == "__main__":
    main()
