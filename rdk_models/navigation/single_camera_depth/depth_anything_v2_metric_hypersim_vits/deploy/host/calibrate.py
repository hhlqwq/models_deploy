"""从 NYU 标准训练划分准备室内量化校准张量."""

import sys

import numpy as np

from context import context


def main():
    """固定样本与预处理,将编译所需校准数据放日期临时目录."""
    ctx = context()
    sys.path.insert(0, str(ctx.model / "deploy"))
    from model import NYUDataset, preprocess

    dataset = NYUDataset(ctx.dataset, ctx.config["evaluation"])
    count = ctx.config["quantization"]["calibration_samples"]
    indices = np.random.default_rng(ctx.config["quantization"]["seed"]).choice(
        dataset.train, count, replace=False)
    destination = ctx.work / "calibration"
    destination.mkdir(parents=True, exist_ok=True)
    expected = {f"{int(index) + 1:04d}.bin" for index in indices}
    for old in destination.glob("*.bin"):
        if old.name not in expected:
            old.unlink()
    try:
        for step, index in enumerate(indices, 1):
            image, _ = dataset.read(int(index))
            tensor = preprocess(image, ctx.config["export"]["input_shape"])
            tensor.tofile(destination / f"{int(index) + 1:04d}.bin")
            if step % 10 == 0 or step == count:
                print(f"[校准准备] {step}/{count},仅使用 NYU train.", flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
