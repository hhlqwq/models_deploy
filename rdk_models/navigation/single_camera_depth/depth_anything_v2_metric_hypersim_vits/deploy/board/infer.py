"""S100 少量通路检查与示例,不生成精度、性能或 RSS 报告."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np

from runtime import Runtime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import fingerprint, preprocess, restore_depth


def main():
    """只运行部署包内固定示例,按统一米制色阶保存输入和结果."""
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if fingerprint(root / config["artifact"]) != config["identity"]["hbm_sha256"]:
        raise ValueError("板端 HBM 与部署配置不一致.")
    engine = Runtime(root / config["artifact"], config["export"]["input_shape"])
    output = root / "examples/output"
    output.mkdir(parents=True, exist_ok=True)
    inputs = sorted((root / "examples/input").glob("sample_*.png"))
    if not inputs:
        raise FileNotFoundError("缺少固定示例输入,请重新准备部署包.")
    for index, path in enumerate(inputs, 1):
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"示例图片不可读: {path.name}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = engine.run(preprocess(image, config["export"]["input_shape"]))
        depth = restore_depth(depth, *image.shape[:2])
        scale = np.clip(depth / config["model"]["max_depth"], 0, 1)
        color = cv2.applyColorMap(np.round(scale * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        destination = output / f"{path.stem}_depth.jpg"
        if not cv2.imwrite(str(destination), np.concatenate([bgr, color], axis=1)):
            raise OSError("示例保存失败.")
        print(f"[通路与示例] {index}/{len(inputs)}: {destination.name},色阶固定 0-20 米.", flush=True)


if __name__ == "__main__":
    main()
