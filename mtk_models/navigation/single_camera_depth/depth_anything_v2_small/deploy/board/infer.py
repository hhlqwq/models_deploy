"""使用 Python 完成图片预处理、NPU 推理与模型后处理."""

import json

import numpy as np

from full_accuracy_board import load_protocol
from depth_utils import preprocess
from tools.board import context, infer_images


def postprocess(values, image, output):
    """只保存 518x518 FP32 深度图,评分和展示时再恢复原图尺寸."""
    depth = values.reshape(518, 518)
    if not np.isfinite(depth).all() or depth.std() <= 0:
        raise ValueError("深度图无效或恒定.")
    np.save(output / "depth.npy", depth.astype("<f4"))


def main():
    """读取本模型配置并使用常驻 SDK Runtime 推理选定图片."""
    ctx = context(__file__, new_run=True)
    names = sorted(load_protocol(ctx.dataset))
    images = [(name, ctx.dataset / name) for name in names]
    metadata = json.loads((ctx.root / "build/reference/model_int8.json").read_text(encoding="utf-8"))
    infer_images(ctx, images, metadata, [1, 3, 518, 518], [1, 518, 518],
                 preprocess, postprocess)


if __name__ == "__main__":
    main()
