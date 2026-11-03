"""使用 Python 完成图片预处理、NPU 推理与模型后处理."""

import json

import cv2
import numpy as np

from full_accuracy_board import load_pairs
from face_utils import load_aligned_face
from tools.board import context, infer_images


def postprocess(values, image, output):
    """保存评分和展示所需的连续 FP32 模型结果."""
    feature = values.reshape(128)
    norm = np.linalg.norm(feature)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("人脸特征无效.")
    (feature / norm).astype("<f4").tofile(output / "feature.bin")


def main():
    """读取本模型配置并使用常驻 SDK Runtime 推理选定图片."""
    ctx = context(__file__, new_run=True)
    rows = load_pairs(ctx.dataset)
    names = sorted({row[field] for row in rows for field in ("image_a", "image_b")})
    images = [(name, ctx.dataset / "images" / name) for name in names]
    metadata = json.loads((ctx.root / "build/reference/metadata.json").read_text(encoding="utf-8"))
    if metadata.get("input_protocol") != "RGB_ImageNet_normalize_aligned_112x112":
        raise ValueError("量化元数据的人脸输入协议不匹配.")
    infer_images(ctx, images, metadata, [1, 3, 112, 112], [1, 128],
                 load_aligned_face, postprocess)


if __name__ == "__main__":
    main()
