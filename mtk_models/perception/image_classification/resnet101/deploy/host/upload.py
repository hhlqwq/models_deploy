"""上传 ResNet101 必要资源与固定三张输入,不执行板端推理."""

import shutil

from tools.deploy import arguments, upload
from tools.files import read_json, require_file


def main() -> None:
    """复用公共上传,从已有 ImageNet 数据准备少量展示输入."""
    ctx = arguments(__file__)
    metadata = read_json(require_file(ctx.work / "quantization.json"))
    if "dla_sha256" not in metadata:
        raise ValueError("缺少 DLA 版本信息,请先执行编译步骤.")
    images = ctx.dataset / "val"
    if not images.is_dir():
        images = ctx.dataset / "ILSVRC2012_img_val"
    destination = ctx.model / "examples/input"
    destination.mkdir(parents=True, exist_ok=True)
    for index in range(1, 4):
        source = require_file(images / f"ILSVRC2012_val_{index:08d}.JPEG")
        shutil.copyfile(source, destination / f"sample_{index}.JPEG")
    upload(ctx, [ctx.output / "model_int8.dla", ctx.work / "quantization.json"])


if __name__ == "__main__":
    main()
