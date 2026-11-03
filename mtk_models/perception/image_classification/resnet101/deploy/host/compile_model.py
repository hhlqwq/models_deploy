"""复用 G720 NCC 配方编译 ResNet101 并绑定实际 DLA 版本."""

import json

from tools.deploy import arguments, compile_dla
from tools.files import read_json, require_file, sha256


def main() -> None:
    """禁止 CPU bridge 编译 MDLA 5.3,保存后续推理的模型版本依据."""
    ctx = arguments(__file__)
    source = require_file(ctx.work / "model_int8.tflite")
    metadata_path = ctx.work / "quantization.json"
    metadata = read_json(metadata_path)
    if metadata["onnx_sha256"] != sha256(require_file(ctx.output / "resnet101.onnx")):
        raise ValueError("ONNX 已改变,请先重新量化.")
    destination = ctx.output / "model_int8.dla"
    compile_dla(ctx, source, destination)
    metadata["dla_sha256"] = sha256(destination)
    metadata["compiler_target"] = "mdla5.3"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
