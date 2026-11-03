"""按官方 V2 预处理校准 ResNet101,保存原生 INT8 I/O 契约."""

import json
from pathlib import Path
import sys

import mtk_converter
import PIL
from tqdm import tqdm

from tools.deploy import arguments
from tools.files import require_file, sha256
from tools.quantization import tensor_metadata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from resnet101 import MODEL, PROTOCOL, TORCHVISION_VERSION, WEIGHTS, preprocess


def main() -> None:
    """使用固定校准样本做 W8A8 PTQ,不运行正式精度或性能评价."""
    ctx = arguments(__file__)
    count = ctx.config["data"]["calibration_samples"]
    offset = ctx.config["data"]["calibration_offset"]
    if type(count) is not int or type(offset) is not int or count <= 0 or offset < 0:
        raise ValueError("校准数量必须为正整数,偏移必须为非负整数.")
    paths = sorted(path for path in ctx.calibration.iterdir()
                   if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    paths = paths[offset:offset + count]
    if len(paths) != count:
        raise ValueError(f"校准图片不足: 需要 {count},实际 {len(paths)}.")
    source = require_file(ctx.output / "resnet101.onnx")
    destination = ctx.work / "model_int8.tflite"

    def calibration_data():
        """复用板端预处理,顺序提供校准输入并显示进度."""
        for path in tqdm(paths, desc="ResNet101 校准"):
            yield [preprocess(path)]

    print("[量化] INT8 激活与逐输出通道 INT8 权重.", flush=True)
    converter = mtk_converter.OnnxConverter.from_model_proto_file(str(source))
    converter.quantize = True
    converter.use_per_output_channel_quantization = True
    converter.append_output_dequantize_ops = False
    converter.calibration_data_gen = calibration_data
    converter.convert_to_tflite(str(destination))
    parser = mtk_converter.TFLiteParser(str(require_file(destination)))
    inputs = parser.get_input_tensor_details()
    outputs = parser.get_output_tensor_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("ResNet101 要求单输入单输出.")
    input_tensor, output_tensor = tensor_metadata(inputs[0]), tensor_metadata(outputs[0])
    if input_tensor["shape"] != [1, 3, 224, 224] or output_tensor["shape"] != [1, 1000]:
        raise ValueError("TFLite 输入输出形状与固定 NCHW 契约不符.")
    metadata = {
        "model": MODEL, "weights": WEIGHTS, "torchvision": TORCHVISION_VERSION,
        "pillow": PIL.__version__,
        "protocol": PROTOCOL, "onnx_sha256": sha256(source),
        "input": input_tensor, "output": output_tensor,
        "precision": "W8A8_INT8_PTQ_per_output_channel_weights",
        "converter": mtk_converter.__version__,
        "native_layout": "contiguous_or_last_dimension_align16",
        "native_layout_board_verified": False,
        "calibration": {"samples": count, "offset": offset,
                        "images": [path.name for path in paths],
                        "source": "configured_calibration_dir"},
    }
    (ctx.work / "quantization.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[量化完成] 已保存实际 I/O dtype、形状和 scale/zero point.", flush=True)


if __name__ == "__main__":
    main()
