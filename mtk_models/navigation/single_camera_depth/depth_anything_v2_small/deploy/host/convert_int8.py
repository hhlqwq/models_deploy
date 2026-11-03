"""使用固定图片集合为 Depth Anything V2 Small 执行 MTK INT8 转换。"""

import argparse
import sys
import json
from pathlib import Path

import mtk_converter
import numpy as np

# 复用板端预处理,保持校准与推理输入一致.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from depth_utils import INPUT_SIZE, preprocess

from tools.quantization import tensor_metadata

from tools.deploy import configure_parser
from tools.storage import storage_roots


def convert_model(onnx: Path, image_dir: Path, output: Path,
                  samples: int) -> None:
    """按文件名排序选取图片，量化并保存输入输出契约。"""
    paths = sorted(path for path in image_dir.iterdir()
                   if path.suffix.lower() in {".jpg", ".jpeg", ".png"})[:samples]
    if samples <= 0 or len(paths) != samples:
        raise ValueError(f"校准图片不足 {samples} 张: {image_dir}")
    calibration = [{"path": str(path)}
                   for path in paths]

    def calibration_data():
        """逐张提供与推理一致的归一化输入,进度由转换器统一显示."""
        for path in paths:
            yield [preprocess(path)]

    output.parent.mkdir(parents=True, exist_ok=True)
    converter = mtk_converter.OnnxConverter.from_model_proto_file(str(onnx))
    converter.quantize = True
    converter.use_per_output_channel_quantization = True
    converter.append_output_dequantize_ops = False
    converter.calibration_data_gen = calibration_data
    print("[CONVERT] 开始 INT8 转换。", flush=True)
    converter.convert_to_tflite(str(output))
    reader = mtk_converter.TFLiteParser(str(output))
    inputs = reader.get_input_tensor_details()
    outputs = reader.get_output_tensor_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("冒烟要求单输入、单输出模型。")
    input_meta = tensor_metadata(inputs[0])
    output_meta = tensor_metadata(outputs[0])
    if input_meta["shape"] != [1, 3, INPUT_SIZE, INPUT_SIZE]:
        raise ValueError(f"输入形状异常: {input_meta}")
    if int(np.prod(output_meta["shape"])) != INPUT_SIZE * INPUT_SIZE:
        raise ValueError(f"输出元素数异常: {output_meta}")
    metadata = {
        "input": input_meta, "output": output_meta,
        "calibration": calibration,
        "converter": mtk_converter.__version__,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] TFLite: {output}", flush=True)


def main() -> None:
    """读取路径与校准样本数并执行转换。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    configure_parser(parser, __file__, {
        "onnx": lambda c: c.output / "depth_anything_v2_vits.onnx",
        "image_dir": lambda c: c.calibration,
        "samples": lambda c: c.config["data"]["calibration_samples"],
        "output": lambda c: c.work / "model_int8.tflite",
    })
    args = parser.parse_args()
    convert_model(args.onnx, args.image_dir, args.output, args.samples)


if __name__ == "__main__":
    main()
