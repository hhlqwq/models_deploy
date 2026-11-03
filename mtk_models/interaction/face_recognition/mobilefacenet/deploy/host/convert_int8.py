"""使用对齐人脸校准 MobileFaceNet 并生成 MTK INT8 TFLite。"""

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import mtk_converter
import numpy as np

# 复用板端预处理,保持校准与推理输入一致.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from face_utils import list_aligned_faces, load_aligned_face

from tools.deploy import configure_parser
from tools.quantization import tensor_metadata


def calibration_data(images: list[Path]) -> Iterator[list[np.ndarray]]:
    """逐张提供真实对齐人脸输入,进度由转换器统一显示."""
    for path in images:
        yield [load_aligned_face(path)]


def convert_model(onnx: Path, image_dir: Path, output: Path,
                  samples: int = 100) -> None:
    """对 ONNX 执行后训练量化。"""
    if samples <= 0:
        raise ValueError("校准图片数量必须大于 0.")
    images = list_aligned_faces(image_dir)
    if len(images) < samples:
        raise ValueError(
            f"校准人脸不足: 需要 {samples} 张,找到 {len(images)} 张: {image_dir}")
    print(f"[CALIBRATION] 找到 {len(images)} 张对齐图,选取前 {samples} 张.",
          flush=True)
    images = images[:samples]
    converter = mtk_converter.OnnxConverter.from_model_proto_file(str(onnx))
    converter.quantize = True
    converter.calibration_data_gen = lambda: calibration_data(images)
    converter.use_per_output_channel_quantization = True
    output.parent.mkdir(parents=True, exist_ok=True)
    converter.convert_to_tflite(str(output))
    save_metadata(output)
    print(f"[OK] INT8 TFLite: {output}")


def save_metadata(tflite: Path) -> None:
    """量化后立即核对 C++ 所需的输入输出契约并保存元数据."""
    parser = mtk_converter.TFLiteParser(str(tflite))
    inputs, outputs = parser.get_input_tensor_details(), parser.get_output_tensor_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("MobileFaceNet 要求单输入单输出.")
    contract = [tensor_metadata(item, np.dtype(item["type"].lower()).name,
                                include_name=False) for item in (inputs[0], outputs[0])]
    if (contract[0]["shape"] != [1, 3, 112, 112] or contract[1]["shape"] != [1, 128] or
            any(info["dtype"] != "int8" for info in contract)):
        raise ValueError(f"MobileFaceNet 张量契约无效: {contract}.")
    destination = tflite.with_name("metadata.json")
    destination.write_text(json.dumps({
        "input": contract[0], "output": contract[1],
        "input_protocol": "RGB_ImageNet_normalize_aligned_112x112",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[元数据] {destination}", flush=True)


def main() -> None:
    """解析量化参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    configure_parser(parser, __file__, {
        "onnx": lambda c: c.output / "mobilefacenet.onnx",
        "image_dir": lambda c: c.calibration,
        "samples": lambda c: c.config["data"]["calibration_samples"],
        "output": lambda c: c.work / "model_int8.tflite",
    })
    args = parser.parse_args()
    convert_model(args.onnx, args.image_dir, args.output, args.samples)


if __name__ == "__main__":
    main()
