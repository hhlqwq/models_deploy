"""通过 MTK ONNX Converter 离线校准 YOLOv8n 并记录张量契约."""

import argparse
import sys
import json
from pathlib import Path

import cv2
import mtk_converter
import numpy as np
from tqdm import tqdm

# 复用板端预处理,保持校准与推理输入一致.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from yolov8_utils import OUTPUT_NAMES, OUTPUT_SHAPES, preprocess

from tools.quantization import tensor_metadata

from tools.deploy import configure_parser
from tools.storage import storage_roots


def main():
    """选取固定校准集,执行转换,按唯一形状映射运行时输出."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    configure_parser(parser, __file__, {
        "onnx": lambda c: c.output / "yolov8n.onnx",
        "calibration_dir": lambda c: c.calibration,
        "samples": lambda c: c.config["data"]["calibration_samples"],
        "output": lambda c: c.work / "model_int8.tflite",
    })
    args = parser.parse_args()
    paths = sorted(path for path in args.calibration_dir.iterdir()
                   if path.suffix.lower() in {".jpg", ".jpeg", ".png"})[:args.samples]
    if args.samples <= 0 or len(paths) != args.samples:
        raise ValueError("校准图片不足或 samples 非正数.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calibration = [{"path": str(path)} for path in paths]

    def calibration_data():
        """按固定顺序提供与板端一致的预处理输入."""
        for path in tqdm(paths, desc="YOLOv8n 校准"):
            yield [preprocess(cv2.imread(str(path)))[0]]

    print("[CONVERT] 开始 INT8 校准和转换.", flush=True)
    converter = mtk_converter.OnnxConverter.from_model_proto_file(str(args.onnx))
    converter.quantize = True
    converter.use_per_output_channel_quantization = True
    converter.append_output_dequantize_ops = False
    converter.calibration_data_gen = calibration_data
    converter.convert_to_tflite(str(args.output))
    reader = mtk_converter.TFLiteParser(str(args.output))
    inputs = reader.get_input_tensor_details()
    if len(inputs) != 1:
        raise ValueError("只支持单输入模型.")
    input_detail = tensor_metadata(inputs[0])
    if input_detail["shape"] != [1, 3, 640, 640]:
        raise ValueError(f"输入布局不是 NCHW: {input_detail}.")
    outputs = []
    for index, detail in enumerate(reader.get_output_tensor_details()):
        output = tensor_metadata(detail)
        if output["shape"] not in OUTPUT_SHAPES:
            raise ValueError(f"输出布局不匹配: {output}.")
        output["semantic"] = OUTPUT_NAMES[OUTPUT_SHAPES.index(output["shape"])]
        output["index"] = index
        outputs.append(output)
    if sorted(item["semantic"] for item in outputs) != sorted(OUTPUT_NAMES):
        raise ValueError("输出头缺失或重复.")
    metadata = {"input": input_detail, "outputs": outputs,
                "calibration": calibration,
                "converter": mtk_converter.__version__,
                "native_layout": "NCHW_INT8_WIDTH_ALIGN16",
                "native_layout_board_verified": False}
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    contract = [input_detail] + outputs
    lines = []
    for item in contract:
        shape = item["shape"]
        lines.append(",".join(map(str, [item.get("semantic", "input"),
                                      item.get("index", 0), *shape[1:],
                                      item["scale"], item["zero_point"]])))
    (args.output.parent / "runtime_config.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("[OK] TFLite 与量化元数据已保存,原生输出布局仍需板端验证.")


if __name__ == "__main__":
    main()
