"""转换 Google YAMNet 为默认 FP16 或第一折校准的 INT8/W8A16 模型."""

import argparse
import csv
from datetime import date
import json
from pathlib import Path
import sys

import mtk_converter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from audio_utils import load_records, load_waveform, preprocess

from tools.quantization import tensor_metadata

from tools.deploy import configure_parser


def audit_w8a16(reader, tail_only=False):
    """检查全 W8A16 或第12层点卷积输出起的混合精度,同时记录偏置精度."""
    graph = reader.as_dict()
    subgraph = graph["subgraphs"][0]
    tensors = subgraph["tensors"]
    tensor_types = {name: sum(tensor["type"] == name for tensor in tensors)
                    for name in sorted({tensor["type"] for tensor in tensors})}
    if not set(tensor_types).issubset({"INT8", "INT16", "INT32", "INT64"}):
        raise ValueError(f"W8A16 模型含非整数张量: {tensor_types}.")
    layers = []
    for operator in subgraph["operators"]:
        code = graph["operator_codes"][operator["opcode_index"]]
        name = code.get("custom_code", code["builtin_code"])
        if name not in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED",
                        "MTKEXT_CONV_2D", "MTKEXT_DEPTHWISE_CONV_2D",
                        "MTKEXT_FULLY_CONNECTED"):
            continue
        activation, weights, bias = [tensors[index] for index in operator["inputs"][:3]]
        output = tensors[operator["outputs"][0]]
        index = len(layers)
        input_type = "INT8" if tail_only and index < 23 else "INT16"
        output_type = "INT8" if tail_only and index < 22 else "INT16"
        if (activation["type"], weights["type"], output["type"]) != (
                input_type, "INT8", output_type):
            raise ValueError(f"主计算层精度不匹配: {name}, {activation}, {weights}, {output}.")
        layers.append({"operator": name, "input_dtype": activation["type"],
                       "weight_dtype": weights["type"], "bias_dtype": bias["type"],
                       "output_dtype": output["type"]})
    if len(layers) != 28:
        raise ValueError(f"YAMNet 应有 27 层卷积与 1 层全连接,实际为 {len(layers)}.")
    return {"verified_affine_layers": len(layers), "tensor_types": tensor_types, "layers": layers}


def configure_tail16(converter, models_dir, work_dir):
    """从SDK真实分组生成尾部INT16配置,避免猜测融合后的张量名称."""
    work_dir.mkdir(parents=True, exist_ok=True)
    seed_config = work_dir / "precision_seed.json"
    converter.precision_proportion = {"sym8W_asym8A": 1.0}
    converter.precision_config_file = str(seed_config)
    print("[量化 1/2] 导出原INT8图的真实精度分组.", flush=True)
    converter.convert_to_tflite(str(work_dir / "precision_seed.tflite"))
    configuration = json.loads(seed_config.read_text())
    selected = 0
    for specification in configuration["precision_specs"]:
        names = (specification["act_names"] + specification["wgt_names"]
                 + specification["param_names"])
        if any(any(part in name for part in (
                "layer13", "layer14", "global_average", "dense_1", "activation_1"))
               for name in names):
            specification["precision_name"] = "sym8W_asym16A"
            selected += 1
    if not selected:
        raise ValueError("未找到YAMNet尾部精度分组.")
    output_config = models_dir / "precision_config_int8_tail16.json"
    output_config.write_text(json.dumps(configuration, indent=2))
    converter.precision_proportion = None
    converter.precision_config_file = str(output_config)
    print(f"[量化 2/2] 应用{selected}个尾部分组,最终逐层核验实际精度.", flush=True)


def main():
    """默认转换全W8A16,保留局部INT16和原有精度入口."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp16", "int8", "w8a16", "int8_tail16"],
                        default="w8a16")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    configure_parser(parser, __file__, {
        "models_dir": lambda ctx: ctx.output,
        "dataset": lambda ctx: ctx.dataset,
        "output_dir": lambda ctx: ctx.work,
        "precision": lambda ctx: ctx.config["conversion"]["precision"],
    })
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.models_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.precision == "fp16":
        converter = mtk_converter.OnnxConverter.from_model_proto_file(
            str(args.models_dir / "yamnet.onnx"))
        converter.convert_float32_weights_to_float16 = True
        converter.convert_to_tflite(str(args.output_dir / "model_fp16.tflite"))
        with (args.output_dir / "runtime_config_fp16.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([1, 0, "fp16"])
            writer.writerow([1, 0, "fp16"])
        (args.output_dir / "quantization_fp16.json").write_text(json.dumps({
            "precision": "fp16", "converter": mtk_converter.__version__,
            "weights": "FP16", "ncc_flags": ["--relax-fp32", "--suppress-input",
                                               "--suppress-output", "--disallow-bridge"],
            "native_input_shape": [1, 1, 96, 64], "native_output_shape": [1, 521],
            "calibration_patches": 0}, indent=2))
        print("[OK] FP16 权重模型与原生 IO 参数已生成,需 NCC 降精度编译.", flush=True)
        return
    records = load_records(args.dataset)
    paths = []
    for target in range(50):
        chosen = [row for row in records if int(row["fold"]) == 1
                  and int(row["target"]) == target][:2]
        if len(chosen) != 2:
            raise ValueError("校准折必须每类至少两条音频.")
        paths.extend(args.dataset / "audio" / row["filename"] for row in chosen)

    def calibration():
        """按固定顺序提供 200 个真实音频特征窗口."""
        for path in tqdm(paths, desc=f"YAMNet {args.precision} 校准"):
            for patch in preprocess(load_waveform(path))[:2]:
                yield [patch[None]]

    converter = mtk_converter.OnnxConverter.from_model_proto_file(
        str(args.models_dir / "yamnet.onnx"))
    converter.quantize = True
    converter.use_per_output_channel_quantization = True
    if args.precision == "w8a16":
        converter.allow_asym16_quantization = True
        converter.default_quantization_bitwidth = 16
        converter.default_weights_quantization_bitwidth = 8
        converter.input_quantization_bitwidths = 16
    converter.append_output_dequantize_ops = False
    converter.calibration_data_gen = calibration
    if args.precision == "int8_tail16":
        converter.allow_asym16_quantization = True
        configure_tail16(converter, args.models_dir, args.work_dir)
    output = args.output_dir / f"model_{args.precision}.tflite"
    converter.convert_to_tflite(str(output))
    reader = mtk_converter.TFLiteParser(str(output))
    inputs, outputs = reader.get_input_tensor_details(), reader.get_output_tensor_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("必须为单输入和单输出.")
    input_dtype = "int16" if args.precision == "w8a16" else "int8"
    output_dtype = "int16" if args.precision in ("w8a16", "int8_tail16") else "int8"
    contract = [tensor_metadata(inputs[0], input_dtype, include_name=False), tensor_metadata(outputs[0], output_dtype, include_name=False)]
    audit = (audit_w8a16(reader, args.precision == "int8_tail16")
             if args.precision in ("w8a16", "int8_tail16") else None)
    if contract[0]["shape"] != [1, 1, 96, 64] or contract[1]["shape"] != [1, 521]:
        raise ValueError(f"输入输出形状不匹配: {contract}.")
    with (args.output_dir / f"runtime_config_{args.precision}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        for item in contract:
            row = [item["scale"], item["zero_point"]]
            if item["dtype"] == "int16":
                row.append("w8a16")
            writer.writerow(row)
    (args.output_dir / f"quantization_{args.precision}.json").write_text(json.dumps({
        "precision": args.precision, "precision_audit": audit,
        "input": contract[0], "output": contract[1], "calibration_fold": 1,
        "calibration_clips": [path.name for path in paths], "patches": 200,
        "converter": mtk_converter.__version__}, indent=2))
    print(f"[OK] {args.precision} 模型与独立运行参数已生成.", flush=True)


if __name__ == "__main__":
    main()
