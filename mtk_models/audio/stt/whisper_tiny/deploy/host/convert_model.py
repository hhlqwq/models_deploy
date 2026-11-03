"""使用真实语音与浮点自回归状态校准 Whisper W8A16 双图."""

import json
from pathlib import Path
import sys

import mtk_converter
import numpy as np
import onnxruntime as ort

from tools.deploy import arguments
from tools.quantization import tensor_metadata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from decoder import CACHE_NAMES, INPUT_SHAPES, OUTPUT_SHAPES, feeds_for_step, next_token
from infer import preprocess


def session(path):
    """限制浮点校准 CPU 线程,不下载资源或修改环境."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def audit_precision(reader):
    """拒绝浮点回退,核对实际仿射层 INT8 权重和 INT16 激活."""
    graph = reader.as_dict()
    subgraph = graph["subgraphs"][0]
    tensors = subgraph["tensors"]
    types = sorted({tensor["type"] for tensor in tensors})
    if not set(types).issubset({"INT8", "INT16", "INT32", "INT64", "BOOL"}):
        raise ValueError(f"W8A16 图含浮点或不支持的类型: {types}.")
    count = 0
    for operator in subgraph["operators"]:
        code = graph["operator_codes"][operator["opcode_index"]]
        name = code.get("custom_code") or code["builtin_code"]
        if name not in {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED",
                        "MTKEXT_CONV_2D", "MTKEXT_DEPTHWISE_CONV_2D",
                        "MTKEXT_FULLY_CONNECTED"}:
            continue
        activation, weights = [tensors[index] for index in operator["inputs"][:2]]
        output = tensors[operator["outputs"][0]]
        actual = (activation["type"], weights["type"], output["type"])
        if actual != ("INT16", "INT8", "INT16"):
            raise ValueError(f"{name} 不满足 W8A16: {actual}.")
        count += 1
    if not count or "INT8" not in types or "INT16" not in types:
        raise ValueError("没有可核实的 W8A16 仿射层.")
    return {"affine_layers": count, "tensor_types": types}


def tensor_contract(details, expected):
    """按名称绑定语义并保留转换器索引,不猜测多输入输出顺序."""
    result = []
    for index, detail in enumerate(details):
        item = tensor_metadata(detail, "int16")
        normalized = item["name"].removesuffix(":0").removeprefix("serving_default_")
        matches = [name for name in expected
                   if normalized == name or normalized.endswith("/" + name)]
        if len(matches) != 1 or tuple(item["shape"]) != expected[matches[0]]:
            raise ValueError(f"转换后张量名称或布局不匹配: {item}; expected={expected}.")
        item.update(index=index, name=matches[0])
        result.append(item)
    if len(result) != len(expected) or {item["name"] for item in result} != set(expected):
        raise ValueError("转换后 I/O 语义覆盖不完整或重复.")
    return result


def main():
    """顺序量化 Encoder 和 1t256c Decoder,保存实际运行契约."""
    ctx = arguments(__file__)
    settings = ctx.config["conversion"]
    if (settings["precision"], settings["cacheSize"], settings["maxTokenLength"]) != (
            "w8a16", 256, 448):
        raise ValueError("当前交付仅支持 W8A16、cacheSize=256、maxTokenLength=448.")
    paths = sorted(ctx.dataset.rglob("*.flac"))[:settings["calibration_samples"]]
    if len(paths) != settings["calibration_samples"] or not paths:
        raise ValueError("Whisper 真实语音校准资源不足.")
    filters = np.fromfile(ctx.work / "mel_filters_f32.bin", dtype="<f4").reshape(80, 201)
    config = dict(line.split("=", 1) for line in
                  (ctx.work / "decode_config.txt").read_text().splitlines())
    encoder = session(ctx.output / "tiny_encoder.onnx")
    decoder = session(ctx.output / "tiny_decoder_step.onnx")

    def calibration(stage):
        """逐条生成真实 Mel 和自回归状态,不保存重复中间张量."""
        for number, path in enumerate(paths, 1):
            print(f"[校准 {stage}] 音频 {number}/{len(paths)}: {path.name}", flush=True)
            mel = preprocess(path, filters)[0][None].astype(np.float32)
            if stage == "encoder":
                yield [mel]
                continue
            features = encoder.run(None, {encoder.get_inputs()[0].name: mel})[0]
            caches = {name: np.zeros(INPUT_SHAPES[name], np.float32) for name in CACHE_NAMES}
            initial = [int(value) for value in config["initial"].split(",")]
            token, generated = initial[0], 0
            steps = int(settings["calibration_decoder_steps"])
            if not len(initial) < steps <= settings["maxTokenLength"]:
                raise ValueError("校准步数必须覆盖提示词且不超过 Decoder 上限.")
            for position in range(steps):
                feeds = feeds_for_step(token, position, features, caches)
                yield [feeds[item.name] for item in decoder.get_inputs()]
                values = dict(zip([item.name for item in decoder.get_outputs()],
                                  decoder.run(None, feeds)))
                caches = {name: values[name.removesuffix("_in") + "_out"] for name in CACHE_NAMES}
                if position + 1 < len(initial):
                    token = initial[position + 1]
                    continue
                token = next_token(values["logits"], config, first=not generated)
                if token == int(config["eot"]):
                    break
                generated += 1

    contract = {"precision": "w8a16", "cacheSize": 256, "maxTokenLength": 448,
                "activation_quantization": "symmetric_int16",
                "layer_norm": "integer_row_scaled_variance",
                "cache_policy": "sliding_window", "calibration_audio": [p.name for p in paths],
                "calibration_decoder_steps": settings["calibration_decoder_steps"],
                "calibration_overlap": "LibriSpeech test-clean 前部音频,与正式评价重叠"}
    for stage, inputs, outputs, prefix in (
            ("encoder", {"mel": (1, 80, 3000)}, {"audio_features": (1, 1500, 384)}, "tiny_encoder"),
            ("decoder_step", INPUT_SHAPES, OUTPUT_SHAPES, "tiny_decoder")):
        suffix = "_w8a16" + ("_1t256c" if stage == "decoder_step" else "")
        filename = prefix + suffix
        converter = mtk_converter.OnnxConverter.from_model_proto_file(
            str(ctx.output / f"tiny_{stage}.onnx"))
        converter.quantize = True
        # MDLA 的注意力 BatchMatMul 第二输入要求对称 INT16.
        converter.allow_asym16_quantization = False
        converter.use_symmetric_quantization = True
        converter.default_quantization_bitwidth = 16
        converter.default_weights_quantization_bitwidth = 8
        converter.input_quantization_bitwidths = 16
        converter.use_per_output_channel_quantization = True
        converter.append_output_dequantize_ops = False
        converter.calibration_data_gen = lambda current=stage: calibration(current)
        target = ctx.work / f"{filename}.tflite"
        converter.convert_to_tflite(str(target))
        reader = mtk_converter.TFLiteParser(str(target))
        contract[stage] = {"file": filename + ".dla", "precision_audit": audit_precision(reader),
            "inputs": tensor_contract(reader.get_input_tensor_details(), inputs),
            "outputs": tensor_contract(reader.get_output_tensor_details(), outputs)}
    (ctx.work / "runtime_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[量化完成] W8A16 双图和 INT16 张量契约已保存,待编译和板端验证.", flush=True)


if __name__ == "__main__":
    main()
