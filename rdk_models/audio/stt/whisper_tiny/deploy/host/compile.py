"""复用浮点图,用真实语音校准后通过官方 hb_compile 编译 S100 HBM."""

import argparse
from collections import Counter
import json
import shutil
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
import yaml

from context import context

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from shared import assets, decode_steps, fingerprint, log_mel, read_audio


def session(path):
    """以 CPU FP32 执行固定 ONNX,保留命名输出."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    runtime = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    names = [item.name for item in runtime.get_outputs()]
    return lambda feeds: dict(zip(names, runtime.run(names, feeds)))


def calibrate(ctx, stage, work):
    """按张量分别保存对齐的真实 Mel 或真实 Decoder 状态."""
    count = ctx.config["quantization"]["calibration_samples"]
    paths = sorted(ctx.dataset.rglob("*.flac"))[:count]
    if len(paths) != count:
        raise ValueError("Whisper 校准音频数量不足.")
    filters, rules = assets(ctx.assets)
    if stage == "decoder_step":
        encoder = session(ctx.output / "tiny_encoder.onnx")
        decoder = session(ctx.output / "tiny_decoder_step.onnx")
    for index, path in enumerate(paths):
        mel = log_mel(read_audio(path, host=True), filters)
        if stage == "encoder":
            records = ({"mel": mel},)
        else:
            features = encoder({"mel": mel})["audio_features"]
            records = (feeds for feeds, _ in decode_steps(decoder, features, rules,
                       ctx.config["quantization"]["decoder_steps"]))
        for step, feeds in enumerate(records):
            for name, value in feeds.items():
                directory = work / "calibration" / name
                directory.mkdir(parents=True, exist_ok=True)
                np.save(directory / f"{index:03d}_{step:03d}.npy", value)
        print(f"[校准 {stage}] {index + 1}/{count} {path.name}", flush=True)


def compile_stage(ctx, stage, precision):
    """请求 INT16 激活,明确指定常量卷积和矩阵乘权重位宽."""
    graph_path = ctx.output / f"tiny_{stage}.onnx"
    model = onnx.load(str(graph_path))
    graph = model.graph
    inputs = list(graph.input)
    names = [item.name for item in inputs]
    shapes = [[dim.dim_value for dim in item.type.tensor_type.shape.dim] for item in inputs]
    if not all(shape and all(dim > 0 for dim in shape) for shape in shapes):
        raise ValueError("S100 Whisper 需要静态输入形状.")
    work = ctx.work / precision / stage
    work.mkdir(parents=True, exist_ok=True)
    calibrate(ctx, stage, work)
    constants = {value.name for value in graph.initializer}
    constants.update(node.output[0] for node in graph.node if node.op_type == "Constant")
    weight = {"w8a16": "int8", "w16a16": "int16"}[precision]
    node_config = {node.name: {"input1": weight} for node in graph.node
                   if node.op_type in {"Conv", "MatMul", "Gemm"} and node.input[1] in constants}
    prefix = f"tiny_{stage}_{precision}"
    config = {
        "model_parameters": {"onnx_model": str(graph_path), "march": ctx.platform["march"],
            "working_dir": str(work / "build"), "output_model_file_prefix": prefix},
        "input_parameters": {"input_name": ";".join(names),
            "input_shape": ";".join("x".join(map(str, shape)) for shape in shapes),
            "input_type_rt": ";".join(["featuremap"] * len(names)),
            "input_type_train": ";".join(["featuremap"] * len(names)),
            "input_layout_rt": ";".join(["NCHW"] * len(names)),
            "input_layout_train": ";".join(["NCHW"] * len(names)),
            "norm_type": ";".join(["no_preprocess"] * len(names))},
        "calibration_parameters": {
            "cal_data_dir": ";".join(str(work / "calibration" / name) for name in names),
            "cal_data_type": ";".join(["float32"] * len(names)), "calibration_type": "max",
            "preprocess_on": False,
            "quant_config": {"model_config": {"all_node_type": "int16"}, "node_config": node_config}},
        "compiler_parameters": {"compile_mode": "latency", "core_num": 1, "optimize_level": "O2"},
    }
    path = work / "compile.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"[编译] {stage} 请求 {precision},待核对实际位宽和执行位置.", flush=True)
    subprocess.run(["hb_compile", "--config", str(path)], cwd=work, check=True)
    artifact = work / "build" / (prefix + ".hbm")
    if not artifact.is_file() or not artifact.stat().st_size:
        raise FileNotFoundError(artifact)
    print(f"[候选 HBM] {artifact}", flush=True)


def audit_stage(ctx, stage, precision):
    """检查实际 PTQ 主计算位宽,单独记录动态注意力的位宽."""
    prefix = f"tiny_{stage}_{precision}"
    build = ctx.work / precision / stage / "build"
    artifact = build / (prefix + ".hbm")
    if not artifact.is_file() or not artifact.stat().st_size:
        raise FileNotFoundError(artifact)
    graph = onnx.load(str(build / (prefix + "_ptq_model.onnx"))).graph
    producers = {output: node for node in graph.node for output in node.output}
    constants = {item.name for item in graph.initializer}

    def qtype(name):
        """返回实际校准节点记录的位宽及原始输入."""
        node = producers.get(name)
        if node is not None and node.op_type == "HzCalibration":
            return next(attr.s.decode() for attr in node.attribute if attr.name == "qtype"), node.input[0]
        return "unrecorded", name

    expected_weight = {"w8a16": "int8", "w16a16": "int16"}[precision]
    affine = Counter()
    attention = Counter()
    for node in graph.node:
        if node.op_type not in {"Conv", "MatMul", "Gemm"}:
            continue
        left, _ = qtype(node.input[0])
        right, original = qtype(node.input[1])
        if original in constants:
            if (left, right) != ("int16", expected_weight):
                raise ValueError(f"{stage} 主计算层位宽不符: {node.name}, {left}/{right}.")
            affine[node.op_type] += 1
        else:
            attention[f"{left}/{right}"] += 1
    if not affine or not attention:
        raise ValueError(f"{stage} 缺少主计算或注意力审计证据.")
    advice = json.loads((build / (prefix + "_advice.json")).read_text())
    backends = dict(Counter(row["backend"] for row in advice))
    return artifact, {"compiler_backend_operations": backends,
        "constant_affine": dict(affine), "activation": "int16",
        "weight": expected_weight, "dynamic_matmul_inputs": dict(attention),
        "hbm_sha256": fingerprint(artifact),
        "onnx_sha256": fingerprint(ctx.output / f"tiny_{stage}.onnx")}


def publish(ctx, precision):
    """仅在双图都通过审计后发布,报告绑定实际文件内容."""
    results = {stage: audit_stage(ctx, stage, precision) for stage in ("encoder", "decoder_step")}
    for artifact, _ in results.values():
        target = ctx.output / artifact.name
        temporary = ctx.output / (artifact.name + ".uploading")
        shutil.copyfile(artifact, temporary)
        temporary.replace(target)
    target = ctx.output / "build" / f"{precision}_precision.json"
    target.write_text(json.dumps({stage: row for stage, (_, row) in results.items()}, indent=2))
    print("[位宽审计]", target.read_text(), flush=True)


def main():
    """默认编译双图,允许对单个子图和指定精度进行诊断."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("encoder", "decoder_step"))
    parser.add_argument("--precision", choices=("w8a16", "w16a16"))
    args = parser.parse_args()
    ctx = context()
    precision = args.precision or ctx.config["quantization"]["precision"]
    for stage in (args.stage,) if args.stage else ("encoder", "decoder_step"):
        compile_stage(ctx, stage, precision)
    if not args.stage:
        publish(ctx, precision)


if __name__ == "__main__":
    main()
