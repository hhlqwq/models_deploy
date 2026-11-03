"""对已有 ResNet101 FP32 ONNX 计算同协议 ImageNet 全量参考."""


import json
import os
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import PIL
from tqdm import tqdm

from tools.deploy import arguments, reference_path
from tools.evaluation import is_test, select_samples, scope_metadata
from tools.files import read_json, require_file, sha256
from tools.output_metrics import record_reference, finish_reference

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from resnet101 import (
    MODEL, PROTOCOL, WEIGHTS, label_signature, preprocess, read_labels,
    validate_contract, validation_images,
)
from tools.evaluation import accuracy_result


def main() -> None:
    """只推理已有浮点模型,不触发导出、量化、编译或板端任务."""
    ctx = arguments(__file__)
    source = require_file(ctx.output / "resnet101.onnx")
    metadata = read_json(ctx.output / "build/quantization.json")
    validate_contract(metadata)
    if sha256(source) != metadata["onnx_sha256"]:
        raise ValueError("已有 ONNX 与部署模型的量化来源不一致.")
    paths = select_samples(validation_images(ctx.dataset), MODEL, key=lambda path: path.name)
    labels = read_labels(ctx.dataset / "val_labels_0based.txt")
    session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or len(outputs) != 1
            or inputs[0].shape != [1, 3, 224, 224] or inputs[0].type != "tensor(float)"
            or outputs[0].shape != [1, 1000] or outputs[0].type != "tensor(float)"):
        raise ValueError("FP32 ONNX 张量契约不符.")
    top1 = 0
    for path in tqdm(paths, total=len(paths), desc="ResNet101 FP32 Top-1 参考"):
        label = labels[int(path.stem.rsplit("_", 1)[1]) - 1]
        tensor = preprocess(path)
        logits = session.run([outputs[0].name], {inputs[0].name: tensor})[0]
        record_reference(source, [tensor], {"output": logits},
                         sample_id=path.name, sample_files=[path])
        if logits.shape != (1, 1000) or not np.isfinite(logits).all():
            raise ValueError(f"ONNX 输出无效: {path.name}.")
        top1 += int(np.argmax(logits[0]) == label)
    report = {
        "model": MODEL, "weights": WEIGHTS, "status": "complete",
        **scope_metadata(), "protocol": PROTOCOL, "samples": len(paths),
        "source": "locally_exported_official_torchvision_onnx",
        "onnx_sha256": metadata["onnx_sha256"], "labels_sha256": label_signature(labels),
        "pillow": PIL.__version__, "onnxruntime": ort.__version__,
        "metric": "Top-1", "accuracy": top1 / len(paths),
    }
    report["evaluation"] = accuracy_result(MODEL, "onnx",
        {"Top-1": report["accuracy"]}, len(paths), "image")
    destination = reference_path(ctx)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[参考完成] {destination}.", flush=True)
    finish_reference()


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    main()
