"""独立全量 FP32 参考,已有同协议参考则直接复用,不重新编译."""

from datetime import date
from tools.evaluation import accuracy_result, record_selection

import json
from pathlib import Path
import sys

from tools.deploy import context
from tools.evaluation import is_test

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from evaluate import evaluate_dataset, identity, validate_manifest
from scoring import score_predictions


def main():
    """使用已有四个 ONNX 和相同 CPU 稀疏主干计算完整浮点参考."""
    ctx = context(__file__)
    manifest = json.loads((ctx.work / "samples.json").read_text(encoding="utf-8"))
    validate_manifest(manifest, ctx.config["data"], is_test())
    samples = manifest["frames"]
    package = ctx.output / "models"
    fingerprint = identity(package)
    tokens = [sample["token"] for sample in samples]
    record_selection(tokens)
    target = ctx.work / "onnx_reference.json"
    if target.is_file():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if (previous.get("identity") == fingerprint and previous.get("sample_tokens") == tokens
                and "evaluation" in previous):
            print("[参考复用] 已有同模型同协议完整 FP32 参考,无需重复运行.", flush=True)
            return
    temporary = Path("/tmp/hailongcodex") / date.today().isoformat() / "mit_bevfusion_reference"
    predictions, _, _, _ = evaluate_dataset(package, ctx.dataset, samples,
                                           temporary / "predictions.json", floating=True)
    accuracy = score_predictions(ctx.work / "ground_truth.json", predictions)
    target.write_text(json.dumps({"identity": fingerprint, "sample_tokens": tokens, "accuracy": accuracy,
        "dataset": {"version": manifest["version"], "split": manifest["split"]},
        "evaluation": accuracy_result("mit_bevfusion", "onnx", accuracy, len(tokens), "frame")}),
                      encoding="utf-8")
    print("[参考完成] 浮点精度已保存,入口接下来自动上传至板端 build/reference/.", flush=True)


if __name__ == "__main__":
    main()
