"""整理音频浮点参考与少量示例,区分快速验证和全量评测."""

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import shutil

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "board"))
from audio_utils import load_records, mapping
from tools.evaluation import audio_records, is_test, scope_metadata


from tools.deploy import arguments, upload


def prepare_reference(args):
    """整理同范围浮点参考、哈希和固定示例,供当前上传步骤使用."""
    records = audio_records(load_records(args.dataset))
    labels = args.models_dir / "original/upstream/yamnet_class_map.csv"
    reference = {}
    for backend in ("tensorflow", "onnx"):
        mode = "test" if is_test() else "full"
        report = json.loads((args.work / "evaluation" / mode / backend / "summary.json").read_text())
        if (report.get("samples") != len(records) or
                report.get("evaluation_scope", "full") != scope_metadata()["evaluation_scope"]):
            raise ValueError(f"{backend} 参考报告与本次样本范围不一致.")
        if is_test() and report.get("sample_ids_sha256") != scope_metadata()["sample_ids_sha256"]:
            raise ValueError(f"{backend} 参考报告的样本指纹不一致.")
        reference[backend] = report
    reference["annotations_sha256"] = hashlib.sha256(
        (args.dataset / "meta/esc50.csv").read_bytes()).hexdigest()
    reference["mapping"] = mapping(labels)
    reference["audio_sha256"] = {
        row["filename"]: hashlib.sha256(
            (args.dataset / "audio" / row["filename"]).read_bytes()).hexdigest()
        for row in records}
    reference.update(scope_metadata())
    filename = "fp32_reference_test.json" if is_test() else "fp32_reference.json"
    (args.work / filename).write_text(json.dumps(reference, indent=2))
    print(f"[参考] {args.work / filename}", flush=True)



def prepare_examples(ctx):
    """只复制固定展示音频和类别表,不计算任何精度参考."""
    records = audio_records(load_records(ctx.dataset))
    shutil.copy2(ctx.output / "original/upstream/yamnet_class_map.csv", ctx.work / "yamnet_class_map.csv")
    inputs = ctx.model / "examples/input"
    inputs.mkdir(parents=True, exist_ok=True)
    samples = []
    for number, target in enumerate((0, 39, 42), 1):
        row = next(row for row in records if int(row["target"]) == target)
        filename = f"sample_{number}.wav"
        shutil.copy2(ctx.dataset / "audio" / row["filename"], inputs / filename)
        samples.append(dict(row, file=filename))
    (inputs / "samples.json").write_text(json.dumps({"samples": samples}, ensure_ascii=False, indent=2))



def main():
    """上传模型、程序和 JSON 配置,保留全量与快速结果的统一部署位置."""
    ctx = arguments(__file__)
    precision = ctx.config["conversion"]["precision"]
    prepare_examples(ctx)
    upload(ctx, [
        ctx.output / f"model_{precision}.dla",
        ctx.work / f"runtime_config_{precision}.csv",
        ctx.work / "yamnet_class_map.csv",
    ])


if __name__ == "__main__":
    main()
