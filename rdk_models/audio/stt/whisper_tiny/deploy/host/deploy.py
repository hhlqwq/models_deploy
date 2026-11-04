"""生成同协议浮点参考,发布 S100 Whisper 运行包和必要样本."""

import argparse
import json
import os
import shutil
from pathlib import Path, PurePosixPath
import sys

import numpy as np

from context import context
from tools.rdk_deploy import connection, upload
from compile import session
from tools.evaluation import accuracy_result, record_selection, select_samples as task_samples, SPEECH_EXAMPLES

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from shared import assets, fingerprint, log_mel, read_audio, token_bytes, transcribe, word_errors
from decoder import INPUT_SHAPES, OUTPUT_SHAPES


def select_samples(ctx, subset):
    """加载官方转录,与 MTK 共用固定子集及示例选择规则."""
    samples = {}
    for path in sorted(ctx.dataset.rglob("*.trans.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            key, text = line.split(" ", 1)
            audio = path.with_name(key + ".flac")
            if key in samples or not audio.is_file():
                raise ValueError(f"音频重复或缺失: {key}.")
            samples[key] = {"id": key, "audio": audio.relative_to(ctx.dataset).as_posix(), "text": text}
    if len(samples) != 2620:
        raise ValueError(f"LibriSpeech test-clean 需要 2620 条,实际 {len(samples)}.")
    os.environ["MTK_EVAL_SCOPE"] = "subset" if subset else "full"
    return task_samples([samples[key] for key in sorted(samples)], "whisper_tiny",
                        key=lambda row: row["id"], required=SPEECH_EXAMPLES)


def prepare_reference(ctx, subset):
    """以已验证的浮点 ONNX 和原版英文规范化计算配套参考."""
    selected = select_samples(ctx, subset)
    stage = "subset" if subset else "full"
    work = ctx.work / stage
    work.mkdir(parents=True, exist_ok=True)
    filters, rules = assets(ctx.assets)
    sys.path.insert(0, str(ctx.assets / "vendor"))
    from whisper.normalizers.english import EnglishTextNormalizer

    normalize = EnglishTextNormalizer()
    ranks = token_bytes(ctx.assets / "whisper/assets/multilingual.tiktoken")
    onnx = {name: ctx.output / f"tiny_{name}.onnx" for name in ("encoder", "decoder_step")}
    encoder, decoder = (session(onnx[name]) for name in onnx)
    reference = {"scope": stage, "samples": selected,
                 "onnx": {name: fingerprint(path) for name, path in onnx.items()}}
    errors = words = 0
    features = {}
    for index, row in enumerate(selected, 1):
        path = ctx.dataset / row["audio"]
        mel = log_mel(read_audio(path, host=True), filters)
        text, values, _, _ = transcribe(encoder, decoder, mel, rules, ranks)
        target = normalize(row["text"])
        error, count = word_errors(target, normalize(text))
        errors += error
        words += count
        row.update(audio_sha256=fingerprint(path), normalized_text=target, fp32_text=text)
        if subset and index <= ctx.config["evaluation"]["numerical_samples"]:
            features[row["id"]] = values
        if index % 5 == 0 or index == len(selected):
            print(f"[ONNX WER] {index}/{len(selected)}", flush=True)
    reference.update(fp32_wer=errors / words, fp32_word_errors=errors, reference_words=words,
        evaluation=accuracy_result("whisper_tiny", "onnx", {"WER": errors / words},
            len(selected), "utterance", selection_id=record_selection(row["id"] for row in selected)))
    if subset:
        reference["numerical_sample_ids"] = list(features)
    report = work / "float_reference.json"
    report.write_text(json.dumps(reference, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    if subset:
        np.savez_compressed(work / "encoder_reference.npz", **features)
    return reference


def main():
    """子集发布现有双 HBM,全量只生成浮点参考和评价输入."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    ctx = context()
    precision = ctx.config["quantization"]["precision"]
    stages = ("encoder", "decoder_step")
    artifacts = {name: ctx.output / f"tiny_{name}_{precision}.hbm" for name in stages}
    if not all(path.is_file() and path.stat().st_size for path in artifacts.values()):
        raise FileNotFoundError("缺少已审计并发布的双 HBM.")
    reference = prepare_reference(ctx, not args.full)
    bind_reference(ctx, reference, precision, artifacts)


def board_configuration(ctx):
    """生成与 MTK 相同布局的板端配置快照."""
    return {"name": "whisper_tiny", "platform": "s100",
        "deployment": {"model_directory": "models"},
        "board": {"dataset_dir": str(PurePosixPath(ctx.platform["board"]["datasets_root"]) /
            ctx.config["board"]["dataset_directory"])},
        "evaluation": ctx.config["evaluation"], "quantization": ctx.config["quantization"]}


def runtime_contract(precision, hbm):
    """记录双图公开契约及已审计的实际 HBM 身份."""
    return {"precision": precision, "cacheSize": 256, "maxTokenLength": 448,
        "cache_policy": "sliding_window", "hbm": hbm,
        "encoder": {"inputs": {"mel": [1, 80, 3000]}, "outputs": {"audio_features": [1, 1500, 384]}},
        "decoder_step": {"inputs": INPUT_SHAPES, "outputs": OUTPUT_SHAPES}, "public_dtype": "float32"}


def bind_reference(ctx, reference, precision, artifacts):
    """核对双图审计与参考,打包并传输约定范围的文件."""
    stage = reference["scope"]
    selected = reference["samples"]
    work = ctx.work / stage
    audit = json.loads((ctx.output / "build" / f"{precision}_precision.json").read_text())
    hbm = {name: fingerprint(path) for name, path in artifacts.items()}
    if any(row["hbm_sha256"] != hbm[name] or row["onnx_sha256"] != reference["onnx"][name]
           for name, row in audit.items()):
        raise ValueError("浮点参考、HBM 和量化审计未绑定同一产物.")
    contract = runtime_contract(precision, hbm)
    contract_path = work / "runtime_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    reference.update(precision=precision, hbm=hbm, precision_audit=audit,
                     artifacts={name: "models/" + path.name for name, path in artifacts.items()})
    report = work / "reference.json"
    runtime_files = {"models/" + name: fingerprint(ctx.assets / name) for name in
        ("mel_filters_f32.bin", "decode_config.txt", "whisper/assets/multilingual.tiktoken")}
    runtime_files.update({"board/" + path.relative_to(ctx.assets / "vendor").as_posix(): fingerprint(path)
        for path in (ctx.assets / "vendor").rglob("*") if path.is_file() and "__pycache__" not in path.parts})
    runtime_files["models/runtime_contract.json"] = fingerprint(contract_path)
    reference["runtime_files"] = runtime_files
    report.write_text(json.dumps(reference, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    files = [(report, f"build/reference/{stage}.json")]
    if stage == "subset":
        numeric = work / "encoder_reference.npz"
        files.append((numeric, "build/reference/encoder_reference.npz"))
        for index, row in enumerate(selected[:ctx.config["evaluation"]["numerical_samples"]], 1):
            example = ctx.model / f"examples/input/sample_{index}.flac"
            example.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ctx.dataset / row["audio"], example)
            files.append((example, f"examples/input/sample_{index}.flac"))
        files.extend((path, "models/" + path.name) for path in artifacts.values())
        files.append((contract_path, "models/runtime_contract.json"))
        for name in ("compile.sh", "evaluate.sh", "board/decoder.py", "board/shared.py", "board/evaluate.py"):
            files.append((ctx.model / "deploy" / name, name))
        for name in ("mel_filters_f32.bin", "decode_config.txt", "whisper/assets/multilingual.tiktoken"):
            files.append((ctx.assets / name, "models/" + name))
        for path in (ctx.assets / "vendor").rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                files.append((path, "board/" + path.relative_to(ctx.assets / "vendor").as_posix()))
        config = work / "config.json"
        config.write_text(json.dumps(board_configuration(ctx)))
        repo = ctx.model.parents[3]
        for module in ("evaluation.py", "files.py", "output_metrics.py"):
            files.append((repo / "tools" / module, "board/tools/" + module))
        files.append((config, "config.json"))
    board = ctx.platform["board"]
    with connection(ctx.platform) as client, client.open_sftp() as sftp:
        upload(sftp, board["datasets_root"], [(ctx.dataset / row["audio"],
               ctx.config["board"]["dataset_directory"] + "/" + row["audio"]) for row in selected])
        upload(sftp, str(PurePosixPath(board["models_root"]) / ctx.config["board"]["deploy_directory"]), files)
    print(f"[上传完成] FP32 WER {reference['fp32_wer']:.4%},样本 {len(selected)}.")


if __name__ == "__main__":
    main()
