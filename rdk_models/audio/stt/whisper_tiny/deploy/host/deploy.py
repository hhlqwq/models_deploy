"""生成同协议浮点参考,发布 S100 Whisper 运行包和必要样本."""

import argparse
import json
from pathlib import Path, PurePosixPath
import posixpath
import sys

import numpy as np
import paramiko

from context import context
from compile import session
from tools.evaluation import fixed_subset, SPEECH_EXAMPLES

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import assets, fingerprint, log_mel, read_audio, token_bytes, transcribe, word_errors


def select_samples(ctx, subset):
    """加载官方转录,子集排除全部校准音频并固定间隔抽取."""
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
    if not subset:
        return [samples[key] for key in sorted(samples)]
    count = ctx.config["quantization"]["calibration_samples"]
    calibration = {p.stem for p in sorted(ctx.dataset.rglob("*.flac"))[:count]}
    candidates = sorted(set(samples) - calibration)
    selected = fixed_subset(candidates, ctx.config["evaluation"]["compile_samples"], required=SPEECH_EXAMPLES)
    return [samples[key] for key in selected]


def connection(ctx):
    """使用仓库本地 S100 配置及固定主机密钥建立连接."""
    board = ctx.platform["board"]
    client = paramiko.SSHClient()
    client.load_host_keys(str(Path(__file__).resolve().parents[6] / "configs/platforms/s100.local.known_hosts"))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(hostname=board["host"], username=board["user"], password=board.get("password"), timeout=15)
    return client


def upload(sftp, destination, files):
    """按明确清单上传所需文件,不遍历目标目录删除内容."""
    for index, (source, name) in enumerate(files, 1):
        target = posixpath.join(destination, name)
        parent = "/"
        for part in PurePosixPath(target).parent.parts[1:]:
            parent = posixpath.join(parent, part)
            try:
                sftp.stat(parent)
            except FileNotFoundError:
                sftp.mkdir(parent)
        sftp.put(str(source), target)
        if index % 25 == 0 or index == len(files) or len(files) <= 20:
            print(f"[上传] {index}/{len(files)} {name}", flush=True)


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
    reference.update(fp32_wer=errors / words, fp32_word_errors=errors, reference_words=words)
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
    reference.update(precision=precision, hbm=hbm, precision_audit=audit,
                     artifacts={name: path.name for name, path in artifacts.items()})
    report = work / "reference.json"
    report.write_text(json.dumps(reference, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    files = [(report, f"build/reference/{stage}.json")]
    if stage == "subset":
        numeric = work / "encoder_reference.npz"
        files.append((numeric, "build/reference/encoder_reference.npz"))
        files.extend((path, path.name) for path in artifacts.values())
        for name in ("compile.sh", "evaluate.sh", "decoder.py", "shared.py", "board/evaluate.py"):
            files.append((ctx.model / "deploy" / name, name))
        for name in ("mel_filters_f32.bin", "decode_config.txt", "whisper/assets/multilingual.tiktoken"):
            files.append((ctx.assets / name, "build/reference/" + name))
        for path in (ctx.assets / "vendor").rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                files.append((path, "build/reference/vendor/" + path.relative_to(ctx.assets / "vendor").as_posix()))
        config = work / "config.json"
        config.write_text(json.dumps({"dataset": str(PurePosixPath(ctx.platform["board"]["datasets_root"]) /
            ctx.config["evaluation"]["dataset"]), "model": "whisper_tiny", "platform": "S100 / nash-e"}))
        files.append((config, "config.json"))
    board = ctx.platform["board"]
    with connection(ctx) as client, client.open_sftp() as sftp:
        upload(sftp, board["datasets_root"], [(ctx.dataset / row["audio"],
               ctx.config["evaluation"]["dataset"] + "/" + row["audio"]) for row in selected])
        upload(sftp, str(PurePosixPath(board["models_root"]) / ctx.config["board"]["deploy_directory"]), files)
    print(f"[上传完成] FP32 WER {reference['fp32_wer']:.4%},样本 {len(selected)}.")


if __name__ == "__main__":
    main()
