"""在 Python 中完成 Whisper 音频前处理与双 DLA 贪心解码."""

import json
import subprocess
import time

import numpy as np

from tools.board import context
from tools.evaluation import is_test, select_samples, SPEECH_EXAMPLES
from tools.runtime import Runtime, peak_rss_kib, quantize
from tools.output_metrics import OutputComparison
from decoder import (CACHE_NAMES, INPUT_SHAPES, OUTPUT_SHAPES, MAX_TOKEN_LENGTH,
                     feeds_for_step, next_token)


def load_samples(root):
    """核对 LibriSpeech test-clean 转录和音频,按固定规则选取样本."""
    samples = []
    for transcript in sorted(root.rglob("*.trans.txt")):
        for line in transcript.read_text().splitlines():
            sample_id, text = line.split(" ", 1)
            audio = transcript.parent / f"{sample_id}.flac"
            if not audio.is_file():
                raise FileNotFoundError(audio)
            samples.append({"sample_id": sample_id, "audio_path": str(audio),
                            "reference_text": text})
    if len(samples) != 2620 or len({row["sample_id"] for row in samples}) != 2620:
        raise ValueError("LibriSpeech test-clean 必须包含 2620 条唯一音频.")
    return select_samples(samples, "whisper_tiny", key=lambda row: row["sample_id"],
                          required=SPEECH_EXAMPLES)


def preprocess(path, filters):
    """沿用 PCM16 解码、30 秒截补齐、400 点 Hann 与 80 维 Mel 协议."""
    started = time.perf_counter()
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1",
                          "-ar", "16000", "-f", "s16le", "pipe:1"],
                         check=True, stdout=subprocess.PIPE).stdout
    values = np.frombuffer(pcm, dtype="<i2")[:480000].astype(np.float32) / 32768
    if not values.size:
        raise ValueError(f"音频为空: {path}.")
    duration = len(values) / 16000
    audio = np.pad(values, (0, 480000 - len(values)))
    decoded = time.perf_counter()
    framed = np.lib.stride_tricks.sliding_window_view(np.pad(audio, (200, 200), mode="reflect"), 400)[::160][:3000]
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)).astype(np.float32)
    spectrum = np.fft.rfft(framed * window, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32).T
    mel = np.log10(np.maximum(filters @ power, 1e-10))
    mel = (np.maximum(mel, mel.max() - 8) + 4) / 4
    return mel.astype(np.float32), duration, {"audio_decode_ms": (decoded - started) * 1000,
        "log_mel_ms": (time.perf_counter() - decoded) * 1000,
        "mel_write_ms": 0.0, "cache_hit": False}


class WhisperRuntime:
    """按实际 INT16 契约运行 W8A16 双图,使用 1t256c 滑动缓存."""

    def __init__(self, encoder, decoder, config, contract):
        """逐张量绑定逻辑视图,保留 SDK 要求的原生补齐存储."""
        self.encoder, self.decoder, self.config = encoder, decoder, config
        if (contract.get("precision"), contract.get("cacheSize"),
                contract.get("maxTokenLength")) != ("w8a16", 256, 448):
            raise ValueError("Whisper 需要重新编译 W8A16 / 1t256c / 448 产物.")
        self.contract = contract
        self.buffers, self.views = {}, {}
        for stage, runtime in (("encoder", encoder), ("decoder_step", decoder)):
            for output, kind in ((False, "inputs"), (True, "outputs")):
                items = contract[stage][kind]
                if len(runtime.sizes[int(output)]) != len(items):
                    raise ValueError(f"Whisper {stage} {kind} 数量不符.")
                buffers, views = [], {}
                for index, item in enumerate(items):
                    if item["dtype"] != "int16" or item["index"] != index:
                        raise ValueError(f"Whisper 张量类型或索引不符: {item}.")
                    try:
                        storage, view = runtime.tensor(index, item["shape"], np.int16,
                            output=output, fill=item["zero_point"])
                    except ValueError as error:
                        raise ValueError(f"Whisper {stage} {kind}[{index}] "
                                         f"{item['name']}: {error}") from error
                    buffers.append(storage)
                    views[item["name"]] = view
                self.buffers[stage, kind], self.views[stage, kind] = buffers, views
        if ({item["name"]: tuple(item["shape"]) for item in contract["decoder_step"]["inputs"]}
                != INPUT_SHAPES or
                {item["name"]: tuple(item["shape"]) for item in contract["decoder_step"]["outputs"]}
                != OUTPUT_SHAPES):
            raise ValueError("Whisper Decoder 逻辑形状与 1t256c 契约不一致.")
        self.initial = [int(value) for value in config["initial"].split(",")]
        self.limit = int(config["max_generated_tokens"])
        if (int(config["cacheSize"]), int(config["maxTokenLength"])) != (256, 448):
            raise ValueError("Whisper 解码配置与模型契约不一致.")
        if not self.initial or not 0 < self.limit <= 448 - len(self.initial):
            raise ValueError("Whisper 生成上限与提示词长度不符.")

    def run(self, stage, runtime, feeds):
        """量化逻辑输入并反量化逻辑输出,跨图和 Cache 不复用错误尺度."""
        for item in self.contract[stage]["inputs"]:
            np.copyto(self.views[stage, "inputs"][item["name"]], quantize(
                feeds[item["name"]], item["scale"], item["zero_point"], np.int16))
        elapsed = runtime.infer(self.buffers[stage, "inputs"], self.buffers[stage, "outputs"])
        outputs = {item["name"]:
                   (self.views[stage, "outputs"][item["name"]].astype(np.float32)
                    - item["zero_point"]) * item["scale"]
                   for item in self.contract[stage]["outputs"]}
        return outputs, elapsed

    def decode(self, mel):
        """使用 448 个绝对位置和 256 槽滑动窗口,只记录实际 SDK 耗时."""
        features, encoder_ms = self.run("encoder", self.encoder, {"mel": mel[None]})
        self.encoder_features = features["audio_features"]
        caches = {name: np.zeros(INPUT_SHAPES[name], np.float32) for name in CACHE_NAMES}
        current, tokens, times, reached, first_ms = self.initial[0], [], [], False, 0.0
        position = -1
        for position in range(MAX_TOKEN_LENGTH):
            feeds = feeds_for_step(current, position, self.encoder_features, caches)
            outputs, elapsed = self.run("decoder_step", self.decoder, feeds)
            times.append(elapsed)
            caches = {name: outputs[name.removesuffix("_in") + "_out"] for name in CACHE_NAMES}
            if position + 1 < len(self.initial):
                current = self.initial[position + 1]
                continue
            if not tokens:
                first_ms = encoder_ms + sum(times)
            current = next_token(outputs["logits"], self.config, first=not tokens)
            if current == int(self.config["eot"]):
                reached = True
                break
            tokens.append(current)
            if len(tokens) >= self.limit:
                break
            if is_test() and (position + 1) % 64 == 0:
                print(f"[解码] {position + 1}/{MAX_TOKEN_LENGTH} 步.", flush=True)
        row = {"tokens": tokens, "reached_eot": reached, "status": "ok",
               "stop_reason": "eot" if reached else "max_token_length",
               "precision": "W8A16", "cacheSize": 256, "maxTokenLength": 448,
               "backend": "python_neuron_runtime_hw"}
        if not is_test():
            total = encoder_ms + sum(times)
            row.update(encoder_ms=encoder_ms, first_token_ms=first_ms,
                decoder_calls=len(times), decoder_total_ms=sum(times), npu_total_ms=total,
                tokens_per_second=len(tokens) * 1000 / total, peak_rss_kb=peak_rss_kib())
        return row


def main():
    """音频逐条在内存准备,保存唯一预测流及评分需要的来源清单."""
    ctx = context(__file__, new_run=True)
    if ctx.result.exists():
        raise FileExistsError(ctx.result)
    samples = load_samples(ctx.dataset)
    filters = np.fromfile(ctx.root / "models/mel_filters_f32.bin", dtype="<f4")
    if filters.size != 80 * 201:
        raise ValueError("Whisper Mel 滤波器大小不符.")
    filters = filters.reshape(80, 201)
    config = dict(line.split("=", 1) for line in (ctx.root / "models/decode_config.txt").read_text().splitlines() if "=" in line)
    cache = ctx.work / "cache/librispeech"
    cache.mkdir(parents=True)
    preprocessing = []
    comparison = OutputComparison(ctx, scope="encoder_output_compile_subset; excludes_autoregressive_decoder")
    contract_path = ctx.root / "models/runtime_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError("缺少 runtime_contract.json,请重新运行 Whisper 编译入口生成 W8A16 双图.")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    with Runtime(ctx.root / "models" / contract["encoder"]["file"]) as encoder, Runtime(
            ctx.root / "models" / contract["decoder_step"]["file"]) as decoder:
        model = WhisperRuntime(encoder, decoder, config, contract)
        with (cache / "source_manifest.jsonl").open("w") as sources, (ctx.work / "board_predictions.jsonl").open("w") as predictions:
            for count, sample in enumerate(samples, 1):
                mel, duration, timing = preprocess(sample["audio_path"], filters)
                row = model.decode(mel)
                comparison.update([mel[None]], {"audio_features": model.encoder_features},
                    sample_id=f"{sample['sample_id']}:encoder", sample_files=[sample["audio_path"]])
                row.update(sample_id=sample["sample_id"], duration_seconds=duration)
                if not is_test():
                    row["npu_rtf"] = row["npu_total_ms"] / (duration * 1000)
                predictions.write(json.dumps(row) + "\n")
                sources.write(json.dumps(dict(sample, duration_seconds=duration)) + "\n")
                if not is_test():
                    preprocessing.append(dict(timing, sample_id=sample["sample_id"], duration_seconds=duration))
                if count % 25 == 0 or count == len(samples):
                    print(f"[推理] Whisper {count}/{len(samples)}", flush=True)

    if not is_test():
        (cache / "preprocess_metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in preprocessing))
    comparison.finish()


if __name__ == "__main__":
    main()
