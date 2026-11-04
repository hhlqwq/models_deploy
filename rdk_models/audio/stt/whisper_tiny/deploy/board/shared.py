"""S100 Whisper 共用音频、静态解码与数据选择协议."""

import base64
import subprocess

import numpy as np

from decoder import CACHE_NAMES, INPUT_SHAPES, MAX_TOKEN_LENGTH, feeds_for_step, next_token


def read_audio(path, *, host):
    """读取真实单声道 PCM16;主机 PyAV,板端 FFmpeg."""
    if host:
        import av

        chunks = []
        with av.open(str(path)) as container:
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            for frame in container.decode(audio=0):
                chunks.extend(item.to_ndarray().reshape(-1) for item in resampler.resample(frame))
            chunks.extend(item.to_ndarray().reshape(-1) for item in resampler.resample(None))
        values = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int16)
    else:
        pcm = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1",
                                       "-ar", "16000", "-f", "s16le", "pipe:1"])
        values = np.frombuffer(pcm, dtype="<i2")
    if not values.size:
        raise ValueError(f"音频为空: {path}.")
    return values


def log_mel(values, filters):
    """按 Whisper 80 Mel、30 秒截补齐协议计算固定输入."""
    audio = values[:480000].astype(np.float32) / 32768
    audio = np.pad(audio, (0, 480000 - len(audio)))
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(audio, (200, 200), mode="reflect"), 400)[::160][:3000]
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)).astype(np.float32)
    power = (np.abs(np.fft.rfft(frames * window, axis=1)) ** 2).astype(np.float32).T
    mel = np.log10(np.maximum(filters @ power, 1e-10))
    return ((np.maximum(mel, mel.max() - 8) + 4) / 4).astype(np.float32)[None]


def assets(path):
    """读取与已有 FP32 图配套的官方滤波器和解码规则."""
    filters = np.fromfile(path / "mel_filters_f32.bin", dtype="<f4").reshape(80, 201)
    config = dict(line.split("=", 1) for line in (path / "decode_config.txt").read_text().splitlines())
    if (int(config["cacheSize"]), int(config["maxTokenLength"])) != (256, 448):
        raise ValueError("Whisper 解码配置必须为 1t256c,上限 448.")
    return filters, config


def decode_steps(run, features, config, limit=MAX_TOKEN_LENGTH):
    """逐步生成真实自回归输入和输出,供校准及推理共用."""
    initial = [int(value) for value in config["initial"].split(",")]
    caches = {name: np.zeros(INPUT_SHAPES[name], np.float32) for name in CACHE_NAMES}
    maximum = int(config["max_generated_tokens"])
    if not initial or not 0 < maximum <= MAX_TOKEN_LENGTH - len(initial):
        raise ValueError("Whisper 提示词与生成长度配置不符.")
    token = initial[0]
    generated_count = 0
    for position in range(min(limit, MAX_TOKEN_LENGTH)):
        feeds = feeds_for_step(token, position, features, caches)
        outputs = run(feeds)
        caches = {name: outputs[name.removesuffix("_in") + "_out"] for name in CACHE_NAMES}
        generated = None
        if position + 1 < len(initial):
            token = initial[position + 1]
        else:
            token = next_token(outputs["logits"], config, first=position + 1 == len(initial))
            generated = token
            generated_count += 1
        yield feeds, generated
        if generated == int(config["eot"]) or generated_count >= maximum:
            break


def token_bytes(path):
    """从同一官方 BPE 表读取 Token 字节,用于无依赖的文本解码."""
    ranks = {}
    for line in path.read_bytes().splitlines():
        value, rank = line.split()
        ranks[int(rank)] = base64.b64decode(value)
    return ranks


def transcribe(encoder, decoder, mel, rules, ranks):
    """运行完整贪心解码并返回文本、特征与解码步数."""
    features = encoder({"mel": mel})["audio_features"]
    tokens = []
    reached = False
    steps = 0
    for _, token in decode_steps(decoder, features, rules):
        steps += 1
        if token == int(rules["eot"]):
            reached = True
            break
        if token is not None:
            tokens.append(token)
    text = b"".join(ranks[token] for token in tokens if token in ranks).decode("utf-8", errors="replace")
    return text, features, steps, reached


def word_errors(reference, prediction):
    """计算单条词级编辑距离和参考词数,供语料级 WER 汇总."""
    words = reference.split()
    result = list(range(len(words) + 1))
    for index, word in enumerate(prediction.split(), 1):
        row = [index]
        for position, expected in enumerate(words, 1):
            row.append(min(result[position] + 1, row[-1] + 1,
                           result[position - 1] + (expected != word)))
        result = row
    return result[-1], len(words)


def fingerprint(path):
    """以流式摘要绑定样本、浮点图和实际 HBM."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
