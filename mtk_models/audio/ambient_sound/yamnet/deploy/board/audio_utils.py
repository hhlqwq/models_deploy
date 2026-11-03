"""YAMNet 音频前处理与 ESC-50 标签投影,主机和板端共用."""

import csv
from pathlib import Path
import subprocess

import numpy as np


# 按 ESC-50 target 顺序映射到 YAMNet 标签,多标签取最大分数.
CLASS_NAMES = [
    ["Dog"], ["Crowing, cock-a-doodle-doo"], ["Pig"], ["Cattle, bovinae"], ["Frog"],
    ["Cat"], ["Cluck"], ["Insect"], ["Sheep"], ["Crow"],
    ["Rain"], ["Waves, surf"], ["Fire"], ["Cricket"], ["Chirp, tweet"],
    ["Drip"], ["Wind"], ["Pour"], ["Toilet flush"], ["Thunder"],
    ["Baby cry, infant cry"], ["Sneeze"], ["Clapping"], ["Breathing"], ["Cough"],
    ["Walk, footsteps"], ["Laughter"], ["Toothbrush"], ["Snoring"], [],
    ["Knock"], ["Mouse"], ["Computer keyboard"], ["Creak"], [],
    [], ["Vacuum cleaner"], ["Alarm clock"], ["Tick-tock"], ["Shatter"],
    ["Helicopter"], ["Chainsaw"], ["Siren"], ["Vehicle horn, car horn, honking"],
    ["Engine"], ["Train"], ["Church bell"], ["Aircraft"], ["Fireworks"], ["Sawing"],
]


def load_records(dataset):
    """验证 ESC-50 全量元数据和音频覆盖,按文件名排序."""
    with (Path(dataset) / "meta/esc50.csv").open(encoding="utf-8") as stream:
        records = sorted(csv.DictReader(stream), key=lambda row: row["filename"])
    if len(records) != 2000 or len({row["filename"] for row in records}) != 2000:
        raise ValueError("要求 ESC-50 全部 2000 条唯一音频.")
    for row in records:
        if not (Path(dataset) / "audio" / row["filename"]).is_file():
            raise FileNotFoundError(row["filename"])
    return records


def load_waveform(path):
    """通过同一 ffmpeg 解码并重采样为 16 kHz 单声道 float32."""
    raw = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
        "-f", "f32le", "pipe:1"], check=True, stdout=subprocess.PIPE).stdout
    values = np.frombuffer(raw, dtype="<f4")
    if not values.size or not np.isfinite(values).all():
        raise ValueError(f"音频为空或含非有限数: {path}.")
    return values


def mel_matrix():
    """复现 TensorFlow 64 维三角 Mel 滤波器,排除直流频率."""
    frequencies = np.linspace(0, 8000, 257)
    mel = 1127.0 * np.log1p(frequencies / 700.0)
    edges = np.linspace(1127.0 * np.log1p(125.0 / 700.0),
                        1127.0 * np.log1p(7500.0 / 700.0), 66)
    lower = (mel[:, None] - edges[:-2]) / (edges[1:-1] - edges[:-2])
    upper = (edges[2:] - mel[:, None]) / (edges[2:] - edges[1:-1])
    matrix = np.maximum(0, np.minimum(lower, upper)).astype(np.float32)
    matrix[0] = 0
    return matrix


def preprocess(waveform):
    """使用 400 点周期 Hann、512 点 FFT、160 点帧移生成 96x64 窗口."""
    minimum = 15600
    padding = max(0, minimum - len(waveform))
    complete_hops = int(np.ceil(max(0, len(waveform) - minimum) / 7680))
    target = minimum + complete_hops * 7680
    padded = np.pad(waveform, (0, max(padding, target - len(waveform))))
    frames = np.lib.stride_tricks.sliding_window_view(padded, 400)[::160]
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, n=512)).astype(np.float32)
    features = np.log(spectrum @ mel_matrix() + 0.001)
    patches = np.stack([features[start:start + 96]
                        for start in range(0, len(features) - 95, 48)])
    return np.ascontiguousarray(patches[:, None], dtype=np.float32)


def mapping(class_map):
    """严格查找固定标签名称,防止标签索引和类别语义静默错位."""
    with Path(class_map).open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 521 or [int(row["index"]) for row in rows] != list(range(521)):
        raise ValueError("YAMNet 标签必须按顺序覆盖 521 类.")
    names = {row["display_name"]: int(row["index"]) for row in rows}
    return [[names[name] for name in group] for group in CLASS_NAMES]


def stream_patches(chunks):
    """验证连续输入的窗口缓存,保留重叠样本并在结束时补齐最后一窗."""
    buffer = np.empty(0, dtype=np.float32)
    produced = 0
    for chunk in chunks:
        buffer = np.concatenate((buffer, chunk))
        while len(buffer) >= 15600:
            yield preprocess(buffer[:15600])[0]
            produced += 1
            buffer = buffer[7680:]
    if not produced or len(buffer) > 7920:
        for patch in preprocess(buffer):
            yield patch


def project_scores(scores, indices):
    """将原始 521 类分数投影到预先固定的 ESC-50 任务类别."""
    return np.stack([np.max(scores[:, group], axis=1) if group else
                     np.full(len(scores), -np.inf) for group in indices], axis=1)


def metrics(scores, records, indices):
    """计算固定映射下的 Top-1 与宏平均 AP,单独报告未校准折."""
    projected = project_scores(scores, indices)
    targets = np.array([int(row["target"]) for row in records])
    held_out = np.array([int(row["fold"]) != 1 for row in records])
    supported = np.array([bool(indices[target]) for target in targets])
    output = {}
    for name, mask in (("all", np.ones(len(records), dtype=bool)),
                       ("held_out", held_out)):
        mask = mask & supported
        subset, labels = projected[mask], targets[mask]
        aps = []
        for category in range(50):
            if not indices[category]:
                continue
            order = np.argsort(-subset[:, category], kind="stable")
            truth = (labels[order] == category).astype(np.float64)
            # 在同分数组末尾计算 precision,避免量化并列分数导致乐观 AP.
            ends = np.r_[np.where(np.diff(subset[order, category]))[0], len(order) - 1]
            positives = np.cumsum(truth)[ends]
            precision = positives / (ends + 1)
            increments = np.diff(np.r_[0, positives])
            aps.append(float(np.sum(precision * increments) / truth.sum()))
        output[name] = {"samples": int(mask.sum()),
                        "top1": float(np.mean(np.argmax(subset, axis=1) == labels)),
                        "map": float(np.mean(aps))}
    return output


def evaluation_ids(records, indices):
    """返回实际参与未校准折宏平均 AP 的音频标识,排除无映射类别."""
    return [row["filename"] for row in records
            if int(row["fold"]) != 1 and indices[int(row["target"])]]
