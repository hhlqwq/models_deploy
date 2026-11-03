"""使用 Python 执行 YAMNet 音频准备、量化与 NPU 窗口推理."""

from argparse import Namespace
import csv
import json
import time

import numpy as np

from full_accuracy_board import prepare
from tools.board import context
from tools.evaluation import is_test
from tools.runtime import Runtime, peak_rss_kib, quantize
from tools.output_metrics import OutputComparison


def read_quantization(row):
    """读取转换器生成的实际 FP16、INT8 或 INT16 I/O 契约."""
    scale, zero = float(row[0]), int(row[1])
    precision = row[2] if len(row) == 3 else "int8"
    dtype = {"int8": np.int8, "w8a16": np.int16, "fp16": np.float16}[precision]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("量化 scale 必须为有限正数.")
    return scale, zero, dtype


def main():
    """复用常驻模型处理缓存窗口,评价阶段保存逐窗口耗时."""
    ctx = context(__file__, new_run=True)
    if ctx.result.exists():
        raise FileExistsError(ctx.result)
    precision = ctx.config["conversion"]["precision"]
    config = ctx.root / "build/reference" / f"runtime_config_{precision}.csv"
    if not config.is_file() and precision in ("int8", "fp16"):
        config = ctx.root / "build/reference/runtime_config.csv"
    with config.open() as stream:
        rows = list(csv.reader(stream))
    if len(rows) != 2:
        raise ValueError("YAMNet 必须有输入和输出两行量化配置.")
    input_scale, input_zero, input_dtype = read_quantization(rows[0])
    output_scale, output_zero, output_dtype = read_quantization(rows[1])
    prepare(Namespace(dataset=ctx.dataset, work=ctx.work))
    counts = [int(value) for value in (ctx.work / "patch_counts.txt").read_text().split()]
    audio_manifest = json.loads((ctx.work / "audio_manifest.json").read_text())
    timing_rows, clip_rows = [], []
    comparison = OutputComparison(ctx)
    with Runtime(ctx.root / f"model_{precision}.dla") as runtime:
        if tuple(map(len, runtime.sizes)) != (1, 1):
            raise ValueError("YAMNet 要求单输入单输出.")
        inputs, view = runtime.tensor(0, (1, 1, 96, 64), input_dtype, fill=input_zero)
        outputs, scores = runtime.tensor(0, (1, 521), output_dtype, output=True)
        with (ctx.work / "features.bin").open("rb") as features, (ctx.work / "scores.bin").open("wb") as results:
            for clip, count in enumerate(counts):
                if count <= 0:
                    raise ValueError("音频窗口数必须为正.")
                started, warmup, comparison_seconds = time.perf_counter(), 0, 0
                average = np.zeros(521, np.float32)
                for patch in range(count):
                    values = np.fromfile(features, dtype="<f4", count=6144)
                    if values.size != 6144 or not np.isfinite(values).all():
                        raise ValueError("音频缓存不足或含无效数值.")
                    tensor = (values.astype(input_dtype) if input_dtype == np.float16 else
                              quantize(values, input_scale, input_zero, input_dtype))
                    floating_input = values.reshape(1, 1, 96, 64)
                    np.copyto(view, tensor.reshape(view.shape))
                    if clip == 0 and patch == 0 and not is_test():
                        tick = time.perf_counter()
                        for _ in range(20):
                            runtime.infer([inputs], [outputs])
                        warmup = time.perf_counter() - tick
                    milliseconds = runtime.infer([inputs], [outputs])
                    values = scores.reshape(521).astype(np.float32)
                    if output_dtype != np.float16:
                        values = (values - output_zero) * output_scale
                    average += values / count
                    comparison_started = time.perf_counter()
                    comparison.update([floating_input], {"output": values.reshape(1, 521)},
                        sample_id=f"{audio_manifest[clip]['filename']}:window:{patch}",
                        sample_files=[ctx.dataset / "audio" / audio_manifest[clip]["filename"]])
                    comparison_seconds += time.perf_counter() - comparison_started
                    if not is_test():
                        timing_rows.append((clip, patch, milliseconds))
                average.astype("<f4").tofile(results)
                if not is_test():
                    clip_rows.append((clip, count, (
                        time.perf_counter() - started - warmup - comparison_seconds) * 1000))
                if (clip + 1) % 50 == 0 or clip + 1 == len(counts):
                    print(f"[推理] YAMNet {clip + 1}/{len(counts)}", flush=True)
            if features.read(1):
                raise ValueError("音频缓存含清单以外的数据.")
    if not is_test():
        for name, header, rows in (("timings.csv", ("clip", "patch", "npu_ms"), timing_rows),
                                   ("clip_times.csv", ("clip", "patches", "inference_stage_ms"), clip_rows)):
            with (ctx.work / name).open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(rows)
        (ctx.work / "peak_rss_kib.txt").write_text(str(peak_rss_kib()))
    comparison.finish()


if __name__ == "__main__":
    main()
