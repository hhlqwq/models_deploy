"""使用 S100 官方 HBM Python 接口完成 Whisper 转写和同协议 WER."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np
from hbm_runtime import HB_HBMRuntime

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "build/reference"
sys.path.insert(0, str(ASSETS / "vendor"))
from decoder import INPUT_SHAPES, OUTPUT_SHAPES
from shared import assets, fingerprint, log_mel, read_audio, token_bytes, transcribe, word_errors
from whisper.normalizers.english import EnglishTextNormalizer


class Runtime:
    """验证公开张量契约,统计同步 HBM 调用耗时."""

    def __init__(self, path, inputs, outputs):
        """加载单图 HBM 并核对名称、形状和数据类型."""
        self.model = HB_HBMRuntime(str(path))
        self.name, = self.model.model_names
        self.inputs = inputs
        self.outputs = outputs
        self.elapsed = 0.0
        for expected, names, shapes, dtypes in (
                (inputs, self.model.input_names, self.model.input_shapes, self.model.input_dtypes),
                (outputs, self.model.output_names, self.model.output_shapes, self.model.output_dtypes)):
            actual = {name: tuple(shapes[self.name][name]) for name in names[self.name]}
            types = {name: dtypes[self.name][name].name for name in names[self.name]}
            if actual != expected or set(types.values()) != {"F32"}:
                raise ValueError(f"HBM 契约不符: {path.name}, shapes={actual}, dtype={types}.")

    def __call__(self, feeds):
        """同步推理,复制输出以保留下一步解码缓存."""
        inputs = {name: np.ascontiguousarray(feeds[name], dtype=np.float32) for name in self.inputs}
        started = perf_counter()
        outputs = self.model.run(inputs)[self.name]
        self.elapsed += perf_counter() - started
        result = {name: np.array(outputs[name], copy=True) for name in self.outputs}
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("HBM 输出包含非有限值.")
        return result


def main():
    """验证产物与样本绑定,完成约定范围的 WER 和少量数值对照."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    scope = "full" if args.full else "subset"
    config = json.loads((ROOT / "config.json").read_text())
    reference = json.loads((ASSETS / f"{scope}.json").read_text())
    if reference["scope"] != scope:
        raise ValueError("浮点参考范围不符.")
    paths = {key: ROOT / name for key, name in reference["artifacts"].items()}
    if {key: fingerprint(path) for key, path in paths.items()} != reference["hbm"]:
        raise ValueError("HBM 与本次浮点参考不匹配.")
    filters, rules = assets(ASSETS)
    ranks = token_bytes(ASSETS / "whisper/assets/multilingual.tiktoken")
    normalize = EnglishTextNormalizer()
    encoder = Runtime(paths["encoder"], {"mel": (1, 80, 3000)}, {"audio_features": (1, 1500, 384)})
    decoder = Runtime(paths["decoder_step"], INPUT_SHAPES, OUTPUT_SHAPES)
    numerical = {} if args.full else dict(np.load(ASSETS / "encoder_reference.npz", allow_pickle=False))
    errors = words = ended = calls = 0
    duration = seconds = 0.0
    comparisons, examples = [], []
    samples = reference["samples"]
    if len(samples) != (2620 if args.full else 50) or len({r["id"] for r in samples}) != len(samples):
        raise ValueError("评价样本数量或唯一性不符.")
    for index, row in enumerate(samples, 1):
        source = Path(config["dataset"]) / row["audio"]
        if fingerprint(source) != row["audio_sha256"]:
            raise ValueError(f"音频与浮点参考不同: {row['id']}.")
        started = perf_counter()
        pcm = read_audio(source, host=False)
        text, features, steps, reached = transcribe(encoder, decoder, log_mel(pcm, filters), rules, ranks)
        seconds += perf_counter() - started
        duration += min(len(pcm), 480000) / 16000
        error, count = word_errors(row["normalized_text"], normalize(text))
        errors += error
        words += count
        ended += reached
        calls += steps
        if row["id"] in numerical:
            expected = numerical[row["id"]].astype(np.float64).ravel()
            actual = features.astype(np.float64).ravel()
            rmse = float(np.sqrt(np.mean((actual - expected) ** 2)))
            comparisons.append({"id": row["id"], "cosine_similarity": float(np.dot(actual, expected) /
                (np.linalg.norm(actual) * np.linalg.norm(expected))), "rmse": rmse,
                "nrmse_percent": 100 * rmse / float(np.sqrt(np.mean(expected ** 2)))})
            examples.append({"id": row["id"], "reference": row["text"], "fp32": row["fp32_text"], "bpu": text})
        if index % 5 == 0 or index == len(samples):
            print(f"[S100 Whisper] {index}/{len(samples)}, WER {errors / words:.4%}, EOT {ended}/{index}", flush=True)
    if words != reference["reference_words"] or len(comparisons) != len(numerical):
        raise ValueError("评分文本或数值对照覆盖不完整.")
    wer = errors / words
    print(f"[任务精度] FP32 WER {reference['fp32_wer']:.4%}, BPU WER {wer:.4%},"
          f"损失 {100 * (wer - reference['fp32_wer']):+.4f} 个百分点.", flush=True)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "model": "whisper_tiny", "platform": config["platform"], "evaluation_scope": scope,
              "status": "complete", "samples": len(samples), "sample_unit": "audio",
              "sample_ids": [row["id"] for row in samples], "hbm": reference["hbm"],
              "precision_requested": reference["precision"], "precision_audit": reference["precision_audit"],
              "protocol": {"dataset": "LibriSpeech test-clean", "decoding": "greedy",
                           "cache_size": 256, "max_token_length": 448,
                           "normalization": "OpenAI Whisper EnglishTextNormalizer v20250625",
                           "aggregation": "corpus_word_edit_distance"},
              "accuracy": {"WER": {"fp32_accuracy": reference["fp32_wer"],
                  "npu_accuracy": wer, "higher_is_better": False,
                  "accuracy_loss": wer - reference["fp32_wer"],
                  "accuracy_loss_percentage_points": 100 * (wer - reference["fp32_wer"])}},
              "reference_words": words, "word_errors": errors, "reached_eot": ended,
              "decoder_calls": calls, "examples": examples,
              "output_errors": {"compared_calls": len(comparisons),
                  "reference_backend": "onnx_cpu", "output": "audio_features",
                  "comparison_basis": "deployment_output_including_preprocessing",
                  "nrmse_definition": "100 * RMSE / RMS(reference)", "per_sample": comparisons},
              "missing": [] if args.full else ["未执行全量2620条、性能和峰值RSS评价"]}
    if args.full:
        report.update(encoder_hbm_call_mean_ms=1000 * encoder.elapsed / len(samples),
                      decoder_hbm_call_mean_ms=1000 * decoder.elapsed / calls,
                      end_to_end_rtf=seconds / duration, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    out = ROOT / "results" / ("summary.json" if args.full else "summary_test.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(out)
    if examples:
        example = ROOT / "examples/output/transcripts.json"
        example.parent.mkdir(parents=True, exist_ok=True)
        example.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[报告] {out}", flush=True)


if __name__ == "__main__":
    main()
