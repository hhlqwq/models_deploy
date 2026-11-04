"""通过 S100 官方 Runtime 推理,保存过程记录并独立汇总同协议精度与性能."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
from time import perf_counter

import numpy as np

from decoder import INPUT_SHAPES, OUTPUT_SHAPES
from shared import assets, fingerprint, log_mel, read_audio, token_bytes, transcribe, word_errors
from tools.evaluation import accuracy_result, compare_accuracy, read_accuracy, record_selection
from tools.output_metrics import Moments


ROOT = Path(__file__).resolve().parents[1]


class Runtime:
    """验证公开张量契约,保存每次同步 HBM 调用的耗时."""

    def __init__(self, path, inputs, outputs):
        """加载 HBM 并检查公开输入输出的名称、形状和 FP32 类型."""
        from hbm_runtime import HB_HBMRuntime

        self.model = HB_HBMRuntime(str(path))
        self.name, = self.model.model_names
        self.inputs = inputs
        self.outputs = outputs
        self.times = []
        for expected, names, shapes, dtypes in (
                (inputs, self.model.input_names, self.model.input_shapes, self.model.input_dtypes),
                (outputs, self.model.output_names, self.model.output_shapes, self.model.output_dtypes)):
            actual = {name: tuple(shapes[self.name][name]) for name in names[self.name]}
            types = {name: dtypes[self.name][name].name for name in names[self.name]}
            if actual != expected or set(types.values()) != {"F32"}:
                raise ValueError(f"HBM 契约不符: {path.name}, shapes={actual}, dtype={types}.")

    def __call__(self, feeds):
        """同步执行一次调用,复制后续解码需要的输出缓存."""
        inputs = {name: np.ascontiguousarray(feeds[name], dtype=np.float32) for name in self.inputs}
        started = perf_counter()
        outputs = self.model.run(inputs)[self.name]
        self.times.append(1000 * (perf_counter() - started))
        result = {name: np.array(outputs[name], copy=True) for name in self.outputs}
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("HBM 输出包含非有限值.")
        return result


def load_reference(scope):
    """读取当前模型配置和同范围参考,核对实际模型包与统一精度接口."""
    os.environ["MTK_EVAL_SCOPE"] = scope
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    reference_path = ROOT / "build/reference" / f"{scope}.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if "evaluation" not in reference:
        raise ValueError("请先在编译服务器执行本模型入口,上传统一协议的浮点参考.")
    read_accuracy(reference, config["name"], "onnx")
    samples = reference["samples"]
    expected = 2620 if scope == "full" else config["evaluation"]["compile_samples"]
    if len(samples) != expected or len({row["id"] for row in samples}) != expected:
        raise ValueError("评价样本数量或唯一性不符.")
    selection = record_selection(row["id"] for row in samples)
    if selection != reference["evaluation"]["samples"]["selection_id"]:
        raise ValueError("参考样本清单与统一评价标识不一致.")
    models = ROOT / config["deployment"]["model_directory"]
    paths = {key: ROOT / name for key, name in reference["artifacts"].items()}
    if any(not path.resolve().is_relative_to(models.resolve()) for path in paths.values()):
        raise ValueError("HBM 必须位于本模型 models 目录.")
    if {key: fingerprint(path) for key, path in paths.items()} != reference["hbm"]:
        raise ValueError("HBM 与浮点参考不匹配.")
    contract = json.loads((models / "runtime_contract.json").read_text())
    if (contract["hbm"] != reference["hbm"] or contract["cacheSize"] != 256
            or contract["maxTokenLength"] != 448 or contract["public_dtype"] != "float32"):
        raise ValueError("Whisper 运行契约与当前参考不符.")
    for name, digest in reference["runtime_files"].items():
        path = ROOT / name
        if not path.resolve().is_relative_to(ROOT.resolve()) or fingerprint(path) != digest:
            raise ValueError(f"运行资源与浮点参考不符: {name}.")
    return config, reference, reference_path, models, paths


def infer(config, reference, reference_path, models, paths, scope):
    """逐条推理并保存恢复汇总所需的文本、性能和少量输出统计量."""
    work = ROOT / "build/work"
    if work.is_symlink() or not work.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("本模型过程目录越界或为符号链接.")
    numerical = {}
    if scope == "subset":
        selected = reference["numerical_sample_ids"]
        count = config["evaluation"]["numerical_samples"]
        if len(selected) != count or len(set(selected)) != count:
            raise ValueError("数值参考数量与模型配置不符.")
        with np.load(ROOT / "build/reference/encoder_reference.npz", allow_pickle=False) as arrays:
            if set(arrays.files) != set(selected):
                raise ValueError("数值参考未完整覆盖选定音频.")
            numerical = {name: arrays[name] for name in selected}
        if any(value.shape != (1, 1500, 384) or not np.isfinite(value).all()
               for value in numerical.values()):
            raise ValueError("数值参考的 Encoder 输出形状或数值不符.")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    run_id = os.environ.get("EVAL_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    manifest = {"run_id": run_id, "scope": scope, "reference_sha256": fingerprint(reference_path)}
    (work / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    filters, rules = assets(models)
    ranks = token_bytes(models / "whisper/assets/multilingual.tiktoken")
    encoder = Runtime(paths["encoder"], {"mel": (1, 80, 3000)}, {"audio_features": (1, 1500, 384)})
    decoder = Runtime(paths["decoder_step"], INPUT_SHAPES, OUTPUT_SHAPES)
    with (work / "board_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index, sample in enumerate(reference["samples"], 1):
            source = Path(config["board"]["dataset_dir"]) / sample["audio"]
            if fingerprint(source) != sample["audio_sha256"]:
                raise ValueError(f"音频与浮点参考不同: {sample['id']}.")
            decoder_start = len(decoder.times)
            started = perf_counter()
            pcm = read_audio(source, host=False)
            text, features, steps, reached = transcribe(encoder, decoder, log_mel(pcm, filters), rules, ranks)
            row = {"sample_id": sample["id"], "text": text, "reached_eot": reached,
                   "stop_reason": "eot" if reached else "max_token_length", "decoder_calls": steps}
            if scope == "full":
                decoder_times = decoder.times[decoder_start:]
                total = encoder.times[-1] + sum(decoder_times)
                duration = min(len(pcm), 480000) / 16000
                row.update(encoder_ms=encoder.times[-1], decoder_total_ms=sum(decoder_times),
                    first_token_ms=encoder.times[-1] + sum(decoder_times[:len(rules["initial"].split(","))]),
                    npu_total_ms=total, npu_rtf=total / (1000 * duration),
                    duration_seconds=duration, end_to_end_ms=1000 * (perf_counter() - started),
                    peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if sample["id"] in numerical:
                moments = Moments()
                moments.update(numerical[sample["id"]], features)
                # 保存充分统计量供独立汇总复用,不保存整张输出张量.
                row["output_statistics"] = vars(moments)
                row["output_errors"] = moments.result()
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            if index % 5 == 0 or index == len(reference["samples"]):
                print(f"[S100 Whisper] {index}/{len(reference['samples'])}", flush=True)
    print(f"[推理完成] {work}", flush=True)


def summarize(config, reference, reference_path, scope):
    """只读取同次过程文件评分并发布报告,不加载 HBM 或重新推理."""
    from whisper.normalizers.english import EnglishTextNormalizer

    work = ROOT / "build/work"
    manifest = json.loads((work / "run.json").read_text())
    if manifest["scope"] != scope or manifest["reference_sha256"] != fingerprint(reference_path):
        raise ValueError("过程数据与当前参考范围或版本不一致.")
    rows = [json.loads(line) for line in (work / "board_predictions.jsonl").read_text().splitlines()]
    samples = reference["samples"]
    if [row["sample_id"] for row in rows] != [sample["id"] for sample in samples]:
        raise ValueError("推理未完整覆盖参考清单,不能发布结果.")
    normalize = EnglishTextNormalizer()
    errors = words = 0
    aggregate = Moments()
    numerical, examples = [], []
    for sample, row in zip(samples, rows):
        error, count = word_errors(sample["normalized_text"], normalize(row["text"]))
        errors += error
        words += count
        if "output_statistics" in row:
            statistics = row["output_statistics"]
            aggregate.count += statistics["count"]
            aggregate.calls += statistics["calls"]
            aggregate.error += statistics["error"]
            aggregate.dot += statistics["dot"]
            aggregate.reference += statistics["reference"]
            aggregate.actual += statistics["actual"]
            numerical.append({"id": sample["id"], **row["output_errors"]})
            examples.append({"id": sample["id"], "reference": sample["text"],
                             "fp32": sample["fp32_text"], "bpu": row["text"]})
    if words != reference["reference_words"] or words <= 0:
        raise ValueError("评分参考词数与浮点参考不一致.")
    if scope == "subset" and [row["id"] for row in numerical] != reference["numerical_sample_ids"]:
        raise ValueError("数值对照没有覆盖全部选定音频.")
    evaluation = accuracy_result(config["name"], "npu", {"WER": errors / words}, len(rows), "utterance",
                                 selection_id=record_selection(row["sample_id"] for row in rows))
    comparison = compare_accuracy(read_accuracy(reference, config["name"], "onnx"), evaluation)
    report = {"model": config["name"], "platform": "s100", "run_id": manifest["run_id"],
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
              "evaluation_scope": scope, "evaluation": evaluation, "accuracy": comparison,
              "samples": len(rows), "sample_unit": "utterance",
              "sample_ids_sha256": evaluation["samples"]["selection_id"],
              "reference_words": words, "word_errors": errors,
              "reached_eot": sum(row["reached_eot"] for row in rows),
              "precision": reference["precision"], "precision_audit": reference["precision_audit"],
              "cacheSize": 256, "maxTokenLength": 448, "cache_policy": "sliding_window",
              "protocol": {"dataset": "LibriSpeech test-clean", "decoding": "greedy",
                  "normalization": "OpenAI Whisper EnglishTextNormalizer v20250625",
                  "aggregation": "corpus_word_edit_distance"},
              "calibration_overlap": "校准为test-clean前8条,固定子集与全量遵循MTK相同选择规则,与校准存在重叠",
              "model_files": [{"file": name, "bytes": (ROOT / name).stat().st_size}
                              for name in reference["artifacts"].values()],
              "missing": [] if scope == "full" else ["未执行全量精度、性能或RSS评价"]}
    if scope == "subset":
        report["output_errors"] = {"compared_calls": aggregate.calls,
            "comparison_basis": "deployment_output_including_preprocessing", "reference_backend": "onnx_cpu",
            "sample_ids": [row["id"] for row in numerical],
            "nrmse_definition": "100 * RMSE / RMS(reference)",
            "outputs": {"audio_features": aggregate.result()}, "per_sample": numerical}
    else:
        metrics = {}
        for key in ("encoder_ms", "decoder_total_ms", "first_token_ms", "npu_total_ms", "npu_rtf", "end_to_end_ms"):
            values = [row[key] for row in rows]
            metrics[key] = {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95))}
        calls = sum(row["decoder_calls"] for row in rows)
        report["performance"] = {"npu_mean_ms": metrics["npu_total_ms"]["mean"],
            "npu_p95_ms": metrics["npu_total_ms"]["p95"], "metrics": metrics,
            "decoder_call_mean_ms": sum(row["decoder_total_ms"] for row in rows) / calls,
            "end_to_end_rtf": sum(row["end_to_end_ms"] for row in rows) /
                (1000 * sum(row["duration_seconds"] for row in rows)),
            "peak_process_rss_mib": max(row["peak_rss_kb"] for row in rows) / 1024,
            "scope": "每条音频的Encoder与全部Decoder同步HBM调用之和,包含Runtime绑定开销",
            "memory_scope": "推理进程峰值RSS,不代表BPU专用内存"}
    target = ROOT / "results" / ("summary.json" if scope == "full" else "summary_test.json")
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    if examples:
        target = ROOT / "examples/output/transcripts.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[评价报告] {ROOT / 'results' / ('summary.json' if scope == 'full' else 'summary_test.json')}", flush=True)


def main():
    """将推理与汇总分开,支持失败后直接重做汇总."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    scope = "full" if args.full else "subset"
    config, reference, reference_path, models, paths = load_reference(scope)
    if args.summarize:
        summarize(config, reference, reference_path, scope)
    else:
        infer(config, reference, reference_path, models, paths, scope)


if __name__ == "__main__":
    main()
