"""汇总 Whisper-Tiny 正式 WER/CER、框架一致性和板端性能."""


import os
import argparse
import json
import math
from pathlib import Path
from typing import Callable

from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
from whisper.tokenizer import get_tokenizer

from tools.board import configure_parser

from tools.evaluation import (
    is_test, record_selection, result_status, scope_metadata,
)
from tools.evaluation import accuracy_result



def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("librispeech", "aishell1"),
                        required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--board-predictions", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path)
    parser.add_argument("--preprocess-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--run-id")
    configure_parser(parser, __file__, {
        "dataset": lambda c: "librispeech",
        "source_manifest": lambda c: c.work / "cache/librispeech/source_manifest.jsonl",
        "board_predictions": lambda c: c.work / "board_predictions.jsonl",
        "preprocess_metrics": lambda c: c.work / "cache/librispeech/preprocess_metrics.jsonl",
        "output_dir": lambda c: c.work / "report",
    })
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    """读取非空 JSONL 行."""
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int,
                                                                       int,
                                                                       int]:
    """计算最小编辑路径的替换、删除和插入数量."""
    previous = [(0, 0, index) for index in range(len(hypothesis) + 1)]
    for ref_index, ref_item in enumerate(reference, 1):
        current = [(0, ref_index, 0)]
        for hyp_index, hyp_item in enumerate(hypothesis, 1):
            if ref_item == hyp_item:
                current.append(previous[hyp_index - 1])
                continue
            substitution = previous[hyp_index - 1]
            deletion = previous[hyp_index]
            insertion = current[hyp_index - 1]
            candidates = (
                (sum(substitution) + 1,
                 (substitution[0] + 1, substitution[1], substitution[2])),
                (sum(deletion) + 1,
                 (deletion[0], deletion[1] + 1, deletion[2])),
                (sum(insertion) + 1,
                 (insertion[0], insertion[1], insertion[2] + 1)),
            )
            current.append(min(candidates, key=lambda item: item[0])[1])
        previous = current
    return previous[-1]


def percentile(values: list[float], percent: float) -> float | None:
    """按线性插值计算百分位数."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (ordered[lower] * (upper - position) +
            ordered[upper] * (position - lower))


def statistics(values: list[float]) -> dict:
    """生成 Mean/P50/P90/P95 统计."""
    return {
        "count": len(values),
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
    }


def duration_bucket(seconds: float) -> str:
    """映射到固定音频时长分组."""
    if seconds <= 5:
        return "le_5s"
    if seconds <= 15:
        return "5_15s"
    if seconds <= 25:
        return "15_25s"
    if seconds <= 30:
        return "25_30s"
    return "gt_30s"


def normalized_units(dataset: str) -> tuple[Callable[[str], str],
                                             Callable[[str], list[str]], str]:
    """返回数据集固定文本规范化、分词函数和指标名."""
    if dataset == "librispeech":
        normalizer = EnglishTextNormalizer()
        return normalizer, lambda text: text.split(), "wer"
    normalizer = BasicTextNormalizer(remove_diacritics=False, split_letters=False)

    def normalize_chinese(text: str) -> str:
        """规范化中文并移除空白,用于字符级 CER."""
        return "".join(normalizer(text).split())

    return normalize_chinese, list, "cer"


def decode_board_texts(dataset: str, board_records: list[dict]) -> None:
    """将板端 Token 原地解码为文本."""
    language = "en" if dataset == "librispeech" else "zh"
    tokenizer = get_tokenizer(
        multilingual=True, language=language, task="transcribe")
    for record in board_records:
        if record.get("status") == "ok":
            record["text"] = tokenizer.decode(record["tokens"]).strip()


def metric_summary(dataset: str, sources: dict[str, dict],
                   predictions: dict[str, dict]) -> tuple[dict, list[dict]]:
    """计算正式 WER/CER 汇总与逐样例错误."""
    normalize, split_units, metric_name = normalized_units(dataset)
    substitutions = deletions = insertions = reference_units = 0
    exact_matches = 0
    details = []
    for sample_id, source in sources.items():
        prediction = predictions[sample_id]
        normalized_reference = normalize(source["reference_text"])
        normalized_prediction = normalize(prediction["text"])
        reference = split_units(normalized_reference)
        hypothesis = split_units(normalized_prediction)
        sub, delete, insert = edit_counts(reference, hypothesis)
        substitutions += sub
        deletions += delete
        insertions += insert
        reference_units += len(reference)
        exact_matches += normalized_reference == normalized_prediction
        details.append({
            "sample_id": sample_id,
            "reference": normalized_reference,
            "prediction": normalized_prediction,
            "substitutions": sub,
            "deletions": delete,
            "insertions": insert,
            "reference_units": len(reference),
            "error_rate": ((sub + delete + insert) / len(reference)
                           if reference else 0.0),
        })
    errors = substitutions + deletions + insertions
    return {
        "metric": metric_name,
        "value": errors / reference_units if reference_units else None,
        "errors": errors,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_units": reference_units,
        "normalized_exact_matches": exact_matches,
        "normalized_exact_match_rate": exact_matches / len(sources),
    }, details


def performance_summary(records: list[dict]) -> dict:
    """汇总整体和分时长板端 NPU 性能."""
    metric_keys = (
        "encoder_ms", "first_token_ms", "decoder_total_ms", "npu_total_ms",
        "npu_rtf", "tokens_per_second", "peak_rss_kb")

    def summarize(items: list[dict]) -> dict:
        """汇总一组成功样例."""
        return {key: statistics([float(item[key]) for item in items])
                for key in metric_keys}

    buckets = {}
    for name in ("le_5s", "5_15s", "15_25s", "25_30s", "gt_30s"):
        buckets[name] = summarize([
            item for item in records
            if duration_bucket(float(item["duration_seconds"])) == name
        ])
    return {"overall": summarize(records), "duration_buckets": buckets}


def preprocessing_summary(records: list[dict]) -> dict:
    """汇总板端 Python 音频读取与 Log-Mel 耗时,Mel 直接在内存中使用."""
    metric_keys = ("audio_decode_ms", "log_mel_ms", "mel_write_ms")
    return {
        "host": "target_board_python", "mel_cache": "in_memory",
        "cache_hits": sum(bool(item["cache_hit"]) for item in records),
        "metrics": {
            key: statistics([float(item[key]) for item in records])
            for key in metric_keys
        },
    }


def main() -> None:
    """执行完整性检查、精度计算和性能汇总."""
    args = parse_args()
    sources = {item["sample_id"]: item
               for item in read_jsonl(args.source_manifest)}
    board_records = read_jsonl(args.board_predictions)
    if any(row.get("backend") != "python_neuron_runtime_hw" for row in board_records):
        raise ValueError("必须使用本次 Python 推理记录.")
    decode_board_texts(args.dataset, board_records)
    successful = {item["sample_id"]: item for item in board_records
                  if item.get("status") == "ok"}
    failed = [item for item in board_records if item.get("status") != "ok"]
    missing = sorted(set(sources) - set(successful))
    complete = not failed and not missing and len(successful) == len(sources)

    evaluated_sources = {sample_id: source for sample_id, source in
                         sources.items() if sample_id in successful}
    accuracy, details = metric_summary(
        args.dataset, evaluated_sources,
        {sample_id: successful[sample_id] for sample_id in evaluated_sources}
    ) if successful else ({}, [])
    truncated_sample_ids = sorted(
        item["sample_id"] for item in successful.values()
        if not item.get("reached_eot", False))
    record_selection(sources)
    summary = {
        "status": result_status("complete" if complete else "incomplete"), **scope_metadata(),
        "dataset": args.dataset, "backend": "python_neuron_runtime_hw",
        "expected_samples": len(sources),
        "successful_samples": len(successful),
        "failed_samples": len(failed),
        "missing_samples": len(missing),
        "accuracy": accuracy,
        "performance": {} if is_test() else performance_summary(list(successful.values())),
        "truncated_samples": len(truncated_sample_ids),
        "truncated_sample_ids": truncated_sample_ids,
    }
    if args.reference_predictions:
        framework = {item["sample_id"]: item for item in
                     read_jsonl(args.reference_predictions)}
        shared = set(successful) & set(framework)
        normalize = normalized_units(args.dataset)[0]
        framework_sources = {
            sample_id: sources[sample_id] for sample_id in sources
            if sample_id in framework
        }
        framework_accuracy, _ = metric_summary(
            args.dataset, framework_sources,
            {sample_id: framework[sample_id]
             for sample_id in framework_sources})
        summary["framework_accuracy"] = framework_accuracy
        normalized_text_exact_matches = sum(
            normalize(successful[item]["text"]) ==
            normalize(framework[item]["text"]) for item in shared)
        token_exact_matches = sum(
            successful[item]["tokens"] == framework[item]["tokens"]
            for item in shared)
        summary["framework_parity"] = {
            "compared_samples": len(shared),
            "normalized_text_exact_matches": normalized_text_exact_matches,
            "normalized_text_exact_match_rate": (
                normalized_text_exact_matches / len(shared)
                if shared else None),
            "token_exact_matches": token_exact_matches,
            "token_exact_match_rate": (
                token_exact_matches / len(shared) if shared else None),
        }
    if args.preprocess_metrics and not is_test():
        summary["host_preprocessing"] = preprocessing_summary(
            read_jsonl(args.preprocess_metrics))

    if not complete:
        raise ValueError("Whisper 任务精度存在失败或缺失样本.")
    summary["evaluation"] = accuracy_result("whisper_tiny", "npu",
        {"WER": accuracy["value"]}, len(successful), "utterance")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("[OK] WER 已计算." if is_test() else "[OK] WER 已计算,准备汇总 NPU 耗时.", flush=True)
    if not complete and not args.allow_incomplete:
        raise SystemExit("评测不完整,请按同一输出文件断点续跑后重新汇总.")


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    main()
