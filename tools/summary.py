"""评价完成后统一发布精度、性能与峰值 RSS,正式结果为 summary.json,保留独立 summary_test.json."""

import json
import math
import os
import platform

from tools.board import context
from tools.evaluation import compare_accuracy, is_test, publish_result, read_accuracy
from tools.files import read_json


def require_count(actual, expected):
    """拒绝不完整覆盖,不把固定子集当作全量评价."""
    if actual != expected:
        raise ValueError(f"全量覆盖不完整: {actual}/{expected}.")


def vision_accuracy(ctx):
    """核对完整任务精度与匹配参考,同时纳入本次硬件性能."""
    model = ctx.config["name"]
    if model in ("yolov5s", "yolov8n", "yamnet"):
        report_path = ctx.result / "summary.json"
    elif model == "rtmpose_body2d":
        report_path = ctx.work / "coco_wholebody_metrics.json"
    else:
        report_path = ctx.work / "report/summary.json"
    report = read_json(report_path)
    if report.get("evaluation_scope") != "full" or report.get("status") not in (
            "complete", "board_verified"):
        raise ValueError(f"原始报告未完成全量评价: {report_path}.")
    evaluation = read_accuracy(report, model, "npu")
    expected = {"rtmpose_body2d": 104125, "yolov5s": 5000, "yolov8n": 5000,
                "yamnet": 1504, "vit_base_patch16_224": 50000, "mobilefacenet": 6000,
                "depth_anything_v2_small": 2068, "fastsam": 5000, "whisper_tiny": 2620}
    samples = evaluation["samples"]["count"]
    require_count(samples, expected[model])
    baseline = read_json(ctx.root / "build/reference/full.json")
    reference_evaluation = read_accuracy(baseline, model, "onnx")
    comparison = compare_accuracy(reference_evaluation, evaluation)
    metric = next(iter(comparison))
    value = comparison[metric]["npu_accuracy"]
    reference = comparison[metric]["fp32_accuracy"]
    source = "本次同模型同协议浮点全量实测"
    change = -comparison[metric]["accuracy_loss_percentage_points"]
    return {
        "evaluation": evaluation,
        "dataset": report.get("dataset", "见本模型评价协议"),
        "samples": samples, "metric": metric, "board_accuracy": value,
        "reference_accuracy": reference, "reference_source": source,
        "accuracy_change_percentage_points": change,
        "comparison_status": "matched_reference" if reference is not None else "reference_unavailable",
        "performance": vision_performance(ctx, report),
    }


def llm_accuracy(ctx):
    """核对约定范围的 MMLU/PPL 协议及模型标识,不读取 Demo 或性能文件."""
    from tools.llm.artifacts import dla_relative_path

    data = ctx.root / "build/reference/full"
    mmlu = read_json(ctx.result / "mmlu/mmlu_accuracy.json")
    ppl = read_json(ctx.result / "wikitext/npu_quality.json")
    reference_mmlu = read_json(data / "reference/mmlu/mmlu_accuracy.json")
    reference_ppl = read_json(data / "quality/pytorch_quality.json")
    for report in (mmlu, ppl, reference_mmlu, reference_ppl):
        if report.get("evaluation_scope") != "full" or report.get("status") != "complete":
            raise ValueError("完整 LLM 评价不能使用子集或未完成报告.")
    for actual, reference, fields in (
        (mmlu, reference_mmlu, ("cases_sha256", "selected_source_question_ids", "evaluated_questions",
                               "scoring", "chat_template", "add_special_tokens", "context")),
        (ppl, reference_ppl, ("protocol", "selected_text_sha256", "corpus_sha256", "input_ids",
                             "blocks", "scored_tokens", "excluded_tail_tokens", "reset_context_each_block")),
    ):
        for field in fields:
            if actual[field] != reference[field]:
                raise ValueError(f"LLM 参考与硬件评价协议不一致: {field}.")
    require_count(mmlu["evaluated_questions"], 1000)
    expected_scope = "first_round_stratified_subset"
    if mmlu["scope"] != expected_scope or ppl["protocol"] != "full_test_block128_v1":
        raise ValueError("MMLU 或 WikiText2 test 不符合本模型约定的评价范围.")
    for row in (mmlu, reference_mmlu):
        if (row.get("sampling_seed") != 17 or row.get("subjects") != 57
                or row.get("source_tokenizer_parity") is not True
                or row.get("choice_continuation_checked") is not True):
            raise ValueError("LLM 千题抽样或 Token 协议核对记录不完整.")
    if (mmlu.get("tokenizer_sha256") != reference_mmlu.get("tokenizer_sha256")
            or reference_mmlu.get("reference", {}).get("parameter_dtypes") != ["torch.float32"]):
        raise ValueError("LLM 参考缺少匹配的 Tokenizer 或实际 FP32 精度记录.")
    manifest = read_json(ctx.models / "manifest.json")
    prompt_name = dla_relative_path(manifest, "prompt")
    prompt = next(row for row in manifest["files"] if row["name"] == prompt_name)
    hardware = read_json(ctx.result / "mmlu/run_manifest.json")
    if hardware["graph_sha256"] != prompt["sha256"] or ppl["dla_sha256"] != prompt["sha256"]:
        raise ValueError("评价 DLA 与已有部署清单不一致.")
    if not all(math.isfinite(row["accuracy"]) and 0 <= row["accuracy"] <= 1
               for row in (mmlu, reference_mmlu)):
        raise ValueError("MMLU Accuracy 无效.")
    if not all(math.isfinite(row["perplexity"]) and row["perplexity"] > 0
               for row in (ppl, reference_ppl)):
        raise ValueError("WikiText2 PPL 无效.")
    # MMLU 已完成约定千题,完整原始基准的覆盖单独标记.
    result = {"evaluation": mmlu["evaluation"],
              "full_benchmark_evaluated": False, "accuracy": {
        "mmlu": {"questions": mmlu["evaluated_questions"],
                 "scope": mmlu["scope"], "subjects": mmlu["subjects"],
                 "sampling_seed": mmlu["sampling_seed"],
                 "scoring": mmlu["scoring"], "cases_sha256": mmlu["cases_sha256"],
                 "fp32_accuracy": reference_mmlu["accuracy"], "npu_accuracy": mmlu["accuracy"],
                 "difference_percentage_points": 100 * (mmlu["accuracy"] - reference_mmlu["accuracy"])},
        "wikitext": {"blocks": ppl["blocks"], "scored_tokens": ppl["scored_tokens"],
                     "protocol": ppl["protocol"], "fp32_ppl": reference_ppl["perplexity"],
                     "npu_ppl": ppl["perplexity"],
                     "difference": ppl["perplexity"] - reference_ppl["perplexity"],
                     "difference_percent": (ppl["perplexity"] / reference_ppl["perplexity"] - 1) * 100}}}
    result.update(
        accuracy_scope="fixed_1000_mmlu_and_full_wikitext2",
        reference=reference_mmlu["reference"],
        quality_assessment={
            "status": "measured_difference_not_automatically_accepted",
            "error_attribution": "unverified_without_matched_sdk_float_and_w4a16_software_comparison",
        })
    return result


def vision_performance(ctx, report):
    """读取同一次硬件推理的性能和进程 RSS,不使用主机参考耗时."""
    model = ctx.config["name"]
    if model == "rtmpose_body2d":
        report = read_json(ctx.work / "timing_summary.json")
    if report.get("backend") != "python_neuron_runtime_hw":
        raise ValueError("性能数据必须来自当前 Python 推理实现.")
    if model == "whisper_tiny":
        overall = report["performance"]["overall"]
        mean = overall["npu_total_ms"]["mean"]
        p95 = overall["npu_total_ms"]["p95"]
        rss = overall["peak_rss_kb"]["max"] / 1024
        scope = "每条音频 Encoder 与全部 Decoder NPU 调用之和"
    else:
        mean = report["npu_mean_ms"]
        p95 = report.get("npu_p95_ms")
        rss = report.get("peak_rss_mib")
        if rss is None:
            value = report.get("peak_rss_kib", report.get("peak_rss_kb"))
            rss = None if value is None else value / 1024
        scope = report.get("timing_scope", "每个人体裁剪的 NPU 调用" if model == "rtmpose_body2d"
                           else "本次实际输入的 NPU 调用")
    if not isinstance(mean, (int, float)) or not math.isfinite(mean) or mean <= 0:
        raise ValueError("缺少有效的板端 NPU 性能记录.")
    if not isinstance(rss, (int, float)) or not math.isfinite(rss) or rss <= 0:
        raise ValueError("缺少有效的推理进程峰值 RSS.")
    result = {"npu_mean_ms": mean, "npu_p95_ms": p95, "scope": scope,
              "peak_process_rss_mib": rss,
              "memory_scope": "推理进程峰值 RSS,不等于设备总内存或驱动专用内存"}
    for key in ("preprocess_mean_ms", "postprocess_mean_ms", "end_to_end_mean_ms",
                "preprocess_mean_ms_per_clip", "inference_stage_mean_ms_per_clip",
                "frontend_peak_rss_mib"):
        if key in report:
            result[key] = report[key]
    if model in ("yolov5s", "yolov8n"):
        timing = read_json(ctx.work / "timing_summary_current_run.json")
        result["npu_p95_ms"] = timing["npu_ms"].get("p95")
        result["end_to_end_ms"] = timing["e2e_ms"]
        result["end_to_end_scope"] = "常驻 Python 进程的逐图预处理、推理和后处理"
    if result["npu_p95_ms"] is None:
        result["unavailable_metrics"] = ["npu_p95_ms: 当前运行器未记录"]
    return result


def llm_performance(ctx):
    """读取定长 Prefill/Decode 评价与评价进程 RSS,拒绝混入其他运行."""
    generation = read_json(ctx.result / "board_metrics.json")
    if generation["run_id"] != ctx.run_id:
        raise ValueError("LLM 性能记录不属于本次评价.")
    speed = generation["performance"]
    required = ("prefill_tokens_per_second", "prefill_npu_ms", "elapsed_seconds")
    if any(not isinstance(speed.get(key), (int, float)) or not math.isfinite(speed[key])
           or speed[key] <= 0 for key in required):
        raise ValueError("Python Prefill 性能记录无效.")
    if speed["decode_calls"] and (not isinstance(speed["decode_tokens_per_second"], (int, float))
            or not math.isfinite(speed["decode_tokens_per_second"]) or speed["decode_tokens_per_second"] <= 0
            or not math.isfinite(speed["decode_npu_ms"]) or speed["decode_npu_ms"] <= 0):
        raise ValueError("Python Decode 性能记录无效.")
    memory = read_json(ctx.result / "evaluation_memory.json")
    if generation.get("backend") != "python_neuron_runtime_hw":
        raise ValueError("性能记录必须来自当前 Python 推理实现.")
    runs = [speed, *generation["samples"], *generation["text_input_checks"]]
    values = [row["peak_process_rss_mib"] for row in runs] + [memory["peak_process_rss_mib"]]
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("LLM 进程 RSS 记录无效.")
    return {"prefill_tokens_per_second": speed["prefill_tokens_per_second"],
            "decode_tokens_per_second": speed["decode_tokens_per_second"],
            "scope": speed["scope"], "input_tokens": speed["prefill_input_tokens"],
            "generated_tokens": len(speed["generated_tokens"]),
            "peak_process_rss_mib": max(values),
            "memory_scope": "本次生成、定长测速和精度评价各进程峰值的最大值",
            "timing_scope": "常驻 Python 生成流程的 Prefill/Decode 阶段,含各阶段输入和 KV 处理",
            "prefill_npu_ms": speed["prefill_npu_ms"],
            "decode_npu_ms": speed["decode_npu_ms"],
            "decode_calls": speed["decode_calls"],
            "generation_elapsed_seconds": speed["elapsed_seconds"],
            "unavailable_metrics": (["decode_tokens_per_second: 首 Token 即停止,未执行 Decode"]
                                    if speed["decode_tokens_per_second"] is None else []),
            "unavailable_checks": generation.get("unavailable_checks", []),
            "warmup_runs": generation.get("warmup_runs", 0),
            "measured_runs": generation.get("measured_runs", 1),
            "functional_failures": generation["functional_failures"]}


def publish(ctx, report):
    """发布唯一正式报告,仅清理当前部署包 results 内已由新报告取代的文件."""

    actual = read_accuracy(report, ctx.config["name"], "npu")
    baseline = read_json(ctx.root / "build/reference/full.json")
    backend = baseline["evaluation"]["backend"]
    reference = read_accuracy(baseline, ctx.config["name"], backend)
    if ctx.config["name"] in ("minicpm5_2b", "deepseek_r1_distill_qwen_1_5b"):
        report["wikitext"] = report["accuracy"]["wikitext"]
    report["accuracy"] = compare_accuracy(reference, actual)
    report["samples"] = actual["samples"]["count"]
    report["sample_unit"] = actual["samples"]["unit"]
    report["sample_ids_sha256"] = actual["samples"]["selection_id"]
    destination = publish_result(ctx.root, report, "summary.json")
    print(f"[评价报告] {destination}; 过程数据保留在 {ctx.result}.", flush=True)


def main():
    """只由 evaluate.sh 发布一个包含实际覆盖、精度、性能和 RSS 的报告."""
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    if is_test():
        raise ValueError("compile.sh 只能发布 summary_test.json,不能改写全量报告.")
    ctx = context(__file__)
    model = ctx.config["name"]
    print(f"[评价汇总] 核对同次精度、性能与峰值 RSS: {ctx.run_id}.", flush=True)
    if model in ("minicpm5_2b", "deepseek_r1_distill_qwen_1_5b"):
        result = llm_accuracy(ctx)
        result["performance"] = llm_performance(ctx)
        result["precision"] = read_json(ctx.models / "manifest.json")["precision"]
        if model == "deepseek_r1_distill_qwen_1_5b":
            manifest = read_json(ctx.models / "manifest.json")
            result.update(context=manifest["context"], prefill=manifest["prefill"],
                          precision_settings=manifest["precision_settings"],
                          source={key: manifest["preparation"]["source"][key]
                                  for key in ("repo", "revision")})
        if result["performance"]["functional_failures"]:
            result["status"] = "functional_checks_failed"
    else:
        result = vision_accuracy(ctx)
    summary = {"schema_version": 1, "model": model, "run_id": ctx.run_id,
               "status": "complete", "evaluation_scope": "full", **result,
               "runtime": {"backend": "python_neuron_runtime_hw", "machine": platform.machine(), "kernel": platform.release()}}
    binding = read_json(ctx.root / "build/reference/full.json")["dla_artifacts"]
    artifacts = [ctx.models / name for name in binding]
    if model in ("minicpm5_2b", "deepseek_r1_distill_qwen_1_5b"):
        artifacts.append(ctx.models / "tokenizer/embedding_int16.bin")
    if model == "whisper_tiny":
        contract = read_json(ctx.root / "models/runtime_contract.json")
        summary.update(precision="W8A16", cacheSize=contract["cacheSize"],
                       maxTokenLength=contract["maxTokenLength"],
                       cache_policy=contract["cache_policy"],
                       calibration_overlap=contract["calibration_overlap"])
        artifacts = [ctx.root / "models" / contract[stage]["file"]
                     for stage in ("encoder", "decoder_step")]
    summary["model_files"] = [{"file": str(path.relative_to(ctx.root)), "bytes": path.stat().st_size}
                              for path in artifacts]
    publish(ctx, summary)


if __name__ == "__main__":
    main()
