"""使用已有 DeepSeek 浮点模型准备固定千题 MMLU、WikiText2 参考与原生输入."""

import os
import sys
from types import SimpleNamespace

from tools.deploy import arguments
from tools.files import read_json, sha256, single_file
from tools.llm import evaluate_mmlu, evaluate_quality
from tools.llm.native import settings


def main():
    """先核对现有模型及完整题目长度,再生成同协议参考,不重新编译或部署."""
    native = settings(arguments(__file__), python_executable=sys.executable)
    if native.prefill != 128 or native.context % 128:
        raise ValueError("评价要求 Prefill=128,上下文为 128 的整数倍.")
    preparation = read_json(native.work / "native_prepare.json")
    manifest = read_json(native.ctx.output / "models/manifest.json")
    quantization = read_json(native.work / "quantization_settings.json")
    if (manifest.get("model") != native.ctx.config["name"]
            or manifest["context"] != native.context
            or manifest["prefill"] != native.prefill
            or manifest["precision_settings"] != quantization
            or preparation["config_sha256"] != sha256(native.model / "config.json")
            or manifest["preparation"] != preparation):
        raise ValueError("现有部署包与浮点工作副本不匹配,评价不能自动重新转换.")
    graph = single_file(native.quantized.with_name(
        f"{native.prefix}_{native.prefill}t{native.context}c"), "*.tflite")
    if sha256(graph) != manifest["graph_contracts"]["prompt"]["tflite_sha256"]:
        raise ValueError("现有静态图与部署包不是同一次转换.")
    data = native.ctx.config["data"]
    datasets = [native.ctx.dataset / data[key] for key in ("mmlu_test", "wikitext_test")]
    for path in datasets:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"缺少 NAS 评价数据: {path}.")
        if not path.resolve().is_relative_to(native.ctx.dataset.resolve()):
            raise ValueError("评价数据必须位于配置的数据集目录.")
    device = native.ctx.config["evaluation"]["reference_device"]
    if device not in ("cuda", "cpu"):
        raise ValueError("reference_device 仅支持 cuda 或 cpu.")
    output = native.ctx.work / "evaluation/mmlu1000"
    mmlu = SimpleNamespace(
        dataset=datasets[0], model=native.model, graph=graph,
        context=native.context, limit=native.ctx.config["evaluation"]["mmlu_questions"],
        seed=native.ctx.config["evaluation"]["mmlu_seed"], source_model=native.source,
        output=output / "mmlu")
    if mmlu.limit != 1000 or mmlu.seed != 17:
        raise ValueError("LLM 正式 MMLU 固定为 1000 题,57 学科,seed=17.")
    print("[参考 1/3] 固定千题 MMLU Token 与长度检查,不截题.", flush=True)
    evaluate_mmlu.prepare(mmlu)
    print("[参考 2/3] 官方权重 FP32 MMLU.", flush=True)
    evaluate_mmlu.evaluate(SimpleNamespace(
        model_name="deepseek_r1_distill_qwen_1_5b", backend="pytorch", data=output / "mmlu", model=native.model,
        output=output / "reference/mmlu", diagnostic_questions=0, device=device))
    print("[参考 3/3] WikiText2 test FP32 与原生输入.", flush=True)
    quality = SimpleNamespace(
        model=native.model, corpus=datasets[1], output=output / "quality",
        backend="pytorch", tflite=graph.parent, decode_tflite=None,
        embedding=native.quantized / "embedding_int16.bin",
        text_rows=128, full_test=True, blocks=None, diagnostic_blocks=0,
        decode_steps=0, skip_demo=True, device=device, cpu_threads=8)
    evaluate_quality.evaluate(quality)
    quality.backend = "prepare_board"
    evaluate_quality.evaluate(quality)
    print(f"[评价资源完成] {output}", flush=True)
    print("[参考就绪] 入口接下来自动上传至板端 build/reference/full/.", flush=True)


if __name__ == "__main__":
    os.environ["MTK_EVAL_SCOPE"] = "full"
    main()
