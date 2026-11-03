"""直接读取评价数据,生成 FP32 参考和板端输入."""

import os
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from tools.deploy import context
from tools.evaluation import is_test
from tools.files import read_json, sha256, single_file
from tools.llm import evaluate_mmlu, evaluate_quality
from tools.llm.native import settings


def evaluate_mode(native, datasets):
    """准备当前模式的输入和参考,同一次流水线使用独立运行目录."""
    config = native.ctx.config
    if native.prefill != 128 or native.context % 128:
        raise ValueError("当前评价协议要求 Prefill=128,上下文为 128 的整数倍.")
    mode = "test" if is_test() else "mmlu1000"
    output = native.ctx.work / "evaluation" / mode
    mmlu, corpus = datasets
    graph = single_file(
        Path(f"{native.quantized}_{native.prefill}t{native.context}c"), "*.tflite")
    manifest = read_json(native.ctx.output / "models/manifest.json")
    if (manifest["precision"] != "asym4W_sym16A"
            or manifest["preparation"]["config_sha256"] != sha256(native.model / "config.json")
            or manifest["graph_contracts"]["prompt"]["tflite_sha256"] != sha256(graph)):
        raise ValueError("现有 W4A16 部署包与参考工作副本或静态图不匹配,不启动参考评价.")
    mmlu_args = SimpleNamespace(
        dataset=mmlu, model=native.model, graph=graph, context=native.context,
        limit=config["evaluation"]["mmlu_questions"],
        seed=config["evaluation"]["mmlu_seed"], output=output / "mmlu",
        source_model=native.source,
        test_questions=config["evaluation"]["test_mmlu_questions"])
    if mmlu_args.limit != 1000 or mmlu_args.seed != 17:
        raise ValueError("MiniCPM 正式 MMLU 协议固定为 1000 题,seed=17.")
    print(f"[评价范围] MMLU {'诊断子集' if is_test() else '固定 1000 题'},"
          f"WikiText2 {'诊断子集' if is_test() else '完整 test'};"
          f"参考设备 {config['evaluation']['reference_device']}.", flush=True)
    evaluate_mmlu.prepare(mmlu_args)
    mmlu_args = SimpleNamespace(
        backend="pytorch", data=output / "mmlu", model=native.model,
        output=output / "reference/mmlu", diagnostic_questions=0,
        device=config["evaluation"]["reference_device"])
    mmlu_args.model_name = "minicpm5_2b"
    evaluate_mmlu.evaluate(mmlu_args)
    quality_args = SimpleNamespace(
        model=native.model, corpus=corpus, output=output / "quality",
        backend="pytorch", tflite=graph.parent, decode_tflite=None,
        embedding=native.quantized / "embedding_int16.bin",
        text_rows=128, full_test=not is_test(), blocks=None,
        diagnostic_blocks=config["evaluation"]["test_wikitext_blocks"] if is_test() else 0,
        decode_steps=0, skip_demo=True,
        device=config["evaluation"]["reference_device"], cpu_threads=8)
    evaluate_quality.evaluate(quality_args)
    quality_args.backend = "prepare_board"
    evaluate_quality.evaluate(quality_args)
    print(f"[评价资源完成] {output}", flush=True)


def main():
    """仅准备当前范围的参考与评价输入,不触发量化、编译或上传."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    options = parser.parse_args()
    ctx = context(__file__, options.config)
    datasets = (ctx.dataset / ctx.config["data"]["mmlu_test"],
                ctx.dataset / ctx.config["data"]["wikitext_test"])
    native = settings(ctx, python_executable=sys.executable)
    evaluate_mode(native, datasets)
    mode = "test" if is_test() else "mmlu1000"
    print(f"[评价资源] {native.ctx.work / 'evaluation' / mode}; 单独放入板端 evaluation/{mode}.",
          flush=True)


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    main()
