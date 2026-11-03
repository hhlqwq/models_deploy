"""使用同协议输入评价 MiniCPM5 的 MMLU、WikiText2 与评价进程 RSS."""

import json
import os
import resource
from types import SimpleNamespace

from tools.board import context
from tools.files import read_json
from tools.llm import evaluate_board_quality, evaluate_mmlu


def check_resources(root):
    """确认正式输入与现有部署包匹配,缺少资源时在模型执行前停止."""
    data = root / "build/reference/full"
    manifest = read_json(root / "models/manifest.json")
    for name in ("mmlu/protocol.json", "mmlu/cases.json", "mmlu/mmlu_inputs.npz",
                 "mmlu/board_input_contract.json", "quality/board_inputs.npz",
                 "quality/board_quality_protocol.json", "quality/board_input_contract.json",
                 "reference/mmlu/mmlu_accuracy.json", "quality/pytorch_quality.json"):
        if not (data / name).is_file():
            raise FileNotFoundError(f"缺少评价资源: evaluation/mmlu1000/{name}.")
    protocol = read_json(data / "mmlu/protocol.json")
    quality = read_json(data / "quality/board_quality_protocol.json")
    references = (read_json(data / "reference/mmlu/mmlu_accuracy.json"),
                  read_json(data / "quality/pytorch_quality.json"))
    if (protocol.get("scope") != "first_round_stratified_subset"
            or protocol["questions"] != 1000 or protocol["subjects"] != 57
            or protocol["sampling_seed"] != 17
            or protocol.get("source_tokenizer_parity") is not True
            or protocol.get("choice_continuation_checked") is not True
            or protocol["maximum_prompt_tokens"] > manifest["context"]
            or quality.get("protocol") != "full_test_block128_v1"
            or any(row.get("evaluation_scope") != "full"
                   or row.get("status") != "complete" for row in references)):
        raise ValueError("需要固定 1000 题、seed=17 的 MMLU 与完整 WikiText2 参考,不能混用旧题集或超长输入.")
    if (references[0].get("evaluated_questions") != 1000
            or references[0].get("reference", {}).get("parameter_dtypes") != ["torch.float32"]):
        raise ValueError("MMLU 参考未完成固定 1000 题或缺少实际 FP32 参数精度记录.")
    if (protocol["context"] != manifest["context"]
            or protocol["model_config_sha256"] != manifest["preparation"]["config_sha256"]
            or quality["model_config_sha256"] != protocol["model_config_sha256"]):
        raise ValueError("评价输入与已部署模型配置不一致,请准备匹配参考.")
    if (references[0]["cases_sha256"] != protocol["cases_sha256"]
            or not protocol.get("tokenizer_sha256")
            or references[0].get("tokenizer_sha256") != protocol["tokenizer_sha256"]
            or references[0]["selected_source_question_ids"] != protocol["selected_source_question_ids"]
            or references[1]["input_ids"] != quality["input_ids"]
            or any(row["model_config_sha256"] != protocol["model_config_sha256"]
                   for row in references)):
        raise ValueError("FP32 参考与已有输入的样本或模型不一致.")
    for folder in ("mmlu", "quality"):
        contract = read_json(data / folder / "board_input_contract.json")
        if contract["tflite_sha256"] != manifest["graph_contracts"]["prompt"]["tflite_sha256"]:
            raise ValueError("评价张量契约来自其他转换图,不能混用.")
    return data


def main():
    """执行约定的千题 MMLU 和完整 WikiText2,中间结果写本模型 work."""
    ctx = context(__file__)
    data = check_resources(ctx.root)
    ctx.result.mkdir(parents=True, exist_ok=True)
    evaluate_mmlu.evaluate(SimpleNamespace(
        model_name="minicpm5_2b", backend="npu", data=data / "mmlu", package=ctx.models,
        output=ctx.result / "mmlu", diagnostic_questions=0))
    evaluate_board_quality.evaluate(SimpleNamespace(
        package=ctx.models, quality_data=data / "quality",
        output=ctx.result / "wikitext", diagnostic_blocks=0, decode_steps=0))
    (ctx.result / "evaluation_memory.json").write_text(json.dumps({
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "scope": "mmlu_and_wikitext_python_process",
    }, indent=2) + "\n", encoding="utf-8")
    print("[精度评价完成] MMLU 和 WikiText2.", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    main()
