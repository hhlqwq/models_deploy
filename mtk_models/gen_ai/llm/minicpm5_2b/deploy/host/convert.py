"""顺序执行 MiniCPM5 模型准备、校准、W4A16 量化和静态图导出."""

import json
import os
from pathlib import Path
import shutil
import sys

from transformers import AutoTokenizer

from tools.deploy import arguments
from tools.files import sha256, single_file
from tools.llm.native import settings
from tools.llm.quantize_native import fix_native_shapes, quantize
from tools.process import run_command
from tools.storage import model_storage


def prepare(source, output, context, calibration_train=None, mask_value=-100.0,
            rotate=False):
    """读取官方资源并生成独立配置、校准输入和英文 Prompt."""
    manifest = json.loads((source / "source_manifest.json").read_text())
    if output.resolve() == source.resolve() or context < 128:
        raise ValueError("输出必须独立于源目录,上下文不得小于 128.")
    output.mkdir(parents=True, exist_ok=True)
    model = output / "MiniCPM5-2B"
    model.mkdir(exist_ok=True)
    for entry in manifest["files"]:
        src = source / entry["name"]
        dst = model / src.name
        if src.suffix == ".safetensors":
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                print(f"[权重复制] {src.name}", flush=True)
                shutil.copyfile(src, dst)
        else:
            shutil.copyfile(src, dst)
    # 仅转换独立副本的 BPE merge 序列化格式,不改变词表和规则.
    token_data = json.loads((model / "tokenizer.json").read_text())
    token_data["model"]["merges"] = [
        " ".join(item) if isinstance(item, list) else item
        for item in token_data["model"]["merges"]]
    (model / "tokenizer.json").write_text(
        json.dumps(token_data, ensure_ascii=False) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tokenizer.chat_template = (source / "chat_template.jinja").read_text()
    tokenizer.save_pretrained(model)
    stops = json.loads((source / "config.json").read_text())["eos_token_id"]
    if not isinstance(stops, list) or tokenizer.eos_token_id not in stops:
        raise ValueError("官方停止 Token 配置不符.")
    config = json.loads((source / "config.json").read_text())
    if not (-10000.0 <= mask_value < 0):
        raise ValueError("掩码负值必须位于 [-10000,0).")
    config.update(mask_value=mask_value, rotate=rotate,
                  rotate_seed=0, rotate_mode="hadamard")
    # SDK 2.7.5 校准生成只兼容单 EOS,使用官方对话结束 Token.
    # 原生交付的 stopToken 仍保留官方两个停止 ID,不修改源配置.
    config.update(bos_token_id=tokenizer.bos_token_id,
                  eos_token_id=130073,
                  pad_token_id=tokenizer.pad_token_id,
                  tokenizer="pretrained_fast", max_position_embeddings=context)
    (model / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": "{instruction}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    (output / "minicpm5_preformatter.json").write_text(json.dumps({
        "description": "MiniCPM5 官方单轮 Chat Template.",
        "prompt_input": template, "prompt_no_input": template,
        "response_split": "<|im_start|>assistant\n",
    }, ensure_ascii=False, indent=2) + "\n")
    # 自编双语校准样例只验证部署流程,不作为正式质量基准.
    questions = [
        "请用中文简单介绍你自己.",
        "Please introduce yourself briefly in English.",
        "计算 17 加 25,说明计算过程.",
        "What is 17 plus 25? Explain the calculation.",
        "解释人工智能如何帮助人们分析信息.",
        "Explain how artificial intelligence helps people analyze information.",
        "把这句话译成英文: 北京是中国的首都.",
        "Translate into Chinese: Beijing is the capital of China.",
        "为什么边缘设备需要节省内存?",
        "Why should an edge device use memory efficiently?",
        "写出整理文件的三个步骤.",
        "Give three steps for organizing files.",
        "总结: 小模型便于部署,但仍需要验证输出质量和速度.",
        "Summarize: Small models are easier to deploy but need quality and speed checks.",
        "如果输入过长,应该怎样安排上下文与回答长度?",
        "How should we budget input and output tokens for a limited context?",
    ]
    records = []
    prompts = output / "prompts"
    prompts.mkdir(exist_ok=True)
    for question in questions:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if tokens.count(tokenizer.bos_token_id) != 1:
            raise ValueError("Prompt 必须且只能包含一个 BOS.")
        # GAI SDK 2.7.5 的 tokens 字段采用空格分隔字符串.
        records.append({"tokens": " ".join(str(token) for token in tokens)})
    calibration_scope = "authored_bilingual_smoke_not_formal_accuracy"
    training_source = None
    if calibration_train is not None:
        # 训练集参与校准,测试集只用于独立 PPL 对比.
        import pyarrow.parquet as parquet

        table = parquet.read_table(calibration_train)
        train_tokens = tokenizer.encode(
            "\n\n".join(table.column("text").to_pylist()),
            add_special_tokens=False)
        mixed_records = []
        for index in range(8):
            segment = train_tokens[index * 256:(index + 1) * 256]
            mixed_records.append({"tokens": " ".join(map(str, segment))})
            mixed_records.append(records[index])
        records = mixed_records
        calibration_scope = "wikitext2_train_8x256_and_authored_bilingual_8"
        training_source = {"revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
                           "split": "train", "sha256": sha256(calibration_train),
                           "text_tokens": 2048}
    # Demo 与官方 FP32 参考使用相同 Prompt,校准输入保持独立且不变.
    question = "Briefly introduce yourself in English."
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    (prompts / "en_demo.txt").write_text(text)
    (prompts / "en_demo.tokens.txt").write_text(" ".join(map(str, tokens)) + "\n")
    demos = [{"id": "en_demo", "prompt": question, "input_ids": tokens}]
    (output / "calibration.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records))
    (output / "native_prepare.json").write_text(json.dumps({
        "source": manifest, "context": context,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id, "stop_token_ids": stops,
        "enable_thinking": False,
        "rope_theta": config["rope_theta"],
        "sdk_adaptation": {"mask_value": mask_value, "rotate": rotate,
                           "rotate_mode": "hadamard", "rotate_seed": 0},
        "calibration_scope": calibration_scope,
        "calibration_training_source": training_source,
        "calibration_samples": len(records),
        "demos": demos,
        "calibration_sha256": sha256(output / "calibration.jsonl"),
        "config_sha256": sha256(model / "config.json"),
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"[准备完成] {model},BOS={tokenizer.bos_token_id},EOS={tokenizer.eos_token_id}",
          flush=True)


def prepare_model(native):
    """由转换步骤直接调用,替换当前工作副本并生成实际转换输入."""
    if native.work.is_symlink() or native.work.resolve() == native.source.resolve():
        raise ValueError("转换目录不能是软链接或原始资源目录.")
    expected = model_storage(Path(__file__).resolve().parents[2]) / "build/native"
    if (native.work.resolve() != expected.resolve()
            or not native.work.resolve().is_relative_to(expected.parents[1].resolve())):
        raise ValueError("MiniCPM 转换目录必须为模型资源下的 build/native.")
    if native.work.exists():
        print("[清理] 替换上一次转换工作目录,保留官方资源及当前部署包.", flush=True)
        shutil.rmtree(native.work)
    prepare(native.source, native.work, native.context,
            Path(native.train) if native.train else None, native.mask, bool(native.rotate))


def calibrate(native):
    """直接执行 SDK 校准命令,成功后记录实际完成的批次数."""
    marker = native.work / "calibration_complete.txt"
    marker.unlink(missing_ok=True)
    command = [
        native.python, native.python.parent / "mtk_make_llm_ptq_calib_dataset",
        "converter", native.model / "config.json",
        native.work / "calibration.jsonl", "-b", str(native.batches), "-m", "128",
    ]
    run_command(command, cwd=native.work)
    marker.write_text(f"{native.batches}\n")


def quantize_model(native):
    """在当前 Python 进程调用 SDK 适配函数并检查全部量化分片."""
    marker = native.work / "calibration_complete.txt"
    if marker.read_text().strip() != str(native.batches):
        raise ValueError("校准尚未完成或批次数不一致.")
    record = native.work / "quantization_settings.json"
    record.unlink(missing_ok=True)
    options = [
        "converter", str(native.model / "config.json"),
        "-p", "asym4W_sym16A", "-tp", "asym4W_sym16A",
        "-d", str(native.work / "calibration_datasets" / native.model.name),
        "-m", "Overall",
    ]
    if native.optimization != "none":
        options.extend(["--weight_optimization", native.optimization])
    if native.extra:
        options.extend(["--extra_converter_options", native.extra])
    quantize(options)
    config = json.loads((native.model / "config.json").read_text())
    expected = {
        f"{native.prefix}_{index}.tflite"
        for index in range(config["num_hidden_layers"] + 1)
    }
    if {path.name for path in native.quantized.glob("*.tflite")} != expected:
        raise ValueError("SDK 量化分片不完整.")
    for name in expected:
        path = native.quantized / name
        if not path.stat().st_size or not path.with_name(path.stem + "_info.json").is_file():
            raise ValueError(f"分片为空或缺少 SDK 元数据: {name}.")
    record.write_text(json.dumps({
        "attention": "asym4W_sym16A", "mlp": "asym4W_sym16A",
        "lm_head": "asym4W_sym16A", "quantized_prefix": native.prefix,
        "weight_optimization": native.optimization,
    }, indent=2) + "\n")


def export_shapes(native):
    """直接导出并核对两张静态图,打印实际文件路径."""
    shapes = [f"{native.prefill}t{native.context}c", f"1t{native.context}c"]
    fix_native_shapes(native.quantized, shapes)
    for phase, shape in zip(("Prefill", "Decode"), shapes):
        source = single_file(
            native.quantized.with_name(f"{native.prefix}_{shape}"), "*.tflite")
        print(f"[{phase} 静态图] {source}", flush=True)


def main():
    """在固定工作目录执行转换函数,不再启动仓库内的 Python 子脚本."""
    native = settings(arguments(__file__), python_executable=sys.executable)
    if native.ctx.config["native"].get("reuse_compiled", False):
        from mtk_models.gen_ai.llm.minicpm5_2b.deploy.host.compile import validate_compiled

        validate_compiled(native, check_files=False)
        print("[转换] 已选配方保持不变,不重新校准、量化或导出.", flush=True)
        return
    print("[转换 1/4] 准备模型副本和输入.", flush=True)
    prepare_model(native)
    previous = os.getcwd()
    try:
        os.chdir(native.work)
        print("[转换 2/4] 生成校准数据.", flush=True)
        calibrate(native)
        print("[转换 3/4] 全 W4A16 量化.", flush=True)
        quantize_model(native)
        print("[转换 4/4] 导出 Prefill 和 Decode 静态图.", flush=True)
        export_shapes(native)
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    main()
