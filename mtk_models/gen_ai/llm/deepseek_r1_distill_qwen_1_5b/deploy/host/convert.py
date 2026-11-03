"""顺序执行 DeepSeek 模型准备、校准、W4A16 量化和静态图导出."""

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
from tools.storage import model_storage, storage_roots


def prepare(source, output, context, calibration_train=None, mask_value=-10000.0,
            rotate=False, calibration_train_samples=8):
    """读取官方文件并生成独立配置、双语校准输入和板端 Prompt."""
    manifest = json.loads((source / "source_manifest.json").read_text())
    if output.resolve() == source.resolve() or context < 128:
        raise ValueError("输出必须独立于源目录,上下文不得小于 128.")
    if calibration_train is not None and calibration_train_samples < 8:
        raise ValueError("训练校准至少需要 8 段文本,并保留 8 条双语输入.")
    output.mkdir(parents=True, exist_ok=True)
    model = output / "DeepSeek-R1-Distill-Qwen-1.5B"
    model.mkdir(exist_ok=True)
    models_root = storage_roots()["models_root"].resolve()
    for entry in manifest["files"]:
        src = source / entry["name"]
        if (Path(entry["name"]).name != entry["name"]
                or not src.resolve().is_relative_to(models_root)
                or not src.is_file() or src.stat().st_size != entry["bytes"]):
            raise ValueError(f"源文件缺失、大小不符或不在模型资源目录: {entry['name']}.")
        dst = model / src.name
        if src.suffix == ".safetensors":
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                print(f"[权重复制] {src.name}", flush=True)
                shutil.copyfile(src, dst)
        else:
            shutil.copyfile(src, dst)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    config = json.loads((source / "config.json").read_text())
    generation = json.loads((source / "generation_config.json").read_text())
    stops = generation.get("eos_token_id", tokenizer.eos_token_id)
    stops = stops if isinstance(stops, list) else [stops]
    if tokenizer.eos_token_id not in stops:
        raise ValueError("官方 tokenizer 与 generation_config 的停止 Token 不一致.")
    if not (-10000.0 <= mask_value < 0):
        raise ValueError("掩码负值必须位于 [-10000,0).")
    config.update(mask_value=mask_value, rotate=rotate,
                  rotate_seed=0, rotate_mode="hadamard")
    # 推理特殊 Token 以官方 tokenizer/generation 配置为准,修正基础配置遗留值.
    config.update(bos_token_id=tokenizer.bos_token_id,
                  eos_token_id=tokenizer.eos_token_id,
                  pad_token_id=tokenizer.pad_token_id,
                  tokenizer="pretrained_fast", max_position_embeddings=context)
    (model / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": "{instruction}"}],
        tokenize=False, add_generation_prompt=True)
    (output / "deepseek_preformatter.json").write_text(json.dumps({
        "description": "DeepSeek 官方单轮 Chat Template.",
        "prompt_input": template, "prompt_no_input": template,
        "response_split": "<｜Assistant｜>",
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
            tokenize=False, add_generation_prompt=True)
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
        training_tokens = calibration_train_samples * 256
        # 默认 8 段保持原取样,新增覆盖复用至少 32 个均匀候选位置.
        pool_samples = (8 if calibration_train_samples == 8
                        else max(32, calibration_train_samples))
        if len(train_tokens) < pool_samples * 256:
            raise ValueError(f"校准训练文本不足 {pool_samples * 256} Token,不能重复填充.")
        mixed_records = []
        pool_offsets = [(len(train_tokens) - 256) * index // (pool_samples - 1)
                        for index in range(pool_samples)]
        offsets = [
            pool_offsets[(pool_samples - 1) * index // (calibration_train_samples - 1)]
            for index in range(calibration_train_samples)]
        for index in range(calibration_train_samples):
            # 均匀覆盖训练文本,避免全部校准片段集中在文件开头.
            segment = train_tokens[offsets[index]:offsets[index] + 256]
            mixed_records.append({"tokens": " ".join(map(str, segment))})
            if index < 8:
                mixed_records.append(records[index])
        records = mixed_records
        calibration_scope = (
            f"wikitext2_train_{calibration_train_samples}x256_uniform_"
            "and_authored_bilingual_8")
        training_source = {"revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
                           "split": "train", "sha256": sha256(calibration_train),
                           "text_tokens": training_tokens,
                           "selection": ("uniform_token_offsets"
                                         if pool_samples == calibration_train_samples
                                         else "uniform_token_grid_subset"),
                           "token_offsets": offsets}
        if pool_samples != calibration_train_samples:
            training_source["sampling_pool_samples"] = pool_samples
    demos = []
    # 展示输入独立于校准记录,使用官方聊天模板生成 Token.
    for name, question in (("zh_demo", "请用中文简单介绍你自己。"),
                           ("en_demo", "Briefly introduce yourself in English.")):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        (prompts / f"{name}.txt").write_text(text)
        (prompts / f"{name}.tokens.txt").write_text(
            " ".join(str(token) for token in tokens) + "\n")
        demos.append({"id": name, "prompt": question, "input_ids": tokens})
    (output / "calibration.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records))
    (output / "native_prepare.json").write_text(json.dumps({
        "source": manifest, "context": context,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id, "stop_token_ids": stops,
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
    roots = storage_roots()
    if (not native.source.resolve().is_relative_to(roots["models_root"].resolve())
            or not native.source.is_dir()):
        raise ValueError("官方源文件必须位于 NAS 模型资源目录.")
    if native.train and not Path(native.train).resolve().is_relative_to(
            roots["dataset_root"].resolve()):
        raise ValueError("校准训练集必须位于 NAS 数据集目录.")
    if native.prefill != 128 or native.context % 128:
        raise ValueError("本模型要求 Prefill=128,上下文为 128 的整数倍.")
    training_samples = int(native.ctx.config["native"].get("calibration_train_samples", 8))
    if native.train and training_samples < 8:
        raise ValueError("训练校准至少需要 8 段文本.")
    expected_samples = training_samples + 8 if native.train else 16
    if native.batches != expected_samples * 2:
        raise ValueError(
            f"当前 {expected_samples} 条校准输入需要 {expected_samples * 2} 批次,"
            "请同步配置 calibration_train_samples 与 calibration_batches.")
    if native.work.is_symlink() or native.work.resolve() == native.source.resolve():
        raise ValueError("转换目录不能是软链接或原始资源目录.")
    expected = model_storage(Path(__file__).resolve().parents[2]) / "build/native"
    if (native.work.resolve() != expected.resolve()
            or not native.work.resolve().is_relative_to(expected.parents[1].resolve())):
        raise ValueError("DeepSeek 转换目录必须为模型资源下的 build/native.")
    if native.work.exists():
        print("[清理] 替换上一次转换工作目录,保留官方资源及当前部署包.", flush=True)
        shutil.rmtree(native.work)
    prepare(native.source, native.work, native.context,
            Path(native.train) if native.train else None, native.mask, bool(native.rotate),
            calibration_train_samples=training_samples)


def calibrate(native):
    """执行 SDK 校准命令,失败时直接停止本次转换."""
    command = [
        native.python, native.python.parent / "mtk_make_llm_ptq_calib_dataset",
        "converter", native.model / "config.json",
        native.work / "calibration.jsonl", "-b", str(native.batches), "-m", "128",
    ]
    run_command(command, cwd=native.work)


def quantize_model(native):
    """在当前 Python 进程调用 SDK 适配函数并检查全部量化分片."""
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
        "weight_optimization_options": (
            json.loads(Path(native.extra).read_text()) if native.extra else {}),
        "rotate": bool(native.rotate),
        "rotate_mode": config["rotate_mode"], "rotate_seed": config["rotate_seed"],
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
