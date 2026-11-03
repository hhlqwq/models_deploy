"""将已编译的原生 MiniCPM5 模型整理为 NAS 同类目录结构."""

import copy
import json
from pathlib import Path
import shutil
import sys


from tools.deploy import arguments
from tools.evaluation import is_test
from tools.files import copy_public_tools, sha256, single_file
from tools.llm.native import settings
from tools.llm.artifacts import dla_filename
from tools.llm.package import generation_resources, graph_contract
from tools.storage import REPO_ROOT


def package(work, output, prefill, quantized_prefix=None):
    """复制真实 DLA 与量化资源,生成配置和待板端验证清单."""
    preparation = json.loads((work / "native_prepare.json").read_text())
    context = preparation["context"]
    model = work / "MiniCPM5-2B"
    config = json.loads((model / "config.json").read_text())
    settings_path = work / "quantization_settings.json"
    saved_settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    prefix = saved_settings.get("quantized_prefix", "MiniCPM5-2B_asym4W_sym16A_Overall_hessian")
    base = quantized_prefix or work / "tflite" / prefix
    if base.resolve().parent != (work / "tflite").resolve():
        raise ValueError("量化目录必须位于本次任务的 tflite 目录.")
    if settings_path.exists():
        precision_settings = json.loads(settings_path.read_text())
        if precision_settings["quantized_prefix"] != base.name:
            raise ValueError("量化精度记录与打包目录不一致.")
        precision_values = [precision_settings[key]
                            for key in ("attention", "mlp", "lm_head")]
        if any(value != "asym4W_sym16A" for value in precision_values):
            raise ValueError("本项目交付仅允许全 W4A16.")
        precision = "asym4W_sym16A"
    else:
        # 兼容现有 W4 包,并核对量化目录的精度标识.
        if "_asym4W_sym16A_Overall_hessian" not in base.name:
            raise ValueError("新精度包缺少量化成功记录.")
        precision = "asym4W_sym16A"
        precision_settings = dict.fromkeys(("attention", "mlp", "lm_head"), precision)
    adapter_path = work / "quantization_adapter.json"
    adapter = json.loads(adapter_path.read_text()) if adapter_path.exists() else None
    # 归档统一精度的实际逐层核对记录,并核对其与交付配置一致.
    uniform_path = work / "weight_precision_audit.json"
    uniform_audit = (json.loads(uniform_path.read_text())
                     if uniform_path.exists() else None)
    if uniform_audit is not None:
        if any(uniform_audit[key] != precision_settings[key]
               for key in ("attention", "mlp", "lm_head")):
            raise ValueError("统一权重精度核对与交付配置不一致.")
        if uniform_audit["parts"] != config["num_hidden_layers"] + 1:
            raise ValueError("统一权重精度核对的分片数量不符.")
    contracts, model_files, sources = {}, {}, {}
    for phase, tokens in (("prompt", prefill), ("decode", 1)):
        folder = Path(f"{base}_{tokens}t{context}c")
        graph = single_file(folder, "*.tflite")
        filename = dla_filename(model.name, config["num_hidden_layers"], tokens, context)
        sources[phase] = folder / filename
        if not sources[phase].is_file():
            raise FileNotFoundError(f"缺少已编译的 DLA: {sources[phase]}.")
        model_files[phase] = f"{context}c/{filename}"
        contracts[phase] = dict(graph_contract(graph), tflite_sha256=sha256(graph))
    embedding = single_file(base, "embedding_int16.bin")
    expected_bytes = config["hidden_size"] * config["vocab_size"] * 2
    if embedding.stat().st_size != expected_bytes:
        raise ValueError("原生 INT16 embedding 大小与官方词表不符.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("交付目录必须为空,避免覆盖既有交付文件.")
    for name in (f"{context}c", "tokenizer", "scripts/prompts"):
        (output / name).mkdir(parents=True, exist_ok=True)
    for src, dst in ((sources["prompt"], model_files["prompt"]),
                     (sources["decode"], model_files["decode"]),
                     (embedding, "tokenizer/embedding_int16.bin")):
        print(f"[打包] {dst}", flush=True)
        shutil.copyfile(src, output / dst)
    for path in (work / "prompts").glob("*.txt"):
        shutil.copyfile(path, output / "scripts/prompts" / path.name)
    generation_resources(model, output, contracts)
    shutil.copyfile(model / "LICENSE", output / "LICENSE")
    (output / "README.md").write_text(
        "# MiniCPM5-2B\n\n"
        "Genio 720 全 W4A16 部署包,支持英文生成、性能和精度评价.\n"
        "本部署包尚待板端评价.\n\n"
        "编译服务器仅编译和上传,不会自动执行板端推理.\n"
        "请登录开发板,在本部署目录运行:\n\n"
        "```bash\nbash compile.sh\n```\n\n"
        "板端 compile.sh 仅检查通路和生成示例,实时显示输入及生成文本.\n"
        "示例最多生成 32 Token,可能只是回答片段;正式评价执行 `bash evaluate.sh`.\n"
        "配置见 config.json,示例见 examples/output,正式报告只有 results/summary.json.\n"
        "具体工具命令及参数由执行代码维护并打印.\n")
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            print(f"[哈希] {path.relative_to(output)}", flush=True)
            files.append({"name": path.relative_to(output).as_posix(),
                          "bytes": path.stat().st_size, "sha256": sha256(path)})
    (output / "manifest.json").write_text(json.dumps({
        "status": "compiled_pending_board", "preparation": preparation,
        "precision": precision, "precision_settings": precision_settings,
        "quantization_adapter": adapter,
        "weight_precision_audit": uniform_audit,
        "weight_optimization": precision_settings.get(
            "weight_optimization",
            "hessian" if "_Overall_hessian" in base.name else
            "gradient" if "_Overall_gradient" in base.name else None),
        "optimized_layers": precision_settings.get("optimized_layers"),
        "optimization_calibration_samples": precision_settings.get("calibration_samples"),
        "cumulative_layer_error": precision_settings.get("cumulative_layer_error"),
        "quantized_prefix": base.name,
        "prefill": prefill, "context": context, "files": files,
        "graph_contracts": contracts, "model_files": model_files,
        "toolkit_sha256": "da10e770e2950542ab17c63182b0b932348ab05ffdceb18cbb097555ca6127f5",
    }, ensure_ascii=False, indent=2) + "\n")


def package_model(native):
    """由上传步骤直接调用,构建本次模式的部署包和同协议输入."""
    output = native.ctx.work / "package"
    if (output.is_symlink() or output.resolve().parent != native.ctx.work.resolve()
            or not output.resolve().is_relative_to(native.ctx.output.resolve())):
        raise ValueError("临时部署包必须位于本模型 build/package.")
    if output.exists():
        print("[清理] 替换上一次临时部署包.", flush=True)
        shutil.rmtree(output)
    package(native.work, output, native.prefill, native.quantized)
    shutil.copyfile(native.ctx.model / "deploy/compile.sh", output / "compile.sh")
    shutil.copyfile(native.ctx.model / "deploy/evaluate.sh", output / "evaluate.sh")
    board = output / "board"
    board.mkdir()
    for name in ("infer.py", "evaluate.py"):
        shutil.copyfile(native.ctx.model / "deploy/board" / name, board / name)
    shutil.copyfile(REPO_ROOT / "tools/summary.py", board / "summary.py")
    shutil.copyfile(REPO_ROOT / "tools/compile_accuracy.py", board / "compile_accuracy.py")
    copy_public_tools(board / "tools", ("files", "evaluation", "board", "runtime", "output_metrics"))
    shared = board / "tools/llm"
    shared.mkdir()
    for name in ("__init__", "artifacts", "evaluate_board_quality", "evaluate_mmlu", "generate"):
        shutil.copyfile(REPO_ROOT / "tools/llm" / f"{name}.py", shared / f"{name}.py")
    snapshot = copy.deepcopy(native.ctx.config)
    snapshot["model"]["source"] = str(native.source)
    snapshot["conversion"]["context"] = native.context
    snapshot["native"].update(
        prefill=native.prefill, calibration_batches=native.batches,
        rotate=bool(native.rotate), mask_value=native.mask,
        weight_optimization=native.optimization, weight_opt_config=native.extra,
        calibration_train=str(native.train))
    (output / "config.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = [
        {"name": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
         "sha256": sha256(path, progress=True)}
        for path in sorted(output.rglob("*")) if path.is_file() and path != manifest_path]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    destination = native.ctx.output / "models"
    if destination.is_symlink():
        raise ValueError("当前部署包不能是软链接.")
    if destination.exists():
        shutil.rmtree(destination)
    output.rename(destination)
    output = destination
    print(f"[打包完成] {output}; 待上传及真实板端验证.", flush=True)
    return output


if __name__ == "__main__":
    package_model(settings(arguments(__file__), python_executable=sys.executable))
