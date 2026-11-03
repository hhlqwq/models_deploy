"""读取本次转换配置,调用 NCC 编译 MiniCPM5 的 Prefill 和 Decode 模型."""

import os
import sys
from pathlib import Path

from tools.deploy import arguments
from tools.files import read_json, sha256, single_file
from tools.llm.native import settings
from tools.llm.artifacts import dla_filename
from tools.process import run_command


def validate_compiled(native, *, check_files=True):
    """核对已选配方与固化产物,禁止退回未经验证的重新量化."""
    preparation = read_json(native.work / "native_prepare.json")
    quantization = read_json(native.work / "quantization_settings.json")
    source = read_json(native.source / "source_manifest.json")
    if (quantization.get("selected_recipe") != "light_hessian16_rotated_w4a16"
            or quantization["quantized_prefix"] != native.prefix
            or preparation["context"] != native.context
            or quantization.get("prefill") != native.prefill
            or native.optimization != "hessian" or not native.rotate
            or preparation["source"]["revision"] != source["revision"]
            or quantization.get("calibration_samples") != 16
            or any(quantization[key] != "asym4W_sym16A"
                   for key in ("attention", "mlp", "lm_head"))):
        raise ValueError("配置与已选 W4A16 Hessian 配方不一致,停止复用.")
    files = quantization.get("compiled_files", [])
    if not files or sum(row["name"].endswith(".dla") for row in files) != 2:
        raise ValueError("缺少已选配方的完整双 DLA 资源清单.")
    for row in files:
        path = native.work / row["name"]
        if (path.is_symlink() or not path.resolve().is_relative_to(native.work.resolve())
                or not path.is_file() or path.stat().st_size != row["bytes"]):
            raise ValueError(f"已选配方资源缺失或无效: {row['name']}.")
        if check_files and sha256(path, progress=True) != row["sha256"]:
            raise ValueError(f"已选配方资源内容变化: {row['name']}.")
    print("[已选配方] 复用固定 16 题板端验证的 Hessian/旋转 W4A16 产物.", flush=True)


def compile_dla(native, phase, tokens):
    """编译一张静态图,完整打印命令并检查最终 DLA 非空."""
    folder = native.quantized.with_name(
        f"{native.prefix}_{tokens}t{native.context}c")
    source = single_file(folder, "*.tflite")
    config = read_json(native.model / "config.json")
    target = folder / dla_filename(
        native.model.name, config["num_hidden_layers"], tokens, native.context)
    ncc = Path(os.environ["NCC_ROOT"])
    if phase == "Prefill":
        extra = ["--broadcast-act-wgt", "--broadcast-flow-distance=63",
                 "--split-large-conv-ic=1536"]
    else:
        extra = ["--split-16a4w-conv-oc"]
    run_command([
        "env", f"LD_LIBRARY_PATH={ncc / 'lib'}", ncc / "bin/ncc-tflite",
        "--arch=mdla5.3", "--l1-size-kb=256", "--num-mdla=1",
        "--opt=3", "--opt-footprint", "--opt-accuracy", "--stable-linearize",
        "--gno=LTS,Inception", "--gno-exp", "--gno-non-4d-tiling", "--mlo",
        "--disable-apusys", "--fc-to-conv", "--mdla-int16-lut",
        "--suppress-input", "--suppress-output", "--show-memory-summary",
        "--mdla-conv-exp", "--intval-color-legacy", "--disallow-bridge",
        *extra, source, "--dla-file", target,
    ])
    if not target.is_file() or not target.stat().st_size:
        raise FileNotFoundError(f"DLA 未生成: {target}.")
    print(f"[{phase} DLA] {target}", flush=True)


def main():
    """核对本次配置,按顺序编译两个阶段,不执行转换或上传."""
    native = settings(arguments(__file__), python_executable=sys.executable)
    if native.ctx.config["native"].get("reuse_compiled", False):
        validate_compiled(native)
        return
    preparation = read_json(native.work / "native_prepare.json")
    quantization = read_json(native.work / "quantization_settings.json")
    if (preparation["context"] != native.context
            or quantization["quantized_prefix"] != native.prefix):
        raise ValueError("当前配置与已转换模型不一致,请重新转换后编译.")
    print("[编译 1/2] Prefill.", flush=True)
    compile_dla(native, "Prefill", native.prefill)
    print("[编译 2/2] Decode.", flush=True)
    compile_dla(native, "Decode", 1)


if __name__ == "__main__":
    main()
