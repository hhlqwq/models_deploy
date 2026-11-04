"""从现有官方 SDK 配方生成 Qwen3 W4 工作配置,不改原始权重或 SDK."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path

import yaml

from tools.rdk_config import load_config


MODEL = Path(__file__).resolve().parents[2]


def context():
    """读取模型与平台配置,只解析已显式配置的 SDK 和资源路径."""
    platform = load_config()
    config = yaml.safe_load((MODEL / "config.yaml").read_text(encoding="utf-8"))
    sdk = Path(os.environ["RDK_LLM_SDK_ROOT"]).resolve()
    resources = (Path(platform["storage"]["models_root"])
                 / config["model"]["resource_directory"])
    dataset = (Path(platform["storage"]["dataset_root"])
               / config["data"]["wikitext_directory"])
    work = (Path("/tmp/hailongcodex") / datetime.now().date().isoformat()
            / "s100/qwen3_0_6b")
    if platform["march"] != "nash-e":
        raise ValueError("本模型仅使用 S100 / nash-e 配方.")
    if config["quantization"]["weight_bits"] != 4:
        raise ValueError("当前已确认 W4;切换 W8 前先对比精度并更新配方.")
    if config["quantization"]["activation_bits"] != 8:
        raise ValueError("QLG 前置配方保持 SDK 默认 A8,最终硬件类型单独审计.")
    if not sdk.is_dir() or not resources.is_dir() or not dataset.is_dir():
        raise FileNotFoundError("缺少配置中的 SDK、模型资源或 WikiText2 目录.")
    return platform, config, sdk, resources, dataset, work


def read_template(sdk, relative):
    """读取 SDK 自带 YAML,失败时停止,不以自编模板替代官方配方."""
    path = sdk / "llm_compression" / relative
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"SDK 配方不是 YAML 映射: {relative}.")
    return value


def write_config(path, value):
    """生成本次执行的独立配置,不覆盖源目录或 SDK 文件."""
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")


def prepare():
    """核对原始模型并生成三步 W4 与 S100 静态校准、编译配置."""
    platform, config, sdk, resources, dataset, work = context()
    source = resources / config["model"]["source_directory"]
    required = ("config.json", "model.safetensors", "tokenizer.json",
                "tokenizer_config.json", "generation_config.json", "LICENSE")
    for name in required:
        path = source / name
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"原始资源缺失或为空: {name}.")
    original = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if (original["model_type"] != "qwen3"
            or original["architectures"] != ["Qwen3ForCausalLM"]):
        raise ValueError("原始权重不是 SDK Qwen3ForCausalLM 支持的架构.")
    train = dataset / config["data"]["wikitext_train"]
    test = dataset / config["data"]["wikitext_test"]
    if not train.is_file() or not test.is_file() or train.resolve() == test.resolve():
        raise ValueError("WikiText2 训练与测试划分缺失或指向相同文件.")
    work.mkdir(parents=True, exist_ok=True)
    configs = work / "configs"
    configs.mkdir(exist_ok=True)
    for index, name in enumerate(("step_1_quarot.yml", "step_2_omniq_lwc.yml",
                                  "step_3_gptq.yml"), 1):
        recipe = read_template(sdk, "lightcompress/configs/released_models/qwen3/" + name)
        if (recipe["model"]["type"] != "Qwen3"
                or recipe["quant"]["weight"]["bit"] != 4
                or recipe["quant"]["act"]["bit"] != 8):
            raise ValueError(f"SDK Qwen3 W4A8 配方已变更,请重新核对: {name}.")
        recipe["model"]["path"] = str(source if index == 1 else
                                        work / f"qlg/step{index - 1}/transformed_model")
        # SDK 默认全量评价不替代仓库协议;此处只执行量化,评价由独立入口负责.
        recipe.pop("eval", None)
        if "calib" in recipe:
            recipe["calib"]["path"] = str(dataset)
            recipe["calib"]["download"] = False
        recipe["save"]["save_path"] = str(work / f"qlg/step{index}")
        if index == 3:
            recipe["quant"]["reload_quant_scale_tensor_path"] = str(
                work / "qlg/step2/transformed_model")
        write_config(configs / f"qlg_step{index}.yaml", recipe)
        print(f"[配方 {index}/3] {name},保持官方算法参数.", flush=True)
    # SDK PPL 校准读取 test-*.parquet;独立视图仅链接 TRAIN,不使用 TEST 校准.
    calibration = work / "calibration_train"
    calibration.mkdir(exist_ok=True)
    link = calibration / "test-train-source.parquet"
    if link.is_symlink():
        if link.resolve() != train.resolve():
            raise ValueError("校准数据链接已指向其他资源,请核对后处理.")
    elif link.exists():
        raise ValueError("校准视图已有非链接文件,停止覆盖.")
    else:
        link.symlink_to(train.resolve())
    if sorted(p.name for p in calibration.iterdir()) != [link.name]:
        raise ValueError("校准视图含有额外文件,不能确认只有 TRAIN 数据.")
    hardware = read_template(sdk, "configs/qwen3_s100.yml")
    hardware["model"].update(march=platform["march"], path=str(source),
                             fake_quant_path=str(work / "qlg/step3/fake_quant_model"))
    hardware["model"]["text_config"].update(
        max_kvcache_len=config["conversion"]["context"],
        max_lm_input_len=config["conversion"]["prefill"])
    hardware["calibration"].update(
        data_path=str(calibration), log_path=str(work / "qconfig"),
        calib_ckpt_save_path=str(work / "calibrated"),
        calibration_step=config["quantization"]["calibration_steps"])
    hardware["compile"].update(
        hbm_save_path=str(work / "compiled"), calib_ckpt_load_path=str(work / "calibrated"),
        cache_path=str(work / "compiler_cache"))
    # 该配置只用于校准/编译,避免意外启动 SDK 默认全量 PPL.
    hardware.pop("evaluation", None)
    write_config(configs / "s100.yaml", hardware)
    print("[配置完成] 三步 QLG + nash-e 静态校准/编译,尚未执行模型.", flush=True)


def main():
    """默认准备配置,可仅查询日期工作路径供 Shell 入口使用."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-work", action="store_true")
    args = parser.parse_args()
    if args.print_work:
        print(context()[-1])
    else:
        prepare()


if __name__ == "__main__":
    main()
