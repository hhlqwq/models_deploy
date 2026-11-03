"""读取不提交 Git 的资源路径配置,代码目录与模型资源目录分离."""

import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import shlex

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def reuse_weight(source, destination, expected_sha256):
    """复用官方权重,共享盘不支持软链接时使用经过哈希校验的副本."""
    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() != source:
        # 只更新引用,不删除引用所指向的原始权重.
        destination.unlink()
    if not destination.exists():
        try:
            destination.symlink_to(source)
            print(f"[权重] 已引用官方资源: {source.name}", flush=True)
        except OSError:
            print(f"[权重] 共享存储不支持软链接,复制: {source.name}", flush=True)
            temporary = destination.with_name(destination.name + ".copy-part")
            copied = 0
            with source.open("rb") as incoming, temporary.open("wb") as outgoing:
                for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
                    outgoing.write(block)
                    copied += len(block)
                    if copied % (512 * 1024 * 1024) == 0:
                        print(f"[权重复制] {copied // 1024**2} MiB", flush=True)
            temporary.replace(destination)
    digest = hashlib.sha256()
    processed = 0
    with destination.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            processed += len(block)
            if processed % (512 * 1024 * 1024) == 0:
                print(f"[权重校验] {processed // 1024**2} MiB", flush=True)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"共享存储权重副本哈希不匹配: {destination.name}")


def storage_roots():
    """读取本地配置或环境变量,返回模型和数据集的绝对路径."""
    config_path = Path(os.environ.get(
        "MTK_STORAGE_CONFIG", REPO_ROOT / "configs/storage.local.json"))
    config = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    roots = {}
    for key, variable in (("models_root", "MTK_MODEL_STORAGE_ROOT"),
                          ("dataset_root", "MTK_DATASET_ROOT")):
        value = os.environ.get(variable) or config.get(key)
        if not value or not Path(value).is_absolute():
            raise ValueError(
                f"请在 configs/storage.local.json 配置 {key} 的绝对路径,"
                f"或设置 {variable}.")
        roots[key] = Path(value)
    return roots


def model_storage(model_directory):
    """按模型的场景、类别、名称映射到外部资源目录."""
    relative = Path(model_directory).resolve().relative_to(REPO_ROOT / "mtk_models")
    if len(relative.parts) != 3:
        raise ValueError("模型代码目录必须为 mtk_models/场景/类别/模型名.")
    return storage_roots()["models_root"] / relative


def load_model_config(config_path):
    """解析各模型 YAML,公共环境与板端根路径统一来自平台配置."""
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    roots = storage_roots()
    model_root = model_storage(config_path.parent)
    config["name"] = config_path.parent.name
    config["model"].setdefault("output_dir", ".")
    for key in ("weights", "output_dir", "source", "source_archive", "patch_archive", "onnx"):
        value = config["model"].get(key)
        if value:
            path = Path(value)
            config["model"][key] = str(path if path.is_absolute() else model_root / path)
    dataset = config.get("data", {}).get("dataset_dir")
    if dataset:
        path = Path(dataset)
        config["data"]["dataset_dir"] = str(
            path if path.is_absolute() else roots["dataset_root"] / path)
    platform = platform_environment()
    board = config.setdefault("board", {})
    for key, base, default in (
            ("deploy_dir", "MTK_BOARD_OPEN_MODELS_ROOT", config["name"]),
            ("dataset_dir", "MTK_BOARD_DATASETS_ROOT", ""),
            ("results_dir", None, "results")):
        path = Path(board.get(key, default))
        board[key] = str(Path(platform[base]) / path if base and not path.is_absolute() else path)
    return config


def platform_environment():
    """合并公开规格与本地环境配置,环境变量具有最高优先级."""
    config = yaml.safe_load(
        (REPO_ROOT / "configs/platforms/genio_720.yaml").read_text(encoding="utf-8"))
    local_path = Path(os.environ.get(
        "MTK_PLATFORM_CONFIG", REPO_ROOT / "configs/platforms/genio_720.local.yaml"))
    if "MTK_PLATFORM_CONFIG" in os.environ or local_path.is_file():
        local = yaml.safe_load(local_path.read_text(encoding="utf-8")) or {}
        if not isinstance(local, dict) or not isinstance(local.get("toolchain", {}), dict):
            raise ValueError("本地平台配置及 toolchain 必须是映射.")
        config["toolchain"].update(local.get("toolchain", {}))
        config.update({key: value for key, value in local.items() if key != "toolchain"})
    toolchain = config["toolchain"]
    sdk_value = os.environ.get("MTK_NP_ROOT") or toolchain["sdk_root"]
    if not sdk_value:
        raise ValueError("请在 configs/platforms/genio_720.local.yaml 配置 toolchain.sdk_root,或设置 MTK_NP_ROOT.")
    sdk = Path(sdk_value)
    ncc = Path(os.environ.get("NCC_ROOT") or sdk / "neuron_sdk/host")
    defaults = {
        "MTK_TOOLCHAIN_ROOT": toolchain["root"],
        "MTK_CONTAINER_NAME": toolchain["container_name"],
        "MTK_GAI_PYTHON": toolchain["gai_python"],
        "MTK_LLM_PYTHON": toolchain["llm_python"],
        "MTK_YAMNET_PYTHON": toolchain["yamnet_python"],
        "MTK_HOST_SDK_ROOT": sdk,
        "MTK_NP_ROOT": sdk,
        "NCC_ROOT": ncc,
        "MTK_NCC_BIN": ncc / "bin",
        "MTK_NCC_LIB": ncc / "lib",
        "MTK_NEURON_INCLUDE": ncc / "include",
        "MTK_BOARD_HOST": config["board_host"],
        "MTK_BOARD_ROOT": config["board_root"],
        "MTK_BOARD_OPEN_MODELS_ROOT": config["board_open_models_root"],
        "MTK_BOARD_DATASETS_ROOT": config["board_datasets_root"],
        "MTK_BOARD_MODELZOO_ROOT": config["board_modelzoo_root"],
    }
    environment = {name: os.environ.get(name) or value for name, value in defaults.items()}
    gai_sdk_root = os.environ.get("MTK_GAI_SDK_ROOT") or toolchain.get("gai_sdk_root")
    if gai_sdk_root:
        environment["MTK_GAI_SDK_ROOT"] = str(gai_sdk_root)
    missing = [name for name, value in environment.items() if not value]
    if missing:
        raise ValueError("请在本地平台配置或环境变量中填写: " + ", ".join(missing))
    for name, key in (("MTK_RESULTS_HOST", "results_host"),
                      ("MTK_RESULTS_ROOT", "results_root")):
        value = os.environ.get(name) or config.get(key)
        if value:
            environment[name] = str(value)
    return environment


def write_upload_config():
    """生成仅供板端回传使用的本地配置,返回路径供 Shell 上传."""
    config = platform_environment()
    values = {name: config.get(name, "") for name in
              ("MTK_RESULTS_HOST", "MTK_RESULTS_ROOT")}
    root = Path(values["MTK_RESULTS_ROOT"])
    if not all(values.values()) or not root.is_absolute() or root == Path("/"):
        raise ValueError("请在本地平台配置填写 results_host 和 results_root 仓库绝对路径.")
    directory = Path("/tmp/hailongcodex") / date.today().isoformat() / f"results_upload_{os.getuid()}"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "platform.local.sh"
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    with os.fdopen(os.open(target, flags, 0o600), "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write("".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items()))
    return target


def main():
    """输出经过 Shell 转义的变量,供 Bash 入口加载."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--write-upload-config", action="store_true",
                        help="生成板端本地回传配置,只输出文件路径.")
    parser.add_argument("--print-build-dir", type=Path, metavar="MODEL_CONFIG",
                        help="只输出模型配置对应的构建目录.")
    args = parser.parse_args()
    if args.write_upload_config:
        print(write_upload_config())
        return
    if args.print_build_dir:
        config = load_model_config(args.print_build_dir)
        print(Path(config["model"]["output_dir"]) / "build")
        return
    roots = storage_roots()
    values = {"MTK_MODEL_STORAGE_ROOT": roots["models_root"],
              "MTK_DATASET_ROOT": roots["dataset_root"]}
    values.update(platform_environment())
    config_path = Path(os.environ.get(
        "MTK_STORAGE_CONFIG", REPO_ROOT / "configs/storage.local.json"))
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("container_image"):
            values["MTK_G720_IMAGE"] = config["container_image"]
    if args.model:
        values["MODEL_ASSETS_DIR"] = model_storage(args.model)
    if args.model_config:
        model_config = load_model_config(args.model_config)
        values["MODEL_BUILD_DIR"] = Path(model_config["model"]["output_dir"]) / "build"
    for name, value in values.items():
        print(f"export {name}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
