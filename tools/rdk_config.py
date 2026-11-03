"""读取 RDK 独立平台配置,不加载 MTK SDK 或平台环境."""

import os
from pathlib import Path
import shlex

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config():
    """合并公开配置和本地覆盖,保留真实连接信息在本地文件."""
    public = ROOT / "configs/platforms/s100.yaml"
    local = Path(os.environ.get(
        "RDK_PLATFORM_CONFIG", ROOT / "configs/platforms/s100.local.yaml"))
    config = yaml.safe_load(public.read_text(encoding="utf-8"))
    if not local.is_file():
        raise ValueError("请先填写 configs/platforms/s100.local.yaml.")
    overrides = yaml.safe_load(local.read_text(encoding="utf-8")) or {}
    for section, value in overrides.items():
        if isinstance(value, dict):
            config[section].update(value)
        else:
            config[section] = value
    if config["id"] != "s100" or config["march"] != "nash-e":
        raise ValueError("当前配置只支持 S100 / nash-e,其他型号需独立配置.")
    return config


def main():
    """输出安全引用的 Shell 环境变量,供 Docker 入口读取."""
    config = load_config()
    values = {
        "RDK_IMAGE": config["toolchain"]["image"],
        "RDK_TOOLCHAIN_ROOT": config["toolchain"]["root"],
        "RDK_CONTAINER_NAME": config["toolchain"]["container_name"],
        "RDK_MODEL_STORAGE_ROOT": config["storage"]["models_root"],
        "RDK_DATASET_ROOT": config["storage"]["dataset_root"],
    }
    for name, value in values.items():
        if not value:
            raise ValueError(f"请在 S100 本地配置填写 {name} 对应字段.")
        print(f"export {name}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
