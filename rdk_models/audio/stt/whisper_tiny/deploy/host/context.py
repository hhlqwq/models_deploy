"""解析 S100 Whisper 专属资源与当前主机工作目录."""

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import yaml

from tools.rdk_config import load_config


def context():
    """读取本模型及 S100 配置,模型和数据集均使用既有 NAS 资源."""
    model = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((model / "config.yaml").read_text(encoding="utf-8"))
    platform = load_config()
    output = Path(platform["storage"]["models_root"]) / config["model"]["resource_directory"]
    return SimpleNamespace(model=model, config=config, platform=platform, output=output,
        assets=output / "build/reference",
        dataset=Path(platform["storage"]["dataset_root"]) / config["evaluation"]["dataset"],
        work=Path("/tmp/hailongcodex") / date.today().isoformat() / "s100/whisper_tiny")
