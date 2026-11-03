"""读取本模型与 RDK 平台配置,不引用 MTK 转换和部署工具."""

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import yaml

from tools.rdk_config import load_config


def context():
    """解析本模型的 NAS 资源、数据和日期临时目录."""
    model = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((model / "config.yaml").read_text(encoding="utf-8"))
    platform = load_config()
    output = Path(platform["storage"]["models_root"]) / config["model"]["resource_directory"]
    work = Path("/tmp/hailongcodex") / date.today().strftime("%Y%m%d") / "s100" / model.name
    return SimpleNamespace(
        model=model, config=config, platform=platform, output=output, work=work,
        source=Path(platform["toolchain"]["root"]) / config["model"]["source_directory"],
        dataset=Path(platform["storage"]["dataset_root"]) / config["evaluation"]["dataset"],
    )
