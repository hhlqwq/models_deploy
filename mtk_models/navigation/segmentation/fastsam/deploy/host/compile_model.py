"""编译 FastSAM DLA,在 Docker 内执行."""

import argparse
import os
from pathlib import Path

from tools.storage import REPO_ROOT, load_model_config

from tools.process import run_command


def compile_model(config: dict, paths: dict, config_path: Path) -> None:
    """调用 Neuron 编译器生成 Genio 720 使用的 DLA."""
    if not os.environ.get("NCC_ROOT"):
        raise RuntimeError("请先通过 docker/mtk_g720/enter_container.sh 进入容器.")
    ncc = Path(os.environ["NCC_ROOT"])
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = f"{ncc / 'lib'}:{environment.get('LD_LIBRARY_PATH', '')}"
    run_command([ncc / "bin/ncc-tflite", "--arch=mdla5.3", "--suppress-output",
                 "--disallow-bridge", paths["work"] / "model_int8.tflite",
                 "-o", paths["output"] / "model_int8.dla"], env=environment)
    print(f"[DLA] 文件路径: {(paths['output'] / 'model_int8.dla').resolve()}", flush=True)


def main() -> None:
    """直接读取 YAML 配置并执行本文件对应的操作."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "config.yaml")
    parser.add_argument("--work-dir", type=Path, help="覆盖默认的模型外部资源 build 目录.")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_model_config(config_path)
    model = Path(__file__).resolve().parents[2]
    work = args.work_dir or Path(config["model"]["output_dir"]) / "build"
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    paths = {"model": model, "repo": REPO_ROOT, "work": work,
             "dataset": Path(config["data"]["dataset_dir"]),
             "output": model / config["model"]["output_dir"]}
    print(f"[步骤] {Path(__file__).stem}; 工作目录: {work}", flush=True)
    compile_model(config, paths, config_path)


if __name__ == "__main__":
    main()
