"""复用公共命令工具上传完整 Python 混合部署包,不执行板端入口."""

import argparse
import json
import os
from pathlib import Path
import shlex

from tools.deploy import context, prune_models
from tools.process import run_command


def upload_inputs(ctx):
    """只上传清单引用的传感器文件,随后核对板端全部输入,不执行推理."""
    manifest_path = ctx.work / "samples.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = sorted({name for frame in manifest["frames"] for name in (
        frame["lidar"], *(item["path"] for item in frame["sweeps"]),
        *(item["path"] for item in frame["cameras"]))})
    if not names:
        raise ValueError("BEVFusion 输入清单为空.")
    sizes = {}
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\n" in name:
            raise ValueError(f"BEVFusion 输入路径无效: {name}.")
        source = ctx.dataset / relative
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(source)
        sizes[name] = source.stat().st_size
    listing = ctx.work / "input_files.txt"
    listing.write_text("\n".join(names) + "\n", encoding="utf-8")
    host = os.environ["MTK_BOARD_HOST"]
    dataset = ctx.config["board"]["dataset_dir"]
    options = ["-o", "BatchMode=yes"]
    print(f"[子集输入] {len(manifest['frames'])} 帧,{len(names)} 个文件,"
          f"{sum(sizes.values()) / 1024 ** 2:.1f} MiB.", flush=True)
    run_command(["ssh", *options, host, "mkdir -p -- " + shlex.quote(dataset)])
    run_command(["rsync", "-rltc", "--no-perms", "--omit-dir-times", "--protect-args",
                 "--info=progress2", "--files-from=" + str(listing),
                 "-e", "ssh -o BatchMode=yes", str(ctx.dataset) + "/", f"{host}:{dataset}/"])
    # 复用已上传的同一份清单,逐个核对文件,不靠一个样例判断整体上传成功.
    script = "\n".join((
        "import json, sys",
        "from pathlib import Path",
        "request = json.load(sys.stdin)",
        f"root = Path({dataset!r})",
        f"manifest = json.loads(Path({str(Path(ctx.config['board']['deploy_dir']) / 'build/reference/compile_samples.json')!r}).read_text())",
        "assert manifest == request['manifest'], '板端与主机子集清单不一致'",
        "missing = [name for name, size in request['sizes'].items() if not (root / name).is_file() or (root / name).stat().st_size != size]",
        "if missing: raise FileNotFoundError('板端输入缺失或大小不符: ' + repr(missing[:10]))",
        "print('[输入上传完成] 全部传感器文件核对通过,未执行推理.', flush=True)",
    ))
    run_command(["ssh", *options, host, "python3 -c " + shlex.quote(script)],
                input=json.dumps({"manifest": manifest, "sizes": sizes}).encode())


def main():
    """上传固定子图,CPU 参数,一帧输入和板端 Python 依赖代码."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", action="store_true", help="上传已评分子集对应的原始输入.")
    args = parser.parse_args()
    if args.inputs:
        os.environ.update(MTK_EVALUATION_WORK="1", MTK_COMPILE_ACCURACY="1")
    ctx = context(__file__)
    if args.inputs:
        upload_inputs(ctx)
        return
    package = ctx.output / "models"
    manifest = json.loads((package / "contract.json").read_text(encoding="utf-8"))
    ctx.config["deployment"] = {"model_directory": "models"}
    config = ctx.work / "config.json"
    config.write_text(json.dumps(ctx.config, ensure_ascii=False, indent=2), encoding="utf-8")
    model_files = [package / name for name in ("contract.json", "sparse_weights.npz", "geometry.npz")]
    for name, item in manifest["stages"].items():
        model_files.append(package / f"bevfusion-det_{name}.onnx")
        if item["backend"] == "mtk_npu":
            model_files.append(package / item["file"])
    groups = {"": [ctx.model / "deploy/compile.sh", ctx.model / "deploy/evaluate.sh",
                     ctx.model / "deploy/requirements-evaluate.txt", config],
              "build/reference": [], "models": model_files, "board": [*list((ctx.model / "deploy/board").glob("*.py")), ctx.repo / "tools/compile_accuracy.py", ctx.repo / "tools/summary.py"],
              "board/tools": [ctx.repo / "tools" / name for name in
                              ("__init__.py", "runtime.py", "board.py", "evaluation.py", "files.py", "output_metrics.py")],
              "examples/input": [ctx.work / "sample.npz", ctx.work / "sample.json"]}
    for files in groups.values():
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
    host, destination = os.environ["MTK_BOARD_HOST"], ctx.config["board"]["deploy_dir"]
    options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    directories = [str(Path(destination) / folder) for folder in groups]
    run_command(["ssh", *options, host, "mkdir -p " + " ".join(map(shlex.quote, directories))])
    for folder, files in groups.items():
        if files:
            run_command(["scp", *options, *files, f"{host}:{destination}/{folder}/"])
    prune_models(ctx, [path.name for path in model_files])
    print("[上传完成] 待板端检查.请登录开发板,在本模型部署目录执行 bash compile.sh.", flush=True)


if __name__ == "__main__":
    main()
