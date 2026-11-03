"""从本地官方 YOLOv5 源码和权重导出 ONNX 与 TorchScript."""

from pathlib import Path
import subprocess
import sys
import shutil
import zipfile

from tools.deploy import arguments
from tools.process import run_command


def main():
    """复用官方源码与既有 MTK 补丁,中间 TorchScript 保存到 build."""
    ctx = arguments(__file__)
    source = ctx.work / "yolov5"
    if not source.exists():
        with zipfile.ZipFile(ctx.config["model"]["source_archive"]) as archive:
            root = archive.namelist()[0].split("/")[0]
            archive.extractall(ctx.work)
        (ctx.work / root).rename(source)
    with zipfile.ZipFile(ctx.config["model"]["patch_archive"]) as archive:
        names = [name for name in archive.namelist() if name.endswith("Fix_yolov5_mtk_tflite_issue.patch")]
        if len(names) != 1:
            raise ValueError("补丁压缩包需要包含唯一 MTK 补丁.")
        patch = archive.read(names[0]).decode()
    applied = subprocess.run(["patch", "-d", str(source), "-p1", "--dry-run", "-R"],
                             input=patch, text=True, capture_output=True)
    if applied.returncode:
        run_command(["patch", "-d", source, "-p1", "--forward"], input=patch, text=True)
    weights = ctx.work / "yolov5s.pt"
    shutil.copyfile(ctx.config["model"]["weights"], weights)
    run_command([sys.executable, source / "export.py", "--weights", weights,
                 "--img-size", "640", "640", "--batch-size", "1", "--device", "0",
                 "--include", "torchscript", "onnx"], cwd=source)
    weights.with_suffix(".onnx").replace(ctx.output / "yolov5s.onnx")
    weights.with_suffix(".torchscript").replace(ctx.work / "yolov5s.torchscript")
    print(f"[ONNX] {ctx.output / 'yolov5s.onnx'}", flush=True)


if __name__ == "__main__":
    main()
