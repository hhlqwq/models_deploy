"""打包并上传 S100 运行所需文件,不通过 SSH 自动启动推理."""

import posixpath
from pathlib import Path, PurePosixPath
import sys
from time import monotonic

import cv2
import numpy as np
import onnxruntime as ort

from context import context


def upload(ctx, files):
    """使用本地设备配置和已知主机密钥上传明确的文件清单."""
    import paramiko

    board = ctx.platform["board"]
    work = PurePosixPath(board["work_dir"])
    if not work.is_absolute() or str(work) == "/" or ".." in work.parts:
        raise ValueError("板端工作目录必须为非根目录的绝对路径.")
    destination = str(work / ctx.config["board"]["deploy_directory"])
    repo = Path(__file__).resolve().parents[6]
    known = repo / "configs/platforms/s100.local.known_hosts"
    client = paramiko.SSHClient()
    client.load_host_keys(str(known))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(
        hostname=board["host"], username=board["user"], password=board.get("password"),
        timeout=15, look_for_keys=True, allow_agent=True)
    try:
        with client.open_sftp() as sftp:
            for index, (source, relative) in enumerate(files, 1):
                target = posixpath.join(destination, relative)
                folder = "/"
                for component in PurePosixPath(target).parent.parts[1:]:
                    folder = posixpath.join(folder, component)
                    try:
                        sftp.stat(folder)
                    except FileNotFoundError:
                        sftp.mkdir(folder)
                if len(files) <= 20 or index % 25 == 0 or index == len(files):
                    print(f"[上传 {index}/{len(files)}] {relative}", flush=True)
                last_progress = [monotonic()]

                def progress(sent, total):
                    """长文件传输每十秒显示一次进度,不创建日志副本."""
                    now = monotonic()
                    if now - last_progress[0] >= 10:
                        print(f"[传输] {relative}: {sent / 1024**2:.1f}/{total / 1024**2:.1f} MiB", flush=True)
                        last_progress[0] = now

                sftp.put(str(source), target, callback=progress)
    finally:
        client.close()
    print("[上传完成] 未自动执行板端入口.", flush=True)


def main():
    """准备精简配置与三个固定示例,上传 HBM 和 Python 运行代码."""
    ctx = context()
    sys.path.insert(0, str(ctx.model / "deploy"))
    from model import NYUDataset, fingerprint, preprocess, save_json

    artifacts = list((ctx.output / "build").glob("*.hbm"))
    if len(artifacts) != 1 or artifacts[0].stat().st_size == 0:
        raise ValueError("编译目录必须有唯一非空 HBM,请核对编译器实际输出.")
    artifact = artifacts[0]
    config = {
        "artifact": "depth_anything_v2_metric_hypersim_vits.hbm",
        "model": ctx.config["model"], "export": ctx.config["export"],
        "evaluation": ctx.config["evaluation"], "quantization": ctx.config["quantization"],
        "dataset_directory": "datasets/" + ctx.config["evaluation"]["dataset"],
        "identity": {
            "hbm_sha256": fingerprint(artifact),
            "onnx_sha256": fingerprint(ctx.output / "depth_anything_v2_metric_hypersim_vits.onnx"),
        },
    }
    package = ctx.work / "package"
    package.mkdir(parents=True, exist_ok=True)
    files = [(artifact, config["artifact"]), (package / "config.json", "config.json")]
    deploy = ctx.model / "deploy"
    for name in ("compile.sh", "evaluate.sh", "model.py"):
        files.append((deploy / name, name))
    for path in (deploy / "board").glob("*.py"):
        files.append((path, "board/" + path.name))
    dataset = NYUDataset(ctx.dataset, config["evaluation"])
    session = ort.InferenceSession(
        str(ctx.output / "depth_anything_v2_metric_hypersim_vits.onnx"),
        providers=["CPUExecutionProvider"])
    predictions, truths = [], []
    config["compile_sample_ids"] = [int(index) + 1 for index in dataset.test[:3]]
    try:
        for index in dataset.test[:3]:
            image, truth = dataset.read(int(index))
            path = package / f"sample_{int(index) + 1:04d}.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise OSError("固定示例输入保存失败.")
            files.append((path, "examples/input/" + path.name))
            tensor = preprocess(image, ctx.config["export"]["input_shape"])
            predictions.append(session.run(["metric_depth"], {"image": tensor})[0][0])
            truths.append(truth)
            print(f"[小批量参考] NYU {int(index) + 1},完成 ONNX 输出.", flush=True)
    finally:
        dataset.close()
    reference = package / "compile_reference.npz"
    np.savez_compressed(reference, ids=np.array(config["compile_sample_ids"]),
                        depth=np.stack(predictions), truth=np.stack(truths))
    config["compile_reference_sha256"] = fingerprint(reference)
    config["compile_reference"] = "build/reference/compile_reference.npz"
    save_json(package / "config.json", config)
    files.append((reference, config["compile_reference"]))
    upload(ctx, files)
    print("[编译部署完成] 请登录 S100,在本模型部署目录执行 bash compile.sh.", flush=True)


if __name__ == "__main__":
    main()
