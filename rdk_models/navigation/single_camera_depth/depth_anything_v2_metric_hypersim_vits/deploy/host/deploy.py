"""打包并上传 S100 运行所需文件,不通过 SSH 自动启动推理."""

import posixpath
from pathlib import Path, PurePosixPath
import sys
from time import monotonic

import cv2
import numpy as np
import onnxruntime as ort

from context import context


def upload(ctx, files, *, dataset=False):
    """分别向公共模型或数据集根目录上传明确的文件清单."""
    import paramiko

    board = ctx.platform["board"]
    work = PurePosixPath(board["datasets_root" if dataset else "models_root"])
    if not work.is_absolute() or str(work) == "/" or ".." in work.parts:
        raise ValueError("板端工作目录必须为非根目录的绝对路径.")
    destination = str(work if dataset else work / ctx.config["board"]["deploy_directory"])
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
    """评价固定百张子集并上传公共数据、参考和模型,保留三张数值示例."""
    ctx = context()
    sys.path.insert(0, str(ctx.model / "deploy"))
    from model import NYUDataset, fingerprint, preprocess, save_json

    onnx = ctx.output / "depth_anything_v2_metric_hypersim_vits.onnx"
    if not onnx.is_file():
        raise FileNotFoundError(f"NAS 模型资源根目录缺少 ONNX: {onnx};请核对文件位置,无需重新编译 HBM.")
    artifacts = list(ctx.output.glob("*.hbm"))
    if len(artifacts) != 1 or artifacts[0].stat().st_size == 0:
        raise ValueError("NAS 模型资源根目录必须有唯一非空 HBM,请核对编译器实际输出.")
    artifact = artifacts[0]
    config = {
        "artifact": "depth_anything_v2_metric_hypersim_vits.hbm",
        "model": ctx.config["model"], "export": ctx.config["export"],
        "evaluation": ctx.config["evaluation"], "quantization": ctx.config["quantization"],
        "dataset_directory": str(PurePosixPath(ctx.platform["board"]["datasets_root"]) / ctx.config["evaluation"]["dataset"]),
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
    from tools.evaluation import fixed_subset
    from model import restore_depth, sample_identity, score_depth

    dataset = NYUDataset(ctx.dataset, config["evaluation"], split="test")
    selected = fixed_subset([int(index) for index in dataset.test],
                            config["evaluation"]["compile_samples"],
                            required=[int(index) for index in dataset.test[:3]])
    config["compile_sample_ids"] = [index + 1 for index in selected]
    config["numerical_sample_ids"] = [int(index) + 1 for index in dataset.test[:3]]
    config["compile_dataset_identity"] = sample_identity(dataset, selected)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    predictions, numeric_ids, metrics = [], [], []
    data_files = []
    try:
        for step, index in enumerate(selected, 1):
            image, truth = dataset.read(index)
            prediction = session.run(["metric_depth"],
                                     {"image": preprocess(image, config["export"]["input_shape"])})[0][0]
            metrics.append(score_depth(restore_depth(prediction, *image.shape[:2]),
                                       truth, config["evaluation"]))
            if index + 1 in config["numerical_sample_ids"]:
                path = package / f"sample_{index + 1:04d}.png"
                if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                    raise OSError("示例输入保存失败.")
                files.append((path, "examples/input/" + path.name))
                predictions.append(prediction)
                numeric_ids.append(index + 1)
            for path in dataset.paths[index]:
                data_files.append((path, config["evaluation"]["dataset"] + "/"
                                   + path.relative_to(ctx.dataset).as_posix()))
            if step % 10 == 0 or step == len(selected):
                print(f"[ONNX 子集精度] {step}/{len(selected)}", flush=True)
    finally:
        dataset.close()
    config["compile_onnx_metrics"] = {
        key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}
    for key, value in config["compile_onnx_metrics"].items():
        print(f"[ONNX 子集精度] {key}={value:.6f}", flush=True)
    indices = package / "compile_indices.txt"
    np.savetxt(indices, config["compile_sample_ids"], fmt="%d")
    data_files.append((indices, config["evaluation"]["dataset"] + "/test/compile_indices.txt"))
    reference = package / "compile_reference.npz"
    np.savez_compressed(reference, ids=np.array(numeric_ids), depth=np.stack(predictions))
    config["compile_reference_sha256"] = fingerprint(reference)
    config["compile_reference"] = "build/reference/compile_reference.npz"
    save_json(package / "config.json", config)
    files.append((reference, config["compile_reference"]))
    upload(ctx, data_files, dataset=True)
    upload(ctx, files)
    print("[编译部署完成] 请登录 S100,在本模型部署目录执行 bash compile.sh.", flush=True)


if __name__ == "__main__":
    main()
