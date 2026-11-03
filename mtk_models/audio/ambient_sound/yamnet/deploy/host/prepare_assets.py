"""下载锁定的 Google YAMNet 和 ESC-50 原始资源,记录来源与哈希."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


SOURCE_REVISION = "34a21326906b9574fa11c4d6d0a5c534ff039267"
DATASET_REVISION = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"


def download(url, path, proxy):
    """下载单个资源,失败保留现场,成功记录真实大小和 SHA256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] {url}", flush=True)
    if not path.exists():
        command = ["curl", "-L", "--fail", "--retry", "5",
                   "--retry-all-errors", "--connect-timeout", "30"]
        if proxy:
            command.extend(["--proxy", proxy])
        partial = path.with_name(path.name + ".part")
        subprocess.run(command + ["-o", str(partial), url], check=True)
        partial.rename(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"url": url, "file": path.name, "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def main():
    """准备模型资源和指定 NAS 数据集,不自动安装依赖."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--proxy", default="")
    args = parser.parse_args()
    records = []
    base = f"https://raw.githubusercontent.com/tensorflow/models/{SOURCE_REVISION}"
    for name in ("yamnet.py", "params.py", "features.py", "yamnet_class_map.csv",
                 "README.md"):
        records.append(download(f"{base}/research/audioset/yamnet/{name}",
                                args.models_dir / "original/upstream" / name, args.proxy))
    records.append(download(f"{base}/LICENSE", args.models_dir / "original/upstream/LICENSE",
                            args.proxy))
    records.append(download("https://storage.googleapis.com/audioset/yamnet.h5",
                            args.models_dir / "original/yamnet.h5", args.proxy))
    archive = args.dataset_dir / "ESC-50.zip"
    url = f"https://codeload.github.com/karolpiczak/ESC-50/zip/{DATASET_REVISION}"
    dataset_record = download(url, archive, args.proxy)
    print("[EXTRACT] ESC-50 原始音频、元数据和许可证.", flush=True)
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            relative = Path(*Path(item.filename).parts[1:])
            if item.is_dir() or not relative.parts:
                continue
            if relative.parts[0] not in {"audio", "meta", "LICENSE", "README.md"}:
                continue
            target = (args.dataset_dir / relative).resolve()
            if not target.is_relative_to(args.dataset_dir.resolve()):
                raise ValueError("压缩包路径超出数据目录.")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(source.read(item))
    count = len(list((args.dataset_dir / "audio").glob("*.wav")))
    if count != 2000:
        raise ValueError(f"ESC-50 音频数量应为 2000,实际 {count}.")
    records.append(dataset_record)
    (args.models_dir / "original/resource_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.dataset_dir / "source_manifest.json").write_text(
        json.dumps(dataset_record, indent=2), encoding="utf-8")
    print("[OK] 原始资源就绪,ESC-50 共 2000 条.", flush=True)


if __name__ == "__main__":
    main()
