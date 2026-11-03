"""从固定 Parquet 快照还原 LFW 图片与完整十折验证对。"""

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import pyarrow.parquet as parquet


EXPECTED_IMAGES = 13233
EXPECTED_PAIRS = 6000
def parse_args() -> argparse.Namespace:
    """解析原始快照、验证对文件与输出目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_pairs(path: Path) -> list[dict[str, str]]:
    """核对完整十折、正负样本数与图片名称。"""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_PAIRS:
        raise ValueError(f"LFW 验证对数量不符: {len(rows)}。")
    folds = Counter(int(row["fold_id"]) for row in rows)
    labels = Counter(int(row["is_same"]) for row in rows)
    if (folds != {fold: 600 for fold in range(1, 11)} or
            labels != {0: 3000, 1: 3000}):
        raise ValueError("LFW 十折或正负样本数量不符。")
    for row in rows:
        for field in ("image_a", "image_b"):
            name = row[field]
            if Path(name).name != name or not name.endswith(".jpg"):
                raise ValueError(f"非法图片名称: {name}。")
    return rows


def prepare_dataset(parquet_path: Path, pairs_path: Path,
                    output_dir: Path) -> None:
    """导出图片，并核对完整验证对清单。"""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    pairs = read_pairs(pairs_path)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True)
    images = {}
    source = parquet.ParquetFile(parquet_path)
    for batch in source.iter_batches(
            batch_size=256, columns=["image", "source_filename"]):
        for record in batch.to_pylist():
            name = record["source_filename"]
            if (Path(name).name != name or not name.endswith(".jpg") or
                    name in images):
                raise ValueError(f"非法或重复图片名称: {name}。")
            image_bytes = record["image"]["bytes"]
            if not image_bytes:
                raise ValueError(f"图片内容为空: {name}。")
            destination = images_dir / name
            destination.write_bytes(image_bytes)
            images[name] = True
            if len(images) % 1000 == 0:
                print(f"[PROGRESS] LFW 图片 {len(images)}/{EXPECTED_IMAGES}。",
                      flush=True)
    if len(images) != EXPECTED_IMAGES:
        raise ValueError(f"LFW 图片数量不符: {len(images)}。")
    for row in pairs:
        if row["image_a"] not in images or row["image_b"] not in images:
            raise ValueError(f"验证对图片缺失: {row['pair_id']}。")
    shutil.copy2(pairs_path, output_dir / "pairs.csv")
    manifest = {
        "source": "https://huggingface.co/datasets/marcelohaps/lfw",
        "parquet_url": (
            "https://hf-mirror.com/api/datasets/marcelohaps/lfw/"
            "parquet/default/train/0.parquet"),
        "image_count": len(images),
        "pair_count": len(pairs),
        "image_variant": "original_non_aligned_250x250",
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"[OK] 完整 LFW 验证集已准备: {output_dir}。", flush=True)


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(args.parquet, args.pairs, args.output_dir)
