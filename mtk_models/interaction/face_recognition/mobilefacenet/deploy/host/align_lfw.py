"""使用锁定上游提交的 MTCNN 五点协议准备 LFW 验证输入。"""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys

import cv2


UPSTREAM_REVISION = "a687c71bea830e70d05fb3b38ddc7c68e1687e94"
UPSTREAM_FILES = (
    "utils.py",
    "align_faces.py",
    "mtcnn/detector.py",
    "mtcnn/weights/pnet.npy",
    "mtcnn/weights/rnet.npy",
    "mtcnn/weights/onet.npy",
)
def load_image_names(dataset_root: Path) -> list[str]:
    """从完整十折验证对读取并核对所有待对齐图片。"""
    with (dataset_root / "pairs.csv").open(newline="", encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    if len(pairs) != 6000:
        raise ValueError(f"LFW 验证对不是 6,000 对: {len(pairs)}。")
    names = sorted({row[field] for row in pairs
                    for field in ("image_a", "image_b")})
    if len(names) != 7701:
        raise ValueError(f"LFW 验证图不是 7,701 张: {len(names)}。")
    for name in names:
        if Path(name).name != name or not (dataset_root / "images" / name).is_file():
            raise ValueError(f"LFW 图片缺失或路径非法: {name}。")
    return names


def load_upstream(upstream_root: Path):
    """检查上游文件存在后加载原版检测与对齐函数。"""
    for name in UPSTREAM_FILES:
        path = upstream_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
    os.chdir(upstream_root)
    sys.path.insert(0, str(upstream_root))
    # 上游源码目录由参数指定并校验后加载,不能在解析参数前导入.
    from utils import align_face, get_central_face_attributes

    return align_face, get_central_face_attributes


def align_dataset(dataset_root: Path, upstream_root: Path,
                  output_root: Path) -> None:
    """逐张对齐完整验证集，记录失败项并支持同目录续跑。"""
    dataset_root = dataset_root.resolve()
    upstream_root = upstream_root.resolve()
    output_root = output_root.resolve()
    names = load_image_names(dataset_root)
    align_face, detect_face = load_upstream(upstream_root)
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    for index, name in enumerate(names, 1):
        source = dataset_root / "images" / name
        destination = images_dir / name
        if destination.is_file():
            aligned = cv2.imread(str(destination), cv2.IMREAD_COLOR)
            if aligned is None or aligned.shape[:2] != (112, 112):
                raise ValueError(f"现有对齐图无效: {destination}。")
        else:
            valid, _, landmarks = detect_face(str(source))
            if not valid or len(landmarks) != 1:
                failures.append(name)
                continue
            aligned = align_face(str(source), landmarks)
            if aligned.shape[:2] != (112, 112):
                raise ValueError(f"对齐图尺寸异常: {name}。")
            temporary = destination.with_suffix(".tmp.jpg")
            if not cv2.imwrite(str(temporary), aligned):
                raise IOError(f"无法保存对齐图: {temporary}。")
            temporary.replace(destination)
        records.append({"image": name})
        if index % 100 == 0 or index == len(names):
            print(f"[PROGRESS] LFW 对齐 {index}/{len(names)} 张，失败 {len(failures)} 张。",
                  flush=True)
    report = {
        "status": "complete" if not failures else "incomplete",
        "dataset": "LFW_original_non_aligned",
        "upstream_url": "https://github.com/foamliu/MobileFaceNet",
        "upstream_revision": UPSTREAM_REVISION,
        "expected_images": len(names),
        "aligned_images": len(records),
        "failed_images": failures,
    }
    (output_root / "alignment_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_root / "alignment_images.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if failures:
        raise RuntimeError(f"未能对齐全部验证图，失败 {len(failures)} 张。")
    shutil.copy2(dataset_root / "pairs.csv", output_root / "pairs.csv")
    source_manifest = {
        "input_protocol": "RGB_ImageNet_normalize_aligned_112x112",
    }
    (output_root / "source_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def main() -> None:
    """读取现有 LFW 数据和锁定上游代码路径并执行对齐。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    align_dataset(args.dataset_root, args.upstream_root, args.output_root)


if __name__ == "__main__":
    main()
