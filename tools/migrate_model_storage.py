"""按统一结构迁移仓库模型资源,复制后校验,同名内容冲突时停止."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import shutil

from storage import REPO_ROOT, storage_roots

from tools.files import sha256 as file_sha256

sha256 = partial(file_sha256, progress=True)


CODE_SUFFIXES = {".py", ".sh", ".cpp", ".cc", ".c", ".h", ".hpp", ".md"}
SOURCE_RECORDS = {"LICENSE", "LICENSE.txt", "source_url.txt",
                  "source_manifest.json", ".gitignore", "CMakeLists.txt"}


def inventory(repo, destination):
    """列出资源映射,文档和源码保留在代码目录并复制配套源码."""
    entries = []
    for model in sorted((repo / "mtk_models").glob("*/*/*/models")):
        target = destination / model.parent.relative_to(repo / "mtk_models")
        for source in sorted(model.rglob("*")):
            if not source.is_file() or source.is_symlink():
                continue
            if "__pycache__" in source.parts or ".git" in source.parts:
                continue
            relative = source.relative_to(model)
            keep = source.suffix in CODE_SUFFIXES or source.name in SOURCE_RECORDS
            if relative.parts[0] not in {"original", "build", "models"}:
                if source.name in SOURCE_RECORDS or source.suffix in {".pt", ".pth", ".h5", ".zip", ".safetensors"}:
                    relative = Path("original") / relative
                elif source.suffix not in {".onnx", ".dla"}:
                    relative = Path("original" if keep else "build") / relative
            entries.append((source, target / relative, not keep))
    # 只迁移标准模型资源目录,散落的临时模型留在本地,不写入 NAS.
    return entries


def append_journal(record, report):
    """移除来源前持久化校验记录,中断后仍可查到已确认的 NAS 副本."""
    with report.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def copy_verified(source, target, remove, apply, label, journal=None):
    """核对副本后迁移,同名不同内容直接停止,不堆积历史副本."""
    size = source.stat().st_size
    digest = sha256(source)
    record = {"source": str(source), "target": str(target), "bytes": size,
              "sha256": digest, "remove_source": remove, "status": "planned"}
    if not apply:
        return record
    target.parent.mkdir(parents=True, exist_ok=True)
    target_digest = sha256(target) if target.exists() else None
    if target_digest is not None and target_digest != digest:
        raise ValueError(f"目标存在不同内容,请先核对,不会创建历史副本: {target}")
    if target_digest != digest:
        partial = target.with_name(target.name + ".migration-part")
        copied = 0
        with source.open("rb") as incoming, partial.open("wb") as outgoing:
            for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
                outgoing.write(block)
                copied += len(block)
                if copied % (512 * 1024 * 1024) == 0:
                    print(f"[复制] {label}: {copied // 1024**2} MiB", flush=True)
        target_digest = sha256(partial)
        if target_digest != digest:
            raise RuntimeError(f"复制校验失败: {label}")
        partial.replace(target)
    # 已核对的临时副本在同目录原子改名,不重复读取整个 NAS 模型.
    if target.stat().st_size != size or target_digest != digest:
        raise RuntimeError(f"最终校验失败: {label}")
    # 再核对来源,防止迁移过程中模型被其他任务改写.
    if remove and sha256(source) == digest:
        record["status"] = "verified_before_removal"
        if journal:
            journal(record)
        source.unlink()
        record["status"] = "moved"
    elif remove:
        raise RuntimeError(f"迁移时源文件发生变化,保留源文件: {label}")
    else:
        record["status"] = "copied_code_or_document"
    return record


def main():
    """保存本地迁移清单并按确认后的模式执行,不进行模型测试."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-tree", type=Path,
                        help="迁移指定模型资源树,须同时指定 destination-relative.")
    parser.add_argument("--destination-relative", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-source", action="store_true")
    args = parser.parse_args()
    destination = storage_roots()["models_root"].resolve(strict=True)
    if args.source_tree:
        relative = args.destination_relative
        if relative is None or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("destination-relative 必须为不包含 .. 的相对路径.")
        source_tree = args.source_tree.resolve(strict=True)
        if source_tree == destination or destination.is_relative_to(source_tree):
            raise ValueError("迁移来源不能包含目标目录.")
        entries = [(p, destination / relative / p.relative_to(source_tree),
                    not args.keep_source) for p in sorted(source_tree.rglob("*"))
                   if p.is_file() and (args.keep_source or not p.is_symlink())
                   and ".git" not in p.parts]
    else:
        entries = inventory(args.repo.resolve(strict=True), destination)
    if any(not target.is_relative_to(destination) for _, target, _ in entries):
        raise ValueError("迁移目标超出配置的模型存储根目录.")
    if any(source.absolute() == target.absolute() for source, target, _ in entries):
        raise ValueError("来源与目标重合,不能迁移.")
    print(f"[清单] {len(entries)} 个文件,"
          f"{sum(p.stat().st_size for p, _, _ in entries) / 1024**3:.3f} GiB", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    records = []
    journal = partial(append_journal, report=args.report.with_suffix(".jsonl"))
    for index, (source, target, remove) in enumerate(entries, 1):
        if index == 1 or index % 100 == 0 or source.suffix not in {".npz", ".npy"}:
            print(f"[迁移 {index}/{len(entries)}] {source.name}", flush=True)
        record = copy_verified(source, target, remove and not args.keep_source,
                               args.apply, source.name, journal=journal)
        records.append(record)
        temporary = args.report.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        temporary.replace(args.report)
    print(f"[完成] 清单保存在本地: {args.report}", flush=True)


if __name__ == "__main__":
    main()
