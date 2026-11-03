"""上传 Whisper 双 DLA、Python 推理代码和原版文本组件."""

from pathlib import Path
import json

from tools.deploy import arguments, upload


def main():
    """在运行板端示例前一并上传 tokenizer 与规范化资源."""
    ctx = arguments(__file__)
    vendor = ctx.work / "whisper_eval_vendor"
    files = [(str(Path("board") / path.relative_to(vendor).parent), path)
             for path in sorted((vendor / "whisper").rglob("*")) if path.is_file()]
    if not files:
        raise FileNotFoundError("缺少导出步骤准备的 Whisper 文本组件.")
    contract = json.loads((ctx.work / "runtime_contract.json").read_text())
    upload(ctx, [ctx.output / contract["encoder"]["file"],
                 ctx.output / contract["decoder_step"]["file"],
                 ctx.work / "runtime_contract.json", ctx.work / "mel_filters_f32.bin",
                 ctx.work / "decode_config.txt"],
           extra_files=files, multi_file=True)


if __name__ == "__main__":
    main()
