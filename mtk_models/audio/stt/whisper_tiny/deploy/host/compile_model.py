"""读取模型配置并使用 NCC 生成最终 DLA."""

import json

from tools.deploy import arguments, compile_dla


def main():
    """执行本模型的编译步骤并打印最终 DLA 路径."""
    ctx = arguments(__file__)
    contract = json.loads((ctx.work / "runtime_contract.json").read_text())
    if (contract["precision"], contract["cacheSize"], contract["maxTokenLength"]) != (
            "w8a16", 256, 448):
        raise ValueError("请先重新生成 W8A16 / 1t256c / 448 量化产物.")
    for name in ("encoder", "decoder_step"):
        filename = contract[name]["file"]
        print(f"[编译] {filename}", flush=True)
        compile_dla(ctx, (ctx.work / filename).with_suffix(".tflite"),
                    ctx.output / filename, ("--suppress-input",))


if __name__ == "__main__":
    main()
