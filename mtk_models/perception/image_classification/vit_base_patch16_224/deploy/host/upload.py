"""上传模型资源与 Python 板端代码,不构建 C++ 程序."""

from tools.deploy import arguments, upload


def main():
    """打包必要资源,由板端 compile.sh 生成少量展示示例."""
    ctx = arguments(__file__)
    upload(ctx, [
        ctx.output / "model_int8.dla",
        ctx.work / "quantization.json",
    ])


if __name__ == "__main__":
    main()
