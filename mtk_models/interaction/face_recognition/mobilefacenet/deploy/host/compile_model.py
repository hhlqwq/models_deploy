"""读取模型配置并使用 NCC 生成最终 DLA."""

from tools.deploy import arguments, compile_dla


def main():
    """执行本模型的编译步骤并打印最终 DLA 路径."""
    ctx = arguments(__file__)
    compile_dla(ctx, ctx.work / "model_int8.tflite", ctx.output / "model_int8.dla", ())



if __name__ == "__main__":
    main()
