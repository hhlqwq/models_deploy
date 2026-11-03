"""使用配置中的 YAMNet 精度编译最终 DLA."""

from tools.deploy import arguments, compile_dla


def main():
    """只为 FP16 权重模式添加原有降精度参数."""
    ctx = arguments(__file__)
    precision = ctx.config["conversion"]["precision"]
    extra = ("--relax-fp32", "--suppress-input") if precision == "fp16" else ()
    compile_dla(ctx, ctx.work / f"model_{precision}.tflite",
                ctx.output / f"model_{precision}.dla", extra)


if __name__ == "__main__":
    main()
