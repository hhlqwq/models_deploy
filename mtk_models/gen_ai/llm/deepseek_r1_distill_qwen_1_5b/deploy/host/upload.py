"""上传已打包的多文件模型,量化、编译和打包均由前序独立步骤完成."""

from tools.deploy import arguments, upload_llm


def main():
    """只上传已有部署包,不重复打包或计算浮点精度."""
    ctx = arguments(__file__)
    upload_llm(ctx, ctx.output / "models")
    print("[模型上传完成] 待数值参考上传及板端 compile 检查.", flush=True)


if __name__ == "__main__":
    main()
