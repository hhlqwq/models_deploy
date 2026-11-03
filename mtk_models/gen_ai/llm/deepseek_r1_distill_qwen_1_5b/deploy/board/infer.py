"""通过 Python 直接调用 SDK 执行原生双 DLA 生成."""

from tools.llm.generate import run


def main():
    """读取本模型的 tokenizer、KV 契约及停止条件后执行生成."""
    run(__file__)


if __name__ == "__main__":
    main()
