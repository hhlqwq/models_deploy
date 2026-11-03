"""根据统一模板创建模型交付目录."""

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = PROJECT_ROOT / "templates" / "model"


def create_model(
        scenario: str, category: str, model_id: str, model_name: str) -> Path:
    """复制模板并替换模型基础信息.

    Args:
        scenario: 应用场景,例如 perception.
        category: 模型类别,例如 image_classification.
        model_id: 仅包含小写字母、数字和下划线的模型标识.
        model_name: 面向用户显示的模型名称.

    Returns:
        新模型目录的绝对路径.

    Raises:
        FileExistsError: 目标模型目录已经存在.
        ValueError: 模型标识包含非法字符.
    """
    if not model_id.replace("_", "").isalnum() or model_id.lower() != model_id:
        raise ValueError("model_id 只能包含小写字母、数字和下划线.")
    for field_name, value in (("scenario", scenario), ("category", category)):
        if not value.replace("_", "").isalnum() or value.lower() != value:
            raise ValueError(f"{field_name} 只能包含小写字母、数字和下划线.")
    target = PROJECT_ROOT / "mtk_models" / scenario / category / model_id
    if target.exists():
        raise FileExistsError(f"模型目录已存在: {target}")

    print(f"[1/2] 创建模型目录: {target}")
    shutil.copytree(TEMPLATE_ROOT, target)
    # 源码按运行位置存放,不增加说明占位文件.
    for directory in ("deploy/host", "deploy/board"):
        (target / directory).mkdir(parents=True, exist_ok=True)
    print("[2/2] 替换模板变量.")
    for path in target.rglob("*"):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        content = content.replace("model_id", model_id)
        content = content.replace("MODEL_NAME", model_name)
        content = content.replace("scenario_name", scenario)
        content = content.replace("category_name", category)
        path.write_text(content, encoding="utf-8", newline="\n")
    return target


def parse_args() -> argparse.Namespace:
    """解析命令行参数.

    Returns:
        已解析的命令行参数.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-name", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    created_path = create_model(
        arguments.scenario,
        arguments.category,
        arguments.model_id,
        arguments.model_name,
    )
    print(f"[OK] 模型模板已创建: {created_path}")
