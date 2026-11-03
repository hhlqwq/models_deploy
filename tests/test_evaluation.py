"""验证快速子集、默认小批量与独立全量行为及参考报告的范围隔离."""

import argparse
import ast
import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import evaluation


class EvaluationTest(unittest.TestCase):
    """不加载模型,覆盖主机与板端共用的样本规则."""

    def test_pose_scoring_independent_selection(self):
        """独立评分须重建集合,两后端及两范围均拒绝缺框和错误图片映射."""
        source = Path(__file__).resolve().parents[1] / (
            "mtk_models/interaction/pose_detection/rtmpose_body2d/deploy/board/full_accuracy_board.py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "evaluate")
        detections = [{"image_id": index // 20 + 1} for index in range(600)]
        for scope in ("full", "subset"):
            os.environ["MTK_EVAL_SCOPE"] = scope
            selected = set(evaluation.pose_image_ids(detections))
            expected_digest = os.environ["MTK_EVAL_SAMPLE_SHA256"]
            grouped = {image: [{"detection_id": index, "image_id": image}
                               for index, row in enumerate(detections) if row["image_id"] == image]
                       for image in selected}
            for backend in ("python_neuron_runtime_hw", "onnxruntime"):
                with self.subTest(scope=scope, backend=backend):
                    args = argparse.Namespace(annotations=mock.MagicMock(), predictions=mock.Mock(),
                                              formatted=mock.MagicMock(), metrics=mock.Mock())
                    path = args.annotations.parent.parent.__truediv__.return_value
                    path.read_text.return_value = json.dumps(detections)
                    namespace = dict(argparse=argparse, json=json, contextlib=contextlib, io=io,
                        SIGMAS=SimpleNamespace(shape=(133,)), load_predictions=mock.Mock(return_value=grouped),
                        is_test=evaluation.is_test, pose_image_ids=evaluation.pose_image_ids,
                        format_results=mock.Mock(return_value=[]), COCO=mock.Mock(),
                        evaluate_component=mock.Mock(return_value={"AP": 0.5}),
                        accuracy_result=evaluation.accuracy_result, result_status=evaluation.result_status,
                        scope_metadata=evaluation.scope_metadata)
                    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
                    os.environ.pop("MTK_EVAL_SAMPLE_SHA256", None)
                    namespace["evaluate"](args, backend)
                    report = json.loads(args.metrics.write_text.call_args.args[0])
                    self.assertEqual(report["evaluation"]["samples"]["selection_id"], expected_digest)
                    self.assertEqual(report["evaluation"]["samples"]["count"], sum(map(len, grouped.values())))
                    image = min(grouped)
                    missing = {key: list(rows) for key, rows in grouped.items()}
                    missing[image] = missing[image][1:]
                    wrong = {key: [dict(row) for row in rows] for key, rows in grouped.items()}
                    wrong[image][0]["image_id"] = -1
                    for broken in (missing, wrong):
                        namespace["load_predictions"].return_value = broken
                        with self.assertRaises(ValueError):
                            namespace["evaluate"](args, backend)

    def setUp(self):
        """隔离每项测试的模式及样本指纹."""
        self.environment = mock.patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.arguments = mock.patch("sys.argv", ["test_evaluation.py"])
        self.arguments.start()

    def tearDown(self):
        """恢复调用方环境与参数."""
        self.arguments.stop()
        self.environment.stop()

    def test_default_subset_and_explicit_full(self):
        """默认固定小批量,只有显式全量范围保留全集."""
        values = list(range(5000))
        self.assertEqual(len(evaluation.coco_ids(values)), 300)
        self.assertEqual(evaluation.result_status(), "smoke_test")
        self.assertEqual(evaluation.scope_metadata()["evaluation_scope"], "subset")
        os.environ["MTK_EVAL_SCOPE"] = "full"
        self.assertEqual(evaluation.coco_ids(values), values)
        self.assertEqual(evaluation.result_status(), "complete")
        self.assertEqual(evaluation.scope_metadata(), {"evaluation_scope": "full"})

    def test_invalid_scope_and_failure_status(self):
        """拒绝拼错的范围,失败状态不能改为成功."""
        self.assertEqual(evaluation.result_status("incomplete"), "incomplete")
        os.environ["MTK_EVAL_SCOPE"] = "invalid"
        with self.assertRaises(ValueError):
            evaluation.is_test()

    def test_coco_subset_deterministic(self):
        """固定 300 张包含示例,输入顺序不影响两端指纹."""
        os.environ["MTK_EVAL_SCOPE"] = "subset"
        values = list(range(5000)) + [210299, 252219, 481386]
        selected = evaluation.coco_ids(values)
        digest = evaluation.scope_metadata()["sample_ids_sha256"]
        self.assertEqual(len(selected), 300)
        self.assertTrue(set(evaluation.COCO_EXAMPLES).issubset(selected))
        self.assertEqual(selected, evaluation.coco_ids(reversed(values)))
        self.assertEqual(digest, evaluation.scope_metadata()["sample_ids_sha256"])

    def test_lfw_balanced_folds(self):
        """十折每折各选二十五个同人和异人配对."""
        os.environ["MTK_EVAL_SCOPE"] = "subset"
        rows = [{"fold_id": fold, "is_same": label,
                 "pair_id": (fold - 1) * 600 + label * 300 + index}
                for fold in range(1, 11) for label in (0, 1)
                for index in range(300)]
        selected = evaluation.lfw_pairs(rows)
        self.assertEqual(len(selected), 500)
        for fold in range(1, 11):
            for label in (0, 1):
                self.assertEqual(sum(int(row["fold_id"]) == fold and
                                     int(row["is_same"]) == label for row in selected), 25)

    def test_audio_coverage(self):
        """ESC-50 子集每类每折各一条,且固定按文件名排序."""
        os.environ["MTK_EVAL_SCOPE"] = "subset"
        rows = [{"target": target, "fold": fold,
                 "filename": f"{fold}-{target:02d}-{index}.wav"}
                for fold in range(1, 6) for target in range(50) for index in range(8)]
        selected = evaluation.audio_records(rows)
        self.assertEqual(len(selected), 250)
        self.assertEqual(len({(row["target"], row["fold"]) for row in selected}), 250)

    def test_reference_scope(self):
        """拒绝全量参考、其他子集参考及缺少指纹的快速报告."""
        os.environ["MTK_EVAL_SCOPE"] = "subset"
        digest = evaluation.record_selection(["a", "b"])
        self.assertFalse(evaluation.matching_reference({"evaluation_scope": "full"}))
        self.assertFalse(evaluation.matching_reference({"evaluation_scope": "subset"}))
        self.assertFalse(evaluation.matching_reference(
            {"evaluation_scope": "subset", "sample_ids_sha256": "wrong"}))
        self.assertTrue(evaluation.matching_reference(
            {"evaluation_scope": "subset", "sample_ids_sha256": digest}))

    def test_llm_cli_limits(self):
        """只调用真实参数解析函数,验证 LLM 诊断上限而不加载模型."""
        root = Path(__file__).resolve().parents[1]
        cases = (
            ("evaluate_mmlu.py", ["prepare", "--model-name", "test_model", "--output", "/tmp/output"],
             "diagnostic_questions", 0, 0),
            ("evaluate_quality.py", ["--model", "/tmp/model", "--corpus", "/tmp/corpus",
             "--output", "/tmp/output", "--backend", "pytorch"],
             "diagnostic_blocks", 0, 4),
            ("evaluate_board_quality.py", ["--package", "/tmp/package",
             "--quality-data", "/tmp/data",
             "--output", "/tmp/output"], "diagnostic_blocks", 0, 4),
        )
        for filename, arguments, field, full_value, quick_value in cases:
            tree = ast.parse((root / "tools/llm" / filename).read_text(encoding="utf-8"))
            main = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "main")
            for quick in (False, True):
                captured = []
                namespace = {"__doc__": "参数检查", "argparse": argparse, "Path": Path,
                             "is_test": evaluation.is_test,
                             "evaluate": captured.append, "prepare": captured.append}
                exec(compile(ast.Module(body=[main], type_ignores=[]), filename, "exec"),
                     namespace)
                os.environ["MTK_EVAL_SCOPE"] = "subset" if quick else "full"
                argv = [filename, *arguments]
                with mock.patch("sys.argv", argv):
                    namespace["main"]()
                self.assertEqual(len(captured), 1)
                args = captured[0]
                self.assertEqual(getattr(args, field), quick_value if quick else full_value)
                self.assertEqual(args.output, Path("/tmp/output/test" if quick else "/tmp/output"))
                if filename == "evaluate_mmlu.py":
                    self.assertEqual(args.limit, 1000)
                    if quick:
                        self.assertEqual(args.test_questions, 100)
                if filename == "evaluate_quality.py" and quick:
                    self.assertTrue(args.skip_demo)
                    self.assertFalse(args.full_test)

    def test_small_or_duplicate_dataset_rejected(self):
        """不把缺少样本或重复标识的数据集当作完整快速评测."""
        with self.assertRaises(ValueError):
            evaluation.fixed_subset([1, 2], 3)
        with self.assertRaises(ValueError):
            evaluation.fixed_subset([1, 1, 2], 2)


if __name__ == "__main__":
    unittest.main()
