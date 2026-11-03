"""验证全部模型的真实精度出口、报告、数值误差和总入口,不运行模型."""

import ast
import copy
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from tools import evaluation
from tools.board import paths
from tools.compile_accuracy import save_subset, subset_report
from tools.output_metrics import Moments, OutputComparison, ReferenceWriter


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "resnet101": ("Top-1", "image"),
    "vit_base_patch16_224": ("Top-1", "image"),
    "yolov5s": ("mAP@0.5:0.95", "image"),
    "yolov8n": ("mAP@0.5:0.95", "image"),
    "mobilefacenet": ("LFW 验证准确率", "face_pair"),
    "rtmpose_body2d": ("WholeBody AP", "person_instance"),
    "depth_anything_v2_small": ("DA-2K 点对准确率", "annotated_point_pair"),
    "fastsam": ("segm AR@100", "image"),
    "yamnet": ("ESC-50 macro AP (folds 2-5)", "audio_clip"),
    "whisper_tiny": ("WER", "utterance"),
    "mit_bevfusion": ("mAP", "frame"),
    "minicpm5_2b": ("MMLU Accuracy", "question"),
    "deepseek_r1_distill_qwen_1_5b": ("MMLU Accuracy", "question"),
}


class DeployFlowTest(unittest.TestCase):
    """只使用小数组、临时文件和命令替身检查接口与控制流程."""

    def test_bev_formal_mini_manifest(self):
        """正式评价接受完整 mini_val,拒绝缺帧,重复帧及旧 trainval 清单."""
        source = ROOT / "mtk_models/pointcloud/three_d_fusion/mit_bevfusion/deploy/board/evaluate.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "validate_manifest")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        validate = namespace["validate_manifest"]
        data = {"version": "v1.0-mini", "split": "mini_val"}
        manifest = {**data, "frames": [{"token": str(index)} for index in range(81)]}
        self.assertEqual(validate(manifest, data, False), 81)
        self.assertEqual(validate(manifest, data, True), 81)
        for broken in ({**manifest, "frames": manifest["frames"][:-1]},
                       {**manifest, "frames": [manifest["frames"][0]] * 81},
                       {**manifest, "split": "val"}):
            with self.assertRaises(ValueError):
                validate(broken, data, False)
        trainval = {"version": "v1.0-trainval", "split": "val"}
        with self.assertRaises(ValueError):
            validate({**trainval, "frames": [{"token": str(i)} for i in range(6019)]},
                     trainval, False)

    def test_yolov8_evaluation_backend_paths(self):
        """执行真实入口,验证配置和命令行路径及两个后端的报告位置."""
        import argparse

        entry = ROOT / "mtk_models/perception/object_detection/yolov8n/deploy/host/evaluate_onnx.py"
        tree = ast.parse(entry.read_text(encoding="utf-8"))
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        ctx = SimpleNamespace(config={"model": {"weights": "/models/original/yolov8n.pt"}},
                              output=Path("/models"), work=Path("/work"))

        def configure(parser, entry, bindings):
            """提供配置默认值,不读取正式资源或运行模型."""
            parser.set_defaults(dataset_root=Path("/dataset"),
                                output=Path("/work/onnx_reference.json"))
            for action in parser._actions:
                if action.dest in ("dataset_root", "output"):
                    action.required = False
            return ctx

        for subset in (False, True):
            calls = []
            namespace = {"argparse": argparse, "Path": Path, "__file__": str(entry),
                         "configure_parser": configure, "reference_path": mock.Mock(),
                         "is_test": lambda: subset, "evaluate": calls.append}
            exec(compile(ast.Module(body=[main], type_ignores=[]), str(entry), "exec"), namespace)
            with mock.patch.object(sys, "argv", [str(entry)]):
                namespace["main"]()
            self.assertEqual(len(calls), 2)
            pytorch, onnx = calls
            self.assertIsInstance(pytorch.weights, Path)
            self.assertEqual(pytorch.weights, Path(ctx.config["model"]["weights"]))
            self.assertIsNone(pytorch.onnx)
            report = "pytorch_reference_test.json" if subset else "pytorch_reference.json"
            self.assertEqual(pytorch.output, ctx.work / report)
            self.assertIsNone(onnx.weights)
            self.assertEqual(onnx.onnx, ctx.output / "yolov8n.onnx")
            self.assertEqual(onnx.output, Path("/work/onnx_reference.json"))
            calls.clear()
            with mock.patch.object(sys, "argv", [str(entry), "--weights", "/custom/model.pt"]):
                namespace["main"]()
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].weights, Path("/custom/model.pt"))

    def setUp(self):
        """隔离范围、样本标识及临时目录,不访问正式模型或数据集."""
        self.environment = mock.patch.dict(os.environ, {"MTK_EVAL_SCOPE": "subset"})
        self.environment.start()
        base = Path("/tmp/hailongcodex") / date.today().isoformat() / "deploy_flow_tests"
        base.mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(dir=base))
        evaluation.record_selection(["sample"])

    def tearDown(self):
        """恢复调用方环境,测试文件留在日期临时目录."""
        self.environment.stop()

    def test_all_models_both_scopes(self):
        """全部模型两端使用相同接口,损失方向、单位和范围逐项校验."""
        for scope in ("subset", "full"):
            os.environ["MTK_EVAL_SCOPE"] = scope
            for model, (metric, unit) in TASKS.items():
                with self.subTest(scope=scope, model=model):
                    backend = "pytorch" if unit == "question" else "onnx"
                    baseline = evaluation.accuracy_result(model, backend, {metric: 0.8}, 1, unit)
                    actual = evaluation.accuracy_result(model, "npu", {metric: 0.7}, 1, unit)
                    self.assertEqual(evaluation.read_accuracy({"evaluation": baseline}, model, backend), baseline)
                    loss = evaluation.compare_accuracy(baseline, actual)[metric]
                    self.assertAlmostEqual(loss["accuracy_loss_percentage_points"], -10 if metric == "WER" else 10)
                    broken = copy.deepcopy(actual)
                    broken["samples"]["unit"] = "wrong"
                    with self.assertRaises(ValueError):
                        evaluation.compare_accuracy(baseline, broken)
                    broken = copy.deepcopy(actual)
                    broken["samples"]["selection_id"] = "other"
                    with self.assertRaises(ValueError):
                        evaluation.compare_accuracy(baseline, broken)

    def test_real_producer_expressions(self):
        """执行真实源码中的接口构造表达式,发现字段名和变量衔接错误."""
        entry = (ROOT / "mtk_models/deploy/compile.sh").read_text(encoding="utf-8")
        models = re.search(r"MODELS=\(\n(.*?)\n\)", entry, re.S).group(1).split()
        generic = {"vit_base_patch16_224", "mobilefacenet", "rtmpose_body2d",
                   "depth_anything_v2_small", "fastsam", "whisper_tiny"}
        for model_path in models:
            model = model_path.rsplit("/", 1)[-1]
            deploy = ROOT / "mtk_models" / model_path / "deploy"
            for backend in ("host", "board"):
                if model in ("minicpm5_2b", "deepseek_r1_distill_qwen_1_5b"):
                    source = ROOT / "tools/llm/evaluate_mmlu.py"
                elif backend == "host" and model in generic:
                    source = ROOT / "tools/accuracy/evaluate_onnx.py"
                elif backend == "host" and model == "yolov5s":
                    source = ROOT / "tools/accuracy/yolov5s_val_coco.py"
                elif backend == "board" and model in ("yolov5s", "yolov8n"):
                    source = ROOT / "tools/coco.py"
                elif model == "mit_bevfusion":
                    source = deploy / backend / ("reference.py" if backend == "host" else "evaluate.py")
                elif model == "resnet101":
                    source = deploy / backend / ("evaluate_onnx.py" if backend == "host" else "evaluate.py")
                else:
                    source = deploy / backend / ("evaluate_onnx.py" if backend == "host" else "full_accuracy_board.py")
                tree = ast.parse(source.read_text(encoding="utf-8"))
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name) and node.func.id == "accuracy_result"]
                self.assertTrue(calls, str(source))
                metric, unit = TASKS[model]
                report = {"accuracy": 0.8, "map_50_95": 0.8, "npu_top1": 0.8,
                          "pairwise_accuracy": 0.8, "AR_100": 0.8,
                          "metrics": {"held_out": {"map": 0.8}}}
                namespace = dict(accuracy_result=evaluation.accuracy_result, record_selection=evaluation.record_selection,
                                 evaluation_ids=lambda records, indices: ["sample"],
                                 MODEL=model, model=model, name=model, metric=metric, unit=unit,
                                 report=report, summary=report, units={name: value[1] for name, value in TASKS.items()},
                                 args=SimpleNamespace(model=model, model_name=model,
                                                      backend="pytorch" if backend == "host" else "npu",
                                                      mode="onnx", onnx=True),
                                 value=0.8, accuracy=0.8, primary=0.8, board_map=0.8,
                                 samples=1, expected=1, total_pairs=1, image_ids=[1], ids=[1],
                                 paths=[1], images=[1], rows=[1], records=[1], done=[1], successful={"a": 1},
                                 tokens=[1], indices=[], expected_ids=[1],
                                 verification={"verification_accuracy": 0.8},
                                 metrics={"wholebody": {"AP": 0.8}}, grouped={1: [1]},
                                 backend="onnxruntime" if backend == "host" else "python_neuron_runtime_hw")
                if model == "whisper_tiny" and backend == "board":
                    namespace["accuracy"] = {"value": 0.8}
                if model == "mit_bevfusion":
                    namespace["accuracy"] = {"mAP": 0.8, "NDS": 0.7}
                with self.subTest(model=model, side=backend, source=str(source)):
                    result = eval(compile(ast.Expression(calls[0]), str(source), "eval"), namespace)
                    self.assertEqual(result["model"], model)
                    self.assertEqual(result["samples"]["unit"], unit)
                    expected_backend = "pytorch" if backend == "host" and unit == "question" else "onnx" if backend == "host" else "npu"
                    self.assertEqual(result["backend"], expected_backend)

    def test_invalid_contract(self):
        """缺失、非有限数、范围和方向错误在汇总前拒绝."""
        with self.assertRaisesRegex(ValueError, "接口不完整"):
            evaluation.read_accuracy({"accuracy": 0.8}, "resnet101", "onnx")
        for value in (float("nan"), float("inf"), -1, True, 1.1):
            with self.assertRaises(ValueError):
                evaluation.accuracy_result("resnet101", "onnx", {"Top-1": value}, 1, "image")
        result = evaluation.accuracy_result("resnet101", "onnx", {"Top-1": 0.8}, 1, "image")
        result["metrics"]["Top-1"]["higher_is_better"] = False
        with self.assertRaises(ValueError):
            evaluation.read_accuracy({"evaluation": result}, "resnet101", "onnx")
        result = evaluation.accuracy_result("resnet101", "onnx", {"Top-1": 0.8}, 1, "image")
        os.environ["MTK_EVAL_SCOPE"] = "full"
        with self.assertRaises(ValueError):
            evaluation.read_accuracy({"evaluation": result}, "resnet101", "onnx")

    def test_official_score_roundoff(self):
        """官方 nuScenes 完全匹配评分可略高于 1,保留原值而不误判非法."""
        score = {"mAP": 1.0000000000000004, "NDS": 1.0000000000000002}
        report = evaluation.accuracy_result("mit_bevfusion", "onnx", score, 1, "frame")
        self.assertEqual(report["metrics"]["mAP"]["value"], score["mAP"])
        self.assertEqual(evaluation.read_accuracy({"evaluation": report}, "mit_bevfusion", "onnx"), report)

    def test_source_tokenizer_merge_formats(self):
        """官方数组和既有字符串 merges 得到相同 Token,加载时不改源文件."""
        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast
        from tools.llm.evaluate_mmlu import load_source_tokenizer

        folder = self.work / "tokenizer"
        backend = Tokenizer(models.BPE(
            vocab={"a": 0, "b": 1, "ab": 2, "[UNK]": 3},
            merges=[("a", "b")], unk_token="[UNK]"))
        expected = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
        expected.save_pretrained(folder)
        path = folder / "tokenizer.json"
        original = json.loads(path.read_text())
        for merge_format in ("string", "array"):
            with self.subTest(merge_format=merge_format):
                data = copy.deepcopy(original)
                if merge_format == "array":
                    data["model"]["merges"] = [["a", "b"]]
                path.write_text(json.dumps(data))
                before = {p.name: p.read_bytes() for p in folder.iterdir()}
                actual = load_source_tokenizer(folder)
                for text in ("ab", "aba", "b", "unknown"):
                    self.assertEqual(actual.encode(text), expected.encode(text))
                self.assertEqual(actual.unk_token_id, expected.unk_token_id)
                self.assertEqual({p.name: p.read_bytes() for p in folder.iterdir()}, before)

        data["model"]["merges"] = [["a", "b"], ["a"]]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "两个字符串"):
            load_source_tokenizer(folder)

    def test_numeric_reference_and_report(self):
        """真实 NPZ 读写、误差累加和子集报告形成闭环,全量不读数值参考."""
        os.environ["MTK_OUTPUT_METRICS"] = "1"
        models = self.work / "models"
        models.mkdir()
        dla = models / "model.dla"
        dla.write_bytes(b"test-dla")
        os.environ["MTK_OUTPUT_DLA_FILES"] = json.dumps({"model.dla": str(dla)})
        source = self.work / "model.onnx"
        source.write_bytes(b"test-onnx")
        reference = self.work / "build/reference/output_reference.npz"
        reference.parent.mkdir(parents=True)
        writer = ReferenceWriter(reference)
        tensor = np.array([[1, 0]], np.float32)
        writer.add(source, [tensor], {"output": tensor},
                   sample_id="sample:window:0", sample_files=[source])
        writer.finish()
        ctx = SimpleNamespace(root=self.work, models=models, work=self.work / "build/work")
        comparison = OutputComparison(ctx)
        comparison.update([tensor + np.float32(1e-6)],
                          {"output": np.array([[0, 1]], np.float32)},
                          sample_id="sample:window:0", sample_files=[source])
        comparison.finish()
        errors = json.loads((ctx.work / "output_metrics.json").read_text())
        values = errors["outputs"]["output"]
        self.assertAlmostEqual(values["cosine_similarity"], 0)
        self.assertAlmostEqual(values["rmse"], 1)
        self.assertAlmostEqual(values["nrmse_percent"], 100 * 2 ** 0.5)
        evaluation.record_selection(["sample"])
        normal = evaluation.accuracy_result("resnet101", "onnx", {"Top-1": 0.8}, 1, "image")
        baseline = {"model": "resnet101", "metrics": {"Top-1": 0.8}, "samples": 1,
                    "evaluation": normal, "dla_artifacts": {}}
        report = subset_report(baseline, {"Top-1": 0.7})
        report["output_errors"] = errors
        save_subset(self.work, report)
        result = json.loads((self.work / "results/summary_test.json").read_text())
        self.assertEqual(result["output_errors"], errors)
        os.environ["MTK_EVAL_SCOPE"] = "full"
        with mock.patch("tools.output_metrics.np.load", side_effect=AssertionError("全量不能读取张量参考")):
            full = OutputComparison(ctx)
            full.update([], {}, sample_id="unused", sample_files=[])
            full.finish()

    def test_numeric_pairing_rejects_wrong_resources(self):
        """允许输入舍入差异,拒绝文件、契约、覆盖和 DLA 错配及旧参考."""
        os.environ["MTK_OUTPUT_METRICS"] = "1"
        source = self.work / "sample.wav"
        source.write_bytes(b"audio")
        graph = self.work / "model.onnx"
        graph.write_bytes(b"onnx")
        dla = self.work / "model.dla"
        dla.write_bytes(b"dla")
        os.environ["MTK_OUTPUT_DLA_FILES"] = json.dumps({dla.name: str(dla)})
        reference = self.work / "build/reference/output_reference.npz"
        tensor = np.ones((1, 2), np.float32)
        writer = ReferenceWriter(reference)
        # 内容相同的两个窗口仍是两个独立样本,不得按输入内容合并.
        for index in range(2):
            writer.add(graph, [tensor], {"output": tensor},
                       sample_id=f"sample.wav:window:{index}", sample_files=[source])
        with self.assertRaisesRegex(ValueError, "重复"):
            writer.add(graph, [tensor], {"output": tensor},
                       sample_id="sample.wav:window:0", sample_files=[source])
        writer.finish()
        ctx = SimpleNamespace(root=self.work, models=self.work, work=self.work / "replay")
        comparison = OutputComparison(ctx)
        with self.assertRaisesRegex(ValueError, "契约"):
            comparison.update([tensor.reshape(2)], {"output": tensor},
                              sample_id="sample.wav:window:0", sample_files=[source])
        changed = self.work / "other" / source.name
        changed.parent.mkdir()
        changed.write_bytes(b"different audio")
        with self.assertRaisesRegex(ValueError, "原始样本"):
            comparison.update([tensor], {"output": tensor},
                              sample_id="sample.wav:window:0", sample_files=[changed])
        for index in reversed(range(2)):
            comparison.update([tensor + np.float32(1e-5)], {"output": tensor},
                              sample_id=f"sample.wav:window:{index}", sample_files=[source])
        with self.assertRaisesRegex(ValueError, "重复"):
            comparison.update([tensor], {"output": tensor},
                              sample_id="sample.wav:window:0", sample_files=[source])
        comparison.finish()
        report = json.loads((ctx.work / "output_metrics.json").read_text())
        self.assertEqual(report["compared_calls"], 2)
        self.assertEqual(report["outputs"]["output"]["rmse"], 0)
        self.assertEqual(report["comparison_basis"], "deployment_output_including_preprocessing")
        missing = OutputComparison(ctx)
        missing.update([tensor], {"output": tensor}, sample_id="wrong-id", sample_files=[source])
        with self.assertRaisesRegex(ValueError, "缺少选定样本"):
            missing.finish()
        dla.write_bytes(b"wrong dla")
        with self.assertRaisesRegex(ValueError, "DLA"):
            OutputComparison(ctx)
        dla.write_bytes(b"dla")
        np.savez(reference, metadata=np.array(json.dumps({"evaluation_scope": "compile_subset"})))
        with self.assertRaisesRegex(ValueError, "旧格式"):
            OutputComparison(ctx)

    def test_numeric_calls_have_explicit_sample_identity(self):
        """核对所有真实生产消费调用均显式提供样本标识和原始文件."""
        calls = 0
        for folder in (ROOT / "mtk_models", ROOT / "tools"):
            for path in folder.rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    reference = isinstance(node.func, ast.Name) and node.func.id == "record_reference"
                    comparison = (isinstance(node.func, ast.Attribute)
                                  and node.func.attr == "update"
                                  and ast.unparse(node.func.value) in ("comparison", "self.comparison"))
                    writer = (isinstance(node.func, ast.Attribute) and node.func.attr == "add"
                              and ast.unparse(node.func.value) in ("writer", "_writer"))
                    if reference or comparison or writer:
                        keywords = {item.arg for item in node.keywords}
                        self.assertTrue({"sample_id", "sample_files"} <= keywords, str(path))
                        calls += 1
        self.assertGreater(calls, 0)

    def test_numeric_producer_consumer_sample_expressions(self):
        """执行各模型真实配对表达式,验证样本及窗口标识在两端一致."""
        shared = "tools/accuracy/evaluate_onnx.py"
        cases = [
            ("resnet101", "tools/prepare_output_reference.py", "classification", "record_reference",
             "mtk_models/perception/image_classification/resnet101/deploy/board/infer.py", "main"),
            ("vit_base_patch16_224", "tools/prepare_output_reference.py", "classification", "record_reference",
             "mtk_models/perception/image_classification/vit_base_patch16_224/deploy/board/infer.py", "main"),
            ("yolov5s", "tools/accuracy/yolov5s_val_coco.py", "build_fp32_infer", "record_reference",
             "tools/detection.py", "infer"),
            ("yolov8n", "mtk_models/perception/object_detection/yolov8n/deploy/host/evaluate_onnx.py", "evaluate", "record_reference",
             "tools/detection.py", "infer"),
            ("fastsam", shared, "evaluate_coco", "infer",
             "mtk_models/navigation/segmentation/fastsam/deploy/board/infer.py", "main"),
            ("mobilefacenet", shared, "evaluate_face", "infer", "tools/board.py", "infer_images"),
            ("depth_anything_v2_small", shared, "evaluate_depth", "infer", "tools/board.py", "infer_images"),
            ("rtmpose_body2d", shared, "evaluate_pose", "infer",
             "mtk_models/interaction/pose_detection/rtmpose_body2d/deploy/board/infer.py", "main"),
            ("yamnet", "mtk_models/audio/ambient_sound/yamnet/deploy/host/evaluate_onnx.py", "evaluate", "record_reference",
             "mtk_models/audio/ambient_sound/yamnet/deploy/board/infer.py", "main"),
            ("whisper_tiny", shared, "evaluate_whisper", "infer",
             "mtk_models/audio/stt/whisper_tiny/deploy/board/infer.py", "main"),
        ]
        for model in ("minicpm5_2b", "deepseek_r1_distill_qwen_1_5b"):
            cases.append((model, "tools/prepare_output_reference.py", "language", "add",
                          "tools/llm/generate.py", "run"))
        dataset = Path("/dataset")
        for model, host_path, host_function, host_call, board_path, board_function in cases:
            with self.subTest(model=model):
                image = "000000000139.jpg"
                name = image
                image_path = dataset / "images" / image
                if model == "depth_anything_v2_small":
                    name = "images/indoor/sample.jpg"
                    image_path = dataset / name
                if model == "mobilefacenet":
                    name = "person/sample.jpg"
                    image_path = dataset / "images" / name
                annotation = dataset / "person_detection_results/COCO_val2017_detections_AP_H_56_person.json"
                path = annotation if model == "rtmpose_body2d" else image_path
                prompt = Path("/models/scripts/prompts/en_demo.tokens.txt")
                if model in ("minicpm5_2b", "deepseek_r1_distill_qwen_1_5b"):
                    path = prompt
                namespace = dict(path=path, image_path=image_path, name=name, index=9,
                    current_id=139, item={"image_id": 139, "detection_id": 9},
                    args=SimpleNamespace(dataset_root=dataset, dataset=dataset),
                    ctx=SimpleNamespace(dataset=dataset), record={"file_name": image},
                    row={"filename": "sample.wav"}, patch_index=4, patch=4, clip=0,
                    audio_manifest=[{"filename": "sample.wav"}], sample_id="sample",
                    sample={"sample_id": "sample", "audio_path": dataset / "sample.flac"},
                    sources={"sample": {"audio": dataset / "sample.flac"}}, language="en", prompt=prompt)
                expressions = []
                for source, function, call_name in (
                        (host_path, host_function, host_call),
                        (board_path, board_function, "generate" if model in
                         ("minicpm5_2b", "deepseek_r1_distill_qwen_1_5b") else "update")):
                    tree = ast.parse((ROOT / source).read_text(encoding="utf-8"))
                    function_node = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == function)
                    calls = [node for node in ast.walk(function_node) if isinstance(node, ast.Call)
                        and ((isinstance(node.func, ast.Name) and node.func.id == call_name)
                             or (isinstance(node.func, ast.Attribute) and node.func.attr == call_name))
                        and {"sample_id", "sample_files"} <= {keyword.arg for keyword in node.keywords}]
                    self.assertEqual(len(calls), 1, source)
                    values = {keyword.arg: eval(compile(ast.Expression(keyword.value), source, "eval"), namespace)
                              for keyword in calls[0].keywords if keyword.arg in ("sample_id", "sample_files")}
                    expressions.append((values["sample_id"], [p.name for p in values["sample_files"]]))
                self.assertEqual(*expressions)

    def test_yamnet_examples_use_uploaded_reference_directory(self):
        """实际无参数运行展示脚本,只在上传目录放标签,确认三个示例生成."""
        board = self.work / "board"
        board.mkdir()
        script = board / "render_examples.py"
        script.write_bytes((ROOT / "tools/render_examples.py").read_bytes())
        (self.work / "config.json").write_text(json.dumps({
            "name": "yamnet", "board": {"dataset_dir": str(self.work)},
            "deployment": {"model_directory": "."}}))
        reference = self.work / "build/reference"
        reference.mkdir(parents=True)
        (reference / "yamnet_class_map.csv").write_text(
            "index,display_name\n" + "".join(f"{i},class_{i}\n" for i in range(521)))
        inputs = self.work / "examples/input"
        inputs.mkdir(parents=True)
        records = [{"filename": f"audio_{i}.wav", "category": f"category_{i}"}
                   for i in range(3)]
        (inputs / "samples.json").write_text(json.dumps({"samples": records}))
        work = self.work / "build/work"
        work.mkdir()
        (work / "audio_manifest.json").write_text(json.dumps(records))
        scores = np.zeros((3, 521), dtype="<f4")
        for i in range(3):
            scores[i, i + 10] = 0.9
        scores.tofile(work / "scores.bin")
        before = (work / "scores.bin").read_bytes()
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT,
            env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        outputs = sorted((self.work / "examples/output").glob("sample_*_events.json"))
        self.assertEqual(len(outputs), 3)
        for i, path in enumerate(outputs):
            report = json.loads(path.read_text())
            self.assertEqual(report["source_filename"], records[i]["filename"])
            self.assertEqual(report["top5_yamnet"][0]["class"], f"class_{i + 10}")
        self.assertEqual((work / "scores.bin").read_bytes(), before)
        self.assertFalse((self.work / "models").exists())

    def test_llm_formal_protocol_has_no_model_exception(self):
        """所有模型接受相同千题协议,拒绝 14042 题及错误学科或随机种子."""
        from tools.llm.evaluate_mmlu import validate_protocol

        protocol = {"evaluation_scope": "full", "questions": 1000, "subjects": 57,
                    "sampling_seed": 17, "scope": "first_round_stratified_subset"}
        validate_protocol(protocol)
        for key, value in (("questions", 14042), ("subjects", 56),
                           ("sampling_seed", 0), ("scope", "full_test")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_protocol(dict(protocol, **{key: value}))
        for model in ("minicpm5_2b", "deepseek_r1_distill_qwen_1_5b"):
            import yaml
            config = yaml.safe_load((ROOT / "mtk_models/gen_ai/llm" / model / "config.yaml").read_text())
            self.assertEqual(config["evaluation"]["mmlu_questions"], 1000)
            self.assertEqual(config["evaluation"]["mmlu_seed"], 17)

    def test_bev_input_transfer_is_model_local(self):
        """本模型清单覆盖当前帧、历史帧和相机,缺文件在传输前拒绝."""
        import importlib.util

        source = ROOT / "mtk_models/pointcloud/three_d_fusion/mit_bevfusion/deploy/host/upload.py"
        spec = importlib.util.spec_from_file_location("bev_inputs_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        dataset = self.work / "dataset"
        names = ["samples/lidar.bin", "sweeps/previous.bin", "samples/camera.jpg"]
        for name in names:
            target = dataset / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"input")
        data = {"frames": [{"token": "frame", "lidar": names[0],
                           "sweeps": [{"path": names[1]}], "cameras": [{"path": names[2]}]}]}
        (self.work / "samples.json").write_text(json.dumps(data))
        ctx = SimpleNamespace(work=self.work, dataset=dataset,
            config={"board": {"dataset_dir": "/data/test", "deploy_dir": "/models/bev"}})
        with mock.patch.dict(os.environ, {"MTK_BOARD_HOST": "test"}), \
                mock.patch.object(module, "run_command") as run:
            module.upload_inputs(ctx)
            self.assertEqual(run.call_args_list[1].args[0][0], "rsync")
            self.assertEqual(set((self.work / "input_files.txt").read_text().splitlines()), set(names))
            self.assertEqual(json.loads(run.call_args_list[2].kwargs["input"])["manifest"], data)
            (dataset / names[1]).unlink()
            run.reset_mock()
            with self.assertRaises(FileNotFoundError):
                module.upload_inputs(ctx)
            run.assert_not_called()

    def test_vit_preprocessing_matches_fp32_calibration(self):
        """执行真实 ViT 前处理和校准函数,验证类型、范围及每个像素值一致."""
        import cv2

        model = ROOT / "mtk_models/perception/image_classification/vit_base_patch16_224/deploy"
        functions = {}
        for file, name in ((model / "board/infer.py", "floating_preprocess"),
                           (model / "host/convert_int8.py", "preprocess_image")):
            tree = ast.parse(file.read_text())
            node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
            namespace = {"np": np, "cv2": cv2, "Path": Path}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(file), "exec"), namespace)
            functions[name] = namespace[name]
        image = np.arange(300 * 400 * 3, dtype=np.uint8).reshape(300, 400, 3)
        path = self.work / "sample.png"
        self.assertTrue(cv2.imwrite(str(path), image))
        tensor = functions["floating_preprocess"](path)
        expected = functions["preprocess_image"](path)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        np.testing.assert_array_equal(tensor, expected)
        self.assertTrue(np.all((tensor >= 0) & (tensor <= 1)))
        # 旧浮点参考先做 FP64 除法再转 FP32,新实现对全部像素值保持一致.
        pixels = np.arange(256)
        np.testing.assert_array_equal((pixels.astype(np.float64) / 255).astype(np.float32),
                                      pixels.astype(np.float32) / np.float32(255.0))

    def test_report_isolation_and_invalid_write(self):
        """只保留两份报告,无效新结果不损坏旧报告."""
        evaluation.publish_result(self.work, {"test": 1}, "summary_test.json")
        directory = self.work / "results"
        (directory / "summary_sub.json").write_text("old")
        evaluation.publish_result(self.work, {"full": 1}, "summary.json")
        self.assertEqual({p.name for p in directory.iterdir()}, {"summary_test.json", "summary.json"})
        with self.assertRaises(ValueError):
            evaluation.publish_result(self.work, {"bad": float("nan")}, "summary.json")
        self.assertEqual(json.loads((directory / "summary.json").read_text()), {"full": 1})

    def test_preflight_before_work_cleanup(self):
        """错误 DLA 在推理前拒绝,不能先删除上次运行现场."""
        root = self.work
        work = root / "build/work"
        work.mkdir(parents=True)
        (work / "old").write_text("keep")
        config = {"name": "resnet101", "deployment": {"model_directory": "."},
                  "board": {"dataset_dir": str(root / "dataset")}}
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config))
        evaluation.record_selection(["sample"])
        reference = {"evaluation": evaluation.accuracy_result("resnet101", "onnx", {"Top-1": 0.8}, 1, "image"),
                     "dla_artifacts": {"model.dla": "wrong"}}
        folder = root / "build/reference"
        folder.mkdir()
        (folder / "subset.json").write_text(json.dumps(reference))
        (root / "model.dla").write_bytes(b"different")
        with self.assertRaises(ValueError):
            paths(root, config_path, "test", new_run=True)
        self.assertTrue((work / "old").is_file())

    def make_shell_environment(self, host=False):
        """构造真实总入口的运行环境,所有远程命令和模型入口均为替身."""
        root = self.work / "flow"
        deploy = root / "mtk_models/deploy" if host else root / "deploy"
        deploy.mkdir(parents=True)
        for name in ("compile.sh", "evaluate.sh", "compile_board.sh", "evaluate_board.sh", "common.sh"):
            (deploy / name).write_bytes((ROOT / "mtk_models/deploy" / name).read_bytes())
        entry = (deploy / "compile.sh").read_text()
        models = re.search(r"MODELS=\(\n(.*?)\n\)", entry, re.S).group(1).split()
        log = self.work / "calls.txt"
        commands = self.work / "commands"
        commands.mkdir()
        for name in ("ssh", "scp", "rsync"):
            script = commands / name
            script.write_text('#!/bin/bash\nif [[ "${1:-}" == -V ]]; then echo OpenSSH >&2; exit 0; fi\nprintf "' + name + ':%s\\n" "$*" >> "$FLOW_LOG"\n')
            script.chmod(0o755)
        key = self.work / "key"
        key.write_text("test-only")
        local = deploy / "platform.local.sh"
        local.write_text(f'MTK_RESULTS_HOST=test-only\nMTK_RESULTS_ROOT={root}/returned\nMTK_RESULTS_KEY={key}\n')
        environment = dict(os.environ, FLOW_LOG=str(log), PATH=str(commands) + ":" + os.environ["PATH"])
        for model in models:
            folder = root / "mtk_models" / model / "deploy" if host else root / model.rsplit("/", 1)[-1]
            folder.mkdir(parents=True)
            for phase in ("compile", "evaluate"):
                (folder / f"{phase}.sh").write_text(
                    '#!/bin/bash\nprintf "' + phase + ':' + model + '\\n" >> "$FLOW_LOG"\n'
                    + '[[ "${FAIL_MODEL:-}" != "' + model + '" ]] || exit 7\n')
            if not host:
                (folder / "examples/output").mkdir(parents=True)
                (folder / "examples/output/sample.txt").write_text("example")
                (folder / "results").mkdir()
                for report in ("summary.json", "summary_test.json"):
                    (folder / "results" / report).write_text("{}")
        if host:
            (root / "tools").mkdir()
            (root / "tools/storage.py").write_text("# 入口检测占位,不会执行.\n")
            python = commands / "python"
            python.write_text('#!/bin/bash\nprintf "%s\\n" "$FLOW_CONFIG"\n')
            python.chmod(0o755)
            environment.update(MTK_MODELS_ROOT=str(root), MTK_BOARD_HOST="test-only",
                               MTK_BOARD_OPEN_MODELS_ROOT="/test-only", FLOW_CONFIG=str(local))
        return deploy, environment, log, models

    def check_entry_logs(self, host):
        """检查两阶段的屏幕与文件输出、覆盖行为、失败码及回传日志归属."""
        deploy, environment, calls, models = self.make_shell_environment(host=host)
        first = (deploy.parent / models[0] / "deploy" if host
                 else deploy.parent / models[0].rsplit("/", 1)[-1])
        for phase in ("compile", "evaluate"):
            with self.subTest(host=host, phase=phase):
                entry = first / f"{phase}.sh"
                entry.write_text(entry.read_text() +
                                 f'printf "stdout-{phase}\\n"\nprintf "stderr-{phase}\\n" >&2\n'
                                 + 'printf "unbuffered=%s\\n" "${PYTHONUNBUFFERED:-}"\n')
                target = deploy / f"{phase}.log"
                other = deploy / ("evaluate.log" if phase == "compile" else "compile.log")
                other.write_text("另一阶段的已有日志\n")
                target.write_text("本阶段的旧日志\n")
                result = subprocess.run(["bash", str(deploy / (f"{phase}.sh" if host else f"{phase}_board.sh"))],
                                        env=environment, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(target.read_text(), result.stdout)
                self.assertEqual(result.stderr, "")
                self.assertIn(f"stdout-{phase}", result.stdout)
                self.assertIn(f"stderr-{phase}", result.stdout)
                self.assertIn("unbuffered=1", result.stdout)
                self.assertNotIn("本阶段的旧日志", target.read_text())
                self.assertEqual(other.read_text(), "另一阶段的已有日志\n")
                if not host:
                    self.assertIn("[回传完成]", target.read_text())

                calls.write_text("")
                failed = dict(environment, FAIL_MODEL=models[1])
                result = subprocess.run(["bash", str(deploy / (f"{phase}.sh" if host else f"{phase}_board.sh"))],
                                        env=failed, capture_output=True, text=True)
                self.assertEqual(result.returncode, 7)
                self.assertEqual(target.read_text(), result.stdout)
                self.assertIn("退出码 7", target.read_text())
                self.assertNotIn(f"{phase}:" + models[2], calls.read_text())
                self.assertEqual(other.read_text(), "另一阶段的已有日志\n")

    def test_four_entries_have_fixed_roles(self):
        """四个总入口固定两端职责,没有运行位置猜测、自调用或隐藏回传参数."""
        expected = None
        for name in ("compile.sh", "evaluate.sh", "compile_board.sh", "evaluate_board.sh"):
            source = (ROOT / "mtk_models/deploy" / name).read_text()
            models = re.search(r"MODELS=\(\n(.*?)\n\)", source, re.S).group(1).split()
            if expected is None:
                expected = models
            self.assertEqual(models, expected)
            self.assertEqual(len(models), 13)
            for old in ("RETURN_ONLY", "--return-results", "MTK_DEPLOY_LOG_ACTIVE", "LOCATION="):
                self.assertNotIn(old, source)
            if "_board" in name:
                self.assertIn("${MODEL##*/}", source)
                self.assertIn("return_results", source)
            else:
                self.assertIn("${MODEL}/deploy/", source)
                self.assertNotIn("return_results", source)

    def test_host_entry_logs(self):
        """编译服务器的两个固定日志完整保存标准输出与错误输出."""
        self.check_entry_logs(host=True)

    def test_board_entry_logs(self):
        """板端的两个固定日志互不覆盖,全量回传继续写 evaluate.log."""
        self.check_entry_logs(host=False)

    def test_log_open_failure_stops_before_models(self):
        """日志路径不可写时在启动模型或远程上传之前停止."""
        deploy, environment, calls, _ = self.make_shell_environment(host=True)
        for phase in ("compile", "evaluate"):
            with self.subTest(phase=phase):
                (deploy / f"{phase}.log").mkdir()
                result = subprocess.run(["bash", str(deploy / f"{phase}.sh")],
                                        env=environment, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(calls.exists())

    def test_log_writer_failure_is_not_success(self):
        """即使模型替身全部成功,日志进程失败也不能返回成功."""
        deploy, environment, _, _ = self.make_shell_environment(host=True)
        writer = self.work / "commands/tee"
        writer.write_text("#!/bin/bash\ncat\nexit 9\n")
        writer.chmod(0o755)
        for phase in ("compile", "evaluate"):
            with self.subTest(phase=phase):
                result = subprocess.run(["bash", str(deploy / f"{phase}.sh")],
                                        env=environment, capture_output=True, text=True)
                self.assertEqual(result.returncode, 9)

    def test_board_compile_and_full_return(self):
        """真实板端总入口顺序执行 13 模型,全量末尾回传不再次触发 compile."""
        deploy, environment, log, models = self.make_shell_environment()
        subprocess.run(["bash", str(deploy / "compile_board.sh")], env=environment,
                       capture_output=True, text=True, check=True)
        rows = log.read_text().splitlines()
        self.assertEqual([row for row in rows if row.startswith("compile:")],
                         ["compile:" + model for model in models])
        transfers = [row for row in rows if row.startswith("rsync:")]
        self.assertEqual(len(transfers), 26)
        self.assertTrue(all("summary.json" not in row and ".dla" not in row for row in transfers))
        log.write_text("")
        subprocess.run(["bash", str(deploy / "evaluate_board.sh")], env=environment,
                       capture_output=True, text=True, check=True)
        rows = log.read_text().splitlines()
        self.assertFalse(any(row.startswith("compile:") for row in rows))
        self.assertEqual([row for row in rows if row.startswith("evaluate:")],
                         ["evaluate:" + model for model in models])
        self.assertEqual(len([row for row in rows if row.startswith("rsync:")]), 39)

    def test_host_order_and_stop_on_failure(self):
        """主机顺序调用原模型入口,失败时停止且不触发板端推理."""
        deploy, environment, log, models = self.make_shell_environment(host=True)
        environment["FAIL_MODEL"] = models[1]
        result = subprocess.run(["bash", str(deploy / "compile.sh")], env=environment,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 7)
        rows = log.read_text().splitlines()
        self.assertEqual([row for row in rows if row.startswith("compile:")],
                         ["compile:" + model for model in models[:2]])
        self.assertFalse(any(row.startswith("rsync:") for row in rows))

    def test_full_entrypoints_upload_without_compile(self):
        """所有单模型全量入口自动上传参考,没有量化编译或数值参考采集命令."""
        for path in (ROOT / "mtk_models").rglob("deploy/evaluate.sh"):
            if path == ROOT / "mtk_models/deploy/evaluate.sh":
                continue
            source = path.read_text(encoding="utf-8")
            host = source.split("\nelse\n", 1)[1]
            with self.subTest(model=str(path)):
                self.assertIn("--upload-reference", host)
                self.assertNotIn("prepare_output_reference.py", host)
                self.assertNotRegex(host, r'python[^\n]*host/(compile|convert|export)')

    def test_all_full_reference_uploads(self):
        """全部模型的真实全量上传函数只发送必要参考,绑定实际 DLA."""
        from tools import deploy
        from tools.files import sha256

        os.environ["MTK_EVAL_SCOPE"] = "full"
        for model, (metric, unit) in TASKS.items():
            with self.subTest(model=model):
                folder = self.work / model
                work = folder / "work"
                output = folder / "assets"
                work.mkdir(parents=True)
                output.mkdir()
                language = unit == "question"
                package = output / "models" if language or model == "mit_bevfusion" else output
                package.mkdir(exist_ok=True)
                (package / "model.dla").write_bytes(b"compiled-test")
                backend = "pytorch" if language else "onnx"
                report = {"evaluation": evaluation.accuracy_result(model, backend, {metric: 0.8}, 1, unit)}
                if language:
                    mode = "mmlu1000"
                    data = work / "evaluation" / mode
                    for name in ("mmlu/protocol.json", "mmlu/cases.json", "mmlu/mmlu_inputs.npz",
                                 "mmlu/board_input_contract.json", "quality/board_quality_protocol.json",
                                 "quality/board_inputs.npz", "quality/board_input_contract.json",
                                 "reference/mmlu/mmlu_accuracy.json", "quality/pytorch_quality.json"):
                        path = data / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(report))
                elif model == "yamnet":
                    path = work / "evaluation/full/onnx/summary.json"
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps(report))
                    (work / "fp32_reference.json").write_text("{}")
                else:
                    (work / "onnx_reference.json").write_text(json.dumps(report))
                    if model == "mit_bevfusion":
                        (work / "samples.json").write_text("{}")
                        (work / "ground_truth.json").write_text("{}")
                ctx = SimpleNamespace(work=work, output=output,
                                      config={"name": model, "board": {"deploy_dir": "/test-only"}})
                with mock.patch.object(deploy, "context", return_value=ctx), \
                        mock.patch.object(deploy, "transfer_files") as transfer, \
                        mock.patch.object(deploy, "deployment_artifacts", return_value={"model.dla": package / "model.dla"}):
                    deploy.upload_evaluation(folder / "deploy")
                transfer.assert_called_once()
                files = transfer.call_args.args[1]
                self.assertIn("build/reference/full.json", files)
                self.assertTrue(all(name.startswith("build/reference/") for name in files))
                self.assertFalse(any(path.suffix == ".dla" for path in files.values()))
                saved = json.loads(files["build/reference/full.json"].read_text())
                self.assertEqual(saved["dla_artifacts"], {"model.dla": sha256(package / "model.dla")})

    def test_full_publish_all_models(self):
        """真实全量发布器统一 13 个模型的精度区,保留子集报告且不混入数值误差."""
        from tools.summary import publish

        os.environ["MTK_EVAL_SCOPE"] = "full"
        for model, (metric, unit) in TASKS.items():
            with self.subTest(model=model):
                root = self.work / model
                reference_dir = root / "build/reference"
                reference_dir.mkdir(parents=True)
                backend = "pytorch" if unit == "question" else "onnx"
                baseline = evaluation.accuracy_result(model, backend, {metric: 0.8}, 1, unit)
                actual = evaluation.accuracy_result(model, "npu", {metric: 0.7}, 1, unit)
                (reference_dir / "full.json").write_text(json.dumps({"evaluation": baseline}))
                evaluation.publish_result(root, {"subset": True}, "summary_test.json")
                report = {"evaluation": actual, "accuracy": {"wikitext": {"fp32_ppl": 2, "npu_ppl": 3}}}
                ctx = SimpleNamespace(root=root, result=root / "build/work", config={"name": model})
                publish(ctx, report)
                saved = json.loads((root / "results/summary.json").read_text())
                self.assertEqual(saved["sample_unit"], unit)
                self.assertNotIn("output_errors", saved)
                self.assertEqual(saved["accuracy"][metric]["fp32_accuracy"], 0.8)
                self.assertEqual({path.name for path in (root / "results").iterdir()},
                                 {"summary.json", "summary_test.json"})

    def test_zero_vector_metrics(self):
        """零向量不能伪造 CosSim 或 NRMSE,有效 RMSE 仍正常记录."""
        metric = Moments()
        metric.update(np.zeros(2), np.ones(2))
        result = metric.result()
        self.assertIsNone(result["cosine_similarity"])
        self.assertIsNone(result["nrmse_percent"])
        self.assertAlmostEqual(result["rmse"], 1)

    def test_artifact_contracts_ignore_history(self):
        """全部模型只绑定契约内 DLA,历史文件不能污染参考或引发板端缺文件."""
        from tools.deploy import deployment_artifacts

        for model in TASKS:
            with self.subTest(model=model):
                output = self.work / model
                output.mkdir()
                config = {"name": model, "conversion": {"precision": "w8a16"}}
                if model in ("minicpm5_2b", "deepseek_r1_distill_qwen_1_5b"):
                    package = output / "models"
                    package.mkdir()
                    names = ["prefill.dla", "decode.dla"]
                    (package / "manifest.json").write_text(json.dumps({
                        "context": 256, "model_files": dict(zip(("prompt", "decode"), names))}))
                elif model == "mit_bevfusion":
                    package = output / "models"
                    package.mkdir()
                    names = ["fusion.dla"]
                    (package / "contract.json").write_text(json.dumps({"stages": {
                        "fusion": {"backend": "mtk_npu", "file": "fusion.dla"},
                        "camera": {"backend": "onnxruntime_cpu", "file": "camera.onnx"}}}))
                elif model == "whisper_tiny":
                    package = output
                    names = ["encoder.dla", "decoder.dla"]
                    (output / "build").mkdir()
                    (output / "build/runtime_contract.json").write_text(json.dumps({
                        "encoder": {"file": names[0]}, "decoder_step": {"file": names[1]}}))
                else:
                    package = output
                    names = ["model_w8a16.dla" if model == "yamnet" else "model_int8.dla"]
                for name in [*names, "obsolete.dla"]:
                    (package / name).write_bytes(b"test-artifact")
                result = deployment_artifacts(SimpleNamespace(output=output, config=config))
                self.assertEqual(set(result), set(names))

    def test_runtime_cleanup_boundaries(self):
        """真实清理脚本只删除 models 内旧文件,不碰结果、参考或链接外目标."""
        from tools import deploy

        root = self.work
        models = root / "models"
        models.mkdir()
        (models / "current.dla").write_bytes(b"current")
        (models / "old").mkdir()
        (models / "old/unused.dla").write_bytes(b"old")
        outside = root / "important.txt"
        outside.write_text("keep")
        ctx = SimpleNamespace(config={"board": {"deploy_dir": str(root)}})
        with mock.patch.dict(os.environ, {"MTK_BOARD_HOST": "test-only"}), \
                mock.patch.object(deploy, "run_command") as command:
            deploy.prune_models(ctx, ["current.dla"])
        arguments = shlex.split(command.call_args.args[0][-1])
        subprocess.run([sys.executable, *arguments[1:]], capture_output=True, check=True)
        self.assertFalse((models / "old").exists())
        self.assertTrue(outside.is_file())
        (models / "outside-link").symlink_to(outside)
        result = subprocess.run([sys.executable, *arguments[1:]], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(outside.is_file())

    def test_host_subset_stage_order_all_models(self):
        """真实子集编排对全部模型先完成浮点评分,再上传模型和参考."""
        from tools import compile_accuracy, deploy, process

        with mock.patch.dict(os.environ, {"MTK_BOARD_HOST": "test-only"}):
            for model, (metric, unit) in TASKS.items():
                with self.subTest(model=model):
                    folder = self.work / model
                    work = folder / "work"
                    work.mkdir(parents=True)
                    dla = folder / "model.dla"
                    dla.write_bytes(b"current")
                    backend = "pytorch" if unit == "question" else "onnx"
                    report = {"evaluation": evaluation.accuracy_result(model, backend, {metric: 0.8}, 1, unit),
                              "sample_ids_sha256": os.environ["MTK_EVAL_SAMPLE_SHA256"],
                              "cases_sha256": "cases", "identity": "identity", "sample_tokens": ["sample"]}
                    source = work / ("onnx_reference.json" if model == "mit_bevfusion" else "onnx_reference_test.json")
                    if model == "yamnet":
                        source = work / "evaluation/test/onnx/summary.json"
                        source.parent.mkdir(parents=True)
                    source.write_text(json.dumps(report))
                    quality = work / "quality.json"
                    quality.write_text(json.dumps({"perplexity": 2, "input_ids": [1], "scored_tokens": 1}))
                    ctx = SimpleNamespace(work=work, output=folder, repo=ROOT,
                                          config={"name": model, "board": {"deploy_dir": "/test-only"}})
                    events = []

                    def step(path, *arguments):
                        """记录独立步骤,不执行模型、量化或数据准备."""
                        events.append("numeric" if Path(path).name == "prepare_output_reference.py" else "float")

                    def command(arguments, **kwargs):
                        """记录上传顺序和工作目录环境,不连接远端."""
                        if len(arguments) > 1 and Path(str(arguments[1])).name == "upload.py":
                            events.append("upload")
                            self.assertEqual(kwargs["env"]["MTK_EVALUATION_WORK"], "0")
                        else:
                            events.append("reference")

                    with mock.patch.object(deploy, "context", return_value=ctx), \
                            mock.patch.object(deploy, "deployment_artifacts", return_value={"model.dla": dla}), \
                            mock.patch.object(compile_accuracy, "language", return_value=(source, quality, work)), \
                            mock.patch.object(compile_accuracy, "run", side_effect=step), \
                            mock.patch.object(process, "run_command", side_effect=command):
                        if unit == "question":
                            events.append("float")
                        compile_accuracy.host_reference(folder / "deploy")
                    self.assertLess(events.index("float"), events.index("upload"))
                    self.assertLess(events.index("upload"), events.index("reference"))
                    if model != "mit_bevfusion":
                        self.assertLess(events.index("upload"), events.index("numeric"))
                        self.assertLess(events.index("numeric"), events.index("reference"))

    def test_yamnet_actual_metric_coverage(self):
        """实际音频评分函数与统一接口覆盖相同的折和可映射类别."""
        source = ROOT / "mtk_models/audio/ambient_sound/yamnet/deploy/board/audio_utils.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("project_scores", "metrics", "evaluation_ids")]
        namespace = {"np": np}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        records = [{"target": target, "fold": fold, "filename": f"{fold}-{target}-{index}.wav"}
                   for target in range(50) for fold in range(1, 6) for index in range(8)]
        indices = [[] if target < 3 else [target] for target in range(50)]
        for scope, expected in (("subset", 188), ("full", 1504)):
            os.environ["MTK_EVAL_SCOPE"] = scope
            chosen = evaluation.audio_records(records)
            identifiers = namespace["evaluation_ids"](chosen, indices)
            actual = namespace["metrics"](np.zeros((len(chosen), 521), np.float32), chosen, indices)
            self.assertEqual(len(identifiers), expected)
            self.assertEqual(actual["held_out"]["samples"], expected)

    def test_bev_upload_skips_empty_groups(self):
        """实际上传函数保留空目录创建,不执行空 scp,缺文件仍在上传前失败."""
        import importlib.util

        model = ROOT / "mtk_models/pointcloud/three_d_fusion/mit_bevfusion"
        source = model / "deploy/host/upload.py"
        spec = importlib.util.spec_from_file_location("bev_upload_test", source)
        uploader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(uploader)
        package = self.work / "models"
        package.mkdir()
        stages = {"camera": {"backend": "onnxruntime_cpu", "file": "bevfusion-det_camera.onnx"},
                  "fusion": {"backend": "mtk_npu", "file": "fusion.dla"}}
        (package / "contract.json").write_text(json.dumps({"stages": stages}))
        for name in ("sparse_weights.npz", "geometry.npz", "bevfusion-det_camera.onnx",
                     "bevfusion-det_fusion.onnx", "fusion.dla"):
            (package / name).write_bytes(b"test-only")
        for name in ("sample.npz", "sample.json"):
            (self.work / name).write_bytes(b"test-only")
        ctx = SimpleNamespace(output=self.work, work=self.work, model=model, repo=ROOT,
                              config={"board": {"deploy_dir": "/test-only/bevfusion"}})
        with mock.patch.dict(os.environ, {"MTK_BOARD_HOST": "test-only"}), \
                mock.patch.object(sys, "argv", [str(source)]), \
                mock.patch.object(uploader, "context", return_value=ctx), \
                mock.patch.object(uploader, "run_command") as command, \
                mock.patch.object(uploader, "prune_models") as prune:
            uploader.main()
            calls = [call.args[0] for call in command.call_args_list]
            self.assertEqual(calls[0][0], "ssh")
            self.assertIn("/test-only/bevfusion/build/reference", calls[0][-1])
            transfers = [args for args in calls if args[0] == "scp"]
            self.assertEqual(len(transfers), 5)
            for arguments in transfers:
                self.assertTrue(arguments[5:-1])
                self.assertTrue(all(Path(path).is_file() for path in arguments[5:-1]))
                self.assertFalse(arguments[-1].endswith("/build/reference/"))
            prune.assert_called_once()

            (package / "fusion.dla").unlink()
            command.reset_mock()
            prune.reset_mock()
            with self.assertRaises(FileNotFoundError):
                uploader.main()
            command.assert_not_called()
            prune.assert_not_called()

    def test_bev_identity_ignores_unused_graphs(self):
        """真实模型标识只包含契约所用图和协议,目录残留不能破坏两端匹配."""
        source = ROOT / "mtk_models/pointcloud/three_d_fusion/mit_bevfusion/deploy/board/evaluate.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "identity")
        namespace = {"Path": Path, "hashlib": hashlib, "json": json, "__file__": str(source)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        (self.work / "contract.json").write_text(json.dumps({"stages": {
            "camera": {"backend": "onnxruntime_cpu", "file": "bevfusion-det_camera.onnx"}}}))
        for name in ("bevfusion-det_camera.onnx", "sparse_weights.npz", "geometry.npz"):
            (self.work / name).write_bytes(b"current")
        baseline = namespace["identity"](self.work)
        (self.work / "obsolete.onnx").write_bytes(b"old")
        self.assertEqual(namespace["identity"](self.work), baseline)


if __name__ == "__main__":
    unittest.main()
