"""在编译主机评测各模型 ONNX 精度,复用板端的数据与后处理协议."""

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np
import onnxruntime as ort
import tqdm

from tools.evaluation import (
    accuracy_result, SPEECH_EXAMPLES, coco_ids, is_test, pose_image_ids, result_status,
    scope_metadata, select_samples,
)
from tools.output_metrics import ReferenceComplete, record_reference, finish_reference


ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "vit_base_patch16_224": "perception/image_classification/vit_base_patch16_224",
    "rtmpose_body2d": "interaction/pose_detection/rtmpose_body2d",
    "mobilefacenet": "interaction/face_recognition/mobilefacenet",
    "depth_anything_v2_small": "navigation/single_camera_depth/depth_anything_v2_small",
    "fastsam": "navigation/segmentation/fastsam",
    "whisper_tiny": "audio/stt/whisper_tiny",
}


def load_module(name: str, path: Path):
    """按明确路径加载已有处理函数,避免不同模型同名模块冲突."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_session(path: Path) -> ort.InferenceSession:
    """加载 ONNX 浮点基准,允许使用编译主机 CPU 或 CUDA."""
    providers = [name for name in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if name in ort.get_available_providers()]
    session = ort.InferenceSession(str(path), providers=providers)
    session.output_reference_source = path
    print(f"[ONNX] {path.name}: {session.get_providers()}", flush=True)
    return session


def infer(session: ort.InferenceSession, tensor: np.ndarray, *, sample_id, sample_files) -> list:
    """按模型声明的浮点输入类型运行推理,不统计主机性能."""
    details = session.get_inputs()
    if len(details) != 1 or details[0].type not in ("tensor(float)", "tensor(float16)"):
        raise ValueError("基准模型必须只有一个 FP32 或 FP16 输入.")
    dtype = np.float16 if details[0].type == "tensor(float16)" else np.float32
    outputs = session.run(None, {details[0].name: tensor.astype(dtype)})
    if any(not np.isfinite(value).all() for value in outputs):
        raise ValueError("ONNX 输出包含非有限数值.")
    record_reference(session.output_reference_source, [tensor.astype(dtype)], {
        item.name: value for item, value in zip(session.get_outputs(), outputs)},
        sample_id=sample_id, sample_files=sample_files)
    return outputs


def evaluate_vit(args, deploy: Path) -> tuple:
    """按板端 OpenCV 缩放和中心裁剪协议计算 ImageNet Top-1."""
    helper = load_module("vit_metrics", deploy / "full_accuracy_board.py")
    preprocessing = load_module("vit_infer", deploy / "infer.py")
    labels = helper.read_labels(args.dataset_root / "val_labels_0based.txt")
    images = args.dataset_root / "val"
    if not images.is_dir():
        images = args.dataset_root / "ILSVRC2012_img_val"
    paths = sorted(images.glob("*.JPEG"))
    if len(paths) != 50000:
        raise ValueError("ImageNet 验证集必须包含 50000 张图片.")
    paths = select_samples(paths, args.model, key=lambda path: path.name)
    session = make_session(args.onnx)
    correct = 0
    for index, path in enumerate(tqdm.tqdm(paths, desc="ONNX Top-1", unit="img")):
        source_index = int(path.stem.rsplit("_", 1)[1]) - 1
        if (not 0 <= source_index < len(labels)
                or path.name != f"ILSVRC2012_val_{source_index + 1:08d}.JPEG"
                or (not is_test() and source_index != index)):
            raise ValueError(f"ImageNet 图片编号或全量顺序不匹配: {path}")
        tensor = preprocessing.floating_preprocess(path).astype(np.float32)
        scores = infer(session, tensor, sample_id=path.name, sample_files=[path])[0].reshape(-1)
        if scores.size != 1000:
            raise ValueError("分类输出必须有 1000 个类别.")
        correct += int(scores.argmax()) == labels[source_index]
    return "Top-1", correct / len(paths), len(paths)


def evaluate_face(args, deploy: Path) -> tuple:
    """复用对齐人脸预处理与 LFW 十折阈值协议."""
    # 仅当前模型或后端需要此依赖,其他评测模式和板端环境不要求安装.
    from face_utils import load_aligned_face
    helper = load_module("face_metrics", deploy / "full_accuracy_board.py")
    pairs = helper.load_pairs(args.dataset_root)
    names = sorted({row[key] for row in pairs for key in ("image_a", "image_b")})
    session = make_session(args.onnx)
    features = {}
    for name in tqdm.tqdm(names, desc="ONNX LFW", unit="img"):
        image_path = args.dataset_root / "images" / name
        values = infer(session, load_aligned_face(image_path),
                       sample_id=name, sample_files=[image_path])[0]
        values = values.astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(values))
        if values.size != 128 or not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"人脸特征无效: {name}")
        features[name] = values / norm
    metrics, _ = helper.evaluate_pairs(pairs, features)
    return "LFW 验证准确率", metrics["verification_accuracy"], len(pairs)


def evaluate_depth(args, deploy: Path) -> tuple:
    """复用 DA-2K 预处理、原图坐标恢复和点对判定."""
    # 仅当前模型或后端需要此依赖,其他评测模式和板端环境不要求安装.
    from depth_utils import preprocess, INPUT_SIZE
    helper = load_module("depth_metrics", deploy / "full_accuracy_board.py")
    annotations = helper.load_protocol(args.dataset_root)
    session = make_session(args.onnx)
    correct = total = 0
    for name in tqdm.tqdm(sorted(annotations), desc="ONNX DA-2K", unit="img"):
        path = args.dataset_root / name
        depth = infer(session, preprocess(path), sample_id=name, sample_files=[path])[0].squeeze()
        if depth.shape != (INPUT_SIZE, INPUT_SIZE) or depth.std() <= 0:
            raise ValueError(f"深度输出形状或数值无效: {name}")
        count, pairs = helper.score_pairs(path, depth, annotations[name])
        correct += count
        total += pairs
    return "DA-2K 点对准确率", correct / total, total


def coco_score(annotation, records: list, kind: str, index: int) -> float:
    """隐藏非核心 COCO 指标打印,同时正确处理没有检测结果的情况."""
    # 仅当前模型或后端需要此依赖,其他评测模式和板端环境不要求安装.
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    print("[ONNX] 计算 COCO 核心精度,请等待.", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        if records:
            prediction = annotation.loadRes(records)
        else:
            prediction = COCO()
            prediction.dataset = {"images": annotation.dataset["images"],
                                  "categories": annotation.dataset["categories"],
                                  "annotations": []}
            prediction.createIndex()
        evaluator = COCOeval(annotation, prediction, kind)
        evaluator.params.imgIds = coco_ids(annotation.getImgIds())
        if kind == "segm":
            evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return float(evaluator.stats[index])


def evaluate_coco(args, deploy: Path) -> tuple:
    """评测 FastSAM 类别无关分割,复用已有后处理."""
    # 仅当前模型或后端需要此依赖,其他评测模式和板端环境不要求安装.
    from pycocotools import mask as mask_utils
    from mtk_models.navigation.segmentation.fastsam.deploy.board.fastsam_utils import (
        preprocess, postprocess)
    helper = load_module("fastsam_metrics", deploy / "full_accuracy_board.py")
    with contextlib.redirect_stdout(io.StringIO()):
        annotation = helper.make_class_agnostic_ground_truth(args.dataset_root /
            "annotations/instances_val2017.json")
    protocol = json.loads((deploy / "accuracy_protocol.json").read_text(encoding="utf-8"))
    paths = sorted((args.dataset_root / "images").glob("*.jpg"))
    if len(paths) != 5000 or {int(path.stem) for path in paths} != set(annotation.getImgIds()):
        raise ValueError("COCO 图片与标注必须覆盖相同的 5000 张图片.")
    selected_ids = set(coco_ids(annotation.getImgIds()))
    paths = [path for path in paths if int(path.stem) in selected_ids]
    session = make_session(args.onnx)
    records = []
    for path in tqdm.tqdm(paths, desc=f"ONNX {args.model}", unit="img", mininterval=1.0):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"无法读取图片: {path}")
        tensor, geometry = preprocess(image)
        _, scores, masks = postprocess(
            infer(session, tensor, sample_id=path.name, sample_files=[path]), geometry,
            protocol["confidence"], protocol["nms_iou"], protocol["max_detections"],
            show_progress=False)
        for score, mask in zip(scores, masks):
            encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            encoded["counts"] = encoded["counts"].decode("ascii")
            records.append({"image_id": int(path.stem), "category_id": 1,
                            "segmentation": encoded, "score": float(score)})
    value = coco_score(annotation, records, "segm", 8)
    return "segm AR@100", value, len(paths)


def evaluate_pose(args, deploy: Path) -> tuple:
    """复用官方人体检测框、SimCC 解码和 WholeBody OKS-NMS 协议."""
    # 仅当前模型或后端需要此依赖,其他评测模式和板端环境不要求安装.
    from rtmpose_utils import preprocess_image, decode_prediction
    metrics = load_module("pose_metrics", deploy / "full_accuracy_board.py")
    detections = json.loads((args.dataset_root /
        "person_detection_results/COCO_val2017_detections_AP_H_56_person.json").read_text())
    if len(detections) != 104125:
        raise ValueError("人体检测框必须有 104125 个.")
    selected_ids = set(pose_image_ids(detections))
    ordered = sorted(((index, item) for index, item in enumerate(detections)
                      if item["image_id"] in selected_ids),
                     key=lambda item: (item[1]["image_id"], item[0]))
    session = make_session(args.onnx)
    current_id = None
    path = args.work_dir / "pose_predictions.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for index, record in tqdm.tqdm(ordered, desc="ONNX WholeBody", unit="det"):
            if record["category_id"] != 1 or min(record["bbox"][2:]) <= 0:
                raise ValueError("无效的人体检测框.")
            if current_id != record["image_id"]:
                current_id = record["image_id"]
                image = cv2.imread(str(args.dataset_root / "images" / f"{current_id:012d}.jpg"))
                if image is None:
                    raise ValueError(f"缺少图片: {current_id}")
            tensor, geometry = preprocess_image(image, record["bbox"])
            outputs = infer(session, tensor,
                sample_id=f"{current_id}:detection:{index}",
                sample_files=[args.dataset_root / "images" / f"{current_id:012d}.jpg",
                              args.dataset_root / "person_detection_results/COCO_val2017_detections_AP_H_56_person.json"])
            if len(outputs) != 2:
                raise ValueError("RTMPose 必须输出 X/Y 两个 SimCC 检测头.")
            # 根据宽度排列两个输出,不依赖导出顺序.
            outputs = sorted(outputs, key=lambda value: value.shape[-1])
            keypoints, area = decode_prediction(*outputs, geometry)
            stream.write(json.dumps({"backend": "onnxruntime",
                "detection_id": index, "image_id": current_id,
                "bbox_score": record["score"], "area": area, "keypoints": keypoints}) + "\n")
    metrics.evaluate(SimpleNamespace(predictions=path,
        annotations=args.dataset_root / "annotations/coco_wholebody_val_v1.0.json",
        formatted=args.work_dir / "formatted.json", metrics=args.work_dir / "metrics.json"),
        backend="onnxruntime")
    value = json.loads((args.work_dir / "metrics.json").read_text())["metrics"]["wholebody"]["AP"]
    return "WholeBody AP", value, len(ordered)


def load_sources(dataset_root: Path) -> dict[str, dict]:
    """从 test-clean 官方转录文件核对全部音频与参考文本。"""
    sources = {}
    for transcript in sorted(dataset_root.rglob("*.trans.txt")):
        for line in transcript.read_text(encoding="utf-8").splitlines():
            sample_id, reference_text = line.split(" ", 1)
            audio = transcript.parent / f"{sample_id}.flac"
            if sample_id in sources or not audio.is_file():
                raise ValueError(f"重复样例或缺少音频: {sample_id}。")
            sources[sample_id] = {"sample_id": sample_id,
                                  "reference_text": reference_text,
                                  "audio": audio}
    if len(sources) != 2620:
        raise ValueError(f"test-clean 样例不完整: {len(sources)}。")
    return sources


def evaluate_whisper(args, deploy: Path) -> tuple:
    """用双 ONNX 和板端相同的固定 Cache、英文 Greedy 规则计算 WER."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "Whisper ONNX 评测需要 ffmpeg,请在编译 Docker 中安装该依赖.")
    # 仅当前模型或后端需要此依赖,其他评测模式和板端环境不要求安装.
    from whisper.tokenizer import get_tokenizer
    helper = load_module("whisper_metrics", deploy / "full_accuracy_board.py")
    assets = load_module("whisper_assets", deploy.parent / "host" / "export_onnx.py")
    sources = load_sources(args.dataset_root)
    chosen = select_samples(sorted(sources), args.model, required=SPEECH_EXAMPLES)
    sources = {key: sources[key] for key in chosen}
    assets.export_assets(args.work_dir)
    protocol = load_module("decoder", deploy / "decoder.py")
    board_infer = load_module("whisper_board_infer", deploy / "infer.py")
    mel_filters = np.fromfile(args.work_dir / "mel_filters_f32.bin", dtype="<f4").reshape(80, 201)
    config = dict(line.split("=", 1) for line in
                  (args.work_dir / "decode_config.txt").read_text().splitlines())
    initial = [int(value) for value in config["initial"].split(",")]
    suppressed = [int(value) for value in config["suppress"].split(",")]
    first = [int(value) for value in config["suppress_first"].split(",")]
    encoder, decoder = make_session(args.onnx), make_session(args.decoder_onnx)
    decoder_inputs = {item.name: item for item in decoder.get_inputs()}
    cache_names = [item.name for item in decoder.get_inputs() if "_cache_" in item.name]
    if len(cache_names) != 8:
        raise ValueError("Whisper-Tiny 必须有四层共八个固定 KV Cache 输入.")
    cache_length = decoder_inputs[cache_names[0]].shape[-1]
    if cache_length != protocol.CACHE_SIZE:
        raise ValueError("Whisper 浮点参考需要重新导出 1t256c 模型.")
    tokenizer = get_tokenizer(multilingual=True, language="en", task="transcribe")
    predictions = {}
    for sample_id in tqdm.tqdm(sorted(sources), desc="ONNX WER", unit="audio"):
        # 浮点参考与板端共用同一 FP32 Log-Mel 和滑动 Cache 协议.
        mel = board_infer.preprocess(sources[sample_id]["audio"], mel_filters)[0][None]
        features = infer(encoder, mel, sample_id=f"{sample_id}:encoder",
                         sample_files=[sources[sample_id]["audio"]])[0]
        caches = {name: np.zeros(decoder_inputs[name].shape, np.float32) for name in cache_names}
        tokens, token = [], initial[0]
        for position in range(protocol.MAX_TOKEN_LENGTH):
            feeds = protocol.feeds_for_step(token, position, features, caches)
            for name, value in feeds.items():
                dtype = np.float16 if decoder_inputs[name].type == "tensor(float16)" else np.float32
                feeds[name] = value.astype(dtype)
            outputs = decoder.run(None, feeds)
            if any(not np.isfinite(value).all() for value in outputs):
                raise ValueError(f"Decoder 输出无效: {sample_id}")
            output_map = dict(zip([item.name for item in decoder.get_outputs()], outputs))
            caches = {name: output_map[name.removesuffix("_in") + "_out"] for name in cache_names}
            if position + 1 < len(initial):
                token = initial[position + 1]
                continue
            logits = outputs[0].astype(np.float32).reshape(-1)
            logits[suppressed] = -np.inf
            if not tokens:
                logits[first] = -np.inf
            token = int(logits.argmax())
            if token == int(config["eot"]):
                break
            tokens.append(token)
            if len(tokens) >= int(config["max_generated_tokens"]):
                break
        predictions[sample_id] = {"sample_id": sample_id, "tokens": tokens,
                                 "text": tokenizer.decode(tokens).strip()}
    accuracy, _ = helper.metric_summary("librispeech", sources, predictions)
    return "WER", accuracy["value"], len(predictions)


def evaluate_model(*, model, onnx, dataset_root, output, decoder_onnx=None, temp_dir=None):
    """直接执行公共浮点评测,CLI 和各模型入口共用同一实现."""
    if model not in MODELS:
        raise ValueError(f"不支持的模型: {model}.")
    if model == "whisper_tiny" and decoder_onnx is None:
        raise ValueError("Whisper 精度评测需要 Decoder ONNX.")
    args = argparse.Namespace(model=model, onnx=Path(onnx), dataset_root=Path(dataset_root),
                              output=Path(output), decoder_onnx=decoder_onnx, temp_dir=temp_dir)
    deploy = ROOT / "mtk_models" / MODELS[args.model] / "deploy" / "board"
    if args.model != "fastsam":
        sys.path.insert(0, str(deploy))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 临时目录由工具创建,成功后仅清理此目录; 失败保留现场.
    temp_root = args.temp_dir or args.output.parent
    temp_root.mkdir(parents=True, exist_ok=True)
    args.work_dir = Path(tempfile.mkdtemp(prefix="onnx_accuracy_", dir=temp_root))
    adapters = {"vit_base_patch16_224": evaluate_vit, "mobilefacenet": evaluate_face,
        "depth_anything_v2_small": evaluate_depth, "rtmpose_body2d": evaluate_pose,
        "fastsam": evaluate_coco, "whisper_tiny": evaluate_whisper}
    try:
        metric, value, samples = adapters[args.model](args, deploy)
    except ReferenceComplete:
        shutil.rmtree(args.work_dir)
        raise
    value = float(value)
    if not math.isfinite(value) or value < 0 or (metric != "WER" and value > 1):
        raise ValueError("ONNX 核心精度无效.")
    units = {"rtmpose_body2d": "person_instance", "mobilefacenet": "face_pair",
             "depth_anything_v2_small": "annotated_point_pair", "whisper_tiny": "utterance",
             "vit_base_patch16_224": "image", "fastsam": "image"}
    evaluation = accuracy_result(args.model, "onnx", {metric: value}, samples, units[args.model])
    args.output.write_text(json.dumps({"evaluation": evaluation, "status": result_status(), **scope_metadata(), "model": args.model,
        "backend": "ONNX Runtime", "metric": metric, "accuracy": value,
        "samples": samples, "source": "本次 ONNX 浮点同范围实测,使用与板端相同的评测协议"},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(args.work_dir)
    finish_reference()
    print(f"[ONNX] {metric}: {value:.6f}; 汇总: {args.output}", flush=True)



def main() -> None:
    """只保存核心精度,成功后清理本次专用临时目录."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--decoder-onnx", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temp-dir", type=Path, help="评测临时文件的父目录.")
    args = parser.parse_args()
    if args.model == "whisper_tiny" and args.decoder_onnx is None:
        parser.error("Whisper 精度评测需要 --decoder-onnx.")
    evaluate_model(model=args.model, onnx=args.onnx, dataset_root=args.dataset_root,
                   output=args.output, decoder_onnx=args.decoder_onnx, temp_dir=args.temp_dir)


if __name__ == "__main__":
    main()
