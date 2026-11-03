"""从官方元数据索引少量编译输入或独立评价清单,不计算模型指标."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from tools.deploy import context
from tools.evaluation import is_test, select_samples

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from dataset import CHANNELS, load_frame, transform


def index_frames(root, version, split):
    """按官方场景划分索引相机,LIDAR 和历史帧,不解压数据包."""
    if version == "v1.0-mini" and split == "mini_val":
        scene_names = {"scene-0103", "scene-0916"}
    elif version == "v1.0-trainval" and split == "val":
        from nuscenes.utils.splits import val
        scene_names = set(val)
    else:
        raise ValueError("仅支持官方 mini_val 通路检查或 v1.0-trainval/val 正式评价.")
    tables = {}
    for name in ("scene", "sample", "sample_data", "calibrated_sensor", "ego_pose", "sensor"):
        print(f"[数据索引] {name}", flush=True)
        with (root / version / f"{name}.json").open(encoding="utf-8") as stream:
            tables[name] = {row["token"]: row for row in json.load(stream)}
    scenes = {key for key, scene in tables["scene"].items() if scene["name"] in scene_names}
    if len(scenes) != len(scene_names):
        raise ValueError("官方划分中的场景不完整.")
    samples = sorted((row for row in tables["sample"].values() if row["scene_token"] in scenes),
                     key=lambda row: (row["timestamp"], row["token"]))
    keyframes = {}
    for row in tables["sample_data"].values():
        if row["is_key_frame"]:
            calibration = tables["calibrated_sensor"][row["calibrated_sensor_token"]]
            channel = tables["sensor"][calibration["sensor_token"]]["channel"]
            keyframes[(row["sample_token"], channel)] = row
    frames = []
    for sample in samples:
        lidar = keyframes[(sample["token"], "LIDAR_TOP")]
        calibration = tables["calibrated_sensor"][lidar["calibrated_sensor_token"]]
        pose = tables["ego_pose"][lidar["ego_pose_token"]]
        lidar_to_ego, ego_to_global = transform(calibration), transform(pose)
        lidar_to_global = ego_to_global @ lidar_to_ego
        frame = {"token": sample["token"], "lidar": lidar["filename"], "sweeps": [], "cameras": [],
                 "lidar_to_ego": lidar_to_ego.tolist(), "ego_to_global": ego_to_global.tolist()}
        previous = lidar["prev"]
        for _ in range(9):
            if not previous:
                break
            sweep = tables["sample_data"][previous]
            sweep_global = (transform(tables["ego_pose"][sweep["ego_pose_token"]])
                            @ transform(tables["calibrated_sensor"][sweep["calibrated_sensor_token"]]))
            frame["sweeps"].append({"path": sweep["filename"],
                                     "to_lidar": (np.linalg.inv(lidar_to_global) @ sweep_global).tolist(),
                                     "time_lag": (lidar["timestamp"] - sweep["timestamp"]) / 1e6})
            previous = sweep["prev"]
        for channel in CHANNELS:
            camera = keyframes[(sample["token"], channel)]
            calibration = tables["calibrated_sensor"][camera["calibrated_sensor_token"]]
            camera_global = transform(tables["ego_pose"][camera["ego_pose_token"]]) @ transform(calibration)
            camera_to_lidar = np.linalg.inv(lidar_to_global) @ camera_global
            intrinsic = np.asarray(calibration["camera_intrinsic"])
            projection = np.eye(4)
            projection[:3, :3] = intrinsic
            frame["cameras"].append({"path": camera["filename"], "intrinsic": intrinsic.tolist(),
                                      "to_lidar": camera_to_lidar.tolist(),
                                      "lidar_to_image": (projection @ np.linalg.inv(camera_to_lidar)).tolist()})
        frames.append(frame)
    expected = 81 if version == "v1.0-mini" else 6019
    if len(frames) != expected:
        raise ValueError(f"划分数量不符: {len(frames)} != {expected}.")
    return frames


def main():
    """编译仅打包一帧,正式评价只索引已有完整验证集."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    ctx = context(__file__)
    data = ctx.config["data"]
    if args.evaluate and not is_test() and (data["version"], data["split"]) != ("v1.0-mini", "mini_val"):
        raise ValueError("本模型正式评价固定使用已解压的 v1.0-mini/mini_val,共 81 帧,2 个场景.")
    frames = index_frames(ctx.dataset, data["version"], data["split"])
    if args.evaluate:
        frames = select_samples(frames, "mit_bevfusion", key=lambda frame: frame["token"])
        for number, frame in enumerate(frames, 1):
            files = [frame["lidar"], *(item["path"] for item in frame["sweeps"]),
                     *(item["path"] for item in frame["cameras"])]
            if any(not (ctx.dataset / name).is_file() for name in files):
                raise FileNotFoundError(f"评价输入缺失,样本={frame['token']}.")
            if number % 100 == 0:
                print(f"[评价输入检查] {number}/{len(frames)}", flush=True)
        target = ctx.work / "samples.json"
        target.write_text(json.dumps({"version": data["version"], "split": data["split"], "frames": frames}),
                          encoding="utf-8")
        prepare_ground_truth(ctx.dataset, ctx.work / "ground_truth.json", data["version"], data["split"])
        print("[评价输入] 已生成完整清单,请与已解压数据一起放入板端配置的数据集目录.", flush=True)
    else:
        np.savez(ctx.work / "sample.npz", **load_frame(ctx.dataset, frames[0]))
        (ctx.work / "sample.json").write_text(json.dumps(frames[0]), encoding="utf-8")
        print(f"[编译输入] 一帧 mini_val,token={frames[0]['token']}.", flush=True)


def prepare_ground_truth(root, target, version="v1.0-trainval", split="val"):
    """主机用官方 devkit 导出 val 标注及 bike-rack 过滤输入,降低板端内存需求."""
    from nuscenes import NuScenes
    from nuscenes.eval.common.loaders import add_center_dist, load_gt
    from nuscenes.eval.detection.data_classes import DetectionBox

    print(f"[官方标注] 在主机加载 {version}/{split},导出评分必要字段.", flush=True)
    database = NuScenes(version=version, dataroot=str(root), verbose=True)
    boxes = add_center_dist(database, load_gt(database, split, DetectionBox, verbose=True))
    bikeracks = {}
    for token in boxes.sample_tokens:
        annotations = [database.get("sample_annotation", key) for key in database.get("sample", token)["anns"]]
        bikeracks[token] = [{key: row[key] for key in ("token", "category_name", "translation", "size", "rotation")}
                           for row in annotations if row["category_name"] == "static_object.bicycle_rack"]
    serialized_boxes = boxes.serialize()
    # 官方 load_gt 的速度字段为 NumPy 数组,转为列表但保留原值及 NaN 缺失标记.
    for sample_boxes in serialized_boxes.values():
        for box in sample_boxes:
            box["velocity"] = np.asarray(box["velocity"]).tolist()
    target.write_text(json.dumps({"boxes": serialized_boxes, "bikeracks": bikeracks}), encoding="utf-8")
    print(f"[官方标注完成] {len(serialized_boxes)} 帧,已保存评分必要字段.", flush=True)


if __name__ == "__main__":
    main()
