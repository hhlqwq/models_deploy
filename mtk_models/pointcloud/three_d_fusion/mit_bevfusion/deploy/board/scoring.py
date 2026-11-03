"""复用 nuScenes 官方评分和过滤代码,板端不加载完整原始数据库."""

import json

import numpy as np

from geometry import CLASSES


class AnnotationIndex:
    """仅提供官方 bike-rack 过滤需要的样本及注释查询."""

    def __init__(self, bikeracks):
        """由主机在原始 nuScenes 数据库导出相关注释."""
        self.samples, self.annotations = {}, {}
        for token, records in bikeracks.items():
            self.samples[token] = {"anns": [row["token"] for row in records]}
            self.annotations.update({row["token"]: row for row in records})

    def get(self, table, token):
        """提供官方 filter_eval_boxes 使用的最小只读接口."""
        if table == "sample":
            return self.samples[token]
        if table == "sample_annotation":
            return self.annotations[token]
        raise KeyError(table)


def format_predictions(frame, boxes, scores, labels):
    """按 MIT 输出约定转为全局框,保持原作者的 yaw 和维度解释."""
    from pyquaternion import Quaternion
    from nuscenes.utils.data_classes import Box

    lidar_to_ego = np.asarray(frame["lidar_to_ego"])
    ego_to_global = np.asarray(frame["ego_to_global"])
    results = []
    defaults = {"car": "vehicle.parked", "pedestrian": "pedestrian.moving", "trailer": "vehicle.parked",
                "truck": "vehicle.parked", "bus": "vehicle.moving", "motorcycle": "cycle.without_rider",
                "construction_vehicle": "vehicle.parked", "bicycle": "cycle.without_rider",
                "barrier": "", "traffic_cone": ""}
    for values, score, label in zip(boxes, scores, labels):
        center = values[:3].copy()
        center[2] += values[5] * 0.5
        box = Box(center, values[3:6], Quaternion(axis=[0, 0, 1], radians=-values[6] - np.pi / 2),
                  velocity=(*values[7:9], 0))
        box.rotate(Quaternion(matrix=lidar_to_ego[:3, :3], atol=1e-6, rtol=1e-6))
        box.translate(lidar_to_ego[:3, 3])
        box.rotate(Quaternion(matrix=ego_to_global[:3, :3], atol=1e-6, rtol=1e-6))
        box.translate(ego_to_global[:3, 3])
        name = CLASSES[int(label)]
        speed = np.linalg.norm(box.velocity[:2])
        if speed > 0.2 and name in ("car", "construction_vehicle", "bus", "truck", "trailer"):
            attribute = "vehicle.moving"
        elif speed > 0.2 and name in ("bicycle", "motorcycle"):
            attribute = "cycle.with_rider"
        elif speed <= 0.2 and name == "pedestrian":
            attribute = "pedestrian.standing"
        elif speed <= 0.2 and name == "bus":
            attribute = "vehicle.stopped"
        else:
            attribute = defaults[name]
        results.append({"sample_token": frame["token"], "translation": box.center.tolist(),
                        "size": box.wlh.tolist(), "rotation": box.orientation.elements.tolist(),
                        "velocity": box.velocity[:2].tolist(), "detection_name": name,
                        "detection_score": float(score), "attribute_name": attribute,
                        "ego_translation": (box.center - ego_to_global[:3, 3]).tolist(), "num_pts": -1})
    return results


def score_predictions(ground_truth, predictions):
    """原样调用官方过滤,匹配,AP 与 TP 计算,只返回 mAP 和 NDS."""
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.loaders import filter_eval_boxes
    from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
    from nuscenes.eval.detection.constants import TP_METRICS
    from nuscenes.eval.detection.data_classes import DetectionBox, DetectionMetrics

    with open(ground_truth, encoding="utf-8") as stream:
        inputs = json.load(stream)
    from tools.evaluation import is_test

    if is_test():
        if not set(predictions).issubset(inputs["boxes"]):
            raise ValueError("子集预测包含未知 nuScenes 帧.")
        inputs["boxes"] = {key: inputs["boxes"][key] for key in predictions}
    gt = EvalBoxes.deserialize(inputs["boxes"], DetectionBox)
    pred = EvalBoxes.deserialize(predictions, DetectionBox)
    if set(gt.sample_tokens) != set(pred.sample_tokens):
        raise ValueError("预测与官方验证集样本未完全对应.")
    index = AnnotationIndex(inputs["bikeracks"])
    config = config_factory("detection_cvpr_2019")
    gt = filter_eval_boxes(index, gt, config.class_range)
    pred = filter_eval_boxes(index, pred, config.class_range)
    metrics = DetectionMetrics(config)
    for name in config.class_names:
        print(f"[官方评分] {name}", flush=True)
        data = {}
        for distance in config.dist_ths:
            data[distance] = accumulate(gt, pred, name, config.dist_fcn_callable, distance)
            metrics.add_label_ap(name, distance, calc_ap(data[distance], config.min_recall, config.min_precision))
        for metric in TP_METRICS:
            excluded = ((name == "traffic_cone" and metric in ("attr_err", "vel_err", "orient_err"))
                        or (name == "barrier" and metric in ("attr_err", "vel_err")))
            value = np.nan if excluded else calc_tp(data[config.dist_th_tp], config.min_recall, metric)
            metrics.add_label_tp(name, metric, value)
    return {"mAP": float(metrics.mean_ap), "NDS": float(metrics.nd_score)}
