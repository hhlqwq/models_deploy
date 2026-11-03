# 模型目录

本目录按照“Scenario → Category → Model”三级结构组织模型，覆盖模型来源、转换、量化、
MTK NPU 部署、板端 Demo、精度与性能评估以及文档交付。Qualcomm AI Hub Models 仅作为
目录结构、模型卡、Demo 和评测方式的 `delivery_reference`，实际模型必须使用官方或原作者
公开源码与权重完成 MTK 侧转换和验证。

## 统一入口与 ONNX 命名

在已有编译容器内执行 `bash mtk_models/deploy/compile.sh`,顺序调用全部 13 个模型的编译上传入口.该入口将本目录 `deploy/` 下的 compile、evaluate、upload 三个 Shell 脚本及本地回传配置上传到板端部署根目录 `deploy/` 下.板端运行同一份 `compile.sh` 做固定子集任务精度对比并保存示例,运行 `evaluate.sh` 做各模型约定的全量评价.容器中的 `bash mtk_models/deploy/evaluate.sh` 仅准备已有浮点模型的全量参考,必要资源按模型提示传输.板端只执行代码,不负责修改或生成脚本.

板端总 compile 成功后自动回传 results/summary_test.json 和示例,总 evaluate 成功后自动回传各模型 `results/summary.json` 和 `examples/output/` 到编译服务器对应模型目录.回传实现位于总 compile.sh 末尾,evaluate.sh 完成后调用仅回传分支,不重跑模型;不再提供 upload.sh.目标账户及仓库根目录由本地 `results_host`、`results_root` 配置,不包含在公开脚本中.缺失文件如实报告,不上传模型或临时过程文件,不自动提交结果.Whisper 新部署目标为全整数 W8A16、1t256c、Decoder 上限 448,本次流程修改不重新运行模型精度,下表历史浮点指标不代表该 W8A16 目标的新测结果.

入口按 ResNet101、ViT、YOLOv5s、YOLOv8n、MobileFaceNet、RTMPose、FastSAM、Depth Anything V2、YAMNet、Whisper、BEVFusion、MiniCPM、DeepSeek 的顺序执行,首个失败即停止.板端部署子目录与模型代码目录末级名称一致,修改布局时同步更新总入口.

| 原始模型资源 | 导出的 ONNX 文件 |
| --- | --- |
| `resnet101-cd907fc2.pth` | `resnet101.onnx` |
| `vit_b_16-c867db91.pth` | `vit_b_16.onnx`、`vit_b_16_mtk_compatible.onnx` |
| `yolov5s.pt` | `yolov5s.onnx` |
| `yolov8n.pt` | `yolov8n.onnx` |
| `mobilefacenet.pt` | `mobilefacenet.onnx` |
| RTMPose 官方 `rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth` | `rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192.onnx` 及同名前缀的 `_mtk_compatible.onnx` |
| `FastSAM-s.pt` | `FastSAM-s.onnx` |
| `depth_anything_v2_vits.pth` | `depth_anything_v2_vits.onnx` |
| `yamnet.h5` | `yamnet.onnx` |
| Whisper `tiny.pt` | `tiny_encoder.onnx`、`tiny_decoder_step.onnx` |
| `bevfusion-det.pth` | `bevfusion-det_camera.onnx`、`bevfusion-det_camera_downsample.onnx`、`bevfusion-det_fusion.onnx`、`bevfusion-det_head.onnx` |

MiniCPM 与 DeepSeek 沿用原生 LLM 双静态图链路,不增加 ONNX 导出.命名调整已同步导出、转换、评价和相关校验引用,未自动改名或删除 NAS 旧产物,需要重新运行编译入口生成新文件.本次入口及命名修改尚未执行编译或板端测试.

## Models List

原有 11 个模型的推理代码已迁移为 Python,尚未做板端验证.这些模型的下表指标和示例保留迁移前 C++/原生 CLI 的历史记录,不能代表当前 Python 实现;当前状态见各模型 summary.json.ResNet101 已完成 Python NPU 全量评价,其下表指标为当前实现的实际结果.

列出计划适配的模型及已完成量化的版本.型号与结果按具体版本对应,不在不同规模模型之间复用数据.具体版本与来源见各模型文档.

原始权重与 ONNX 为公共参考;量化方式、编译后大小、性能和精度按平台分别记录.G5100 暂无已回收实测,预留独立列.`—` 表示尚无可引用记录,不代表零或已完成适配.点击模型名称查看完整版本、来源和限制,点击“报告”查看实测依据.平台“编译与性能”单元格依次列出量化方式、DLA 大小、NPU 耗时和 RSS.

| 模型 / 版本 | 场景 / 类别 | 输入 / 参数量 | 原始权重 / FP32 ONNX | 数据集 / 指标 / 参考基线 | G720 量化方式 / 编译与性能 | G720 **CosSim**/RMSE/NRMSE/精度 | G5100 量化方式 / 编译与性能 | G5100 **CosSim**/RMSE/NRMSE/精度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5n-v7.0（640×640） | 感知 / 目标检测 | — | — | — | — | — | — | — |
| YOLOv5m-v7.0（640×640） | 感知 / 目标检测 | — | — | — | — | — | — | — |
| [YOLOv8n（640×640）](perception/object_detection/yolov8n/README.md)<br>YOLOv8-N,Ultralytics `v8.0.111` | 感知 / 目标检测 | RGB `1×3×640×640`<br>3.16 M | 原始: 6.53 MB<br>ONNX: 12.66 MB | COCO val2017,5000 张<br>bbox mAP@0.5:0.95<br>ONNX: 36.63% | **W8A8**<br>DLA: 3.53 MB<br>NPU: 6.283 ms<br>RSS: 31.32 MiB | —/—/—/35.35%<br>(-1.28%)<br>[报告](perception/object_detection/yolov8n/results/summary.json) | — | — |
| YOLOv8m（640×640） | 感知 / 目标检测 | — | — | — | — | — | — | — |
| RT-DETRv2-S（640×640） | 感知 / 目标检测 | — | — | — | — | — | — | — |
| SqueezeNet-1.1 | 感知 / 分类 | — | — | — | — | — | — | — |
| ResNet50 | 感知 / 分类 | — | — | — | — | — | — | — |
| [ResNet101](perception/image_classification/resnet101/README.md)<br>TorchVision `IMAGENET1K_V2` | 感知 / 分类 | RGB `1×3×224×224`<br>44.55 M | 原始: 178.81 MB<br>ONNX: 178.04 MB | ImageNet val2012,50000 张<br>Top-1 Accuracy<br>ONNX: 81.90% | **W8A8**<br>DLA: 45.38 MB<br>NPU: 5.157 ms<br>RSS: 243.74 MiB | —/—/—/81.54%<br>(-0.35%)<br>[报告](perception/image_classification/resnet101/results/summary.json) | — | — |
| MobileNetV3-Large | 感知 / 分类 | — | — | — | — | — | — | — |
| [ViT-B/16](perception/image_classification/vit_base_patch16_224/README.md)<br>ViT-B/16,TorchVision `IMAGENET1K_V1` | 感知 / 分类 | RGB `1×3×224×224`<br>86.57 M | 原始: 346.33 MB<br>ONNX: 346.44 MB | ImageNet val2012,50000 张<br>Top-1 Accuracy<br>ONNX: 80.66% | **W8A8**<br>DLA: 88.00 MB<br>NPU: 51.754 ms<br>RSS: 141.30 MiB | —/—/—/79.40%<br>(-1.27%)<br>[报告](perception/image_classification/vit_base_patch16_224/results/summary.json) | — | — |
| MediaPipe Gesture Recognizer（Palm+Hand Landmark+Gesture Classifier） | 交互 / 手势识别 | — | — | — | — | — | — | — |
| DepthAI Hand Tracker（PalmDetection+HandLandmark 21点） | 交互 / 手部关键点 | — | — | — | — | — | — | — |
| [MobileFaceNet](interaction/face_recognition/mobilefacenet/README.md)<br>foamliu MobileFaceNet,`v1.0` 权重,128 维特征 | 交互 / 人脸识别 | RGB 对齐人脸 `1×3×112×112`<br>1.00 M | 原始: 4.14 MB<br>ONNX: 4.00 MB | LFW,MTCNN 五点对齐,6000 对<br>验证准确率<br>ONNX: 99.38% | **W8A8**<br>DLA: 1.41 MB<br>NPU: 0.515 ms<br>RSS: 8.25 MiB | —/—/—/99.32%<br>(-0.07%)<br>[报告](interaction/face_recognition/mobilefacenet/results/summary.json) | — | — |
| MoveNet SinglePose Lightning（192×192） | 交互 / 姿态估计 | — | — | — | — | — | — | — |
| [RTMPose-m WholeBody（256×192，133点）](interaction/pose_detection/rtmpose_body2d/README.md)<br>RTMPose-M WholeBody 133 点,MMPose `v1.3.2` | 交互 / 姿态估计 | RGB 人体裁剪 `1×3×256×192`<br>17.95 M | 原始: 72.01 MB<br>ONNX: 71.83 MB | COCO-WholeBody val2017,104125 个人体<br>WholeBody AP<br>ONNX: 57.04% | **W8A8**<br>DLA: 19.14 MB<br>NPU: 3.862 ms<br>RSS: 32.74 MiB | —/—/—/53.24%<br>(-3.80%)<br>[报告](interaction/pose_detection/rtmpose_body2d/results/summary.json) | — | — |
| HRNet-W32（COCO 17点） | 交互 / 姿态估计 | — | — | — | — | — | — | — |
| BlazePose GHUM Lite（33点） | 交互 / 姿态估计 | — | — | — | — | — | — | — |
| PROCR v4 | 交互 / 文本识别 | — | — | — | — | — | — | — |
| Fast-SCNN（Cityscapes 19类） | 导航 / 语义分割 | — | — | — | — | — | — | — |
| DeepLabV3-MobileNetV3-Large（VOC 21类） | 导航 / 语义分割 | — | — | — | — | — | — | — |
| segformer-b0-finetuned-ade-512-512 | 导航 / 语义分割 | — | — | — | — | — | — | — |
| [FastSAM-s](navigation/segmentation/fastsam/README.md)<br>官方 `FastSAM-s.pt`,Ultralytics `v8.0.111` | 导航 / 语义分割 | RGB `1×3×640×640`<br>11.79 M | 原始: 23.83 MB<br>ONNX: 47.18 MB | COCO val2017,5000 张<br>类别无关 segm AR@100<br>ONNX: 39.02% | **W8A8**<br>DLA: 12.29 MB<br>NPU: 14.725 ms<br>RSS: 64.55 MiB | —/—/—/37.57%<br>(-1.45%)<br>[报告](navigation/segmentation/fastsam/results/summary.json) | — | — |
| FastSAM-x | 导航 / 语义分割 | — | — | — | — | — | — | — |
| EdgeSAM（Image Encoder+Prompt Decoder） | 导航 / 语义分割 | — | — | — | — | — | — | — |
| [Depth Anything V2-Small（ViT-S）](navigation/single_camera_depth/depth_anything_v2_small/README.md)<br>Depth Anything V2 Small,ViT-S/14 | 导航 / 单目深度 | RGB `1×3×518×518`<br>24.79 M | 原始: 99.22 MB<br>ONNX: 98.97 MB | DA-2K,2068 个点对<br>点对准确率<br>ONNX: 94.83% | **W8A8**<br>DLA: 25.83 MB<br>NPU: 133.017 ms<br>RSS: 33.04 MiB | —/—/—/85.78%<br>(-9.04%)<br>[报告](navigation/single_camera_depth/depth_anything_v2_small/results/summary.json) | — | — |
| StereoNetPlus-v1.2.1（MixVarGENet + FPN，SceneFlow） | 导航 / 双目深度 | — | — | — | — | — | — | — |
| DINOv2 ViT-S/14（dinov2_vits14） | 导航 / 视觉基础模型 | — | — | — | — | — | — | — |
| CLIP ViT-B/16 | 生成式AI / 视觉语言 | — | — | — | — | — | — | — |
| [MiniCPM5-2B](gen_ai/llm/minicpm5_2b/README.md) | 生成式AI / 大语言模型 | Token + KV Cache<br>上下文 1024 Token<br>Prefill 每批 128 Token<br>逐 Token 解码<br>约 2520 M | 原始: 5033.56 MB | MMLU test,57 学科固定 1000 题,0-shot<br>Accuracy,FP32: 55.90%<br>WikiText2 test 前 128 行/7366 Token<br>PPL,FP32: 40.04 (均为子集) | **W4A16 (INT4/INT16)**<br>DLA: 2292.52 MB<br>embedding: 534.77 MB<br>部署合计: 2827.29 MB<br>Prefill: 288.67 Token/s<br>Decode: 8.15 Token/s<br>RSS: 1152.42 MiB | —/—/—/MMLU: 41.30%<br>(-14.60%)<br>PPL: 48.38<br>(+8.34)<br>[报告](gen_ai/llm/minicpm5_2b/results/summary.json) | — | — |
| [DeepSeek-R1-Distill-Qwen-1.5B（W4A16）](gen_ai/llm/deepseek_r1_distill_qwen_1_5b/README.md) | 生成式AI / 大语言模型 | Token + KV Cache<br>上下文 1024 Token<br>Prefill 每批 128 Token<br>逐 Token 解码<br>1777.09 M | 原始: 3554.21 MB | MMLU 固定 16 题,FP32: 75.00%<br>WikiText2 固定 4 块,PPL: 93.3811<br>少量诊断,非全量 | **W4A16 (INT4/INT16)**<br>DLA: 1579.20 MB<br>embedding: 466.75 MB<br>部署合计: 2045.94 MB<br>标准性能/RSS 未测 | —/—/—/MMLU: 31.25%<br>(-43.75%)<br>PPL: 135.6519<br>(+42.2708)<br>[报告](gen_ai/llm/deepseek_r1_distill_qwen_1_5b/results/summary.json) | — | — |
| DeepSeek-R1-Distill-Qwen-7B | 生成式AI / 大语言模型 | — | — | — | — | — | — | — |
| InternVL3-1B | 生成式AI / VLM | — | — | — | — | — | — | — |
| InternVL3-8B | 生成式AI / VLM | — | — | — | — | — | — | — |
| Gemma 4 E2B | 生成式AI / VLM | — | — | — | — | — | — | — |
| Gemma 4 E4B | 生成式AI / VLM | — | — | — | — | — | — | — |
| MeloTTS | 音频 / 语音合成 | — | — | — | — | — | — | — |
| PiperTTS | 音频 / 语音合成 | — | — | — | — | — | — | — |
| WebRTC AEC3 | 音频 / 环境音频 | — | — | — | — | — | — | — |
| DeepFilterNet v3 | 音频 / 环境音频 | — | — | — | — | — | — | — |
| [YAMNet（TF Hub，521类）](audio/ambient_sound/yamnet/README.md)<br>Google AudioSet 521 类 | 音频 / 环境音频 | Log-Mel `1×1×96×64`,单声道音频<br>3.73 M | 原始: 15.30 MB<br>ONNX: 14.94 MB | ESC-50 第 2-5 折,47 类/1504 条有效音频<br>标签投影宏平均 AP<br>ONNX: 74.15% | **W8A16**<br>DLA: 3.99 MB<br>NPU: 0.516 ms<br>RSS: 10.75 MiB | —/—/—/73.90%<br>(-0.25%)<br>[报告](audio/ambient_sound/yamnet/results/summary.json) | — | — |
| YOLOv8n + OSNet_x0_25 + BoT-SORT（BoxMOT） | SLAM / 目标跟踪 | — | — | — | — | — | — | — |
| MobileNetVLAD（mobilenetvlad_depth-0.35） | SLAM / 定位 | — | — | — | — | — | — | — |
| HF-Net | SLAM / 定位 | — | — | — | — | — | — | — |
| SuperPoint（MagicLeap pretrained） | SLAM / 定位 | — | — | — | — | — | — | — |
| LightGlue（SuperPoint，固定1024关键点） | SLAM / 定位 | — | — | — | — | — | — | — |
| PointPillar-Tiny | 点云 / 点云 | — | — | — | — | — | — | — |
| CenterPoint-Pillar 0.2（双 PFN,10 sweeps）<br>[tianweiy/CenterPoint](https://github.com/tianweiy/CenterPoint) | 点云 / 点云 | 每点 5 维,10 sweeps<br>参数量待统计 | — | nuScenes,10 类<br>mAP / NDS | — | — | — | — |
| [MIT BEVFusion](pointcloud/three_d_fusion/mit_bevfusion/README.md)<br>camera+lidar,swint_v0p075 | 点云 / 3D融合 | 六相机 256×704<br>5维 LiDAR,最多10帧 | 原始: 163.72 MB<br>四个 FP32 子图 | nuScenes trainval val,6019 帧<br>mAP / NDS,待评价 | **CPU FP32 + NPU FP16**<br>三 DLA: 13.43 MB<br>一帧板端完整通路通过 | — | — | — |
| [YOLOv5s-v7.0](perception/object_detection/yolov5s/README.md)<br>YOLOv5-S,官方 `v7.0` 权重 | 感知 / 目标检测 | RGB `1×3×640×640`<br>7.24 M | 原始: 14.81 MB<br>ONNX: 28.94 MB | COCO val2017,5000 张<br>bbox mAP@0.5:0.95<br>ONNX: 37.09% | **W8A8**<br>DLA: 7.66 MB<br>NPU: 9.757 ms<br>RSS: 33.23 MiB | —/—/—/35.86%<br>(-1.23%)<br>[报告](perception/object_detection/yolov5s/results/summary.json) | — | — |

### 数据说明

- 输入按 `N×C×H×W` 表示,图像 batch=1;音频采样率为 16 kHz.参数量 `1 M = 100 万`,不含 BatchNorm 运行统计量等缓冲区.
- 文件大小采用十进制 MB,`1 MB = 1,000,000 bytes`;RSS 为 MiB,与文件大小分开记录.原始权重为当前官方/原作者检查点文件实际大小,保留格式,不将其等同于 FP32 网络大小.多分片、双模型按实际必需文件合计;DLA 不含外置 embedding,有外置权重时另列部署合计.
- `WnAm` 表示权重 n 位、激活 m 位.浮点必须同时标明 FP16/FP32,不可与同位宽整数定点量化混淆;W8A8 为 INT8 PTQ,主要权重采用逐输出通道量化,不表示偏置、全部算子或文件 I/O 都是 8 位.本表不统计仅有浮点部署结果的版本.
- 精度仅比较同模型、同数据集与同评测协议的参考基线.精度下一行的括号记录 NPU 减参考的有符号差值,百分比精度的差值采用百分点,沿用 `%` 展示;PPL 使用绝对差值.准确率/AP/AR 越高越好,WER/PPL 越低越好.差值按报告原始数值计算,不由展示后舍入值相减.MiniCPM5 使用原框架 FP32 参考,不是 ONNX 基线.
- CosSim/RMSE/NRMSE 比较同输入下浮点参考与 NPU 的后处理前输出,NRMSE = RMSE / RMS(参考输出) × 100%.缺少实测时用 `—`,不从任务精度推算;多输出分别记录,不混合不同量纲.计算、覆盖范围及资源说明见[输出误差统计](../docs/output_metrics.md).
- NPU 耗时按一次模型推理统计.图像模型 batch=1;RTMPose 一次输入为一个裁剪好的人体,YAMNet 一次输入为 96 帧 Log-Mel 特征,不等同于整段音频处理耗时.
- MiniCPM 的 Prefill 是输入处理阶段,Decode 是逐 Token 生成阶段.表中速度来自 128 Token 输入、最多生成 128 Token 的测速协议;当前 RSS 来自该次生成性能进程,不是设备总内存.
- 指标来源为各模型结果报告.不同任务的加载、计时和内存范围不同,不作为统一性能排名.


## 目录结构

```text
models/
├── perception/
│   ├── object_detection/
│   └── image_classification/
├── interaction/
│   ├── gesture/
│   ├── hand_landmark/
│   ├── face_recognition/
│   ├── pose_detection/
│   └── text_recognition/
├── navigation/
│   ├── segmentation/
│   ├── single_camera_depth/
│   ├── dual_camera_depth/
│   └── vision_foundation_model/
├── gen_ai/
│   ├── vision_language/
│   ├── llm/
│   └── vlm/
├── audio/
│   ├── stt/
│   ├── tts/
│   └── ambient_sound/
├── slam/
│   ├── object_tracking/
│   └── localization/
├── pointcloud/
│   └── pointcloud/
└── three_d/
    └── three_d_fusion/
```

未开始的模型只保留在 `registry/target_models.yaml`，不预先创建大量空目录。模型通过
来源、许可证、图结构和 MTK 算子预检查并进入实现阶段后，使用 `tools/create_model.py`
创建完整交付目录，再加入 `registry/models.yaml`。

## 使用与资源

各模型的 README 提供配置、编译上传及板端运行方法.公共环境、NAS 资源规则和新增模型要求见[仓库说明](../README.md),板端目录及评价要求见[Genio 720 规范](../docs/genio_720.md).

型号来源由 `registry/target_models.yaml` 和各模型的 `models/source_url.txt` 记录.原始权重、转换产物和数据集保存在 NAS,不放进代码目录或临时目录作为正式资源.
