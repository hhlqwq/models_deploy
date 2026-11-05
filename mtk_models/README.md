# 模型目录

本目录按场景、任务和模型组织，主要维护 G720 的模型来源、转换、量化、目标端推理和按需评价。独立应用见 [Demo 总览](../demo/README.md)。模型源码和权重来自作者或官方项目，旧的 Qualcomm AI Hub Models 记录作为 `delivery_reference`。

## 统一入口与 ONNX 命名

单模型优先按其 README 选择入口.需要批量运行时,在已有编译容器内执行 `bash mtk_models/deploy/compile.sh`,顺序调用 13 个模型的编译上传入口,同时上传 `compile_board.sh`、`evaluate_board.sh`、`common.sh` 与本地回传配置.板端总入口分别编排模型的 `compile.sh` 与 `evaluate.sh`.需要独立评价时,容器端 `bash mtk_models/deploy/evaluate.sh` 使用已有模型准备所需参考.

板端总入口通过 `deploy/common.sh` 回传已有摘要和示例.目标账户及仓库根目录由本地 `results_host`、`results_root` 配置.结果范围按各模型记录说明.

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

MiniCPM 与 DeepSeek 沿用原生 LLM 双静态图链路。

## Models List

G720 的 13 个模型均已完成 Python 板端子集及约定范围正式评价。各模型 README 汇总精度、性能、进程峰值 RSS 和子集输出误差，并链接板端示例与两份报告。

列出计划适配的模型及已完成量化的版本.型号与结果按具体版本对应,不在不同规模模型之间复用数据.具体版本与来源见各模型文档.

原始权重与 ONNX 为公共参考;量化方式、编译后大小、性能和精度按平台分别记录.G5100 暂无已回收实测,预留独立列.`—` 表示尚无可引用记录,不代表零或已完成适配.点击模型名称查看完整版本、来源和限制,点击“报告”查看实测依据.平台“编译与性能”单元格依次列出量化方式、DLA 大小、NPU 耗时和 RSS.

| 模型 / 版本 | 场景 / 类别 | 输入 / 参数量 | 原始权重 / FP32 ONNX | 数据集 / 指标 / 参考基线 | G720 量化方式 / 编译与性能 | G720 正式精度与子集数值对比 | G5100 量化方式 / 编译与性能 | G5100 **CosSim**/RMSE/NRMSE/精度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5n-v7.0（640×640） | 感知 / 目标检测 | — | — | — | — | — | — | — |
| YOLOv5m-v7.0（640×640） | 感知 / 目标检测 | — | — | — | — | — | — | — |
| [YOLOv8n（640×640）](perception/object_detection/yolov8n/README.md)<br>YOLOv8-N,Ultralytics `v8.0.111` | 感知 / 目标检测 | RGB `1×3×640×640`<br>3.16 M | 原始: 6.53 MB<br>ONNX: 12.66 MB | COCO val2017，5,000 张图片<br>mAP@0.5:0.95 浮点: 36.63% | **W8A8**<br>DLA: 3.53 MB<br>NPU 平均 6.107 ms，P95 6.172 ms<br>RSS: 499.43 MiB | mAP@0.5:0.95: 35.35% (+1.28 pp)<br>[正式](perception/object_detection/yolov8n/results/summary.json) · [子集](perception/object_detection/yolov8n/results/summary_test.json) | — | — |
| YOLOv8m（640×640） | 感知 / 目标检测 | — | — | — | — | — | — | — |
| RT-DETRv2-S（640×640） | 感知 / 目标检测 | — | — | — | — | — | — | — |
| SqueezeNet-1.1 | 感知 / 分类 | — | — | — | — | — | — | — |
| ResNet50 | 感知 / 分类 | — | — | — | — | — | — | — |
| [ResNet101](perception/image_classification/resnet101/README.md)<br>TorchVision `IMAGENET1K_V2` | 感知 / 分类 | RGB `1×3×224×224`<br>44.55 M | 原始: 178.81 MB<br>ONNX: 178.04 MB | ImageNet val2012，50,000 张图片<br>Top-1 浮点: 81.90% | **W8A8**<br>DLA: 45.38 MB<br>NPU 平均 5.109 ms，P95 5.243 ms<br>RSS: 244.16 MiB | Top-1: 81.54% (+0.35 pp)<br>[正式](perception/image_classification/resnet101/results/summary.json) · [子集](perception/image_classification/resnet101/results/summary_test.json) | — | — |
| MobileNetV3-Large | 感知 / 分类 | — | — | — | — | — | — | — |
| [ViT-B/16](perception/image_classification/vit_base_patch16_224/README.md)<br>ViT-B/16,TorchVision `IMAGENET1K_V1` | 感知 / 分类 | RGB `1×3×224×224`<br>86.57 M | 原始: 346.33 MB<br>ONNX: 346.44 MB | ImageNet val2012，50,000 张图片<br>Top-1 浮点: 80.66% | **W8A8**<br>DLA: 88.00 MB<br>NPU 平均 51.898 ms，P95 51.997 ms<br>RSS: 225.59 MiB | Top-1: 79.40% (+1.27 pp)<br>[正式](perception/image_classification/vit_base_patch16_224/results/summary.json) · [子集](perception/image_classification/vit_base_patch16_224/results/summary_test.json) | — | — |
| MediaPipe Gesture Recognizer（Palm+Hand Landmark+Gesture Classifier） | 交互 / 手势识别 | — | — | — | — | — | — | — |
| DepthAI Hand Tracker（PalmDetection+HandLandmark 21点） | 交互 / 手部关键点 | — | — | — | — | — | — | — |
| [MobileFaceNet](interaction/face_recognition/mobilefacenet/README.md)<br>foamliu MobileFaceNet,`v1.0` 权重,128 维特征 | 交互 / 人脸识别 | RGB 对齐人脸 `1×3×112×112`<br>1.00 M | 原始: 4.14 MB<br>ONNX: 4.00 MB | LFW 十折，6,000 对人脸<br>验证准确率 浮点: 99.38% | **W8A8**<br>DLA: 1.41 MB<br>NPU 平均 0.631 ms，P95 0.712 ms<br>RSS: 67.55 MiB | 验证准确率: 99.32% (+0.07 pp)<br>[正式](interaction/face_recognition/mobilefacenet/results/summary.json) · [子集](interaction/face_recognition/mobilefacenet/results/summary_test.json) | — | — |
| MoveNet SinglePose Lightning（192×192） | 交互 / 姿态估计 | — | — | — | — | — | — | — |
| [RTMPose-m WholeBody（256×192，133点）](interaction/pose_detection/rtmpose_body2d/README.md)<br>RTMPose-M WholeBody 133 点,MMPose `v1.3.2` | 交互 / 姿态估计 | RGB 人体裁剪 `1×3×256×192`<br>17.95 M | 原始: 72.01 MB<br>ONNX: 71.83 MB | COCO-WholeBody val2017，104,125 个人体框<br>WholeBody AP 浮点: 57.04% | **W8A8**<br>DLA: 19.14 MB<br>NPU 平均 3.892 ms，P95 4.051 ms<br>RSS: 163.08 MiB | WholeBody AP: 53.22% (+3.82 pp)<br>[正式](interaction/pose_detection/rtmpose_body2d/results/summary.json) · [子集](interaction/pose_detection/rtmpose_body2d/results/summary_test.json) | — | — |
| HRNet-W32（COCO 17点） | 交互 / 姿态估计 | — | — | — | — | — | — | — |
| BlazePose GHUM Lite（33点） | 交互 / 姿态估计 | — | — | — | — | — | — | — |
| PROCR v4 | 交互 / 文本识别 | — | — | — | — | — | — | — |
| Fast-SCNN（Cityscapes 19类） | 导航 / 语义分割 | — | — | — | — | — | — | — |
| DeepLabV3-MobileNetV3-Large（VOC 21类） | 导航 / 语义分割 | — | — | — | — | — | — | — |
| segformer-b0-finetuned-ade-512-512 | 导航 / 语义分割 | — | — | — | — | — | — | — |
| [FastSAM-s](navigation/segmentation/fastsam/README.md)<br>官方 `FastSAM-s.pt`,Ultralytics `v8.0.111` | 导航 / 语义分割 | RGB `1×3×640×640`<br>11.79 M | 原始: 23.83 MB<br>ONNX: 47.18 MB | COCO val2017，5,000 张图片<br>segm AR@100 浮点: 39.01% | **W8A8**<br>DLA: 12.29 MB<br>NPU 平均 14.537 ms，P95 14.592 ms<br>RSS: 177.04 MiB | segm AR@100: 37.57% (+1.44 pp)<br>[正式](navigation/segmentation/fastsam/results/summary.json) · [子集](navigation/segmentation/fastsam/results/summary_test.json) | — | — |
| FastSAM-x | 导航 / 语义分割 | — | — | — | — | — | — | — |
| EdgeSAM（Image Encoder+Prompt Decoder） | 导航 / 语义分割 | — | — | — | — | — | — | — |
| [Depth Anything V2-Small（ViT-S）](navigation/single_camera_depth/depth_anything_v2_small/README.md)<br>Depth Anything V2 Small,ViT-S/14 | 导航 / 单目深度 | RGB `1×3×518×518`<br>24.79 M | 原始: 99.22 MB<br>ONNX: 98.97 MB | DA-2K，2,068 个标注点对<br>点对准确率 浮点: 94.83% | **W8A8**<br>DLA: 25.83 MB<br>NPU 平均 133.478 ms，P95 133.937 ms<br>RSS: 243.45 MiB | 点对准确率: 85.78% (+9.04 pp)<br>[正式](navigation/single_camera_depth/depth_anything_v2_small/results/summary.json) · [子集](navigation/single_camera_depth/depth_anything_v2_small/results/summary_test.json) | — | — |
| StereoNetPlus-v1.2.1（MixVarGENet + FPN，SceneFlow） | 导航 / 双目深度 | — | — | — | — | — | — | — |
| DINOv2 ViT-S/14（dinov2_vits14） | 导航 / 视觉基础模型 | — | — | — | — | — | — | — |
| CLIP ViT-B/16 | 生成式AI / 视觉语言 | — | — | — | — | — | — | — |
| [MiniCPM5-2B](gen_ai/llm/minicpm5_2b/README.md) | 生成式AI / 大语言模型 | Token + KV Cache<br>上下文 1024 Token<br>Prefill 每批 128 Token<br>逐 Token 解码<br>约 2520 M | 原始: 5033.56 MB | MMLU 固定千题、完整 WikiText2 test，1,000 题<br>MMLU Accuracy 浮点: 55.90%<br>PPL 浮点: 37.63 | **W4A16 (INT4/INT16)**<br>DLA: 2292.52 MB<br>embedding: 534.77 MB<br>部署合计: 2827.29 MB<br>Prefill 284.55 Token/s，Decode 6.65 Token/s<br>RSS: 1254.62 MiB | MMLU Accuracy: 41.30% (+14.60 pp)<br>PPL: 45.10 (+7.48)<br>[正式](gen_ai/llm/minicpm5_2b/results/summary.json) · [子集](gen_ai/llm/minicpm5_2b/results/summary_test.json) | — | — |
| [DeepSeek-R1-Distill-Qwen-1.5B（W4A16）](gen_ai/llm/deepseek_r1_distill_qwen_1_5b/README.md) | 生成式AI / 大语言模型 | Token + KV Cache<br>上下文 1024 Token<br>Prefill 每批 128 Token<br>逐 Token 解码<br>1777.09 M | 原始: 3554.21 MB | MMLU 固定千题、完整 WikiText2 test，1,000 题<br>MMLU Accuracy 浮点: 37.20%<br>PPL 浮点: 108.96 | **W4A16 (INT4/INT16)**<br>DLA: 1579.20 MB<br>embedding: 466.75 MB<br>部署合计: 2045.94 MB<br>Prefill 373.33 Token/s，Decode 8.93 Token/s<br>RSS: 903.81 MiB | MMLU Accuracy: 34.50% (+2.70 pp)<br>PPL: 134.75 (+25.79)<br>[正式](gen_ai/llm/deepseek_r1_distill_qwen_1_5b/results/summary.json) · [子集](gen_ai/llm/deepseek_r1_distill_qwen_1_5b/results/summary_test.json) | — | — |
| DeepSeek-R1-Distill-Qwen-7B | 生成式AI / 大语言模型 | — | — | — | — | — | — | — |
| InternVL3-1B | 生成式AI / VLM | — | — | — | — | — | — | — |
| InternVL3-8B | 生成式AI / VLM | — | — | — | — | — | — | — |
| Gemma 4 E2B | 生成式AI / VLM | — | — | — | — | — | — | — |
| Gemma 4 E4B | 生成式AI / VLM | — | — | — | — | — | — | — |
| [Whisper-Tiny](audio/stt/whisper_tiny/README.md) | 音频 / 语音识别 | 16 kHz 单声道 | 官方 `tiny.pt`<br>Encoder / Decoder FP32 ONNX | LibriSpeech test-clean，2,620 条音频<br>WER 浮点: 7.56% | **W8A16**<br>双 DLA: 61.21 MB<br>NPU 平均 311.436 ms，P95 556.279 ms<br>RSS: 87.29 MiB | WER: 11.20% (+3.64 pp)<br>[正式](audio/stt/whisper_tiny/results/summary.json) · [子集](audio/stt/whisper_tiny/results/summary_test.json) | — | — |
| MeloTTS | 音频 / 语音合成 | — | — | — | — | — | — | — |
| PiperTTS | 音频 / 语音合成 | — | — | — | — | — | — | — |
| WebRTC AEC3 | 音频 / 环境音频 | — | — | — | — | — | — | — |
| DeepFilterNet v3 | 音频 / 环境音频 | — | — | — | — | — | — | — |
| [YAMNet（TF Hub，521类）](audio/ambient_sound/yamnet/README.md)<br>Google AudioSet 521 类 | 音频 / 环境音频 | Log-Mel `1×1×96×64`,单声道音频<br>3.73 M | 原始: 15.30 MB<br>ONNX: 14.94 MB | ESC-50 第 2–5 折，47 类，1,504 条有效音频<br>macro AP 浮点: 74.15% | **W8A16**<br>DLA: 3.99 MB<br>NPU 平均 0.601 ms，P95 0.710 ms<br>RSS: 50.73 MiB | macro AP: 73.90% (+0.25 pp)<br>[正式](audio/ambient_sound/yamnet/results/summary.json) · [子集](audio/ambient_sound/yamnet/results/summary_test.json) | — | — |
| YOLOv8n + OSNet_x0_25 + BoT-SORT（BoxMOT） | SLAM / 目标跟踪 | — | — | — | — | — | — | — |
| MobileNetVLAD（mobilenetvlad_depth-0.35） | SLAM / 定位 | — | — | — | — | — | — | — |
| HF-Net | SLAM / 定位 | — | — | — | — | — | — | — |
| SuperPoint（MagicLeap pretrained） | SLAM / 定位 | — | — | — | — | — | — | — |
| LightGlue（SuperPoint，固定1024关键点） | SLAM / 定位 | — | — | — | — | — | — | — |
| PointPillar-Tiny | 点云 / 点云 | — | — | — | — | — | — | — |
| CenterPoint-Pillar 0.2（双 PFN,10 sweeps）<br>[tianweiy/CenterPoint](https://github.com/tianweiy/CenterPoint) | 点云 / 点云 | 每点 5 维,10 sweeps<br>参数量待统计 | — | nuScenes,10 类<br>mAP / NDS | — | — | — | — |
| [MIT BEVFusion](pointcloud/three_d_fusion/mit_bevfusion/README.md)<br>camera+lidar,swint_v0p075 | 点云 / 3D融合 | 六相机 256×704<br>5维 LiDAR,最多10帧 | 原始: 163.72 MB<br>四个 FP32 子图 | nuScenes v1.0-mini / mini_val，2 个场景，81 帧<br>mAP 浮点: 58.44%<br>NDS 浮点: 58.33% | **CPU FP32 + NPU FP16**<br>三 DLA: 13.43 MB<br>评测模型资源: 155.92 MB<br>NPU 平均 218.924 ms<br>RSS: 959.89 MiB | mAP: 58.41% (+0.03 pp)<br>NDS: 58.32% (+0.01 pp)<br>[正式](pointcloud/three_d_fusion/mit_bevfusion/results/summary.json) · [子集](pointcloud/three_d_fusion/mit_bevfusion/results/summary_test.json) | — | — |
| [YOLOv5s-v7.0](perception/object_detection/yolov5s/README.md)<br>YOLOv5-S,官方 `v7.0` 权重 | 感知 / 目标检测 | RGB `1×3×640×640`<br>7.24 M | 原始: 14.81 MB<br>ONNX: 28.94 MB | COCO val2017，5,000 张图片<br>mAP@0.5:0.95 浮点: 37.09% | **W8A8**<br>DLA: 7.66 MB<br>NPU 平均 9.723 ms，P95 9.794 ms<br>RSS: 583.54 MiB | mAP@0.5:0.95: 35.88% (+1.21 pp)<br>[正式](perception/object_detection/yolov5s/results/summary.json) · [子集](perception/object_detection/yolov5s/results/summary_test.json) | — | — |

### 数据说明

- 输入按 `N×C×H×W` 表示,图像 batch=1;音频采样率为 16 kHz.参数量 `1 M = 100 万`,不含 BatchNorm 运行统计量等缓冲区.
- 文件大小采用十进制 MB,`1 MB = 1,000,000 bytes`;RSS 为 MiB,与文件大小分开记录.原始权重为当前官方/原作者检查点文件实际大小,保留格式,不将其等同于 FP32 网络大小.多分片、双模型按实际必需文件合计;DLA 不含外置 embedding,有外置权重时另列部署合计.
- `WnAm` 表示权重 n 位、激活 m 位.浮点必须同时标明 FP16/FP32,不可与同位宽整数定点量化混淆;W8A8 为 INT8 PTQ,主要权重采用逐输出通道量化,不表示偏置、全部算子或文件 I/O 都是 8 位.本表不统计仅有浮点部署结果的版本.
- 精度对比使用同版本、同协议的本地浮点参考。括号内为报告中的损失，正值表示退化。百分比指标的差值用百分点，PPL 用绝对差值。
- CosSim/RMSE/NRMSE 比较同输入下浮点参考与 NPU 的后处理前输出,NRMSE = RMSE / RMS(参考输出) × 100%.缺少实测时用 `—`,不从任务精度推算;多输出分别记录,不混合不同量纲.计算、覆盖范围及资源说明见[输出误差统计](../docs/output_metrics.md).
- NPU 耗时按一次模型推理统计.图像模型 batch=1;RTMPose 一次输入为一个裁剪好的人体,YAMNet 一次输入为 96 帧 Log-Mel 特征,不等同于整段音频处理耗时.
- LLM 的 Prefill 是输入处理阶段,Decode 是逐 Token 生成阶段.表中速度来自 128 Token 输入、最多生成 128 Token 的测速协议;RSS 取生成、测速和精度评价进程峰值中的最大值。
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

后续方向见 [路线图](../docs/roadmap.md)。进入实现阶段后，按 [开发规范](../docs/development.md) 确认来源、许可和实际任务范围，再选择已有模板或同类模型目录。需要纳入现有模型检查索引时更新 `registry/models.yaml`。

## 使用与资源

各模型的 README 提供配置、编译上传及板端运行方法。公共环境和资源规则见 [开发与测试环境](../docs/environment.md)，板端目录与评价口径见 [G720 板端指南](../docs/genio_720.md)。

实际模型来源见各模型的 `models/source_url.txt` 和模型卡。原始权重、转换产物和数据集通过本地配置定位。
