# MTK 官网 model zoo
[来源](https://genio.mediatek.com/doc/iot-aihub/ai_hub/model_zoo/overview.html) 于MTK官方的modelzoo模型部署, 以下仅以G720为例

官网分成三大块(其中TFLite为 TFLite 模型和预编译的深度学习归档文件 (DLA), onnx runtime是onnx运行时模型推理)

### 1. TFLite(LiteRT) - Analytical AI
   专注于三大主要视觉 AI 任务：图像分类、目标检测和识别, 测试脚本[benchmark_dla.sh](01_AnalyticalAI/benchmark_dla.sh)，实测结果 [01_Analytical_result.md](01_AnalyticalAI/01_Analytical_result.md).

| task | name | source | type | inputsize | Infer(ms) | 实测 |
| ------ | ------ | ------ | ------ | ------ | ------ |  ------ |
| 检测 | yolov5s | FAE | Quant8 | 640x640 | 10.10 | todo |
| 检测 | yolov5s | FAE | Float32 | 640x640 | 32.30 | todo |
| 检测 | yolov8s | FAE | Quant8 | 640x640 | 17.00 | todo |
| 检测 | yolov8s | FAE | Float32 | 640x640 | 50.30 | todo |
| 分类 | DenseNet | [densenet121-quant](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/tflite/genio-520-720/densenet121-quant.zip) | Quant8 | 224x224 | 3.83 | 3.87 |
| 分类 | DenseNet | [densenet121-float](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/tflite/genio-520-720/densenet121-float.zip) | Float32 | 224x224 | 7.72 | 7.80 |
| 分类 | EfficientNet | [efficientnet_b0-quant](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/tflite/genio-520-720/efficientnet_b0-quant.zip) | Quant8 | 224x224 | 1.56 | 1.49 |
| 分类 | EfficientNet | [efficientnet_b0-float](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/tflite/genio-520-720/efficientnet_b0-float.zip) | Float32 | 224x224 | 3.15 | 3.25 |
| 分类 | MobileNetV2 | [mobilenet_v2-quant](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/tflite/genio-520-720/mobilenet_v2-quant.zip) | Quant8 | 224x224 | 0.97 | 0.85 |
| 分类 | MobileNetV2 | [mobilenet_v2-float](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/tflite/genio-520-720/mobilenet_v2-float.zip) | Float32 | 224x224 | 1.71 | 1.71 |
| 分类 | MobileNetV3 | [mobilenet_v3-quant](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/tflite/genio-520-720/mobilenet_v3-quant.zip) | Quant8 | 224x224 | 0.74 | 0.63 |
| 分类 | MobileNetV3 | [mobilenet_v3-float](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/tflite/genio-520-720/mobilenet_v3-float.zip) | Float32 | 224x224 | 1.20 | 1.14 |
| 分类 | ResNet | [resnet-quant](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/tflite/genio-520-720/resnet-quant.zip) | Quant8 | 224x224 | 1.48 | 1.39 |
| 分类 | ResNet | [resnet-float](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/tflite/genio-520-720/resnet-float.zip) | Float32 | 224x224 | 3.86 | 3.97 |
| 分类 | SqueezeNet | [squeezenet-quant](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/tflite/genio-520-720/squeezenet-quant.zip) | Quant8 | 224x224 | 1.00 | 0.99 |
| 分类 | SqueezeNet | [squeezenet-float](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/tflite/genio-520-720/squeezenet-float.zip) | Float32 | 224x224 | 3.00 | 2.33 |
| 分类 | VGG | [vgg-quant](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/tflite/genio-520-720/vgg-quant.zip) | Quant8 | 224x224 | 11.54 | 11.22 |
| 分类 | VGG | [vgg-float](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/tflite/genio-520-720/vgg-float.zip) | Float32 | 224x224 | 33.87 | 33.24 |
| 识别 | VGGFace | [vggface-quant](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/tflite/genio-520-720/vggface-quant.zip) | Quant8 | 224x224 | 11.50 | 11.54 |
| 识别 | VGGFace | [vggface-float](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/tflite/genio-520-720/vggface-float.zip) | Float32 | 224x224 | 33.80 | 33.87 |



### 2. TFLite(LiteRT) - Generative AI
   大型语言模型 (LLM)、视觉语言模型 (VLM)、语音(ASR) 和图像生成模型(仅安卓), 以下模型[来源于](https://genio.mediatek.com/doc/iot-aihub/ai_hub/model_zoo/litert_gai/supported_models.html)
   
开发板[LLM工具](https://genio.mediatek.com/doc/iot-aihub/ai_hub/supported_os/yocto/litert_gai/llm_cmdline_tool.html), [VLM工具](https://genio.mediatek.com/doc/iot-aihub/ai_hub/supported_os/yocto/litert_gai/vlm_cmdline_tool.html), 新版本 [发行说明](https://genio.mediatek.com/doc/iot-yocto/latest/sw/yocto/release-notes/iot-yocto-v26.0-release-note.html)：iot-yocto-v26.0-release  v26.0 - 2026 Jul 29  
[刷机指南](https://genio.mediatek.com/doc/iot-yocto/latest/sw/yocto/get-started/flash.html)
#### LLM
| 模型 | source | 性能 token/s | 实测 token/s |
| ------ | ------ | ------ | ------ |
| Qwen3-0.6B | [Qwen3-0.6B](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/gai/genio-360-360p-420-520-720/qwen3-0.6b.zip) | 535.94 / 22.93 | [470 / 18.52](02_Generative\qwen3-0.6b\benchmark_qwen3-0.6b.log) |
| Qwen3-1.7B | [Qwen3-1.7B](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/gai/genio-360-360p-420-520-720/qwen3-1.7b.zip) | 262.73 / 13.84 | [245.56 / 10.34](02_Generative\qwen3-1.7b\benchmark_qwen3-1.7b.log) |
| Qwen3-4B | [Qwen3-4B](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/gai/genio-360-360p-420-520-720/qwen3-4b.zip) | 125.58 / 7.25 | [118.45 / 5.29](02_Generative\qwen3-4b\benchmark_qwen3-4b.log) |
| Qwen3-8B | [Qwen3-8B](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/gai/genio-360-360p-420-520-720/qwen3-8b.zip) | 80.17 / 4.76 | [76.68 / 4.12](02_Generative\qwen3-8b\benchmark_qwen3-8b.log) |
| Qwen2.5-1.5B-Instruct | [Qwen2.5-1.5B-Instruct](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/gai/genio-360-360p-420-520-720/qwen2.5-1.5b.zip) | 337.06 / 19.25 | [321.1 / 16.1](02_Generative\qwen2.5-1.5b\benchmark_qwen2.5-1.5b.log) |
| Qwen2.5-3B-Instruct | [Qwen2.5-3B-Instruct](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/gai/genio-360-360p-420-520-720/qwen2.5-3b.zip) | 163.23 / 10.64 | [145.79 / 7.8](02_Generative\qwen2.5-3b\benchmark_qwen2.5-3b.log) |
| Qwen2.5-7B-Instruct | [Qwen2.5-7B-Instruct](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/gai/genio-360-360p-420-520-720/qwen2.5-7b.zip) | 69.47 / 4.73 | [64.74 / 4.05](02_Generative\qwen2.5-7b\benchmark_qwen2.5-7b.log) |
| gemma3-1B (Text-Only) | [gemma3-1B (Text-Only)](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/gai/genio-360-360p-420-520-720/gemma3-1b.zip) | 583.01 / 26.44 | [failed](02_Generative\gemma3-1b\gemma3_1b_single.log) |
| gemma3-4B (Text-Only) | [gemma3-4B (Text-Only)](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/gai/genio-360-360p-420-520-720/gemma3-4b.zip) | 176.79 / 5.93 | [failed](02_Generative\gemma3-4b\benchmark_gemma3-4b.log) |
| llama3.2-1B-Instruct | [llama3.2-1B-Instruct](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/gai/genio-360-360p-420-520-720/llama3.2-1b.zip) | 400.57 / 24.开发板 | [380.4 / 18.56](02_Generative/llama3.2-1b/benchmark_llama3.2-1b.log) |
| llama3.2-3B-Instruct | [llama3.2-3B-Instruct](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/gai/genio-360-360p-420-520-720/llama3.2-3b.zip) | 153.56 / 10.36 |  [144.58 / 6.58](02_Generative\llama3.2-3b\benchmark_llama3.2-3b.log) |
| Phi-3-mini-4k-instruct | [Phi-3-mini-4k-instruct](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/gai/genio-360-360p-420-520-720/phi-3-mini-4k-instruct.zip) | 127.56 / 7.28 | [127.91 / 6.62](02_Generative\Phi-3-mini-4k-instruct\benchmark_phi-3-mini-4k-instruct.log) |
| Phi-3.5-mini-instruct | [Phi-3.5-mini-instruct](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/gai/genio-360-360p-420-520-720/phi-3.5-mini-instruct.zip) | 136.63 / 6.29 | [138.42 / 6.05](02_Generative\Phi-3.5-mini-instruct\benchmark_phi-3.5-mini-instruct.log) |


#### VLM
| 模型 | source | vit/Prompt/Generative | 实测 |
| ------ | ------ | ------ | ------ |
| Qwen3VL-2B | [Qwen3VL-2B](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/gai/genio-360-360p-420-520-720/qwen3vl-2b.zip) | 0.43s / 199.75 / 14.43 | [0.499s / 322.65 / 13.52](02_Generative\qwen3vl-2b\benchmark_qwen3vl-2b.log) |
| Qwen3VL-4B | - | - | - |

#### ASR
| 模型 | source | Performance | 实测 |
| ------ | ------ | ------ | ------ |
| Whisper-base (8w16a) | [Whisper-base (8w16a)](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/gai/genio-360-360p-420-520-720/whisper-base-8w16a.zip) | 160.6 ms / 174.8 tok/s | 未包含asr_cmdline_tool |
| Qwen3-ASR-0.6B | [Qwen3-ASR-0.6B](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/gai/genio-360-360p-420-520-720/qwen3-asr-0.6b.zip) | 127.6 / 7.1 tok/s | 未包含asr_cmdline_tool |
| Moonshine-Tiny | [Moonshine-Tiny](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/gai/genio-360-360p-420-520-720/moonshine-tiny.zip) | V | 未包含asr_cmdline_tool |


### 3. ONNX Runtime - Analytical AI

##### 3.1. ONNX 批量测试
```bash
cd /opt/mtk/ort
tar -xzf onnx_benchmark_autorun.tar.gz
```
##### 3.2 安装并启动服务
```bash
chmod +x benchmark_onnx_watchdog.sh

cp onnx-benchmark.service /etc/systemd/system/onnx-benchmark.service

systemctl daemon-reload
systemctl enable onnx-benchmark.service
systemctl restart onnx-benchmark.service
```
##### 3.3. 查看运行状态
```bash
systemctl status onnx-benchmark.service
```
##### 3.4. 查看实时日志
```bash
tail -f /opt/mtk/ort/benchmark_onnx_service.log
```
##### 3.5. 查看测试结果
```bash
cat /opt/mtk/ort/benchmark_onnx_summary.csv
```
##### 3.6. 查看异常模型
```bash
cat /opt/mtk/ort/.benchmark_state/kernel_oops_models.txt
```
##### 3.7. 停止自动测试
```bash
systemctl disable --now onnx-benchmark.service
```
说明
- 默认每个模型测试 10 次。
- 普通卡住超过 120 秒会超时跳过。
- 如果出现 Kernel Oops，会记录该模型并自动重启。
- 重启后服务会自动继续测试后续模型

TAO Related Models: TAO相关模型源自NVIDIA TAO工具包或使用基于TAO的预训练权重
| task | name | source | type | inputsize | npu_Infer(ms) | 实测 |
| ------ | ------ | ------ | ------ | ------ | ------ |  ------ |
| 检测 | PeopleNet (ResNet34) | [resnet34_peoplenet](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/resnet34_peoplenet.onnx) | Float32 | 3x544x960 | 77.63 | 78.36 |
| 检测 | PeopleNet (ResNet34) | [resnet34_peoplenet_int8](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/resnet34_peoplenet_int8.onnx) | Quant8 | 3x544x960 | 19.44 | 19.94 |
| 识别 | Action Recognition Net (ResNet18) | [resnet18_2d_rgb_hmdb5_32](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/resnet18_2d_rgb_hmdb5_32.onnx) | Float32 | 96x224x224 | 14.75 | 15.02 |
| 姿态估计 | BodyPoseNet | [bodypose](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/bodypose.onnx) | Float32 | 224x320x3 | 40.56 | 41.54 |
| 检测 | LPDNet (USA Pruned) | [LPDNet_usa_pruned_tao5](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/LPDNet_usa_pruned_tao5.onnx) | Float32 | 3x480x640 | 3.42 | 3.53 |
| 分割 | PeopleSemSegNet_AMR | [peoplesemsegnet_amr](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/peoplesemsegnet_amr.onnx) | Float32 | 3x576x960 | X | todo |
| 分割 | PeopleSemSegNet_AMR (Rel) | [peoplesemsegnet_amr_rel](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/peoplesemsegnet_amr_rel.onnx) | Float32 | 3x544x960 | 13.29 | 16.57 |
| 分割 | PeopleSemSegNet (ShuffleSeg) | [peoplesemsegnet_shuffleseg](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/peoplesemsegnet_shuffleseg.onnx) | Float32 | 3x544x960 | 13.41 | 16.69 |
| 分割 | PeopleSemSegNet (Vanilla Unet) | [peoplesemsegnet_vanilla_unet_dynamic_etlt_int8_fp16](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/peoplesemsegnet_vanilla_unet_dynamic_etlt_int8_fp16.onnx) | Float32 | 3x544x960 | 163.45 | 166.83 |
| 重识别 | ReIdentificationNet (ResNet50) | [resnet50_market1501_aicity15](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/resnet50_market1501_aicity156.onnx) | Float32 | 3x256x128 | 6.87 | 7.09 |
| OCR | Ocrnet_resnet50 | [ocrnet_resnet50](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/ocrnet_resnet50.onnx) | Float32 | 1x32x100 | 18.16 | 27.43 |
| OCR | Ocrnet_resnet50 (Pruned) | [ocrnet_resnet50_pruned](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/ocrnet_resnet50_pruned.onnx) | Float32 | 1x32x100 | 13.94 | 23.39 |
| OCR | ocd_resnet50 | [model_ocdnet_736x1280](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/model_ocdnet_736x1280.onnx) | Float32 | 3x736x1280 | 149.85 | 151.80 |
| OCR | ocd_resnet50 | [model_ocdnet_640x640](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/model_ocdnet_640x640.onnx) | Float32 | 3x640x640 | 68.10 | 69.46 |
| OCR | ocdnet_mixnet | [ocdnet_mixnet_640x640](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/ocdnet_mixnet_640x640.onnx) | Float32 | 3x640x640 | 340.09 | 342.58 |
| 分类 | Pose Classification (ST-GCN) | [st-gcn_3dbp_nvidia](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/st-gcn_3dbp_nvidia.onnx) | Float32 | 3x300x34x1 | 207.00 | 291.96 |
| 姿态估计 | Centerpose (Chair DLA34) | [chair_DLA34](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/chair_DLA34.onnx) | Float32 | 3x512x512 | X | X |
| 姿态估计 | Centerpose (Camera FAN) | [camera_FAN_small](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/camera_FAN_small.onnx) | Float32 | 3x512x512 | X | X |
| 检测 | LPDNet (CCPD Pruned) | [LPDNet_CCPD_pruned_tao5](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/LPDNet_CCPD_pruned_tao5.onnx) | Float32 | 3x1168x720 | 6.73 | 7.14 |
| 姿态估计 | Foundation Pose (Refiner) | [refiner_net](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/refiner_net.onnx) | Float32 | 6x160x160 | 60.97 | 66.98 |
| 姿态估计 | Foundation Pose (Score) | [score_net](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/score_net.onnx) | Float32 | 6x160x160 | 34.63 | 38.94 |
| 姿态估计 | Multi 3D Centerpose | [Multiclass_CenterPose_DLA34](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/tao/Multiclass_CenterPose_DLA34.onnx) | Float32 | 3x512x512 | X | X |


Legacy Analytical Models: 传统分析模型是经典的视觉骨干和网络，仅广泛用于基准测试和参考。模型的准确性没有得到解决。
| task | name | source | type | inputsize | Infer(ms) | 实测 |
| ------ | ------ | ------ | ------ | ------ | ------ |  ------ |
| 检测 | yolov5s | FAE | Quant8 | 640x640 | X | todo |
| 检测 | yolov5s | FAE | Float32 | 640x640 | 32.37 | todo |
| 检测 | yolov8s | FAE | Quant8 | 640x640 | 80.57 | todo |
| 检测 | yolov11s | FAE | Quant8 | 640x640 | 90.99 | todo |
| 分类 | ConvNeXt | [convnext_base_quant](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/analytical/convnext_base_quant.onnx) | Quant8 | 224x224 | X | X |
| 分类 | ConvNeXt | [convnext_base_fp32](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/analytical/convnext_base_fp32.onnx) | Float32 | 224x224 | X | X |
| 分类 | DenseNet | [densenet121_quant](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/analytical/densenet121_quant.onnx) | Quant8 | 224x224 | X | X |
| 分类 | DenseNet | [densenet121_fp32](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/analytical/densenet121_fp32.onnx) | Float32 | 224x224 | 7.49 | 7.67 |
| 分类 | EfficientNet | [efficientnet_b0_quant](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/analytical/efficientnet_b0_quant.onnx) | Quant8 | 224x224 | 30.52 | 48.74 |
| 分类 | EfficientNet | [efficientnet_b0_fp32](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/analytical/efficientnet_b0_fp32.onnx) | Float32 | 224x224 | 2.81 | 2.77 |
| 分类 | MobileNetV2 | [mobilenet_v2_quant](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/analytical/mobilenet_v2_quant.onnx) | Quant8 | 224x224 | 1.26 | 2.46 |
| 分类 | MobileNetV2 | [mobilenet_v2_fp32](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/analytical/mobilenet_v2_fp32.onnx) | Float32 | 224x224 | 1.47 | 1.54 |
| 分类 | MobileNetV3 | [mobilenet_v3_small_quant](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/analytical/mobilenet_v3_small_quant.onnx) | Quant8 | 224x224 | X | X |
| 分类 | MobileNetV3 | [mobilenet_v3_small_fp32](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/analytical/mobilenet_v3_small_fp32.onnx) | Float32 | 224x224 | 12.81 | 22.15 |
| 分类 | ResNet | [resnet18_quant](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/analytical/resnet18_quant.onnx) | Quant8 | 224x224 | 1.78 | 1.82 |
| 分类 | ResNet | [resnet18_fp32](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/analytical/resnet18_fp32.onnx) | Float32 | 224x224 | 3.49 | 3.49 |
| 分类 | SqueezeNet | [squeezenet_quant](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/analytical/squeezenet_quant.onnx) | Quant8 | 224x224 | 8.38 | 13.42 |
| 分类 | SqueezeNet | [squeezenet_fp32](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/analytical/squeezenet_fp32.onnx) | Float32 | 224x224 | 8.81 | 14.51 |
| 分类 | VGG | [vgg16_quant](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/analytical/vgg16_quant.onnx) | Quant8 | 224x224 | 11.62 | 11.94 |
| 分类 | VGG | [vgg16_fp32](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/analytical/vgg16_fp32.onnx) | Float32 | 224x224 | 32.24 | 32.78 |
| 识别 | VGGFace | [vggface_quant](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/analytical/vggface_quant.onnx) | Quant8 | 224x224 | 291.22 | 137.17 |
| 识别 | VGGFace | [vggface_fp32](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/analytical/vggface_fp32.onnx) | Float32 | 224x224 | 32.84 | 33.40 |
| Omni6DPose | scale_policy | [ScalePolicy](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/robotic/ScalePolicy.onnx) | Float32 | 1x3x3 | X | X |
| Diffusion Policy(扩散策略) | model_diffusion_sampling | [model_diffusion_sampling](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/robotic/model_diffusion_sampling.onnx) | Float32 | trajectory:1x16x12, global_cond:1x800 | 57.68 | 103.96 |
| MobileSam(SAM轻量版) | mobilesam_encoder | [transformed_mobilesam_encoder_tiny](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/robotic/transformed_mobilesam_encoder_tiny.onnx) | Float32 | 3x448x448 | X | X |
| RegionNormalizedGrasp(抓取检测) | anchornet | [anchornet](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/robotic/anchornet.onnx) | Float32 | 4x640x360 | 12.13 | 19.34 |
| RegionNormalizedGrasp | localnet | [localnet](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/robotic/localnet.onnx) | Float32 | 64x64x6 | 19.78 | 27.53 |
| YoloWorld | yoloworld_xl | [yoloworld_xl](https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/robotic/yoloworld_xl.onnx) | Float32 | 3x640x640 | 403.15 | 454.43 |
