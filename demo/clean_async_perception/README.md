# 清洁机器人异步感知 Demo

`async_perception` 是运行在 Linux 上的 C++ 感知程序。它对同一帧彩色图像运行 YOLOX 目标检测和 MobileNetV2 地面分类，结合相机内参估算地面覆盖率和脏污等级，并提供网页预览。

| 构建目标 | 默认输入 | 推理后端 | 默认结果输出 |
| --- | --- | --- | --- |
| MTK Genio 720（AArch64） | Orbbec 实时彩色流，640×360 @ 30 fps | Neuron Runtime，MDLA 5.3，INT8 DLA | JSON Lines、网页预览 |
| Intel（Linux x86_64） | ROS 2 图像和相机内参 | OpenVINO AsyncInfer SDK | ROS 2 `/perception/result`、网页预览 |

## 构建和运行

以下命令都从 `demo/clean_async_perception` 目录执行。

### MTK Genio 720

首次构建需要 Bash、CMake、`aarch64-linux-gnu-g++`、与目标板镜像匹配的 `g720-cpp-sdk.tar.gz`，以及 `models/g720/` 下的 `yolox.dla`、`mobilenetv2.dla` 和 `tensors.json`。SDK 包的组成和从 ONNX 生成 DLA 的步骤见 [G720 构建指南](docs/build_guide.md#mtk-genio-720)。

将 SDK 包放到 `dep/g720-cpp-sdk.tar.gz`，首次构建前解压一次：

```bash
mkdir -p dep/linux-arm64
tar -xzf dep/g720-cpp-sdk.tar.gz -C dep/linux-arm64
```

```bash
bash build.sh Release --platform mtk-g720
```

构建脚本从 `dep/linux-arm64/` 读取 Neuron、OpenCV 和 Orbbec 的头文件与动态库。将 `output/linux/mtk-g720/async_perception/` 整个目录复制到板端，插入相机，从运行包目录执行：

```bash
./async_perception
```

默认逐帧向标准输出写入 JSON Lines。G720 的 ROS 2 集成仅发布结果，启用方式见 [G720 构建指南](docs/build_guide.md#mtk-genio-720)。

### Intel Linux x86_64

按 [Intel 构建指南](docs/build_guide.md#1-构建) 准备 ROS 2 Humble、OpenVINO、AsyncInfer SDK 和 XML/BIN 模型，然后构建并启动：

```bash
bash build.sh Release
cd output/linux/amd64/async_perception
./async_perception
```

默认订阅 `/camera/color/image_raw`（`rgb8`）和 `/camera/color/camera_info`，将结果发布到 `/perception/result`。

## 预览和配置

程序启动后，在浏览器打开 `http://<设备IP>:8080/`。左侧显示原始画面，右侧显示检测叠加图，页面同时展示实时性能。

运行配置位于包内的 `config/async_perception.ini`。可直接修改该文件，也可用 `./async_perception --config /path/to/async_perception.ini` 指定配置。

| 配置项 | 用途 |
| --- | --- |
| `[application]` | 选择 `live`（Orbbec 直连）、`playback`（Orbbec BAG）或 `ros`（Intel 的 ROS 2 图像输入） |
| `[live]` / `[playback]` | 选择相机设备或设置 `bag_path` |
| `[web_preview]` | 设置监听地址、端口、JPEG 质量和预览帧率 |
| `[ipm]` / `[dirt_assessment]` | 设置相机安装参数、内参回退值、脏污权重和等级阈值 |

## 感知结果

- `perception.objects` 给出五类杂物的检测框与置信度。类别为 `liquid`、`debris`、`paper`、`cable`、`bottle`。
- `perception.ground_classification` 给出 `industrial_floor` 或 `stone_floor` 及分类置信度。该帧分类失败时值为 `null`。
- `perception.coverage_percent` 是检测框在地面投影的并集面积占比。 `perception.dirt_level` 由覆盖率和五类目标计数加权计算。缺少有效 IPM 参数时两项均为 `null`。

## G720 验证记录

Genio 720 EVK 上用 Orbbec Gemini 2 L 短测了实时取流、双模型 NPU 推理、JSON Lines 和网页预览。连续处理 33 帧，端到端约 17 FPS。该记录对应实时彩色输入和 JSON Lines 输出。G720 的 BAG 回放和 ROS 2 话题发布待实测。

## 工程入口

| 入口 | 内容 |
| --- | --- |
| [构建指南](docs/build_guide.md) | SDK 与模型准备、构建和部署 |
| [Linux 平台层](src/platform/linux/) | 采集、运行配置、平台推理后端及结果输出 |
| [共享感知逻辑](src/shared/) | 推理任务、帧处理及 JSON 结果编码 |
| [YOLOX 与 ROS 2 设计说明](docs/async_perception_app_design_doc_yolox_ros.md) | Intel 侧的图像订阅、推理链路和结果协议 |
