# Orbbec Async Perception

`async_perception` 是清扫机器人 POC 的异步感知应用。支持 Orbbec 相机、BAG 回放和 ROS2 原始图像输入；YOLOX 执行目标检测，MobileNetV2 对同一画面执行地面分类，
结果发布到 ROS2 `/perception/result`。

## 架构

```text
src/main.cpp
  ├─ src/platform/linux/        INI 配置、控制台/service 生命周期
  ├─ src/platform/linux/ros2/   图像订阅与结果发布
  └─ src/shared/                YOLOX 与 MobileNetV2 异步推理、结果编码
```

Pipeline 固定按 `数据采集 → 感知处理 → 结果整合 → 结果编码上报` 运行。平台层只负责配置读取和生命周期管理。

`src/platform/linux/config/async_perception.ini` 的 `application.mode` 可选 `ros`、`live`、`playback`。默认 `ros` 使用 `/camera/color/image_raw` (`sensor_msgs/msg/Image`, `rgb8`) 和 `/camera/color/camera_info` (`sensor_msgs/msg/CameraInfo`)；`live` 与 `playback` 保留原 Orbbec 路径。详情见 [Part 2 接入说明](docs/async_perception_app_design_doc_yolox_ros.md)。

## 地面分类

每个实际处理并发布的画面分别提交 YOLOX 与 MobileNetV2 任务。MobileNetV2
使用训练链路的中心正方形裁剪、320 → 256 → 224 缩放裁剪及 ImageNet
RGB 归一化；运行设备在 `src/platform/linux/config/mobilenetv2_config.json`
中设为 NPU。类别 ID 0 映射 `industrial_floor`，类别 ID 1
（模型训练时命名为 `store`）映射 `stone_floor`。结果中的
`confidence` 是两个 logits 经 softmax 后获选类别的分数。

`/perception/result` 的 `perception.ground_classification` 为
`{"category_code":"industrial_floor","confidence":0.93}` 或
`{"category_code":"stone_floor","confidence":0.94}`；逐帧分类提交或推理失败时为
`null`，YOLOX 检测、覆盖率和脏污等级仍照常发布。模型加载或 NPU 编译失败时，
应用初始化失败。

构建前须将当前量化模型 `mobilenetv2.xml` 和 `mobilenetv2.bin` 放到
`models/mobilenetv2/`。该目录不提交 Git；构建会将 XML/BIN 复制到交付目录的
`models/mobilenetv2/`。

## 脏污等级试验规则

ROS 结果中的 `coverage_percent` 仍是检测框 IPM 地面投影并集与整幅画面地面投影的面积比，不是融合评分。`dirt_level` 使用同一帧的覆盖率与 YOLOX 后处理后的五类原始检测框数计算：

```text
score = area_weight × coverage_percent
      + weight_liquid × liquid_count + weight_debris × debris_count
      + weight_paper × paper_count + weight_cable × cable_count
      + weight_bottle × bottle_count
```

覆盖率和评分分别做五帧滑动平均；等级根据平滑后的评分与四个阈值比较，等于阈值时归入较高等级。若没有检测框且覆盖率为零，等级为 1；缺少有效 IPM 信息时，覆盖率和等级为 `null`，并输出 `IPM info missing`。

试验权重和等级阈值位于 `src/platform/linux/config/async_perception.ini` 的 `[dirt_assessment]`；YOLOX 置信度阈值位于主工程 `src/platform/linux/config/yolox_config.json` 的 `conf_th`，当前为 `0.5`。相关配置在构建时复制到交付目录，修改配置后需要重启应用。试验数值可调整，不把参考样图视为固定标定标准。

## 构建

```bash
cd async_perception
./build.sh              # 构建
./build.sh Test         # 构建后运行
./build.sh Release      # 构建并打包
./build.sh clean        # 清理构建中间物和输出目录
```

`clean` 不删除 SDK、缓存、发布包、BAG、日志或录制结果。

## 网页推理预览

应用默认在 `0.0.0.0:8080` 启动只读预览服务。运行后访问
`http://<设备IP>:8080/`，可查看左侧原始 RGB、右侧 YOLO 推理叠加图及实时性能指标。
JPEG 编码和网络发送在后台线程执行，慢客户端不会阻塞取流和推理；端口占用时仅禁用
预览，不影响 ROS2 感知主流程。

监听地址、端口、JPEG 质量和最高预览帧率可在
`src/platform/linux/config/async_perception.ini` 的 `[web_preview]` 中配置。

详见 [构建指南](docs/build_guide.md)。

## 目录

```text
async_perception/
  build.sh                       构建入口
  scripts/                       SDK 恢复脚本
  src/shared/                    跨平台 Pipeline
  src/platform/linux/            Linux 适配与配置
  models/yolo/                   YOLO INT8 推理模型
  models/mobilenetv2/            MobileNetV2 量化 XML/BIN（不提交）
  config/                        推理配置（JSON）
  cache/                         SDK 下载与中间物，不提交
  dep/                           SDK 最终产物，不提交
```

## Docker 容器环境

```bash
docker run -it \
  --name intel-perception \
  --hostname intel-perception \
  --network host \
  --privileged \
  --device /dev/dri:/dev/dri \
  --device /dev/accel:/dev/accel \
  -v /dev/bus/usb:/dev/bus/usb \
  -v /path/to/async_perception:/data \
  intel-perception:22.04
```
