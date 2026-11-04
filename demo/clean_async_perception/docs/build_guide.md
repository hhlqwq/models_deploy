# Async Perception 构建与 SDK 恢复指南

本文件是 `async_perception` 的构建、SDK 恢复、验证与部署指南。目标平台为 **Linux x86_64**（Docker 容器或原生 Ubuntu 22.04）。

## 1. 构建

```bash
cd async_perception
./build.sh              # 构建
./build.sh Test         # 构建后运行
./build.sh Release      # 构建并打包
./build.sh clean        # 清理构建中间物和输出目录
```

`Test` 前确认 `src/platform/linux/config/async_perception.ini` 中的模式和设备配置正确。
`clean` 仅删除 `build/linux/amd64` 和 `output/linux/amd64`，不会删除 SDK、缓存、发布包、BAG、日志或录制结果。

输出目录：`output/linux/amd64/async_perception/`

运行后默认访问 `http://<设备IP>:8080/` 查看原始 RGB、推理叠加图和实时性能指标。
`[web_preview]` 可配置开关、监听 IPv4 地址、端口、JPEG 质量和最高帧率。若端口被占用，
应用会打印警告并继续运行 ROS2 感知主流程。

## 2. 依赖目录

```text
async_perception/
  cache/linux-x86_64/      # SDK 下载包、解压中间物，可删除后由恢复脚本重建
  dep/linux-x86_64/        # SDK 最终产物（orbbec_sdk/ + async_infer_sdk/），由恢复脚本生成
```

构建脚本会检查 `dep/` 是否包含 MobileNetV2 SDK 接口、配置及 XML/BIN；
缺失时自动触发恢复。模型原件须预先放在
`models/mobilenetv2/mobilenetv2.xml` 和 `models/mobilenetv2/mobilenetv2.bin`。
恢复脚本不会替已有 SDK 缓存仓库切换提交；缓存提交与
`origin/intel` 不一致时会报错，由仓库维护者更新。

## 3. SDK 恢复

通常由构建脚本自动触发。也可手动运行：

```bash
./scripts/restore_sdk.sh linux-x86_64
```

### 前提条件

| 项目 | 说明 |
| --- | --- |
| Git | 可访问 `ssh://git@git.realthon.com:2222/aitc/customer/jack/poc/async_infer_lib.git`（`intel` 分支） |
| 网络 | 可访问 GitHub Release API 下载 Orbbec SDK |
| 工具 | `curl`、`tar`、`unzip`、`git`、`rsync`、`cmake`、`g++` |

Orbbec SDK 版本：`v2.8.7`（Linux x86_64），从 [GitHub Releases](https://github.com/orbbec/OrbbecSDK_v2/releases) 获取。

### 人工预下载

网络不可用时，下载 Orbbec 官方 Release 整包（`OrbbecSDK_v2.8.7_*_linux_x86_64.tar.gz`），放到 `cache/linux-x86_64/orbbec.tar.gz`，再重跑恢复脚本。

AsyncInfer 离线接续：在可联网机器 `git clone --branch intel <仓库地址>`，将完整克隆目录（含 `.git/`）放到 `cache/linux-x86_64/async_infer_lib/`，再重跑。

### 恢复后校验

`dep/linux-x86_64/` 应包含：

```text
orbbec_sdk/
  include/libobsensor/hpp/      # 头文件
  lib/libOrbbecSDK.so*          # 运行时库
  lib/extensions/               # depthengine、filters 等
  lib/OrbbecSDKConfig.xml       # 设备配置
async_infer_sdk/
  sdk/include/async_infer.h     # 头文件
  sdk/lib/libasync_infer.so     # 运行时库
  example/files/config/         # 推理配置
  example/files/models/mobilenetv2/mobilenetv2.xml
  example/files/models/mobilenetv2/mobilenetv2.bin
```

当前开发板在 SDK 独立仓库完成提交和推送后，若 `cache/linux-x86_64/async_infer_lib`
已有新提交，可在 `intel-perception:22.04` 环境中运行
`BUILD_TYPE=Release ./build.sh sdk` 重建 SDK，并将
`output/sdk/` 同步到主工程 `dep/linux-x86_64/async_infer_sdk/sdk/`。
当前恢复脚本也会在缺失依赖时重建和同步；已有完整依赖时直接跳过。
SDK 源码始终留在独立仓库的 `cache/` 下，主工程只使用生成的头文件、
动态库和配置。

## 4. Docker 容器

### 启动

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

| 参数 | 用途 |
| --- | --- |
| `--privileged` | 允许访问 GPU/NPU/USB 设备 |
| `--device /dev/dri:/dev/dri` | Intel GPU（OpenVINO） |
| `--device /dev/accel:/dev/accel` | Intel NPU（VPU 推理） |
| `-v /dev/bus/usb:/dev/bus/usb` | USB 摄像头直通 |

### USB 设备权限

```bash
# 查看 USB 总线号
ls /dev/bus/usb/

```

### 容器内构建运行

```bash
source /opt/ros/humble/setup.bash
cd /data/async_perception
./build.sh Test
```

## 5. 部署

将 `output/linux/amd64/async_perception/` 复制到 `/opt/async_perception`，配置放到 `/etc/async_perception/async_perception.ini`。

systemd 服务（`/etc/systemd/system/async-perception.service`）：

```ini
[Unit]
Description=Async perception service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=robot
WorkingDirectory=/opt/async_perception
Environment=LD_LIBRARY_PATH=/opt/async_perception
ExecStart=/opt/async_perception/async_perception --config /etc/async_perception/async_perception.ini
Restart=on-failure
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now async-perception
journalctl -u async-perception -f
```

## 6. 常见故障

| 现象 | 处理 |
| --- | --- |
| Git clone/fetch 失败 | 执行 `git ls-remote` 确认权限；离线时按第 3 节放置缓存 |
| Orbbec 下载失败 | 在可联网机器下载 Release 整包放到 `cache/linux-x86_64/orbbec.tar.gz`，重跑恢复 |
| Docker 中 `No Orbbec device found` | `uid()` 返回 USB 路径而非序列号，代码已修复为同时匹配 uid 和序列号。确保 USB 权限：`chmod 666 /dev/bus/usb/003/*` |
| 链接成功但程序无法启动 | 检查输出目录包含 Orbbec so、extensions、AsyncInfer、模型和 `config/`；设置 `LD_LIBRARY_PATH` |
| NPU 推理报错 | 确认 `/dev/accel/accel0` 存在且有权限 |
