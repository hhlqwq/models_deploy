# 开发与测试环境

按任务准备对应环境。写文档、检查配置或开发应用逻辑可以在本地完成。转换、目标端推理和真实外设测试分别使用匹配的环境。

| 环境 | 用途 | 需要确认的内容 |
| --- | --- | --- |
| 本地开发机 | 编辑、Git、静态检查、已有参考后端或离线逻辑开发 | OS、Shell、Python 或 CMake，是否具备当前步骤需要的依赖。Windows 下编写的 Bash 入口应在项目指定的 Linux 环境运行。 |
| 编译主机和容器 | 导出、量化、编译模型或交叉构建应用 | 镜像标识、SDK 和编译器版本、资源挂载、当前用户权限。按 [G720 Docker](../docker/mtk_g720/README.md) 或 [S100 Docker](../docker/rdk_s100/README.md) 准备。 |
| 目标设备 | 加载制品、运行模型和按需评价 | 设备、OS/固件、CPU 架构、Runtime、驱动、依赖、实际后端及可写工作目录。 |
| Demo 测试环境 | 文件回放、实际相机或麦克风、Web 或 ROS 2 联调 | 文件格式和版本、外设连接与权限、ROS 2 发行版及话题，按应用 README 选择。 |

## 配置与数据

- G720 使用 `configs/storage.local.json` 和 `configs/platforms/genio_720.local.yaml`，公开字段见 [存储示例](../configs/storage.example.json) 和 [G720 模板](../configs/platforms/genio_720.yaml)。
- S100 使用 `configs/platforms/s100.local.yaml`，公开字段见 [S100 模板](../configs/platforms/s100.yaml)。模型、数据集、板端目录和工具链字段以该平台文档为准。
- 主机与容器按现有工具链约定使用一致的资源路径。模型资源读写挂载，数据集只读挂载。真实路径、SSH 配置和凭据留在本地。手工临时文件放 `/tmp/hailong/当天日期/`，Windows 用 `D:/tmp/hailong/当天日期/`，无 D 盘用 C 盘。已有脚本的过程目录以入口代码为准。
- 复现实验时记录代码版本、目标和后端、模型版本或制品标识、输入范围、实际命令及结果位置。离线文件运行、目标硬件运行和真实外设测试分别记载。

## G720 现有环境记录

### 配置与版本

[G720 平台配置](../configs/platforms/genio_720.yaml) 只保存公共规格和空的环境字段.复制为同目录的 `genio_720.local.yaml`,填写开发板连接信息和 `toolchain` 工具链路径;加载时本地值覆盖公共配置,环境变量优先级最高.可通过 `MTK_PLATFORM_CONFIG` 指定其他本地平台文件.模型与数据集根目录保存在 `configs/storage.local.json`,两类本地配置均不提交 Git,缺少必需环境字段时入口会明确报错.

- `toolchain.host_sdk_root`: 构建镜像使用的主机 SDK 来源.
- `toolchain.sdk_root`: 容器中的 SDK、NCC 与 Runtime 头文件.

以下版本来自项目已有配置与模型记录,各模型兼容性以对应运行报告为准:

| 工具 | 版本 | 用途 |
| --- | --- | --- |
| NeuroPilot SDK Basic | 8.0.11 build 20260211 | 主机端工具集合 |
| mtk_converter | 8.16.0 cp311 | PyTorch/ONNX 到 TFLite 转换和 PTQ |
| mtk_quantization | 8.2.1 | 量化工具 |
| ncc-tflite | Neuron 8.2.31 | TFLite 到 DLA 编译 |
| neuronrt | 板端 8.2.16 | DLA NPU 推理 |
| ONNX Runtime | 板端 1.20.2 | Neuron/XNNPACK/CPU Execution Provider 在线推理 |

编译器与板端运行时版本必须同时记录,不能只写“NeuroPilot SDK 8”.项目采用 NCC 8.2.31 和板端 Runtime 8.2.16,该组合不能仅凭版本信息推断所有模型兼容.

### 进入编译环境

在项目根目录执行:

```bash
bash docker/mtk_g720/enter_container.sh
```

入口按平台配置准备 SDK、NCC 和板端环境变量.模型转换、DLA 编译与上传在容器内执行,模型精度和性能测试按对应入口在目标开发板执行.完整准备方式见 [Docker 说明](../docker/mtk_g720/README.md).

容器使用 Python 3.11,与现有 SDK 的 cp311 wheel 对齐.普通模型环境使用 Torch 2.0.0+cu118 和 ONNX 1.13.1; 需要其他框架或版本的模型按其说明使用独立环境,避免覆盖共享依赖.

### GPU 与性能口径

GPU 可用于原始框架或 ONNX 浮点基准.实际使用的设备取决于任务及执行提供程序,不能仅凭已安装 CUDA 或 GPU 包判断.数据读取、预处理、后处理和指标计算可能仍在 CPU 上执行.

开发板 NPU 的性能与编译主机性能分别记录.设备可见、版本查询成功或运行时可加载,都不能代替真实模型推理与全量精度结果.

### 板端 Python 推理

Python 通过 `tools/runtime.py` 的 ctypes 接口直接调用 Neuron Runtime,动态库和 APUSYS 驱动必须匹配开发板.默认加载 `libneuronusdk_runtime.mtk.so.8`,不同安装位置可在本地运行环境设置 `MTK_RUNTIME_LIBRARY`.不再编译自有 C++ 程序或桥接库.

板端需要 Python3、NumPy、OpenCV 及各模型的评分依赖.COCO 指标使用 pycocotools,Whisper 使用 ffmpeg 和随包文本组件,两个 LLM 另需 tokenizers.依赖在部署环境中提前准备,运行模型时不自动安装或下载.板端无需安装模型转换 SDK.

### 目录与结果

正式模型和数据集统一保存在 NAS,主机与容器使用相同绝对路径;临时目录仅存放可丢弃的中间文件.模型资源读写挂载,数据集只读挂载; 完整内部路径仅保存在本地配置及运行记录中.

校准集和正式评测集分别记录,不能将校准子集或单张示例的结果描述为全量精度.更换固件、Runtime 或关键依赖后,已有历史结果仍按原环境标识,需重新评测才能更新结论.

## 按需检查

文档修改核对引用和 Git 差异。代码、配置、构建、模型推理及外设测试根据本次任务选择范围。先阅读对应入口，确认是否会上传、覆盖过程目录或启动设备任务。

在已准备依赖的编译容器内执行 Python 回归测试:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

LLM 契约检查依赖 Converter.测试按用户授权执行,代码迁移不代表已完成板端推理验证.
