# Models Deploy

个人的多平台模型适配、实验和 Demo 开发总仓库。模型源码与权重来自作者或官方项目。模型适配和应用开发可以独立推进，具体项目的交付由各项目另行管理。

## 从哪里开始

| 想做的事 | 入口 |
| --- | --- |
| 找现有模型、运行目标和结果 | [MTK 模型](mtk_models/README.md) · [RDK 模型](rdk_models/README.md) |
| 找应用 Demo | [Demo 总览](demo/) |
| 配置开发机、编译容器和目标设备 | [开发与测试环境](docs/environment.md) |
| 查看仓库结构和多平台设计 | [项目架构](docs/architecture.md) |
| 找开发规范、示例和学习资料 | [文档导航](docs/README.md) |

## 当前平台入口

| 目标 | 模型目录 | 平台配置 | 环境 |
| --- | --- | --- | --- |
| MTK Genio 720 | [mtk_models](mtk_models/README.md) | [genio_720.yaml](configs/platforms/genio_720.yaml) | [G720 板端](docs/genio_720.md) · [G720 Docker](docker/mtk_g720/README.md) |
| RDK S100 | [rdk_models](rdk_models/README.md) | [s100.yaml](configs/platforms/s100.yaml) | [S100 Docker](docker/rdk_s100/README.md) |
| MTK Genio Pro 5100 | 待接入 | [genio_5100.yaml](configs/platforms/genio_5100.yaml) 占位 | 待核实实际设备与工具链 |

从对应平台的 Docker 文档进入已有环境，再看模型 README 中的命令。现有部分 `deploy/compile.sh` 会同时准备子集参考并上传到目标设备，运行前先核对本地配置和入口行为。`deploy/evaluate.sh` 用于按需评价已有制品。模型、数据集和 SDK 通过本地配置定位，内部路径与连接信息留在 Git 忽略的本地文件中。

模型适配使用 [model-deploy](.agents/skills/model-deploy/SKILL.md)，编译容器维护使用 [platform-docker](.agents/skills/platform-docker/SKILL.md)。
