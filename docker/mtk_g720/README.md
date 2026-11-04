# Docker 编译环境

基于 Ubuntu 22.04,预装 Python 3.11、Torch、NeuroPilot Converter、LLM SDK 和 NCC,用于视觉模型及原生 LLM 的转换与评价.依赖版本由 Dockerfile 和 requirements.txt 固定.

默认自建镜像名称为 `mtk/ai_toolchain_ubuntu_22_g720_gpu:np8.0.11`,标签对应 NeuroPilot SDK 8.0.11,`mtk/` 仅作为本地镜像分类.可通过本地 `configs/storage.local.json` 的 `container_image` 或环境变量 `MTK_G720_IMAGE` 指定镜像名称.给已有镜像增加标签后,现有容器仍可能显示创建时使用的名称,后续重新创建容器时才会使用新名称.

## 配置

平台路径和连接信息放在 `configs/platforms/genio_720.local.yaml`,模型与数据集路径放在 `configs/storage.local.json`,本地配置不提交 Git.
基础 SDK 和 LLM SDK 使用已准备的本地资源;LLM SDK wheel 目录由本地平台配置的 `toolchain.gai_sdk_root` 指定.模型和数据集不打入镜像.

## 创建与使用

先显式构建完整镜像,并用全新容器检查依赖与编译器可用性.基础 SDK 与 mtk-llm-sdk 2.7.5 均在镜像构建阶段安装:

```bash
bash docker/mtk_g720/build_image.sh
```

镜像就绪后,在仓库根目录创建容器.镜像缺失或 LLM 版本标签不符时入口直接报错,不会自动构建、拉取镜像或补依赖:

```bash
bash docker/mtk_g720/create_container.sh
```

进入已经运行的容器:

```bash
bash docker/mtk_g720/enter_container.sh
```

进入后直接使用 `python`、编译器和其他工具,执行模型入口或独立步骤.依赖在镜像构建时安装,项目路径和平台环境在创建容器时设置;进入脚本只初始化普通账户并打开终端,不安装依赖或切换解释器.

创建和进入入口必须由宿主机普通用户执行,不要使用 `sudo bash`.新容器默认使用该用户的 UID/GID,已有容器通过 `docker exec --user UID` 使用同一身份,主组从账户读取,无需重启.只指定 UID 才能保留账户的附加组,不要在进入命令中附加 `:GID`.共享的 `docker/mtk_g720/user.sh` 为该身份配置 `/home/` 下的 HOME、可写缓存和必要附加组,新生成的本地挂载文件归宿主机用户所有.必要的账户和系统配置步骤显式使用 root,模型任务使用普通用户.

首次配置普通账户时,将调用者已有的 `~/.ssh` 复制到容器普通用户 HOME,设置为该用户可读的私有目录,供现有上传步骤使用.不会覆盖已有容器用户的 SSH 配置,密钥不进入代码仓库.后续 SSH 配置变更需单独同步.共享临时根目录 `/tmp/hailongcodex` 和当天日期目录使用 sticky bit,不递归修改历史任务数据.

已经打开的 root 终端和 VS Code 会话不会自动切换身份,请退出旧终端后通过上述入口重新进入.连接现有容器的编辑器也应选择对应普通用户;直接执行 `docker exec` 时需显式指定宿主机 UID,旧容器的默认用户仍为 root.已有 root 属主文件需要核对范围后单独修复,入口不会对仓库、SDK 或 NAS 递归执行 `chown`.

## 维护

项目、模型、数据集和SDK 资源按本地配置挂载.现有容器不会因修改构建文件而自动更新,更换镜像或容器前需保留必要资源并确认维护安排.
构建过程和最终依赖清单保存在镜像内,SDK 镜像按其许可使用,不推送公共镜像仓库.最终 NPU 性能和精度以开发板实测为准.

## 全平台统一验收

遵守 [platform-docker skill](../../.agents/skills/platform-docker/SKILL.md).MTK 镜像包含全部现有模型的导出、转换及评分依赖,包括 nuScenes、MMPose、Whisper 和 YAMNet TensorFlow CPU.模型入口使用默认 python,不依赖用户 HOME 中临时安装的包.板端依赖独立管理,不使用主机镜像替代板端验证.

build_image.sh 仅使用本目录 Dockerfile,先生成候选镜像,在不挂载模型数据的全新普通用户容器中验证全部模型依赖、轻量评分接口及 GPU,通过后才更新正式标签并移除候选标签.失败不会替换正式标签.镜像记录配方指纹和最终包清单,create_container.sh 拒绝过时配方,并检查实际容器的依赖与 GPU.进入入口不安装依赖.

依赖变化同时更新 requirements.txt 和 verify_environment.py.实际包来自对应项目的 PyPI 发布页,PyTorch CUDA wheel 来源为 https://download.pytorch.org/whl/cu118,Python 源码下载地址见 Dockerfile.不使用 docker commit、临时 Dockerfile 或 --no-deps 绕过依赖解析.环境验收不加载模型权重、不执行模型编译或全量测试.
