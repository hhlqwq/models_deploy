# S100 编译环境

使用官方 S100/S600 OpenExplorer 镜像,编译目标固定为 S100 的 `nash-e`.
镜像名称、固定版本、容器名称及 NAS 路径填写在 `configs/platforms/s100.local.yaml`.
toolchain.root 指向 NAS 的 S100/tools,统一归档 SDK、离线安装包及依赖资源.每项资源记录具体下载地址、固定版本、文件名和用途;不将二进制依赖放入仓库.
宿主机需要 Python 3 与 PyYAML.镜像应提前准备,入口不拉取镜像或安装依赖.

在仓库根目录执行:

```bash
bash docker/rdk_s100/create_container.sh
bash docker/rdk_s100/enter_container.sh
```

创建入口只创建独立新容器,同名容器存在时失败,不会修改、重启已有容器.
创建和进入入口会在目标容器内补齐当前宿主机 UID/GID 对应的普通用户、组和附加组,并设置独立的 HOME 与缓存目录.请使用宿主机普通用户执行,不要使用 `sudo bash`.
用户初始化沿用 MTK 入口的方式: 准备公共日期临时目录,首次复制宿主机 SSH 配置到容器用户目录且不覆盖已有配置,进入时使用登录 Shell.凭据仅保存在容器内,不进入仓库.
已有容器出现 `I have no name!` 或找不到组名时,退出容器后重新执行进入入口即可补齐账户,无需重建或重启容器.新建账户显示为 `rdk_<UID>`,文件权限仍按宿主机 UID/GID 对应.
模型资源读写挂载,数据集只读挂载,容器内外路径一致.
NAS 工具目录按同一路径只读挂载,依赖从归档安装到容器或匹配的板端环境,不把 x86 主机库复制到 AArch64 板端.
模型入口不重复设置公共环境.S100 模型适配状态见 `rdk_models/README.md`.

工具链来源: https://developer.d-robotics.cc/rdk_doc_center/
具体镜像版本必须与板端 Runtime 核对,镜像存在不代表已经验证兼容.

板端资源路径在本地配置中分别填写 `board.models_root` 和 `board.datasets_root`,用于模型部署包与跨模型复用的公共数据集.原 `board.work_dir` 不再使用.
