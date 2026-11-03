# Models Deploy

面向 MediaTek Genio 720 和 Genio 5100 的开源模型适配仓库,包含模型转换、量化、NPU 部署及精度和性能评价.模型必须来自作者或官方开源项目;Qualcomm AI Hub Models 仅作为目录和交付方式参考.

- [MTK 模型部署 skill](.agents/skills/mtk-model-deploy/SKILL.md): 复用本仓库适配与独立评价流程.
- [模型总览](mtk_models/README.md): 型号、输入、量化方式和已归档指标.
- [编译环境](docs/environment.md)与 [Docker](docker/mtk/README.md): 工具链、依赖和容器配置.
- [Genio 720 板端规范](docs/genio_720.md): 部署布局、运行要求和评价口径.
- [YOLOv5s 学习文档](docs_study/Genio720_MT8189部署YOLOv5s总览.md): 分阶段理解完整部署链路.

## 快速开始

在 `configs/storage.local.json` 配置 NAS 的 `models_root` 和 `dataset_root`,格式见 [配置示例](configs/storage.example.json).将 [G720 平台配置](configs/platforms/genio_720.yaml) 复制为同目录的 `genio_720.local.yaml`,在本地副本中填写开发板地址、容器名称和工具链路径.两份本地配置均不提交 Git;环境变量可覆盖本地平台配置,也可用 `MTK_PLATFORM_CONFIG` 指定本地平台文件.

在 Ubuntu 编译主机的仓库根目录进入已有容器:

```bash
bash docker/mtk/enter_container.sh
```

在容器的仓库根目录执行所选模型入口,例如:

```bash
bash mtk_models/perception/object_detection/yolov5s/deploy/compile.sh
```

需要顺序处理全部 13 个已有模型时,在容器中执行:

```bash
bash mtk_models/deploy/compile.sh
```

总入口顺序调用全部模型已有入口.四个阶段依次执行:

1. 编译服务器: `bash mtk_models/deploy/compile.sh`,完成各模型导出、量化、编译、部署和固定子集浮点精度评估,自动上传必要参考.
2. 板端部署根目录: `bash deploy/compile_board.sh`,完成相同子集 NPU 精度、精度损失、CosSim/RMSE/NRMSE 和示例,保存并回传 `results/summary_test.json` 与 `examples/output/`.
3. 编译服务器: `bash mtk_models/deploy/evaluate.sh`,使用已有模型完成约定全量浮点评估,自动上传结果及必要评价输入,不重新量化或编译.
4. 板端部署根目录: `bash deploy/evaluate_board.sh`,完成相同全量范围的 NPU 精度、精度损失、性能和峰值 RSS,保存 `results/summary.json`,最后自动回传两份已有报告及示例.

四个总入口将标准输出和错误输出同时显示在终端并保存在本地日志.编译服务器保存为 `mtk_models/deploy/compile.log`、`mtk_models/deploy/evaluate.log`;板端保存为部署根目录下的 `deploy/compile.log`、`deploy/evaluate.log`.每次运行覆盖对应日志,日志开头标明日志文件.板端 evaluate 末尾的结果回传也保存在 evaluate.log,不会覆盖 compile.log.日志不存 NAS、不纳入 Git、不自动回传;流程失败保留原退出码,日志写入失败也返回失败.

两端均不混用子集和全量结果.普通模型数值对比最多选择 5 次输入;LLM 对比示例首个 Prefill 的有效输出,Whisper 对比 Encoder,BEVFusion 对比单帧各 NPU 子图.数值误差写入 summary_test.json,全量评价不采集输出张量或重复执行数值比较.

板端普通 DLA 放模型根目录,多文件运行包放 models/,代码放 board/,浮点参考及必要输入放 build/reference/,当前运行数据放 build/work/ 并覆盖.配置中的 deployment.model_directory 明确声明模型位置.原始资源只在运行确有需要时上传,不重复上传训练权重或主机编译缓存.结果目录只保留 summary_test.json 和 summary.json,没有额外 Markdown/CSV 报告.

结果回传只在 compile.sh 末尾维护,evaluate.sh 全部完成后调用其仅回传分支,不会再次编译或推理.不再提供 upload.sh.连接信息由本地平台配置 results_host、results_root 提供,生成板端 deploy/platform.local.sh,不提交 Git.专用 SSH 密钥及 known_hosts 仍由本地环境管理.回传只同步报告和示例,不传模型和中间文件,不自动提交 Git.

评分、参考及模型版本不匹配时明确失败,不生成成功报告.主机上传不会自动执行板端推理;上传成功只能标记待板端检查.任务范围及接口说明见[输出误差与精度](docs/output_metrics.md).

新建编译环境时使用 `docker/mtk/build_image.sh` 和 `docker/mtk/create_container.sh`,具体资源要求见 Docker 文档.运行模型时不自动下载权重、数据集或安装依赖.

## 存储规则

仓库保存代码、配置模板、来源记录、文档和精简结果.所有正式资源位于 NAS:

| 资源 | 位置 |
| --- | --- |
| 原始权重、离线源码、下载包、许可证和来源记录 | 对应模型资源目录的 `original/` |
| 当前导出 ONNX、最终 DLA | 对应模型资源目录最外层 |
| TFLite、校准数据和转换编译中间产物 | 对应模型资源目录的 `build/` |
| 多文件模型的完整部署包 | 对应模型资源目录的 `models/` |
| 主机评价、数值对照、缓存与临时文件 | 主机本地日期临时目录,不写 NAS |
| 校准与评价数据集 | `dataset_root/` |
| 一次性临时文件 | `/tmp/hailongcodex/当天日期/`,Windows 使用 `D:/tmp/hailongcodex/当天日期/` |

主机与容器使用相同绝对路径,模型资源读写挂载、数据集只读挂载.正式模型和数据集不得以临时目录为唯一存放位置,也不得依赖指向临时目录的链接.来源记录应包含官方地址、固定版本和文件信息,内部完整路径仅保存在本地配置及运行记录中.

板端使用独立部署包,无需访问源码仓库.所有模型的过程文件统一保存在各自部署目录的 `build/work/`,compile 和 evaluate 开始推理时覆盖上次中间结果,后续评分和展示复用同一目录.板端不按日期或运行编号建立目录,也不写入 /tmp 内存盘;运行编号仅作为报告内部记录.子集与全量报告分别保存在 `results/summary_test.json`、`results/summary.json`,示例保存在 `examples/output/`.编译服务器的一次性诊断仍使用日期临时目录.

## 仓库结构

```text
configs/             平台、工具链及本地存储配置模板
mtk_models/          按场景和任务组织的 MTK 模型适配
tools/               公共配置、转换、Python Runtime、评价及文件工具
docker/              编译镜像与容器入口
registry/            模型范围、来源索引与状态
reports/             平台报告入口
docs/                环境及开发板使用规范
docs_study/          分阶段学习材料
templates/model/     新模型交付模板
tests/               公共代码回归测试
```

模型入口只编排独立步骤:主机实现在 `deploy/host/`,板端代码在 `deploy/board/`.公共配置和工具集中维护,不在模型脚本重复设置 SDK 或 Docker 环境.板端推理、前后处理与评价统一使用 Python,直接调用 MTK SDK 动态库,不构建自有 C++ 推理程序.

## 模型与结果

[模型总览](mtk_models/README.md)是指标的统一索引.各模型 README 说明使用方式、当前结果和必要的协议限制;`results/summary.json` 保存机器可读汇总.不在首页重复模型指标、实验经过或逐次运行日志.

精度比较须使用同一模型、输入和评价协议.全量、固定子集和功能示例分别标识,运行成功不等于精度合格.性能注明纯 NPU 或端到端范围,RSS 不等同于全部驱动内存.更换权重、量化配方或设备环境后须重新评价才能更新结论.

原有 11 个已适配模型的 Python 推理代码已迁移,尚未进行板端验证.这些模型文档中的历史精度、性能、RSS 和示例不能作为新实现的验证结果.[ResNet101](mtk_models/perception/image_classification/resnet101/README.md)已完成 G720 Python NPU 全量精度、性能与 RSS 评价,同协议 ONNX 参考已核对,当前结果见模型总览和报告.
