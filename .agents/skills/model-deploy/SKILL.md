---
name: model-deploy
description: 在 models_deploy 仓库适配、量化、编译和部署官方开源模型到 MTK Genio 720/G5100 或地瓜 RDK S100,通过 compile.sh 和 evaluate.sh 交付子集精度、独立精度性能及 RSS 双报告.用于新增模型、部署链路修复与精度差异排查;Docker 环境变更配合 platform-docker skill.
---

# 通用模型适配与部署

共用交付和评价规范,各平台独立维护网络适配、量化、编译与 Runtime.所有模型交付明确张量契约、两个主要入口、必要示例及双报告.优先复用现有工具,不为单个模型另造环境管理或流水线框架.仓库规范由 skills 维护,不要求创建或恢复根 AGENTS.md.

## 平台选择

| 平台 | 模型与 Docker | 本地配置与公共工具 | 目标边界 |
| --- | --- | --- | --- |
| MTK Genio 720 | mtk_models/、docker/mtk_g720/ | genio_720.local.yaml、tools/storage.py、tools/deploy.py、tools/board.py | MT8189 / MDLA 5.3、NCC、DLA、Neuron Runtime |
| MTK G5100 | mtk_models/、已配置的独立目标环境 | genio_5100.local.yaml | 先核实实际硬件及工具链,不沿用 G720 指标或认定已适配 |
| 地瓜 RDK S100 | rdk_models/、docker/rdk_s100/ | s100.local.yaml、tools/rdk_config.py、tools/rdk_deploy.py | nash-e、匹配的 OpenExplorer/LLM SDK、HBM 与板端 Runtime |

平台配置均在 configs/platforms/.SDK、依赖、编译产物和实测结果不得跨平台混用.S100P / nash-m 与 S600 / nash-p 的例程只能参考,不等于 S100 支持.旧 MTK 文档的顶层 models/ 按 mtk_models/ 理解,模型内部资源包 models/ 不改名.

## 开始工作

1. 检查 git status --short,保留用户改动.先只读定位平台配置、本地覆盖、模型 README、config.yaml、模型卡、来源、deploy/ 和 results/.新增模型先查 registry/target_models.yaml 与 registry/models.yaml.
2. 修改前确认计划,已有明确批准不重复询问.区分代码修改、环境准备、模型编译、板端测试及评价范围.未要求的测试、大文件下载或耗时任务先确认;工作区外写入、升级、重启和停止其他进程按用户授权执行.
3. 部署时读取 [deployment.md](references/deployment.md);评价时读取 [evaluation.md](references/evaluation.md);核对外部来源时读取 [sources.md](references/sources.md).只读任务所需参考.涉及镜像或依赖变更时读取 [platform-docker skill](../platform-docker/SKILL.md).

## 共用约束

- 原始源码与权重来自作者或官方项目.Qualcomm 仅作 delivery_reference,不能作为实际移植源.记录非代码资源的具体下载 URL、版本、文件名及用途;来源未核实保持 pending,不在文档堆叠哈希.
- 使用现成原始权重或用户导出的完整微调权重,不新增业务训练/微调流程.LoRA 等 adapter 先与对应基座合并并核对配置.官方量化配方必需的阈值优化与业务微调分开说明.校准与正式评价划分独立,核对 SDK 默认的数据读取逻辑,不得用测试集泄漏改善精度.
- 优先采用目标平台官方 SDK 已有网络适配和量化规则,只补必要的资源、目标及入口配置.区分权重原始 dtype、前置量化规则、硬件校准与最终存储/计算类型,不把 W4 配方等同原生 INT4,不把 MTK W4A16 强加给 S100.按用户确认的顺序选择精度,不得擅自提高位宽掩盖错误.
- 推理、前后处理、采样、精度评价和性能统计统一使用 Python,通过官方 Python 接口或可直接调用的 SDK 动态库执行 NPU/BPU 推理.不新增自有 C++ 推理程序,不以 Python 启动自有 C++ CLI 替代迁移.只有 C++ 类接口时不能假设可直接用 ctypes 调用.已有 C++ 实现如实标为待迁移,迁移后重新实测.
- 编译在 Ubuntu Docker 执行,真实硬件验证在对应开发板执行.运行前核对本地配置、SDK、编译器、板端 Python、Runtime、固件、硬件和内存,公开模板或历史成绩不能替代实时环境证据.不得跨 SoC 覆盖系统库.
- Docker 依赖和版本由正式构建声明管理.创建/进入容器和模型默认入口不安装依赖、不自动下载资源.新增依赖同时更新配方和轻量验收;新容器验收通过后才替换正式环境.其余环境要求仅维护在 platform-docker skill.
- 主机地址、凭据和内部绝对路径只放被 Git 忽略的本地配置.正式模型与数据集保存在 NAS;临时数据放约定日期目录,Windows 为 D:/tmp/hailongcodex/当天日期,无 D 盘用 C 盘,Linux 为 /tmp/hailongcodex/当天日期.不默认清理,不触及其他任务资源.
- 入口直接顺序调用命令,只补必要阶段与进度提示,工具输出和错误直接显示.不默认增加日志副本、包装函数、重试框架或管理机制.独立步骤默认读取本模型配置,不重复传配置路径和运行编号,仅保留确实改变行为的参数.公共环境由 Docker 与本地配置提供,模型不重复设置.
- 新代码按 Google 风格、4 空格缩进,函数包含中文注释并使用英文符号.耗时步骤显示进度或关键节点.保持文件、目录和步骤精简,不保留无关文件或一堆历史版本;不删除未审阅的用户改动或运行必需资源.

## 双入口与双报告

| 阶段 | 主机职责 | 板端职责 | 发布内容 |
| --- | --- | --- | --- |
| deploy/compile.sh | 转换、量化、编译、固定子集浮点参考、打包和上传 | 相同子集硬件精度、损失、少量数值对比与示例 | results/summary_test.json、examples/ |
| deploy/evaluate.sh | 使用已有浮点模型准备约定范围参考及输入 | 使用已有 DLA/HBM 独立评价精度、性能和进程峰值 RSS | results/summary.json |

- 单模型仅维护上述两个主要入口,不保留旧 run.sh 兼容入口.evaluate 不导出、不量化、不编译、不构建、不重新部署.
- 主机上传后结束并给出板端命令.默认由用户登录板端运行,不通过 SSH 自动启动推理或评价;用户明确授权自动板端执行时按其范围执行.上传成功不宣称板端验证通过.
- results/ 只保留 summary_test.json 与 summary.json,互不覆盖.默认正式评价范围见评价参考;用户明确要求少量评价时执行该范围并标 subset,不自动扩大为全量.没有实际结果不填数字.
- 数值对比最多 5 次输入: 普通模型核对实际输出,LLM 仅示例首个 Prefill,Whisper 对比 Encoder,BEVFusion 对比单帧各子图并说明参考执行位置.正式全量评价不重复收集大输出张量.
- 示例与必要输入放 examples/,预测、评分中间结果及必要排错信息放日期临时目录或板端现有 build/work/.不保存同义格式、逐步状态文件或不必要逐样本日志.失败现场不发布到 results/.
- 多文件模型包放 models/,Python 代码放 board/,必要参考放 build/reference/;单文件产物沿用本平台布局.只上传运行必需的模型、tokenizer、参数与数据,不复制训练权重或编译缓存.
- 平台总入口分别在 mtk_models/deploy/ 与 rdk_models/deploy/.主机 compile/evaluate 与板端 compile_board/evaluate_board 分离,顺序调用单模型入口并按约定回传报告/示例,不递归调用或借 compile 的隐藏参数回传评价.

## 评价与真实性

- 同版本、同输入、同 Token/预后处理、同计分协议对比浮点与目标板.明确样本集合、覆盖数量、单位、范围和指标方向;缺匹配参考时只报绝对精度,差值留空.不得沿用其他模型、平台或旧版本分数.
- 公共精度接口复用 tools.evaluation.accuracy_result,生产端与汇总端字段明确一致,不按模型名或字段存在性猜格式.修改共享链路须核对全部受影响调用方,不以单模型补丁代替公共接口修复.
- AP/AR/Accuracy/WER 的差值使用百分点,PPL 使用绝对差或相对变化.缺数据、上下文不足、程序错误时停止并说明,不截题、丢样本、改阈值或在评价中自动重编译.
- LLM 正式范围为固定 1000 题、57 学科、seed=17 的 0-shot MMLU 和完整 WikiText2 test,不称为完整 MMLU 14042 题.子集范围由模型配置与本次授权确定.MMLU 验证选项 Token 延续,不自动套聊天模板;PPL 使用统一分块和上下文重置协议,不能混用 SDK 默认窗口.详见 evaluation.md.
- LLM 统一指标包括 MMLU Accuracy/损失/预测一致率、PPL/相对增幅、首个 Prefill 数值误差、TTFT、Prefill/Decode token/s、端到端耗时、Python 峰值 RSS 和完整部署体积.沿用指标与协议,不要求厂商采用相同量化算法.
- 性能与 RSS 由独立 evaluate 测量,必要预热、实际输入/生成数、计时和内存边界写清楚.示例生成耗时不冒充定长性能,Python 同步调用不冒充纯设备耗时,RSS 不等于全部驱动/BPU 内存.
- 生成示例显示输入与实际聊天模板、Prefill 提示,增量刷新完整 UTF-8 文本,结束时显示停止原因、生成数与示例位置.独立测速关闭流式打印,不逐题输出全量评分文本.
- 精度退化先区分原框架、SDK 网络转换、软件量化与硬件执行.先用已授权的小样本定位,不擅自增加耗时实验,无验收依据不判定损失合理或量化已达极限.

## 修改审查与收尾

- 修复保持最小必要范围,保留已正确的流程、模型与真实结果.汇总入口只编排,不顺手改变各模型配方或预后处理.语法检查不等于接口验证或模型实测,已有结果可复用时优先验证接口,不让用户反复全量运行来排查基本错误.
- skill 及参考文件作为共享规范,普通模型任务不得自行修改或放宽;需调整时说明原因、具体改动并获确认.用户明确要求维护 skill 时可按该范围修改.
- 新模型参照 templates/model 并核对 tools/check_registry.py 的当前约束;模板和检查器仍有平台专用假设时如实处理,不虚构已接入.完成后同步模型文档、模型总览、注册状态及用户指定的进度入口,不把代码完成写成实测通过.
- 中文说明修改、静态检查、接口验证、真实编译、小批量/完整实测及剩余问题.缺协议或资源时标 pending/partial,不拼接版本.提交与推送按既有用户授权执行,使用中文规范前缀和具体说明,不混入其他未提交改动,推送后核对远端.
