# 部署参考

先按目标平台选择对应章节,共用流程仅在本文件与 SKILL.md 维护.不同平台不共享 SDK 参数、编译产物或硬件成绩.

## 配置与资源

- 从 skill 所在目录定位仓库根目录,不写死主机路径.MTK 模型在 mtk_models/,RDK 模型在 rdk_models/,模型内部的资源包 models/ 不改名.
- MTK 存储使用 configs/storage.example.json、本地 storage.local.json 与 tools/storage.py;平台配置使用 configs/platforms/<platform>.yaml、本地覆盖与 MTK_PLATFORM_CONFIG,以当前实现为准.
- S100 使用 configs/platforms/s100.yaml、s100.local.yaml、RDK_PLATFORM_CONFIG 与 tools/rdk_config.py;传输复用 tools/rdk_deploy.py.不通过 MTK 配置解析 S100 SDK.
- 模型独立步骤默认读取所属 config.yaml,板端读取 config.json.主机实现放 deploy/host/,板端实现放 deploy/board/.入口不重复传配置路径和运行编号,保留真正改变行为的参数.
- SDK、容器、挂载及专用解释器由公共环境和本地配置提供.主机、板端各用已有匹配的 Python,不在模型脚本安装依赖或自动切换解释器.
- 原始权重、源码和正式产物保存在 NAS 模型资源目录,评价输入保存在数据集目录.过程文件放日期临时目录,不把临时目录作为正式资源的唯一存放位置.资源需明确下载 URL、版本、文件名及用途.
- 编译环境新增依赖时先按 platform-docker skill 更新正式配方并验收全新容器,不在运行容器手工补包交付.主机和板端动态库分别核对架构及版本.

## 双入口与发布

compile.sh 顺序执行平台所需的导出、量化、编译、固定子集浮点参考、打包和上传.板端同名入口运行相同子集,计算任务精度、浮点损失及少量数值误差,生成 summary_test.json 和必要示例,不做独立测速.

evaluate.sh 使用已有模型生成约定范围浮点参考并上传必要数据,板端同名入口独立评价精度、性能和峰值 RSS,写 summary.json;不得重新量化、编译、构建或部署.用户明确要求少量评价时标 subset,不扩大为完整基准.

主机上传后提示板端目录和命令,默认由用户登录板端执行;明确获准自动板端执行时遵守授权范围.上传成功不等于推理成功.部署包包括入口、配置、模型、必要参数、tokenizer、Python 代码与匹配的运行依赖,不上传训练权重或主机编译缓存.

两个平台的四个总入口分别放在 mtk_models/deploy/ 和 rdk_models/deploy/,只顺序调用单模型入口并执行约定传输.报告/示例使用共用回传实现,不递归调用,不通过 compile 隐藏参数完成评价回传.

## MTK Genio

普通视觉/音频复用 tools/deploy.py、tools/board.py、tools/evaluation.py.先进入对应 Docker 再运行模型入口,例如:

```bash
bash docker/mtk_g720/enter_container.sh
bash mtk_models/perception/object_detection/yolov5s/deploy/compile.sh
```

G720 普通模型已使用的基础 NCC 配方:

```bash
ncc-tflite --arch=mdla5.3 --suppress-output --disallow-bridge input.tflite -o model.dla
```

该配方不是其他芯片或全部 LLM 的通用命令.核对实际 dtype、layout、stride、scale/zero point 与缓冲区字节数,不能用 ONNX 逻辑形状推断原生布局.不把 G720 的编译目标和指标用于 G5100.

MTK 原生 LLM 先读模型 ADAPTATION.md、配置及转换实现.MiniCPM5 和 DeepSeek 复用各自 host/convert.py、compile.py、package.py 与板端入口,共享 tools/llm/ 的量化适配、KV 转移和生成工具,不跨模型复制 BOS/EOS、模板或 KV Head 配置.

交付包含 Prefill/Decode 双 DLA、外置 embedding、tokenizer、配置和 Python 推理代码.按实际图核对 W/A 类型与 I/O,重新量化后不得复用旧契约.KV 转移必须处理尺度差异,不能直接复制不同量化尺度的缓存.运行使用匹配的 Neuron 动态库.

## 地瓜 RDK S100

先核对目标 nash-e、板端系统/Runtime 及实际 SDK.S100 普通模型使用匹配 OpenExplorer 的 hb_compile,LLM 使用匹配的官方 LLM SDK.普通 OpenExplorer 镜像可用不证明 LLM 的 Torch、插件、编译器和依赖已满足.

- 优先官方网络适配.从指定目录加载原始 config、tokenizer 和完整权重,检查参数名称、结构、形状及缺失/多余键.只有预编译模型或 benchmark 不证明可从原始权重重现.
- S100P/nash-m、S600/nash-p 配方必须核对后改为 S100 实际目标,不直接复用 HBM.厂商页面只展示 W8 不证明不支持 W4,也不能从其他芯片的 W4 成绩推断 S100 原生 INT4.
- 量化顺序和方法按已确认方案.当前已审阅 LLM 2.0 SDK 的 Qwen3 示例提供 QuaRot → OmniQuant LWC → GPTQ 前置 W4A8 配方,小模型保留 LWC;随后执行 S100 静态校准、导出和编译.这只是该版本的例子,后续 SDK 以实际源码和配方为准,不强制其他模型沿用此算法.
- 记录前置量化位宽和最终硬件类型.当前已审阅 SDK 的 nash-e/nash-m 路径将来源权重量化参数映射为 qint8,部分激活和 KV 仍是混合类型;不得把 W4 网格直接写成原生 W4A16 或承诺体积/速度收益.
- 校准核对数据读取代码.某些 PPL 读取器按 test-*.parquet 查找,应提供仅包含训练划分的独立校准视图并说明文件命名;真实测试集不能直接传给该校准器.不为校准重命名或覆盖原始数据集.
- SDK 默认全量评价不替代本仓库协议.保留量化算法必要参数,评价单独按仓库确定的题集、Token、上下文和计时口径执行;必要时关闭配方中的默认评分.
- 板端优先使用已验证的官方 Python HBM Runtime 或 SDK 动态库接口.检查原生 I/O 名称、shape、dtype、缓存顺序、内存生命周期与同步方式.只有 C++ 头文件/类接口时需继续核实 Python/C ABI,不能臆造 ctypes 绑定.
- Whisper 的英语子集结果不证明中文可用;深度模型的绝对米制协议不能替换为相对深度协议.具体差异保留在模型文档,不写成跨任务默认行为.

## LLM 共用契约

明确上下文预算、Prefill 分块、Decode 长度、mask、RoPE、KV Cache 与停止规则.原模型上下文上限不等于编译产物容量.官方 tokenizer、聊天模板及 BOS/EOS 与浮点参考一致,记录文本/Token 一致性.多轮由明确历史消息构造,是否复用 KV 取决于已验证接口.

生成示例显示输入、实际聊天模板和 Prefill 提示,流式刷新完整 UTF-8 文本,记录停止原因、生成数和示例位置.独立预热/测速关闭文本打印.性能输入记录真实 Token 与补齐执行长度,不能把厂商固定 256 分块与仓库 128 Token 输入混成同一计时口径.

compile 准备同子集 MMLU/PPL 浮点参考和示例输入,evaluate 准备约定独立评价参考.统一指标及计分见 evaluation.md.体积包含全部编译产物、外置 embedding 和必要辅助权重,不只统计单个 DLA/HBM.

## 故障定位

| 现象 | 先核对 | 执行边界 |
| --- | --- | --- |
| 不支持的目标或算子 | 实际硬件、SDK、编译目标、官方支持范围 | 不跨芯片复用,不在评价中自动重编译 |
| 模型或库加载失败 | ELF 架构、运行库、驱动、固件及版本组合 | 不覆盖系统库;版本查询不代替推理 |
| 输出错位/精度异常 | 契约、尺度、缓存及前后处理 | 同输入逐阶段定位,不改阈值掩盖问题 |
| 题目超上下文 | 完整题目 Token 与编译容量 | 不截题或丢题,另行计划扩大容量 |
| 参考缺失/不匹配 | 样本、权重、协议、版本和覆盖 | 不混用历史、子集、全量和其他平台结果 |

保留最小必要失败现场、原始错误和阶段.组件不匹配时先规划官方兼容组合的维护,升级、重启、停止进程和范围外操作按用户授权执行.
