# 部署参考

## 配置和资源

从 skill 位置向上定位仓库根目录,不写死用户机器路径.

- 存储入口: `configs/storage.example.json`、本地 `configs/storage.local.json`、`tools/storage.py` 和 `tools/storage.sh`.
- 平台入口: `configs/platforms/<platform>.yaml`、同目录本地覆盖、`MTK_PLATFORM_CONFIG`.加载顺序以 `docs/environment.md` 和当前实现为准.
- 模型入口: `config.yaml` 路径由公共工具解析,不在模型脚本重复设置 NAS、SDK、容器和板端根目录.
- 正式权重、离线源码、ONNX/TFLite/DLA 位于模型资源目录;校准/评价数据位于数据集目录.临时目录不能是正式资源的唯一存储位置.
- 复用已通过全新容器环境检查的 Docker 镜像,进入后直接使用.缺依赖时报告具体版本和用途并更新镜像构建配方,不得进入后手工补包或让模型入口自动安装.资源下载遵守用户授权并记录具体来源;不要擅自覆盖共享 Python 环境或下载大文件.

## 普通视觉和音频

先进入编译容器,再执行已选择的模型入口:

```bash
bash docker/mtk_g720/enter_container.sh
bash models/perception/object_detection/yolov5s/deploy/compile.sh
```

入口只编排独立步骤:主机实现放 `deploy/host/`,板端 Python 实现放 `deploy/board/`.复用 `tools/deploy.py`、`tools/board.py` 与 `tools/evaluation.py`.推理、前后处理、评价和性能统计使用 Python,通过 Python 调用 MTK SDK 动态库执行 NPU 推理.已有 C++ 程序和自编桥接库逐模型迁移,不作为新增模型的依赖.

Shell 入口直接调用 Python,只补充必要阶段提示,工具输出与错误直接显示在终端,不默认增加日志文件或包装函数.不增加 C++ 推理程序的 make 或交叉编译步骤.公共环境复用 Docker 配置,板端临时路径由 tools.board.paths 计算;只保存后续步骤必需的产物.

主流程 Python 文件无参数时读取所属模型配置.普通容器步骤用 python,专用解释器沿用本地配置,板端用 python3.已有参数仅用于覆盖默认值,不在入口重复传递.

G720 已适配的普通 DLA 配方:

```bash
ncc-tflite --arch=mdla5.3 --suppress-output --disallow-bridge input.tflite -o model.dla
```

此配方不是全部 MTK 芯片或模型的通用命令.浮点、多输入、多输出及 LLM 按模型编译实现处理.核对编译后真实 dtype、layout、stride、scale/zero point 和缓冲区字节数,不能直接用 ONNX 逻辑 shape 解释原生输出.

板端部署包包括 compile.sh、evaluate.sh、配置快照、DLA、必要参数、Python 脚本和公共模块,依赖匹配板端架构的 SDK 动态库.所有模型的主机入口完成编译、打包和上传后结束,提示板端部署目录及命令,不通过 SSH 自动执行推理、示例生成或评价,也不自动取回尚未生成的示例.用户登录开发板,在部署目录执行 bash compile.sh,运行固定子集任务精度、损失及少量数值误差并保存 summary_test.json 和示例,不执行全量评价或专项测速.上传成功只能标为待板端检查,不能宣称示例已生成或板端已验证.校准与量化配方不随评价范围变化.

## 原生 LLM

先读模型 `ADAPTATION.md`、配置和转换实现,再选环境及分片方式:

- LLM 遵守前述所有模型统一的服务器上传、用户登录板端执行流程.板端 compile.sh 保存少量文本示例,正式评价由板端 evaluate.sh 执行.
- 板端生成示例先显示输入文本及实际聊天模板,给出 Prefill 提示,随后增量解码并立即刷新生成文本.按完整词或完整字符输出,避免未完成的 UTF-8 字符和特殊 Token 造成乱码;结束时显示停止原因、生成 Token 数和示例保存位置.生成上限由模型示例配置确定,短片段如实说明,不统一扩大为完整回答.
- 复用 tools/llm/generate.py 的流式生成逻辑,保留各模型的 Token 与停止条件.正式评价中的展示输入同样可流式输出,独立预热和性能测量关闭文本打印,仅保留测量外的阶段提示.不将流式展示耗时标为独立性能测量,不逐题打印 MMLU/WikiText2 全量评分文本.
- 明确上下文、Prefill 批长、Decode 单 Token、KV Cache、mask 和 RoPE 规则.
- tokenizer、chat template、BOS/EOS 和停止 Token 与官方参考一致.文本/Token 输入一致性须有实测记录.
- 双 DLA、外置 embedding、tokenizer、配置及 Python 推理代码共同构成交付,依赖匹配的 SDK 动态库,不能只统计单 DLA.生成循环、KV 转移及采样均使用 Python.
- W4A16 与 INT16 原生 I/O 按实际图记录,重新量化后不能复用旧契约.
- MiniCPM 和 DeepSeek 使用各自的 host/convert.py、compile.py、package.py 及板端入口,共享量化适配、KV 转移和评价工具.不得跨模型复制特殊 Token、聊天模板或 KV Head 配置.
- 编译阶段准备同子集 MMLU/PPL 浮点参考、生成示例所需 Token 与运行参数.按约定范围的 MMLU/PPL 参考和评价输入在 evaluate.sh 阶段准备;所有 LLM 使用固定 1000 题、57 学科、seed=17 的正式 MMLU,资源统一放板端 build/reference/full/,WikiText2 均为完整 test.
- Prefill/Decode 测速及 RSS 采集只属于 evaluate.sh,输入长度、生成计数及内存范围统一写入 summary.json.
- 生成文本放 examples/output,日志和中间指标放临时目录,results 仅保留 summary_test.json 和 summary.json.

## 故障定位

| 现象 | 优先核对 | 边界 |
| --- | --- | --- |
| `EDPA_1_2` / unsupported target | 当前 DLA 参数、MDLA 目标及 Runtime | G720 检查禁止桥接配方;重新编译属于部署修复,不从评价自动触发 |
| `libcmdl.so` / 模型加载失败 | ELF 架构、Runtime、APUSYS、驱动和固件 | 不跨 SoC 覆盖系统库或创建假链接;版本查询不代替真实推理 |
| 输出错位 / 精度异常 | shape/dtype/stride、缓冲区大小、scale/zero point 和预后处理 | 用同一输入逐阶段定位,不改评价阈值掩盖问题 |
| 全量题目超上下文 | 完整题目长度和已有 DLA 上下文 | 报告阻塞,另行计划部署变更,不得截题或在评价中重新转换 |
| 缺样本 / 参考不匹配 | 样本 ID、覆盖、权重及协议 | 保留错误和预测,不混用子集与全量指标 |

保留原始错误、失败阶段和最小复现输入.组件不匹配时规划官方匹配发布组合的维护,不能为单模型覆盖系统库.维护前确认设备任务及授权.

总入口固定为主机 compile.sh/evaluate.sh 和板端 compile_board.sh/evaluate_board.sh,四者均位于 models/deploy/.主机 compile 上传两个板端入口、common.sh 和本地回传配置至板端 deploy/.总入口只顺序调用单模型入口、保存本地覆盖日志及执行约定传输;不互相递归或通过 compile 的隐藏参数完成评价回传.
