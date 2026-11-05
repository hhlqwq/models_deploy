# Qwen3-0.6B / S100

当前状态: **按用户最新决定停止原始权重适配,不再作为语音 Demo 的前置任务.** 此前仅完成官方配方配置准备,未量化、编译、部署或板端实测.语音 Demo 复用板端原有 DeepSeek-1.5B、旧 SDK 与本仓库已部署的 Whisper tiny,ASR 来源已由用户明确确认.方案与进度见 [语音 Demo](../../../../demo/smart_car/voice_control/README.md).下文保留已提交配置的用途说明,不代表当前仍在执行该路线.

使用 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 原始权重和已有地瓜 LLM 2.0 SDK.原始权重位于本地平台配置的模型资源根目录下 `llm/qwen3_0.6b/original/`,不使用预编译 HBM 代替原始适配.具体下载版本仍需核实,模型许可为原目录的 LICENSE.

## 官方部署路线

复用 SDK 的 Qwen3 网络实现和官方小模型 W4 量化配方:

1. QuaRot Hadamard 旋转.
2. OmniQuant LWC 量化阈值优化,保留 SDK 默认参数.
3. GPTQ 权重量化,保留 SDK 默认参数.
4. S100 / nash-e 静态校准、图导出及 HBM 编译.

前三步默认 W4A8;S100 静态阶段使用官方混合类型规则,主要激活 INT16,部分 Cache/注意力 INT8,来源 W4 权重量化参数按 SDK 映射为 qint8.最终类型与体积必须在编译后审计,不宣称原生 W4A16.先评估 W4,明显退化时排查后再考虑 W8.

`deploy/host/prepare.py` 只生成日期工作目录中的配置,不修改 SDK 或 original.通过公共环境 `RDK_LLM_SDK_ROOT` 指向已准备的 SDK 根目录.正式编译和评价入口及板端实现尚在接入,当前不能宣称完整流程可运行.

## 校准与评价

QLG 使用 WikiText2 TRAIN.静态校准通过独立的 `test-train-source.parquet` 链接提供同一 TRAIN 文件: SDK PPL 读取器按 `test-*.parquet` 查找,该校准目录中不含真实 TEST 文件.测试数据仅供独立评价.

沿用 MTK LLM 的协议: 子集 MMLU 100 题、57 学科、seed=17、0-shot,WikiText2 前 16 个 128 Token 块、每块重置上下文.报告浮点/BPU Accuracy 与下降百分点、预测一致率、PPL 及相对增幅,最多 5 次首个 Prefill logits 的 CosSim/RMSE/NRMSE.

独立性能采用 128 个真实输入 Token、最多生成 128 Token,报告 TTFT、Prefill/Decode token/s、端到端耗时和 Python 进程峰值 RSS.记录 Prefill 补齐长度、实际生成数和停止原因.统计 HBM、必要辅助权重和整个部署包大小.完整评价不在本轮默认执行.

子集精度写 `results/summary_test.json`,独立评价写 `results/summary.json`,本轮独立评价标注 subset.尚无本轮实际报告,不填入厂商或其他平台数字.

## 来源

- [官方原始模型](https://huggingface.co/Qwen/Qwen3-0.6B).
- [地瓜 S100 LLM 2.0 工具链说明](https://d-robotics.github.io/rdk_s_doc/en/Advanced_development/toolchain_development/LLM_Toolchain/rdk_s100/LLM_Toolchain_v2_0_0/).
- SDK 内 `llm_compression/lightcompress/configs/released_models/qwen3/`、`models/qwen3/` 与 `configs/qwen3_s100.yml`.

已归档 SDK 文件名为 `D-Robotics_LLM_2.0.0_S600_S100_SDK.tar.gz`;下载具体地址以 NAS 资源记录及官网版本核实结果为准.连接信息和内部绝对路径仅保存在本地配置.
