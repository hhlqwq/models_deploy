# MiniCPM5-2B 模型卡

## 来源与许可

- 官方权重: [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B).
- 固定 revision: `f97400052a43d642bbc6e9975e2397e3ae6a6b52`.
- 官方项目: [OpenBMB/MiniCPM](https://github.com/OpenBMB/MiniCPM).
- 官方模型卡声明 Apache-2.0.本目录的 [LICENSE](LICENSE) 保留来源清单中固定项目版本的许可证原文.
- 原始实现为 Transformers `LlamaForCausalLM`,权重以 BF16 Safetensors 存储.具体下载地址和文件记录见 [source_url.txt](models/source_url.txt) 与 [source_manifest.json](models/source_manifest.json).

## 模型与部署接口

| 项目 | 配置 |
| --- | --- |
| 参数量 | 2516756480 |
| 层数 / Hidden Size | 42 / 2048 |
| Query Heads / KV Heads | 16 / 2,GQA |
| Head Dimension / Intermediate Size | 128 / 6144 |
| Vocab Size | 130560 |
| Embedding / LM Head | 不共享权重 |
| 官方 Max Position Embeddings | 131072,不代表板端支持该长度 |
| 当前部署上下文 | 1024 Token |
| Prefill / Decode | 每块 128 Token / 每步 1 Token |
| 量化 | Attention、MLP、LM Head 全 W4A16,外置 embedding 为 INT16 |
| 编译与执行 | MDLA 5.3,禁止 CPU 桥接;Python 直接调用 MTK Neuron Runtime |

原生静态图输入包含 embedding、mask、RoPE 和历史 KV,输出包含 logits 和当前 KV.张量顺序、形状和尺度以部署包 `manifest.json` 的 `graph_contracts` 为准,板端核对实际 Runtime 缓冲区.不能把 INT16 embedding 当作 FP16,也不能混用不同配方的 DLA、embedding 和分词器.

当前配置复用 `build/native_hessian16/` 中已选定的量化图和双 DLA;缺失或不匹配时停止,不自动重新量化.配方与历史对照见 [ADAPTATION.md](ADAPTATION.md),运行参数见 [config.yaml](config.yaml).

## 输入与评价

文本生成使用官方聊天模板,默认关闭 thinking,Greedy 解码,停止 Token 为 `1` 和 `130073`.示例最多生成 32 Token,可能只是回答片段.

`compile.sh` 的任务精度子集为固定 100 题 MMLU 和 WikiText2 的 16 个诊断块.主机计算浮点参考,板端使用同题同 Token 评分并统计精度损失;另比较示例首个 Prefill 的有效输出,记录 CosSim、RMSE、NRMSE.成功后发布 `results/summary_test.json`.

`evaluate.sh` 的正式范围为 MMLU 固定 1000 题、57 学科、seed=17,以及完整 WikiText2 raw test.这里的全量指完整执行该约定范围,不是全部 14042 题 MMLU.MMLU 使用无聊天模板的 0-shot 单 Token 选项评分;输入超出部署上下文时停止,不截题或丢题.板端正式结果包含精度、损失、性能与推理进程峰值 RSS,发布为 `results/summary.json`.

当前配方的固定 16 题诊断为 NPU 8/16、同题浮点 13/16,不能外推正式范围精度.现有 `summary.json` 保留迁移前历史记录,不代表当前 Python 链路的新实测.本模型卡补齐不产生新的精度、性能或内存结论.
