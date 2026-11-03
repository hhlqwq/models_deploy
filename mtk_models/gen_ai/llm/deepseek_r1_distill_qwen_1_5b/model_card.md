# DeepSeek-R1-Distill-Qwen-1.5B 模型卡

## 来源与许可

- 官方权重: [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).
- 固定 revision: `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`.
- 官方项目: [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1).
- 基础模型: Qwen/Qwen2.5-Math-1.5B.模型代码与权重许可为 MIT,基础模型许可为 Apache-2.0.
- 原始实现: Transformers `Qwen2ForCausalLM`,BF16 Safetensors.具体文件下载地址见 [source_url.txt](models/source_url.txt),文件记录见 [source_manifest.json](models/source_manifest.json).

## 规格与部署接口

| 属性 | 值 |
| --- | --- |
| 参数量 | 1777088000 |
| 层数 / Hidden Size | 28 / 1536 |
| Attention Heads / KV Heads | 12 / 2,GQA |
| Head Dimension | 128 |
| Intermediate Size | 8960 |
| Vocab Size | 151936 |
| RoPE Theta | 10000 |
| 官方 Max Position Embeddings | 131072,不代表板端上下文容量 |
| Embedding / LM Head | 不共享权重 |
| 量化 | Attention、MLP、LM Head 为 W4A16,非对称 INT4 权重与对称 INT16 激活 |
| 部署 | Prefill 128 Token / Decode 1 Token,上下文由转换配置固定 |
| 辅助资源 | 外置 INT16 embedding、官方词表和 merges、特殊 Token、运行配置 |

两个 DLA 输入包含 embedding、mask、RoPE 和历史 KV,输出包含词表 logits 和当前 KV.实际张量数量、顺序、形状及量化尺度从静态图读取,板端核对 Runtime 缓冲区大小;不把 Q Head 数当作 KV Head 数.

分片连接时若 INT16 尺度不同,使用显式 Requantize.板端 KV 转移同样按实际尺度处理.禁止 CPU 桥接,编译目标为 MDLA 5.3.

## 输入与评价

生成使用官方 Chat Template 和预先编码的 Token,运行停止列表来自官方 generation 配置.生成样例采用固定解码路径,不用于宣称正式准确率.

MMLU 与 WikiText2 使用独立的原始文本协议,不加聊天模板,两端 Token 与评分规则一致.完整题目超上下文时停止.部署与正式评价分别使用 `compile.sh` 和 `evaluate.sh`,详见 [README](README.md).

## 交付参考

[D-Robotics Model Zoo](https://github.com/D-Robotics/rdk_model_zoo_s) 仅作为模型选型与展示参考,不使用其预编译模型作为 MTK 转换输入.
