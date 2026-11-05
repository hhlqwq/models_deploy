# MiniCPM5-2B 适配流程：从首次跑通到精度定位与速度优化

本文在原《MiniCPM5-2B适配流程》基础上修订，依据 2026-10-22 本地仓库实现及仓库保留的历史评价记录编写。目标平台为 MT8189 / Genio 720。

本文要解决的是：拿到官方模型后，怎样逐步完成适配；发现精度下降后，怎样找到下降发生的位置；怎样决定下一次改什么；怎样确认优化确实有效。核心配置、关键函数和必要的数值例子均放在正文，不要求跳到源码阅读。

先约定三种证据：

| 标记 | 含义 |
|---|---|
| **当前实现** | 已阅读当前代码，能够说明它会怎样执行；不代表本次运行过 |
| **历史记录** | 仓库保存的过去实测或诊断结论，保留其模型、协议和实现范围 |
| **建议实验** | 为继续定位或优化而设计的步骤，尚未执行，没有改善幅度承诺 |

本次只是文档修订，没有重新转换、编译、运行模型或测量性能。当前 Python 推理实现尚待板端验证；文中的历史指标属于迁移前的 C++ / 原生 CLI 实现，不能当作当前 Python 实现成绩。

## 目录

1. [先看实际问题：模型跑起来以后，究竟还差什么](#1-先看实际问题模型跑起来以后究竟还差什么)
2. [拿到模型后的第一轮工作：先确定参考、输入和验收方式](#2-拿到模型后的第一轮工作先确定参考输入和验收方式)
3. [准备模型、校准、量化与导出：代码真正执行了什么](#3-准备模型校准量化与导出代码真正执行了什么)
4. [编译与运行：数值契约怎样延续到板端](#4-编译与运行数值契约怎样延续到板端)
5. [怎样评价：把“感觉变差”变成可定位的问题](#5-怎样评价把感觉变差变成可定位的问题)
6. [精度不够时怎样一步一步查：本项目的诊断主线](#6-精度不够时怎样一步一步查本项目的诊断主线)
7. [怎样调整量化流程：每个优化项应该怎样做实验](#7-怎样调整量化流程每个优化项应该怎样做实验)
8. [怎样优化速度和内存：先拆开时间，再动代码和配方](#8-怎样优化速度和内存先拆开时间再动代码和配方)
9. [把下一轮工作排成可执行顺序](#9-把下一轮工作排成可执行顺序)
10. [交付边界、原文修正与资料说明](#10-交付边界原文修正与资料说明)

## 1. 先看实际问题：模型跑起来以后，究竟还差什么

### 1.1 已有成绩告诉了我们什么

**历史记录。** 已归档实验为 `20261014_minicpm5_w4_light_hessian_v8`，量化为 `asym4W_sym16A`，Prefill=128，context=1024，记录了使用 16 个样本的 Hessian 权重优化。它不是当前默认 `weight_optimization: none` 配置的一次实测。

| 项目 | FP32 参考 | 历史 W4A16 | 正确理解 |
|---|---:|---:|---|
| MMLU | 559/1000，55.9% | 413/1000，41.3% | 固定 1000 题，57 学科，0-shot，下降 14.6 个百分点 |
| WikiText2 PPL | 40.0411 | 48.3821 | 原始测试文本前 128 行，58 块，共评分 7366 个 Token，相对上升约 20.83% |
| Prefill | — | 288.666 Token/s | 原生实现，固定 128 Token 输入 |
| Decode | — | 8.14848 Token/s | 原生实现，历史测量生成 128 Token |
| 峰值进程 RSS | — | 1152.42 MiB | 原生 CLI 进程范围，不能代表 Python 实现或全部 NPU 内存 |

这里的 FP32 成绩是本项目同协议计算的参考，不是拿官方模型卡上的成绩直接比较。1000 题也不是完整的 14042 题 MMLU 测试集。

两个精度指标同时退化，说明需要研究部署带来的质量差异。它们还不能单独告诉我们：到底是 tokenizer 错、结构转换错、量化损失大，还是 NPU 输入输出解释错。

### 1.2 “能回答问题”为什么不够

历史上另有 4 条自编功能样例，结果分别出现预期答案、遇到停止 Token，且文本输入与预先编码的 Token 输入得到一致结果。但严格遵循“只回答结果”的只有 3 条。

其中一条输入是：

```text
Reply with only the result: 3 + 4 = ?
```

FP32 回答为 `7`，量化模型回答为 `3 + 4 = 7`。答案内容正确，格式要求没有满足。历史记录还显示 4 条中有 3 条与 FP32 的生成 Token 完全一致。

由此可以分清四层判断：

1. 程序成功运行：证明调用链没有直接失败。
2. 输出包含正确答案：证明该样例具备基本功能。
3. 遵循指令与停止规则：证明该样例的行为符合要求。
4. 固定协议下的整体指标：说明指定评价范围内的质量。

功能样例通过不能抵消 MMLU 下降 14.6 个百分点；一次生成不同，也不能直接证明所有差异都来自量化。

### 1.3 实际适配需要建立两个闭环

```text
正确性闭环：
官方参考 → 输入一致 → 浮点转换一致 → 量化误差可解释
         → 板端数值可解释 → MMLU/PPL/生成回归

效率闭环：
固定质量与工作负载 → 拆分耗时/内存 → 找最大开销
                  → 只改一个因素 → 重测速度并检查质量回退
```

当前项目已经保留“主要退化发生在量化阶段”的诊断结论，具体涉及后段 MLP 的 INT4 近似损失与部分 A16 激活截断。但没有保存完整的逐次实验时间线和所有中间数值。因此，下文会讲清这些结论对应的排查方法，不编造“某次调参提高了多少分”的过程。

## 2. 拿到模型后的第一轮工作：先确定参考、输入和验收方式

### 2.1 第一件事是冻结一个可以比较的对象

当前来源记录指向官方 `openbmb/MiniCPM5-2B`，固定 revision 为 `f97400052a43d642bbc6e9975e2397e3ae6a6b52`。资源包括结构配置、Safetensors 权重、tokenizer、聊天模板和许可证。转换使用独立副本，原始资源不直接修改。

需要同时固定以下内容，否则后面可能拿两个不同实验互相比较：

| 必须固定的内容 | 不固定会造成什么问题 |
|---|---|
| 权重与模型配置版本 | 同名模型可能已经变过；差异未必来自量化 |
| tokenizer、聊天模板与特殊 Token | 看起来相同的文本，实际送入的 Token 可能不同 |
| SDK / Converter / NCC / Runtime 组合 | 不同版本可能改变转换、算子实现和缓冲区要求 |
| Prefill、context、mask、RoPE 与 KV 布局 | 影响静态图、数值和运行逻辑 |
| 评价数据、样本顺序、评分规则 | 不同题集、不同 Prompt 不能直接计算精度下降 |
| 采样策略和生成上限 | 随机采样会妨碍逐 Token 排错；上限会影响停止表现 |

资源地址及版本应记录，执行主机地址和本地资源路径只进入本地配置。正文无需铺满校验值，但运行时必须避免混用不同转换的 DLA、embedding 和契约。

### 2.2 在量化前先准备浮点参考

学习和排错的顺序应该是：先知道官方模型在同一输入上应当表现怎样，再研究量化输出。当前入口为了将编译和评价分开，把正式 FP32 参考放到独立 `evaluate.sh` 中；这不改变参考先行的诊断原则。

当前严格加载浮点权重的核心逻辑如下。它直接按索引读取 Safetensors，避免旧加载器因元数据格式报错；不是重新训练，也没有修改权重：

```python
def load_reference(model_path):
    """按官方索引严格加载 FP32 权重,省略环境和进度设置."""
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    index = json.loads(
        (model_path / "model.safetensors.index.json").read_text()
    )
    files = sorted(set(index["weight_map"].values()))
    state = {}
    for name in files:
        state.update(load_file(model_path / name, device="cpu"))
    model.load_state_dict(state, strict=True)
    return model.eval()
```

这是对当前有索引权重分支的节选整理；正文省略 import 和路径读取，不作为独立运行脚本。

`strict=True` 保证参数名称和形状严格匹配，不能证明 SDK 对模型语义的解释也完全正确。后面仍要核对 SDK 浮点图与原框架的输出。

### 2.3 第一轮先把要回答的问题写明白

第一次适配先回答：

- 短输入、跨 128 Token 的输入和多步 Decode 是否正确？
- MMLU 和 PPL 相对同协议 FP32 退化多少？
- Prefill、Decode 各自多快，计时包含什么？
- 模型驻留与运行过程占多少内存？

在没有用户指定质量阈值和速度目标时，先报告真实差异，不自行宣布“已经合格”，也不把当前成绩当作 W4A16 的极限。优先修正正确性问题，再讨论量化配方和执行优化。

## 3. 准备模型、校准、量化与导出：代码真正执行了什么

### 3.1 当前真正使用的配置

下面是当前模型配置中影响核心计算和评价的部分，保留实际层级：

```yaml
conversion:
    context: 1024
    max_new_tokens: 512
native:
    work_dir: build/native
    prefill: 128
    calibration_batches: 32
    mask_value: -100
    rotate: false
    attention_precision: asym4W_sym16A
    mlp_precision: asym4W_sym16A
    tail_precision: asym4W_sym16A
    weight_optimization: none
    weight_opt_config: ''
    calibration_train: ''
evaluation:
    test_mmlu_questions: 16
    test_wikitext_blocks: 4
    reference_device: cuda
```

这组配置有几个容易影响判断的细节：

| 配置 | 实际作用 | 调整时必须注意 |
|---|---|---|
| `context=1024` | 本次导出的上下文容量 | 输入和生成共同占用，改大要重新导出、编译并配套资源 |
| `prefill=128` | Prefill 图每次计算 128 个位置 | 长 Prompt 分块；不是输入长度只能为 128 |
| `calibration_batches=32` | 请求 SDK 执行的校准批次数 | 不等于有 32 条互不相同的输入 |
| `mask_value=-100` | SDK mask 使用的负值 | 不是越负越好，必须同时满足语义与接口可表示范围 |
| `rotate=false` | 默认不启用旋转 | 不能将历史旋转实验结论自动套到当前默认配置 |
| `weight_optimization=none` | 默认不进行额外权重优化 | 历史 Hessian 成绩不属于这套默认配方 |
| `weight_opt_config` | 作为 SDK 额外 Converter 选项文件传入 | 不是仓库已定义好语义的通用 Hessian 参数表 |
| `test_*` | 小范围诊断数据量 | 当前板端 `compile.sh` 不会据此自动跑精度；正式入口要求全量 |
| `reference_device=cuda` | WikiText2 浮点参考的设备设置 | 当前 MMLU 调用未显式传 `device`，评分函数默认 CPU，不是所有参考都自动走 CUDA |

配置之外还有环境覆盖，例如 `CONTEXT_SIZE`、`PREFILL_TOKENS`、`CALIBRATION_BATCHES`、`ROTATE_WEIGHTS`、`WEIGHT_OPTIMIZATION`、`WEIGHT_OPT_CONFIG` 和 `CALIBRATION_TRAIN`。因此比较实验时要记录**实际生效值**，只看 YAML 不足以确认配方。

### 3.2 转换入口的四步与各自输入输出

当前转换 `main()` 的主干是：

```python
def main():
    """依次准备模型、校准、量化和导出,省略进度提示与目录恢复."""
    native = settings(arguments(__file__), python_executable=sys.executable)
    prepare_model(native)
    os.chdir(native.work)
    calibrate(native)
    quantize_model(native)
    export_shapes(native)
```

实际实现还会打印四个阶段提示，并用 `try/finally` 恢复工作目录。这里截取计算顺序，不能用这段节选替换正式入口。

| 阶段 | 读取什么 | 留下什么 | 要解决的问题 |
|---|---|---|---|
| 准备 | 官方权重、配置、tokenizer、模板 | 转换副本、校准 Token、示例 Token、实际准备记录 | SDK 能读取，输入语义不变 |
| 校准 | 副本配置、校准 Token | SDK 校准数据 | 为后续量化提供代表性输入与所需信息 |
| PTQ | 浮点模型、校准数据、精度及优化选项 | 逐层量化 TFLite、元数据、embedding 等 | 构造 W4A16 计算 |
| 固定形状 | 逐层分片、Prefill/context | 两张静态 TFLite | 明确形状、拼接接口和计算边界 |

当前准备函数会替换上次 `build/native` 转换工作目录，并保留官方源资源和当前部署包。需要比较候选时，先保留诊断必需的候选产物，不能指望同一个工作目录自动保留上一轮结果；也不需要复制所有缓存和日志。

### 3.3 模型准备：先修兼容性，避免改掉原模型含义

当前转换副本包含以下处理：

```python
# 只改 BPE merges 的序列化形式,不改合并规则含义.
token_data["model"]["merges"] = [
    " ".join(item) if isinstance(item, list) else item
    for item in token_data["model"]["merges"]
]

# 使用官方模板构造校准聊天输入,关闭思考模式.
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": question}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
tokens = tokenizer.encode(text, add_special_tokens=False)
if tokens.count(tokenizer.bos_token_id) != 1:
    raise ValueError("Prompt 必须且只能包含一个 BOS.")

# 校准记录中的 tokens 是空格分隔字符串,不是 JSON 整数数组.
record = {"tokens": " ".join(str(token) for token in tokens)}
```

为什么这里必须关心 `add_special_tokens=False`？因为聊天模板可能已经包含 BOS；再次自动加特殊 Token 会改变真实输入。看到终端文本相同，并不能说明 Token 序列相同。

副本配置还会写入：

```python
config.update(
    mask_value=mask_value,
    rotate=rotate,
    rotate_seed=0,
    rotate_mode="hadamard",
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=130073,
    pad_token_id=tokenizer.pad_token_id,
    tokenizer="pretrained_fast",
    max_position_embeddings=context,
)
```

SDK 2.7.5 的校准生成接口使用单 EOS=130073；运行时停止列表仍保留原配置中的 `1`、`130073`。两者作用阶段不同，不能为了兼容 SDK 就把运行停止规则也丢掉。

这些格式调整应先通过同一文本的 Token ID 与模板核对，再进入数值比较。如果此处已经变化，后面的“量化误差”比较就失去了相同输入前提。

### 3.4 校准：从原始文本到 SDK 输入，实际经过了什么

默认校准数据为 16 条自编双语问题，内容包括介绍、算术、翻译、总结和边缘部署等。每条先套聊天模板再编码，写成一行：

```json
{"tokens": "<第1个Token ID> <第2个Token ID> ..."}
```

上面仅示意格式，尖括号内容不是可执行输入。

显式提供训练语料时，当前实现的混合方式是：

```python
# 从训练文本连续取 8 段,每段最多 256 Token,与前 8 条自编问题交替排列.
mixed_records = []
for index in range(8):
    segment = train_tokens[index * 256:(index + 1) * 256]
    mixed_records.append({"tokens": " ".join(map(str, segment))})
    mixed_records.append(records[index])
records = mixed_records
```

这不是覆盖整个 WikiText2 训练集，也不是随机采样全语料；最多只取前 2048 个训练 Token。前 8 条自编问题也不等于全部 16 条双语问题。当前片段没有逐段断言长度必须为 256，使用该方案还应确认训练文本确实足够。

真正启动 SDK 校准的函数如下，保留当前调用参数：

```python
def calibrate(native):
    """执行 SDK 校准命令,成功退出后记录请求的批次数."""
    marker = native.work / "calibration_complete.txt"
    marker.unlink(missing_ok=True)
    command = [
        native.python,
        native.python.parent / "mtk_make_llm_ptq_calib_dataset",
        "converter",
        native.model / "config.json",
        native.work / "calibration.jsonl",
        "-b", str(native.batches),
        "-m", "128",
    ]
    run_command(command, cwd=native.work)
    marker.write_text(f"{native.batches}\n")
```

实际流程是：

```text
16 条记录或显式混合记录
  → SDK 读取模型与 Token
  → 按 SDK 校准流程执行并生成 calibration_datasets
  → 命令成功返回
  → 写入批次数标记
  → PTQ 读取这些校准数据
```

这里需要知道仓库控制到哪里为止：

- `-b 32` 是请求批次数，标记文件写的也是请求值，不能代替 SDK 内部逐批覆盖审计。
- `-m 128` 属于校准命令参数，和 Prefill 静态图 128 是两个接口。当前未读取安装中的 SDK 参数定义，不把它进一步解释成已确认的具体语义。
- SDK 具体保存哪些中间量、如何选统计范围、是否生成后续 Token，应由对应 SDK 版本的实现或文档确认；仓库这个调用函数本身没有给出这些细节。
- 32 个批次不说明输入分布充分。重复运行相似短问题，并不能补上长输入、领域词汇和 Decode 状态的覆盖。

**优化时真正要改的是模型看到的分布。** 例如业务常用中文长文，而校准基本是短问答，就应先设计含中文长段落、任务词汇与合理长度分布的训练/开发样本，而不是直接把批次数翻倍并宣称校准更充分。

### 3.5 PTQ：实际参数如何进入 SDK

当前量化函数中的核心调用为：

```python
options = [
    "converter", str(native.model / "config.json"),
    "-p", "asym4W_sym16A",
    "-tp", "asym4W_sym16A",
    "-d", str(native.work / "calibration_datasets" / native.model.name),
    "-m", "Overall",
]
if native.optimization != "none":
    options.extend(["--weight_optimization", native.optimization])
if native.extra:
    options.extend(["--extra_converter_options", native.extra])
quantize(options)
```

参数与行为的对应关系：

| 参数 | 本项目作用 |
|---|---|
| `converter` 与配置路径 | 指定 SDK 转换入口及模型配置 |
| `-p asym4W_sym16A` | Attention / MLP 投影采用当前统一配方 |
| `-tp asym4W_sym16A` | LM Head 同样保持 W4A16 |
| `-d .../calibration_datasets/...` | 读取刚才准备的 SDK 校准数据 |
| `-m Overall` | 传入 SDK 的校准方法名称；不能仅凭名称推断统计和优化算法 |
| `--weight_optimization` | 可选 `hessian` 或 `gradient`，默认不传 |
| `--extra_converter_options` | 可选 Converter 额外配置文件，不在本文杜撰内部字段 |

仓库包装函数随后做三件事：检查精度约束；临时设置 SDK 命令行参数后直接调用 `ptq.main()`；成功后核对实际权重位宽。原始 SDK 异常会向外传播，入口停止，不把失败隐藏成一轮“完成”。

这里的执行边界是：**模型结构解释、数值统计、权重量化及可选优化由 SDK 完成；仓库负责输入、参数、兼容适配和产物检查。** 从外层这段代码不能断言 SDK 采用了 GPTQ、AWQ 或某种特定分组算法。

### 3.6 W4A16 在这里到底量化了哪些东西

| 部分 | 当前要求 |
|---|---|
| 每层 Attention 的 Q/K/V/O 投影权重 | 非对称 INT4 |
| 每层 MLP 的三个投影权重 | 非对称 INT4 |
| LM Head 权重 | 非对称 INT4 |
| 激活配方 | 对称 INT16 |
| 外置 embedding | INT16 |
| 静态图输入输出与 KV | INT16 原生接口 |

**A16 在这里是整数 INT16，不是 FP16。** 这一表描述的是当前可核对的投影配方和接口，不表示所有内部算子及累加器都用相同位宽。

最基本的量化关系是：

```text
q = clip(round(x / scale) + zero_point, qmin, qmax)
x_hat = (q - zero_point) × scale
```

可以用一个只说明数值关系的 INT4 例子理解：假定某个非对称 4 bit 表示采用 0…15，scale=0.1，zero_point=5，那么 x=0.26 会编码为 q=8，反量化为 0.3。该例不代表 SDK 内部 INT4 打包格式或实际分组参数。

为什么不能只看权重本身的平均误差？对线性层，有：

```text
原输出：Y = X × Wᵀ
量化后：Y_hat = X × W_hatᵀ
误差：ΔY = X × (W_hat - W)ᵀ
```

同样大小的权重误差，在不同输入激活 X 上影响可能完全不同。这就是为什么校准样本、权重优化和后段 MLP 的输入分布需要一起研究。

当前后续检查不是只相信配置字符串：

1. 应有 `num_hidden_layers + 1` 个分片，包含 Transformer 层及输出头。
2. 每个分片必须非空，并有对应 SDK 元数据。
3. 若有量化精度配置，逐个核对 Attention、MLP、LM Head 的精度声明。
4. 缺少该类配置时，读取实际 TFLite，核对投影权重 INT4 和图接口 INT16。
5. 核对投影数量应为 `层数×4`、`层数×3`、`1`。

这些检查证明配方和文件结构符合预期，**不证明精度已合格**。

### 3.7 固定形状：为什么要两张图，拼图时具体修了什么

当前导出形状为：

```text
128t1024c → Prefill：一次处理 128 个位置
1t1024c   → Decode：一次处理 1 个位置
```

代码主干：

```python
shapes = [f"{native.prefill}t{native.context}c", f"1t{native.context}c"]
fix_native_shapes(native.quantized, shapes)
```

shape fixer 通过 `-n 1` 为每个阶段合成单张图。两张图使用同一套模型权重，区别在输入形状、缓存与执行安排；外置 embedding 不在这两张图中查表。

图接口可以概括为：

```text
input_embeds + mask + pos_emb + past_key/value
          ↓ 全部 Transformer 层与 LM Head
logits + curr_key/value
```

拼接分片时，**dtype 相同不表示量化尺度相同**。假设前段输出 scale=0.02、zero_point=0，后段输入 scale=0.01、zero_point=0，同一个实数 2.0 在前者中编码为 100，在后者中应编码为 200。直接把 100 接过去，会被解释为 1.0。

当前适配代码在导入分图时：

1. 比较两侧类型和量化参数，相同就直接连接。
2. 不同则要求逻辑 shape 相同。
3. 只接受两侧均为 INT16、单尺度的已支持接口。
4. 创建具有目标输入量化参数的新张量，并插入真正的 `Requantize` 算子。
5. 将分片输入映射到该新张量；不支持的情况直接报错。

数值上相当于：

```text
q_target = clip(round(((q_source - z_source) × s_source) / s_target)
                + z_target)
```

这修正的是分片连接的数值解释问题。它不会恢复已经被 INT4 舍入或 A16 截断丢失的信息，不能把“补了 Requantize”理解成量化精度问题已经全部解决。

## 4. 编译与运行：数值契约怎样延续到板端

### 4.1 NCC 编译什么，哪些参数真实存在

量化决定数值表示，静态化决定输入形状，NCC 才把图编译成目标 NPU 可执行的 DLA。当前两张图分别编译成 Prefill / Decode DLA。

当前 NCC 参数的核心部分如下，输入、输出及工具路径用占位符表示，不包含执行主机信息：

```bash
ncc-tflite \
    --arch=mdla5.3 --l1-size-kb=256 --num-mdla=1 \
    --opt=3 --opt-footprint --opt-accuracy --stable-linearize \
    --gno=LTS,Inception --gno-exp --gno-non-4d-tiling --mlo \
    --disable-apusys --fc-to-conv --mdla-int16-lut \
    --suppress-input --suppress-output --show-memory-summary \
    --mdla-conv-exp --intval-color-legacy --disallow-bridge \
    <阶段附加参数> <静态图.tflite> --dla-file <输出.dla>
```

阶段附加参数为：

```text
Prefill:
    --broadcast-act-wgt
    --broadcast-flow-distance=63
    --split-large-conv-ic=1536

Decode:
    --split-16a4w-conv-oc
```

这些是当前代码实际传递的配方，不能仅根据选项名称保证某个选项一定提速或保精度；准确语义和支持范围需要匹配当前 NCC 版本。最关键的执行约束是：

- `--arch=mdla5.3`：当前 Genio 720 的目标，不能直接外推到 G5100。
- `--disallow-bridge`：禁止依靠 CPU 桥接来接受不支持的图；编译成功仍须真实板端执行。
- `--suppress-input/--suppress-output`：按原生接口准备和读取数据，需要携带 scale / zero point，不把浮点数组直接当成 INT16 缓冲区。

当前编译前会核对准备记录中的 context 以及量化前缀；编译后检查 DLA 是否存在且非空。这是产物检查，不能替代板端数值检查。

### 4.2 打包时必须绑定在一起的资源

```text
Prefill DLA + Decode DLA
INT16 embedding 表
tokenizer 与已经套好模板的示例 Token
每张图的 I/O 契约
同版 SDK 生成的 mask / RoPE 输入库
Python 生成与评价程序
配置及资源清单
```

I/O 契约中每个输入输出至少包含：

```json
{
    "name": "input_embeds",
    "dtype": "INT16",
    "shape": [1, 128, "hidden_size的实际整数值"],
    "scale": "实际读取的正数",
    "zero_point": "实际读取的整数"
}
```

这是结构示意，真实契约的 shape、scale、zero_point 均为数值。当前代码从实际静态 TFLite 读取这些值，要求接口为 INT16、单尺度，并检查 scale 有限且大于 0，不在文档中编造固定数值。

embedding 文件的预期字节数为：

```text
vocab_size × hidden_size × 2
```

因为每个元素为 INT16。当前历史包中 embedding 约 534.77 MB，双 DLA 合计约 2292.52 MB，总计约 2827.29 MB；这是文件大小，采用历史报告中的十进制 MB，不是 RSS。不能仅用“2B×4 bit”估算整个部署包，因为这里还有两张阶段图、外置 embedding 和其他资源。

打包阶段还在主机上用 SDK 生成各位置的 mask 和 RoPE，按对应图的 scale / zero point 量化，保存为输入库。核心数值转换是：

```python
# values 是 SDK 生成的浮点 mask 或 RoPE,参数取自对应图的输入契约.
data = values / item["scale"] + item["zero_point"]
if list(data.shape) != item["shape"] or not np.isfinite(data).all():
    raise ValueError("SDK 输入形状或数值不符.")
if np.any((data < -32768.5) | (data > 32767.5)):
    raise ValueError("SDK 输入超出 INT16 可表示范围.")
encoded = np.clip(np.rint(data), -32768, 32767).astype("<i2")
```

这段是节选整理，重点是先检查再编码，而不是一律截断后假装合法。Prefill 与 Decode 分别使用自己的输入量化参数；重新量化后也要重新生成这些资源。

### 4.3 当前板端已经不需要交叉编译自有桥接库

当前执行路径为：

```text
Python：生成循环、embedding、mask/RoPE、KV、选词与评价
  ↓ ctypes，直接绑定 SDK 的 C ABI
libneuronusdk_runtime.mtk.so.8 或本地配置指定的兼容库
  ↓ NeuronRuntime API
DLA → NPU
```

所以原文“交叉编译 libneuron_bridge.so”的章节不再适用于当前实现。这里仍然依赖 SDK 原生动态库，但不再维护仓库自己的 C++ 推理程序。

当前 Runtime 做的核心操作按顺序为：

```text
ctypes.CDLL 加载 SDK
  → 设置函数参数类型与返回类型
  → NeuronRuntime_create
  → NeuronRuntime_loadNetworkFromFile
  → 查询输入输出数量及各自 padded size
  → NumPy 分配存储，建立逻辑视图
  → NeuronRuntime_setInput / setOutput
  → NeuronRuntime_inference
  → NeuronRuntime_release
```

SDK 返回非零错误码时直接失败。INT16 逻辑张量的 `元素数×2` 不一定等于 Runtime 要求的存储大小，因此不能只按逻辑 shape 分配后盲目绑定。

当前 `HardwareModel` 只针对已知 Decode mask 的尾部补齐放行，其余接口按预期大小严格约束；补齐区与逻辑数据视图区分别处理。若平台返回新的 stride / padding 规则，应先核实契约，不能任意放宽检查。

### 4.4 一次生成如何执行：把 Prefill、首 Token 和 Decode 分开

当前生成逻辑采用贪心 `argmax`，主干如下。为突出顺序，以下是教学伪代码；实际实现还包含资源校验、停止标记、流式显示、计时和异常释放：

```python
def generate_core(tokens, limit):
    """展示当前生成顺序,所有辅助操作均为语义占位."""
    check_token_range_and_context(tokens, limit)
    reset_all_inputs_to_quantized_zero()

    for offset in range(0, len(tokens), prefill):
        chunk = tokens[offset:offset + prefill]
        feed_embedding_mask_and_rope("prompt", offset, chunk)
        outputs = run_prompt_dla()
        if offset + prefill < len(tokens):
            append_prompt_kv(outputs)

    next_token = argmax(outputs.logits[0, len(chunk) - 1])
    seed_decode_with_valid_prompt_kv(len(chunk))

    generated = []
    for step in range(limit):
        generated.append(next_token)
        if is_stop_token(next_token) or step + 1 == limit:
            break
        feed_embedding_mask_and_rope("decode", len(tokens) + step, [next_token])
        outputs = run_decode_dla()
        next_token = argmax(outputs.logits[0, 0])
        append_decode_kv(outputs)
    return generated
```

拿 150 个输入 Token 举例：

1. Prefill 第一次处理 128 个 Token，位置从 0 开始，更新历史 KV。
2. 第二次处理余下 22 个有效 Token，图仍为 128 个位置，剩余位置填充。
3. 读取第二块第 21 号位置的 logits，预测第一个回答 Token。不能读取填充部分最后一个位置。
4. 转入 Decode 时，合并前块历史 KV 与最后一块的 22 个有效 KV，不能把 106 个填充位置带入有效历史。
5. 将第一个回答 Token 作为位置 150 的输入交给 Decode，预测第二个回答 Token。
6. 后续每次输入一个已选 Token，更新 KV 并向后移动一个位置。

因此，生成 N 个 Token 通常只需要 N−1 次 Decode 调用：第一个来自 Prefill。遇到停止 Token 时可能提前结束。计数差一会同时影响速度报告和逐步 logits 对齐。

当前实现要求 `len(input_tokens)+limit<=context`，不能先输入 900 Token、再无条件申请生成 512 Token。`max_new_tokens=512` 是上限配置，不是每个输入都能使用的独立额外容量。

### 4.5 KV 转移为什么也是精度问题

当前 KV 约定以末端为有效历史区，张量为四维，序列轴为 axis=2。每个 `past_*` 对应一个 `curr_*`，不能按输出次序猜测含义。

当前核心函数的执行顺序为：

```text
按名称找到 curr_* 输出与 past_* 输入
  → 检查四维 shape、head 等非序列维度一致
  → 追加时将历史窗口左移，腾出本次新增长度
  → scale / zero point 相同则复制原始 INT16
  → 不同则先反量化、再按目标尺度重量化
  → 写入缓存末端
```

实际重量化函数如下：

```python
def requantize(values, source, target):
    """按源与目标量化参数转换 INT16 张量,保留形状."""
    if (source["scale"] == target["scale"]
            and source["zero_point"] == target["zero_point"]):
        return values
    values = (values.astype(np.float32) - source["zero_point"]) * source["scale"]
    return np.clip(
        np.rint(values / target["scale"] + target["zero_point"]),
        -32768,
        32767,
    ).astype("<i2")
```

两个常见错误是：把整数 0 当成任意量化张量的实数零；把两个都是 INT16 的 KV 直接复制而不比较尺度。前者在 zero point 非零时错误，后者会改变 KV 的实数含义。

即使首 Token 正确，KV 更新错误也可能让后续越来越差。因此只检查 Prefill 输出、只算重置上下文的 PPL，不能证明 Decode 链路正确。

## 5. 怎样评价：把“感觉变差”变成可定位的问题

### 5.1 三类评价分别回答什么

| 评价 | 当前怎么做 | 不能代替什么 |
|---|---|---|
| 生成样例 | 官方聊天模板，贪心，检查文本与停止 | 不能代替完整精度 |
| MMLU | 0-shot Prompt，比较带前导空格的 A/B/C/D 单 Token 分数 | 不全面覆盖自由生成、多步 Decode 与长回答 |
| WikiText2 PPL | 真实文本分成 128 Token 块，每块重置上下文 | 不覆盖跨块长上下文及完整 Decode 生成 |

聊天、MMLU、PPL 不必套同一个模板。要统一的是**同一项评价的各后端输入与评分规则**。硬给 PPL 加聊天模板或把 MMLU 改成解析自由回答，会改变评价协议。

### 5.2 MMLU：最后一行 logits 怎样成为准确率

当前 Prompt 格式为：

```text
The following are multiple choice questions (with answers) about <subject>.

<question>
A. <choice A>
B. <choice B>
C. <choice C>
D. <choice D>
Answer:
```

候选并不是随便找词表里的 `A`：

```python
# continuation 含前导空格,并要求每个候选恰好为单 Token.
choices = [
    tokenizer.encode(" " + letter, add_special_tokens=False)
    for letter in "ABCD"
]
if any(len(ids) != 1 for ids in choices):
    raise ValueError("候选不是单 Token,需要采用序列评分.")
choice_token_ids = [ids[0] for ids in choices]
```

FP32 读取完整 Prompt 最后一个位置的四个候选分数；NPU 对完整 Prompt 分块 Prefill，读取最后一块最后有效位置：

```python
# NPU 原始输出先按同一个输出契约反量化,再记录四个候选分数.
raw = outputs[0][0, len(chunk) - 1, choice_token_ids]
scores = (raw.astype(np.float32) - spec["zero_point"]) * spec["scale"]
prediction = int(np.argmax(scores))
```

四个分数共享正 scale 和同一个 zero point 时，直接对它们取 argmax 与反量化后排序一致。但误差诊断和概率计算仍需使用正确实数尺度，不能据此省略所有契约处理。

准确率为 `正确题数/实际题数`。当前正式入口检查 14042 题、57 学科；完整 Prompt 超出 DLA 上下文会停止。不能删掉长题或截断后仍称全量成绩。

### 5.3 PPL：计算的是哪些位置，为什么是 127

对一块真实 Token `x0...x127`，模型在位置 0 的输出预测 x1，位置 126 的输出预测 x127。该块前 127 个输出参加评分，位置 127 的输出没有该块内的下一个真实标签，因此不计入这块 PPL。

```python
# 按当前块重置上下文的评分关系,省略设备和进度处理.
all_logits = model(ids, use_cache=False).logits.float()
logits = all_logits[:, :127]
loss_sum = torch.nn.functional.cross_entropy(
    logits.reshape(-1, logits.shape[-1]),
    ids[:, 1:128].reshape(-1),
    reduction="sum",
)
```

整体计算为：

```text
NLL = 所有参与评分位置的 -log p(真实下一个 Token)
PPL = exp(NLL / scored_tokens)
scored_tokens = 完整块数 × 127
```

当前协议用连续原文编码后取完整 128 Token 块，末尾不足一块的 Token 数单独记录；每块重置上下文。这是协议定义，不能把它说成跨整篇文章连续缓存的 PPL，更不能直接对比采用不同上下文窗口的外部成绩。

历史 58 块恰好有 `58×127=7366` 个评分 Token。正式全测试集则使用全部测试文本可组成的完整块，不能继续沿用“前 128 行”这一历史范围名称。

### 5.4 逐步诊断要补上最后一行与 Decode

PPL 没使用块末最后一行，但**生成首 Token 正好依赖该行**。所以当前 logits 诊断工具会额外保存和比较 Prefill 最后一行；不能因为前 127 个位置的 PPL 接近，就假定第 128 行也正确。

Decode 排错应采用 teacher forcing：每个后端都输入同一段真实 Token，位置一致，各自维护缓存，再比较每步输出。若让各后端自由 argmax 后继续生成，某一步选词不同之后，输入历史已经不同，后续 logits 差异就无法只归因于后端误差。

当前诊断工具的 Decode 模式是“同一真实续写 Token + 各后端自己的 Prefill KV”。它保留端到端缓存误差，适合检查完整链路；若要进一步隔离单个 Decode 算子的数值差异，还需要额外控制可兼容的输入 KV，当前默认正式评价没有执行这个隔离实验。

## 6. 精度不够时怎样一步一步查：本项目的诊断主线

### 6.1 先问：错误最早在哪个阶段出现

理想诊断链是四个检查点：

```text
A. 原框架 FP32
        ↓ 结构解释与浮点转换
B. SDK 浮点模型/浮点图
        ↓ 校准、量化、静态拼接
C. 量化静态 TFLite 软件执行
        ↓ NCC 编译、Runtime、板端数据管理
D. 板端 DLA 执行
```

| 比较 | 若这里首次明显变差 | 优先查什么 |
|---|---|---|
| A → B | 量化前已经不同 | 模型缩放、归一化、RoPE、Attention/KV Head 配置、模板与 Token |
| B → C | 浮点转换正常，量化后退化 | 权重近似、激活范围、校准代表性、分片尺度、embedding |
| C → D | 软件量化图正常，硬件输出退化 | 编译配方、算子实现、I/O 顺序、padding、KV 转移与尺度 |

**当前实现边界：** 仓库已有 A/C/D 的三后端 logits 比较工具；B 是进一步区分“结构转换”和“量化”的必要检查点，但本文未运行 SDK 浮点导出，也没有一组新的 A/B 对比结果。只有 A/C 差异时，应先称“转换与量化链路差异”，不能直接跳到“就是 4 bit 的错”。

**历史记录：** 适配说明已归纳“主要退化发生在量化阶段”，并确认后段 MLP INT4 近似损失与少量 A16 截断。这是保留的诊断结论，不是本次重新执行完整四点比较后的结论。

### 6.2 第一步：固定最小复现输入

建议保留一个很小但能区分问题的诊断集合，覆盖短问答、真实文本、跨 128 Token 的 Prompt，以及若干真实续写 Token。先看相同输入，避免一开始用完整基准淹没错误位置。

对齐至少包括：

```text
同一 Token ID 序列
同一有效长度与输出位置
同一 mask、RoPE 位置和历史范围
同一 embedding 实数含义
同一输出 scale / zero point 解释
同一图版本与缓存布局
```

短题正确、长题错误时，首先查多块 Prefill 与缓存；Prefill 正确、Decode 第一步错误时，首先查首 Token 后的位置、Prefill→Decode 的 KV 初始化与量化尺度。这些症状比先增加校准数据更有定位价值。

### 6.3 第二步：比较 logits，不只比较最终文字

当前比较工具计算余弦相似度、去均值余弦、RMSE、MAE、最大绝对误差、Top-1 一致率以及参考到候选的 KL。

| 观察量 | 怎样帮助判断 | 单独使用的局限 |
|---|---|---|
| RMSE / MAE | 反映整体数值偏差 | 数值量级不同会影响可比性 |
| 最大绝对误差 | 暴露少量极端位置 | 不能代表所有任务损失 |
| 原始余弦 | 看整体方向变化 | 公共偏置可能掩盖有意义的差异 |
| 去均值余弦 | 减少公共偏移影响 | 依然不是任务准确率 |
| Top-1 一致率 | 看最大分数 Token 是否改变 | 不能说明真实标签概率和四选一准确率 |
| KL | 看概率分布变化 | 对尺度解释和归一化非常敏感 |

MMLU 还应看四个候选的 margin。下面是假设数值，不是实际题目成绩：

```text
FP32 候选分数：A=3.10，B=3.08，C=1.00，D=0.50
量化候选分数：A=3.06，B=3.09，C=1.01，D=0.50
```

整体误差很小，但 A/B 排名已经翻转。因此“余弦很高”不能直接解释为“准确率不会掉”。反过来，整组 logits 同加一个常数可能使数值误差变大，但 softmax 概率与排序不变，需要结合去均值比较。

同时将 MMLU 变化分成：FP32 对→量化错、FP32 错→量化对、两者都错、两者都对。历史总体正确率差为 14.6 个百分点，不能据此推算究竟有多少题从对变错，因为也可能存在反向修正。

### 6.4 第三步：把误差缩小到层和投影

若量化链路是主要来源，下一步不能只说“INT4 精度低”，应定位哪个子模块最敏感。

建议在少量固定样本上观察以下边界：Attention Q/K/V/O、MLP 三个投影、残差相加、归一化以及 LM Head。不要默认导出所有样本所有层的巨大激活；先逐层汇总，保存最早突变点和必要重现张量。

逐层误差需要两种实验，二者回答不同问题：

| 实验 | 输入如何给 | 回答的问题 |
|---|---|---|
| 局部同输入对比 | 浮点与量化子层都输入同一个 X | 当前层自己引入多大误差 |
| 整链累计对比 | 量化层接收前面量化层的真实输出 | 到当前层时，前面积累及传播后的误差多大 |

如果只看整链误差，后层很大不代表后层本身最差；如果只看同输入误差，也可能遗漏层间分布漂移。历史配置记录中 `cumulative_layer_error=false`，不能把那轮实验描述成已经开启逐层累计误差优化。

当前仓库保留的是后段 MLP 敏感这一归纳，没有完整逐层数表；本节介绍的是复核与继续定位时的实验设计。

### 6.5 第四步：分离权重近似与激活截断

对一个敏感投影，建议进行以下受控比较：

```text
固定浮点输入 X：
    Y_float = X × Wᵀ
    Y_weight = X × W_hatᵀ
        → 观察权重近似引入的 ΔY

固定对应权重条件：
    比较输入/输出量化前后的数值与饱和位置
        → 观察 A16 表示及范围造成的额外损失

最后：
    放回真实量化整链，检查叠加后的误差与任务指标
```

这是诊断隔离实验，不是把部署模型永久改成高位宽。项目当前交付保持全 W4A16，不通过把敏感层提升到 W8/FP16 来绕过问题。

A16 截断的诊断至少应统计：

- 超出可表示范围的元素比例，按层、Token 和通道汇总。
- 被截断部分的幅度或能量，而不只看数量。
- 这些位置是否对应 logits、候选 margin 或真实标签概率的明显变化。

假设 100 万个元素里只有 10 个越界，但恰好在敏感通道且幅度很大，比例低也可能造成明显影响。反过来，单个激活越界不等于已经证明它是 MMLU 退化主因。

### 6.6 第五步：验证“修了局部”能否改善整模

历史记录明确：全局范围保护未采用，当前交付未因此被替换，主要质量差距仍未解决。正确解释是：不能从某层误差下降推断整模更好，未获整模评价支持的局部修复没有替代当前模型。

若一个范围调整使该层 RMSE 降低，应继续依次检查：

```text
敏感层同输入误差
  → 整链 logits 与 teacher-forcing
  → 固定开发集 MMLU/PPL
  → 板端同输入数值
  → 质量、速度、RSS 综合回归
```

不要在这一链条中途跳到“优化完成”。也不要为了追求本次诊断样本的分数，修改候选答案规则、截掉长题或把校准数据换成正式测试答案。

## 7. 怎样调整量化流程：每个优化项应该怎样做实验

### 7.1 先明确哪些地方已经有开关，哪些还需要实现

| 方向 | 当前已有能力 | 尚不能声称具备的能力 |
|---|---|---|
| 校准数据 | 默认 16 问题；可显式切换训练文本混合 | 自动覆盖业务分布、自动挑选最佳样本 |
| 校准批次 | 配置与环境覆盖 | 批次增加必然产生新内容或必然提升精度 |
| 权重优化 | `none/hessian/gradient`，可传额外配置文件 | 已核实所有 SDK 内部参数，或默认已优化 |
| 旋转 | Hadamard，seed=0 的配置分支 | 当前候选一定改善，或等同某篇论文完整实现 |
| 分片尺度 | INT16 单尺度接口的 Requantize | 任意 dtype、任意布局和逐通道接口自动适配 |
| 激活范围 | 已有 SDK 配方与额外选项入口 | 一个可直接使用且已验证的全局范围调优器 |
| Prefill / context | 支持转换时配置 | 当前所有评价代码支持任意 Prefill 长度 |

不清楚 SDK 内部字段时，先阅读该版本帮助与实现，再补充实验参数；不能在文档中编一个 `clip_ratio` 字段，让读者以为改 YAML 就会生效。

### 7.2 实验一：先提高校准代表性，再增加数量

**触发条件：** 输入和硬件链路正确；量化差异集中在默认短问答未覆盖的内容、长度或激活状态。

按下面顺序做，能够知道改善来自哪里：

1. 固定当前模型、量化配方、旋转状态、优化状态和诊断集，建立默认校准的基线。
2. 改为明确的训练/开发输入分布，例如保持样本预算不变，替换为业务语言、领域和输入长度更接近的内容。
3. 先看 SDK 实际消费的样本和有效长度，再比较量化 logits、PPL 与候选分数变化。
4. 若代表性改善有效，再单独增加样本覆盖或批次预算，观察收益是否稳定。
5. 候选值得保留时，再执行对应板端和正式评价。

当前提供的“WikiText2 前 2048 Token + 8 个问题”可以作为对照组，但它偏英文且只来自语料开头，不应被称为最终的通用校准方案。中文业务需要另外设计相应输入；这将涉及模型准备逻辑修改，本文只解释实验，不执行代码改动。

校准集合、调参开发集和最终测试集应分开。已有历史 MMLU/WikiText2 子集已经被反复观察，继续用它们可做回归，却不能再声称它们是从未用于方案选择的独立留出集。正式测试集不用于构造校准输入或针对答案调参。

**保留条件：** 在固定开发协议上有稳定改善，未引入输入兼容问题；最终还要有相同候选的板端与正式指标。只看到校准命令执行成功，不构成改善证据。

### 7.3 实验二：用 Hessian 类优化处理敏感权重

**触发条件：** 同一浮点输入下，敏感层 `X×W_hatᵀ` 相比 `X×Wᵀ` 已出现显著偏差，且误差会传播到任务输出。

权重优化要解决的不是单纯让 `W_hat` 接近 W，而是更重视实际输入上输出的接近程度：

```text
简单权重误差目标：min ||W - W_hat||²
输入感知输出目标：min ||X × (W - W_hat)ᵀ||²
```

在线性层重构中，`XᵀX` 一类二阶信息描述输入方向的重要程度及相关性。这解释了为什么两个绝对值差不多的权重量化误差，可能值得不同的补偿。

使用近似二阶信息做低位权重量化是 GPTQ 的核心思路之一，但 SDK 的 `hessian` 开关不能直接等同 GPTQ 的全部算法与超参数；这里的公式是理解输入感知重构的数学依据。[GPTQ 原论文](https://arxiv.org/abs/2210.17323)

当前开关示意为：

```yaml
native:
    weight_optimization: hessian
    weight_opt_config: ''
```

这会让 PTQ 多传 `--weight_optimization hessian`。**它不会自动复现历史的“16 样本 light_hessian_v8”。** 样本数、逐层配置和 SDK 内部默认值，必须由那一轮实际配置与该版本实现确定。

建议实验步骤：

1. 固定校准数据与旋转状态，先比较 `none` 和明确参数的 `hessian`，避免同时更换三项配方。
2. 保持全部 Attention / MLP / LM Head 为 INT4，接口仍为 INT16。
3. 对已定位的后段 MLP，比较同输入重构误差与整链输出；同时看其他层是否退化。
4. 在开发集做 MMLU/PPL 回归，检查是否只改善少量诊断位置。
5. 记录离线优化耗时、候选文件大小，板端确认计算图与速度是否发生变化。

历史记录支持的是“有一轮使用了 16 样本 Hessian 优化，得到 41.3% / 48.38 的固定子集结果”。它没有提供可比的无 Hessian 基线，所以不能写成“Hessian 将 MMLU 从某值提高到了 41.3%”。

`gradient` 也是当前允许的 SDK 选项，但没有足够结果支持它优于 Hessian。若要尝试，应作为单独候选核实该版本的重构行为，而不是扩展为新增训练或微调项目。

### 7.4 实验三：处理 A16 截断时，别只扩大所有范围

**触发条件：** 在敏感层捕获到超范围激活，并能看到它与输出误差的联系。

假设对称 INT16 用大约 ±32767 的范围表示浮点 ±3.2767，则 scale≈0.0001。某激活为 4.0 时会被截到上界。如果将范围扩为 ±6.5534，scale≈0.0002，可以覆盖 4.0，但正常值的量化步长也扩大到两倍。

这说明同时存在两个方向：

```text
扩大范围 → 减少大值截断 → 正常区间刻度变粗
缩小范围 → 正常区间刻度更细 → 更多大值截断
```

建议按层或已确认的张量边界试验：

1. 记录该位置的原范围、饱和数量、越界幅度、输出重构误差。
2. 在 SDK 当前支持的配置层面调整目标范围；先确认参数确实作用于目标张量。
3. 只改变该因素，比较局部误差和下游 logits。
4. 若改善，继续整模 MMLU/PPL 与板端回归；若只改变少数样本，则检查是否过拟合诊断集。
5. 连同新尺度重新导出、编译和打包，不让旧 I/O 契约留在新模型旁。

**历史记录：** 某些内部激活确有罕见截断，全局范围保护未被采用，主质量差距没有因此解决。不能把“有截断”进一步写成“只要范围放大就能恢复 FP32”。

### 7.5 实验四：旋转是重分配数值分布，需要先验证等价性

**触发条件：** 已观察到离群通道或分布不均，且 SDK 支持该模型对应的旋转路径。

对行向量输入 X、投影 W，设 R 为正交矩阵，数学上有：

```text
X' = X × R
W' = W × R
X' × W'ᵀ = X × R × Rᵀ × Wᵀ = X × Wᵀ
```

这是单个线性投影的等价关系。整套 Transformer 还包含残差、归一化、非线性、Attention、embedding 和 LM Head，不能把某个权重单独右乘 R，就宣称整模等价。

旋转用于改变数值分布、减轻离群影响的思路可参考 QuaRot；论文方案不是本项目已完整复现的功能，也不能用其其他模型成绩预测这里的收益。[QuaRot 原论文](https://arxiv.org/abs/2404.00456)

当前配置分支为：

```yaml
native:
    rotate: true
```

准备阶段会写入 `rotate_mode="hadamard"`、`rotate_seed=0`。建议比较流程为：

1. 先验证旋转后的浮点路径与原浮点路径足够一致；若这里出问题，暂停量化调参。
2. 固定输入、校准数据、权重优化设置和实际旋转矩阵。
3. 比较旋转开/关的分布、截断、重构误差和整模指标。
4. 再观察两张静态图、DLA 大小与板端速度是否变化。

只记录 seed 还不足以跨 SDK 版本证明矩阵相同；需固定生成规则，并在必要时核对实际 R1/R2。历史适配说明也明确要求旋转对比时固定实际矩阵与样本。

### 7.6 为什么实验顺序很重要

假设同时把校准批次从 32 改到 64、打开 Hessian、打开旋转并改变 mask，随后 MMLU 提高了 2 个百分点。即使这个提升真实，也无法知道哪个改动有用，下一轮就难以继续改进。

更有效的顺序是：

```text
先修输入/契约错误
  → 确认结构浮点路径
  → 校准代表性
  → 权重优化
  → 激活范围或旋转的独立对照
  → 最有希望的组合
  → 板端与最终评价
```

单因素筛选以后可以做有依据的组合实验。例如旋转改变了激活分布，Hessian 统计也可能随之变化，两者组合需重新校准并重新验证，不能简单把两轮独立收益相加。

正式评价入口不会替你改量化配方并重新编译。选定新候选后，应回到编译部署链生成配套资源，再用评价入口验证已有候选。

## 8. 怎样优化速度和内存：先拆开时间，再动代码和配方

### 8.1 先理解当前计时到底包括什么

当前 Python Runtime 的核心计时为：

```python
# setInput / setOutput 已完成后,仅计时 SDK 同步调用.
started = time.perf_counter()
self.call("inference", self.handle)
milliseconds = (time.perf_counter() - started) * 1000
```

这个时间是 Python 观察到的同步 SDK `inference` 调用耗时，包含该调用内部的运行时成本；没有硬件计数器证据时，不应称为严格的 NPU 核心纯计算时间。Python 侧绑定、embedding、mask 拷贝、KV 搬运和选词不都在其中。

生成器还记录两个外层区间：

| 计时 | 当前覆盖 | 解释时注意 |
|---|---|---|
| `prefill_seconds` | 从 Prefill 循环开始，到所有 Prompt 块执行和中间 KV 转移完成 | 不含模型加载；结束点在首 Token argmax 与 `seed_decode` 之前 |
| `prefill_npu_ms` | Prefill 各次同步 SDK 调用耗时之和 | 不含 Python 准备与缓存操作 |
| `decode_seconds` | 生成循环，包括每次 Decode 输入准备、调用、argmax 和 KV 更新 | 若流式显示开启，也会带入文本处理，因此独立测速关闭显示 |
| `decode_npu_ms` | Decode 各次同步 SDK 调用耗时之和 | 不是整段 Decode 的端到端延迟 |
| `elapsed_seconds` | 从 Prefill 开始到生成记录附近的总时长 | 排除加载，也不是从用户点击开始的完整延迟 |

当前 `prefill_tokens_per_second = 输入Token数 / prefill_seconds`；`decode_tokens_per_second = 实际Decode调用数 / decode_seconds`。需要第一 Token 延迟 TTFT 时，应另外计入定义范围内的分词、输入准备、排队、Prefill、首 Token 选择及显示成本，不能直接把 Prefill Token/s 当 TTFT。

### 8.2 先测同一个工作负载，再讨论谁更快

当前性能流程使用 128 个真实文本 Token，先独立预热一次，再测一次最多生成 128 Token，测量时关闭流式输出。停止 Token 仍可能提前终止，因此“最多 128”不等于每次必然生成 128。

要做可信的候选比较，应固定：

```text
同一板卡与 Runtime / 固件
同一输入 Token 与上下文容量
同一生成策略、生成计数与停止行为
同一预热和加载范围
同一计时边界
尽可能一致的温度、频率、后台负载
```

当前实现只有一次预热、一次测量，足以给出该次结果，尚不能描述波动分布。需要稳定性能结论时，另行设计重复测量并报告中位数等统计；本文没有执行这些耗时任务。

短输入真实只有 25 Token，但图仍一次执行 128 个位置，用有效 Token 数除以几乎相同的图执行时间，得到的 Token/s 会低于 128 Token 输入。历史短样例约 54～59 Token/s 与定长约 289 Token/s 可以同时存在，不能据此断言短样例跑得更慢。

### 8.3 从时间差判断先查哪一层

先观察以下差额，作为定位线索：

```text
Prefill 外层时间 - Prefill 同步 SDK 调用合计
Decode 外层时间 - Decode 同步 SDK 调用合计
```

差额大时，优先拆 Python 的输入准备、缓存复制、重量化和词表 argmax；SDK 调用本身占主要时间时，再研究编译配方、图形状和执行配置。差额只是成本范围，不是自动证明某段代码就是瓶颈。

**当前代码中值得观察的具体位置：**

1. 每个 Prefill 块和 Decode 步都调用 NumPy 填充或拷贝输入。
2. KV 追加使用切片复制移动历史，再写新 KV；它不是零拷贝循环缓存。
3. KV 尺度不同时会转换成 FP32 临时数组，再重量化为 INT16。
4. Prefill→Decode 会合并历史与末块有效 KV，并进行必要的重新编码。
5. 每次 `inference` 前重新绑定所有输入输出缓冲区。
6. 每步从整词表 logits 取 argmax；流式展示还涉及文本解码，但独立测速关闭该项。

这些都只是可以测量的候选开销，本文未做 profiler，不能凭阅读源码宣布它们各占多少比例。

### 8.4 如果 Python 侧 KV 搬运占比高，怎样改才有依据

建议先统计每步 KV 实际读写字节数和耗时，再尝试：

- 复用输入、输出和重量化临时缓冲区，减少频繁分配。
- 避免重复处理无效缓存位置；有效长度仍必须符合 SDK 布局。
- 对量化参数确实相同的接口保留直接复制路径。
- 若 SDK 和图接口支持，再研究更少数据移动的缓存布局；不要在不改 mask / RoPE 解释的情况下私自改成环形缓存。

注意：当前 dtype、尺度和后缀有效布局是正确性契约。为了减少一次转换而强行改两边 scale，或者为了省复制而乱接旧缓冲区，可能把速度优化变成新的精度错误。

验收应同时看同输入 KV/logits、teacher-forcing、多块 Prefill、MMLU/PPL 和速度。只测一个短 Prompt，看输出“差不多”，无法覆盖缓存问题。

### 8.5 如果 SDK 调用占比高，怎样改形状和编译配方

Prefill 批长影响一次图的计算规模及调用次数：

```text
短 Prompt：较小批长可能减少填充计算，但实际收益由编译结果决定。
长 Prompt：较大批长可能减少调用与缓存搬运，但单次延迟和内存可能增加。
Decode：每步仍是 1 Token，不能用 Prefill 的收益直接推断 Decode 收益。
```

当前配置能改 Prefill，但评价准备函数明确要求 `prefill=128`，且 PPL 和多块输入准备多处按 128 编写。因此不能建议“把配置改到 64/256 就完成实验”。要改批长，需要同时规划评价协议、输入库、缓存契约和公平比较方式，属于代码适配工作。

context 同样有取舍：增大容量可以容纳更长题目，但可能提高 KV、mask 输入库和运行开销；减小容量可能降低开销，却会减少可评价或可服务的输入范围。因缩短上下文而无法完成相同任务的候选，不能只比较更高的 Token/s。

NCC 选项的对照应固定量化静态图，一次只改已核实支持的选项，编译后重新比较 TFLite / DLA 输出和板端耗时。保留 `--disallow-bridge` 的执行约束，不通过混入 CPU 计算制造看起来能跑的结果。

### 8.6 内存要分清文件、进程与运行时存储

当前 INT16 KV 的粗略逻辑容量可以表示为：

```text
2 × L × B × Hkv × C × Dhead × 2 bytes
│   │   │    │    │     │
K/V 层数 batch KV头 缓存长 每头维度
```

这是采用缓存长度 C 的一份逻辑 K/V 容量估计。实际要按每张图的 `past_*` / `curr_*` shape 计算，可能还有 Prefill 与 Decode 两套输入输出、Runtime 对齐缓冲区、FP32 临时数组和驱动分配。GQA 模型应使用 KV Head 数，不能把 Query Head 数直接代入。

还有几个容易漏掉的内存来源：

- embedding 的文件映射与实际驻留页面。
- Prefill 输出完整词表 logits，逻辑大小约为 `B×T×V×2 bytes`，另有 KV 输出。
- 为全部位置预生成的 mask / RoPE 输入库，按数组实际 shape 和元素大小求和；context 增大时应重新核算。
- 两个 Runtime 实例及其工作区。
- Python、NumPy、tokenizer 以及评价期间的额外对象。

当前 Python RSS 读取 `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`，在目标 Linux 以 KiB 换算 MiB。它是当前进程的历史高水位，不是某一步新增内存；生成进程与 MMLU/PPL 评分进程也要说明各自范围。

历史 1152.42 MiB 属于原生 CLI，不能沿用为 Python 数值；系统 `MemAvailable` 的下降也不能当成进程 RSS 或全部 NPU 占用。量化后的 DLA 文件变小，同样不保证 RSS 按相同比例下降。

### 8.7 精度方案与速度方案怎样一起选择

用同一开发协议比较候选时，建议把结果放在一张临时工作表中，而不是每个指标单独出一份正式报告：

| 候选 | 改动 | MMLU/PPL | Prefill / Decode | RSS | 决策 |
|---|---|---|---|---|---|
| 基线 | 固定配方 | 填同协议实测 | 填同边界实测 | 填同进程范围 | 作为对照 |
| 候选 A | 只改校准覆盖 | 待测 | 待测 | 待测 | 没有结果前不下结论 |
| 候选 B | 固定数据，改权重优化 | 待测 | 待测 | 待测 | 同上 |
| 候选 C | 固定图，改编译选项 | 待测 | 待测 | 待测 | 同上 |

这张表是实验设计示意，不含本次实测。若某方案离线量化更慢，但板端质量更好且运行速度不变，这仍可能值得采用；若 Decode 更快但 MMLU 明显下降，则需要对照用户的质量要求，不能单凭一个指标选“最好”。

## 9. 把下一轮工作排成可执行顺序

### 9.1 当前最优先解决的事情

**当前代码已迁移而板端尚未验证。** 下一轮首先要确认 Python 的生成、KV 和评价与已有 DLA 契约匹配，再重建当前实现的精度、性能和 RSS 基线。否则同时更换量化配方和运行实现，出现差异时很难定位。

旧历史结果仍可提供问题线索，但不能直接当作 Python 运行基线；当前默认无权重优化的转换也不能冒充旧 Hessian 候选。

### 9.2 推荐执行顺序与每一步的继续条件

下表是下一轮获准执行后的工作安排，本次没有执行其中的模型任务：

| 步骤 | 做什么 | 看什么证据 | 什么时候停下来处理问题 |
|---|---|---|---|
| 1 | 固定现有 DLA、配置、契约和运行实现 | 实际生效配方一致 | 资源版本不匹配时先处理配套关系 |
| 2 | 核对 tokenizer、模板、BOS/EOS、候选 Token | 同输入 ID 与停止规则 | Token 不一致时不进入量化归因 |
| 3 | Python 板端做少量短输入和跨块输入 | 真实文本、首 Token、KV、停止原因 | 短输入/跨块/Decode 任一不一致先定位 |
| 4 | 固定诊断样本做 FP32/SDK浮点/TFLite/DLA 对比 | 找到最早出现差异的检查点 | 结构或接口错误不靠调校准掩盖 |
| 5 | 分层看后段 MLP 与 A16 范围 | 同输入重构与累计传播 | 只有现象、没有隔离证据时不认定根因 |
| 6 | 用开发集单因素比较校准、权重优化、旋转或范围 | 每项变化对应可解释结果 | 局部改善但整模回退则不采用 |
| 7 | 对选定候选编译部署并重新评价 | 同一候选的板端精度与有效覆盖 | 超上下文或资源不足时报告阻塞，不删题 |
| 8 | 固定质量和负载拆解性能 | 同步 SDK 与外层耗时、实际调用数、RSS | 先测开销，再决定改运行器还是 NCC |
| 9 | 最终回归与发布 | 完整范围、同协议参考、质量及效率 | 未完成项写明，不借用其他候选数字 |

这个顺序并不要求每一个无希望候选都跑全量评价。可以用固定开发集和小范围数值诊断先排除明显退化方案；正式交付才需要对最终候选完成约定范围，且准确标记覆盖情况。

### 9.3 每次实验记录到什么程度才够用

一条有用记录应能回答：

```text
问题：哪类样本或哪一层出了什么差异？
假设：认为原因是什么，有什么证据？
改动：这次只改了哪些生效参数？
控制：哪些权重、输入、矩阵、协议和图保持一致？
结果：局部误差、整模质量、板端行为和效率分别怎样？
决策：保留、继续隔离还是放弃；尚缺什么证据？
```

过程只保留能复现、评分和排错的数据，不重复导出多种同义格式，不默认保存所有巨大激活。候选实验放日期临时目录，最终对外正式结果仍集中到模型唯一 `results/summary.json`。

没有保留下来的历史过程就写“缺少记录”。不能把当前建议实验倒写为过去已经完成的实验，更不能补造调参前后的分数。

## 10. 交付边界、原文修正与资料说明

### 10.1 当前两个入口的实际边界

| 入口与执行位置 | 负责内容 | 不应误解为 |
|---|---|---|
| 容器 `deploy/compile.sh` | 准备、校准、量化、导出、编译、打包、上传 | 上传完成不等于板端通过 |
| 板端 `compile.sh` | 少量通路检查，流式生成文本并保存示例 | 不计算正式精度、性能和 RSS |
| 容器 `deploy/evaluate.sh` | 用已有模型准备 FP32 参考与完整评价输入 | 不重新量化、编译或自动上传 |
| 板端 `evaluate.sh` | 使用已有部署包运行生成、独立性能测量、MMLU/PPL 与 RSS，写统一报告 | 不自动修配方、补编译或混用旧成绩 |

容器编译完成后，由用户登录开发板运行板端入口。完整评价资源需单独放入已有部署包对应位置；缺资源、上下文不足或协议不匹配时停止，并报告缺失项。

当前板端示例最多生成 32 Token，可能只是回答片段。虽然 YAML 保留小范围 MMLU=16 / WikiText2=4 块设置，板端 `compile.sh` 已经只做生成示例，不会自动进行原文描述的“生成、性能、小批量精度”三项检查。

### 10.2 原文需要修正的关键说法

| 原文表达 | 本文修正 |
|---|---|
| 关键细节让读者去配置或某个代码行查看 | 将配置、函数、参数、数值关系和排错顺序直接写入正文 |
| 量化主要讲舍入与截断原理 | 补充输入记录格式、SDK 调用、优化开关、分片检查、尺度连接及诊断闭环 |
| 正式参考像是编译流程中的固定步骤 | 明确当前参考属于独立评价入口，诊断时仍应先建立浮点参考 |
| 需要交叉编译自己的桥接库 | 当前为 Python ctypes 直接绑定 SDK C ABI |
| 日常生成走原生 CLI | 当前由 Python 管理 Prefill、Decode、KV 与 greedy 生成 |
| 默认检查包含性能和小批量精度 | 当前编译阶段只做少量生成检查；指标属于独立评价 |
| 历史成绩容易被理解成当前默认结果 | 明确历史 Hessian 候选与默认 `none` 配方不同，旧实现与 Python 也不同 |
| 流程结束于“能生成” | 延伸到误差定位、候选对照、质量回归与速度/内存取舍 |

### 10.3 本文事实与方法的来源

本地事实核对范围包括 MiniCPM5-2B 的当前配置、转换/编译/打包/评价实现，公共量化适配、生成、KV 转移、三后端 logits 比较与 Runtime 实现；历史实测来自模型保留的统一汇总，以及 Git 中迁移前归档的功能证据。这里列出核对范围，不把理解正文依赖转回源码。

官方模型固定版本的资源入口为 [MiniCPM5-2B 固定 revision](https://huggingface.co/openbmb/MiniCPM5-2B/tree/f97400052a43d642bbc6e9975e2397e3ae6a6b52)。模型来源、版本和 SDK 2.7.5 / Toolkit 2.0.8 信息依据本地资源记录；本次在线工具未能读取该固定版本的配置与模板，因此未把远端内容当作新核实证据，也没有下载模型。

GPTQ 与 QuaRot 的链接仅用于本文相应优化思路的来源说明。当前没有重新审计安装中的 SDK PTQ 内部源码，因此没有将 SDK 的 `Overall`、`hessian` 或 `rotate` 标签扩大解释为已确认的完整算法实现。

读完本文后，判断下一步不应只问“改哪个参数”，而应能说明：**差异首先出现在哪里、这个参数为什么可能影响它、怎样保持对照公平，以及什么结果足以决定保留这次修改。**


