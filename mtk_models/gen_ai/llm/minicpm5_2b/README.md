# MiniCPM5-2B

面向 MediaTek Genio 720 / MT8189 的文本生成模型,使用官方 [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B),许可证 Apache-2.0.来源见 [source_url.txt](models/source_url.txt),完整规格和评价边界见 [模型卡](model_card.md),资源布局见 [模型资源](models/README.md).

| 项目 | 当前配置 |
|---|---|
| 结构 | 42 层,hidden size=2048,Query/KV heads=16/2 |
| 量化 | 全 W4A16,包含 LM Head;外置 embedding 为 INT16 |
| 配方 | 16 批次 Hessian 重构 + Hadamard 旋转 |
| 上下文 / Prefill | 1024 / 128 Token,Decode 每步 1 Token |
| 后端 | Python 直接调用 MTK Neuron Runtime,MDLA 5.3 |
| 生成 | Greedy,关闭 thinking,停止 Token 为 1/130073 |

配方对照和适配过程见 [ADAPTATION.md](ADAPTATION.md)。

## 使用

参数和离线资源位置由 [config.yaml](config.yaml) 及本地平台配置管理,运行时不自动下载.在已配置的编译容器中,进入本模型目录执行:

```bash
bash deploy/compile.sh
```

默认核对并复用 `build/native_hessian16/` 的已选产物,不重新量化或编译;缺少固化资源时停止.随后打包、上传并准备固定小子集参考,服务器不自动执行板端推理.登录开发板后在部署目录执行:

```bash
bash compile.sh
```

板端检查少量输入,显示输入、实际聊天模板及流式文本,示例最多生成 32 Token,可能仅为回答片段;输出位于 `examples/output/`.compile 在两端对相同子集评分,板端将任务精度、精度损失及 CosSim/RMSE/NRMSE 写入 `results/summary_test.json`;总入口完成后自动回传报告和示例.[板端生成示例](examples/output/sample_1.txt)。

`deploy/evaluate.sh` 是独立正式评价入口,不重新转换或编译,结果只写入 `results/summary.json`.正式 MMLU 为固定 1000 题、57 学科、seed=17,WikiText2 为完整 test。使用与板端目录规范见 [Genio 720 文档](../../../../docs/genio_720.md).

## G720 实测结果

正式评价覆盖 MMLU 固定千题、完整 WikiText2 test。主指标计分 1,000 题，运行编号 `20261104_072603_2241265`。[正式报告](results/summary.json)

| 指标 | 浮点参考 | G720 | 损失 |
| --- | ---: | ---: | ---: | ---: |
| MMLU Accuracy | 55.900% | 41.300% | +14.600 个百分点 |
| WikiText2 PPL（2,250 块，285,750 个计分 Token） | 37.626 | 45.103 | +7.477（+19.87%） |

板端性能为 Prefill 284.55 Token/s，Decode 6.65 Token/s，Python 进程峰值 RSS 为 1254.62 MiB。
MMLU 使用固定 1000 题、57 学科、seed=17。性能使用 128 Token 输入和实际生成的 128 Token。

编译子集主指标计分 100 题，运行编号 `20261030_091512_1750273`。[子集报告](results/summary_test.json)

| 指标 | 浮点参考 | G720 | 损失 |
| --- | ---: | ---: | ---: | ---: |
| MMLU Accuracy | 53.000% | 34.000% | +19.000 个百分点 |
| WikiText2 PPL（16 块，2,032 个计分 Token） | 37.936 | 46.546 | +8.610（+22.70%） |

损失为正表示退化。子集报告中的输出误差仅覆盖下表所列调用。

| 输出 | 调用次数 | CosSim | RMSE | NRMSE |
| --- | ---: | ---: | ---: | ---: |
| `output` | 1 | 0.709780 | 2.26639 | 76.369% |

## DLA 文件

- `1024c/MiniCPM5-2B_asym4W_sym16A_42layer_prefill_128t_1024c.dla`
- `1024c/MiniCPM5-2B_asym4W_sym16A_42layer_decode_1t_1024c.dla`

实际加载路径由 `manifest.json` 的 `model_files` 指定.两个 DLA、embedding、分词器和张量契约必须配套,不能混用不同配方的资源.

## 量化参数与编译命令

参数来自 [config.yaml](config.yaml)、[转换实现](deploy/host/convert.py) 和 [编译实现](deploy/host/compile.py).实际配方记录保存在模型资源的 `build/native_hessian16/quantization_settings.json`.

### 量化设置

| 参数 | 当前设置 |
| --- | --- |
| 当前执行策略 | `reuse_compiled=true`,校验并复用 `build/native_hessian16/` 中已选产物;正常 compile 不重新校准、量化或编译 |
| 精度 | Attention、MLP、LM Head 均为 `asym4W_sym16A`,外置 embedding 与 KV 接口为 INT16 |
| 校准生成配置 | `calibration_batches=128`,SDK 命令参数 `-b 128 -m 128` |
| 已选 Hessian 配方 | `selected_recipe=light_hessian16_rotated_w4a16`,从已有批次中按 seed=17 选择 16 个批次,奇偶批次各 8 个,覆盖 42 层及 LM Head;该 16 批指 Hessian 重构输入,与 MMLU 题数独立 |
| PTQ 参数 | `-p asym4W_sym16A -tp asym4W_sym16A -m Overall`,Hessian 权重重构;当前配方参数从 `quantization_settings.json` 核对 |
| 旋转与掩码 | `rotate=true`,`rotate_mode=hadamard`,`rotate_seed=0`,`mask_value=-100` |
| 静态形状 | `context=1024`,`prefill=128`,Decode=1 Token,每阶段 1 张静态图 |
| 新转换配置边界 | 当前配置中的 `weight_opt_config` 为空;已选配方由资源记录和校验固定,不能把禁用复用当成可直接重建相同配方的开关 |

本模型使用 SDK 原生分层量化与静态 TFLite 路线,不经 ONNX.转换副本仅兼容 BPE merges 序列化和 SDK 配置,官方资源保持不变.分片合并对不同 INT16 单尺度接口插入显式 Requantize,不能通过改 dtype 解释内存.

下列命令完整列出当前 [compile.py](deploy/host/compile.py) 中的 NCC 配方,用于明确重编译已有静态图时审阅.默认入口复用已选 DLA,不会执行这些命令.当前选定配方与校准批次信息已按资源中的 `quantization_settings.json` 核对,不将配置默认值冒充新的精度实测.

### 完整 NCC 编译命令

在仓库根目录的已配置编译容器中,只编译已有 TFLite.原生 I/O 形状和量化尺度以静态图及部署契约为准:

```bash
source tools/storage.sh mtk_models/gen_ai/llm/minicpm5_2b
WORK="${MODEL_ASSETS_DIR}/build/native_hessian16"
PREFIX="MiniCPM5-2B_asym4W_sym16A_Overall_hessian_rotate_hadamard_0"
PREFILL_DIR="${WORK}/tflite/${PREFIX}_128t1024c"
DECODE_DIR="${WORK}/tflite/${PREFIX}_1t1024c"

# Prefill 阶段: 每块 128 Token.
env LD_LIBRARY_PATH="${NCC_ROOT}/lib" \
    "${NCC_ROOT}/bin/ncc-tflite" \
    --arch=mdla5.3 --l1-size-kb=256 --num-mdla=1 \
    --opt=3 --opt-footprint --opt-accuracy --stable-linearize \
    --gno=LTS,Inception --gno-exp --gno-non-4d-tiling --mlo \
    --disable-apusys --fc-to-conv --mdla-int16-lut \
    --suppress-input --suppress-output --show-memory-summary \
    --mdla-conv-exp --intval-color-legacy --disallow-bridge \
    --broadcast-act-wgt --broadcast-flow-distance=63 --split-large-conv-ic=1536 \
    "${PREFILL_DIR}/${PREFIX}_42layer_128t1024c_0.tflite" \
    --dla-file "${PREFILL_DIR}/MiniCPM5-2B_asym4W_sym16A_42layer_prefill_128t_1024c.dla"

# Decode: 每次 1 Token.
env LD_LIBRARY_PATH="${NCC_ROOT}/lib" \
    "${NCC_ROOT}/bin/ncc-tflite" \
    --arch=mdla5.3 --l1-size-kb=256 --num-mdla=1 \
    --opt=3 --opt-footprint --opt-accuracy --stable-linearize \
    --gno=LTS,Inception --gno-exp --gno-non-4d-tiling --mlo \
    --disable-apusys --fc-to-conv --mdla-int16-lut \
    --suppress-input --suppress-output --show-memory-summary \
    --mdla-conv-exp --intval-color-legacy --disallow-bridge \
    --split-16a4w-conv-oc \
    "${DECODE_DIR}/${PREFIX}_42layer_1t1024c_0.tflite" \
    --dla-file "${DECODE_DIR}/MiniCPM5-2B_asym4W_sym16A_42layer_decode_1t_1024c.dla"
```

编译输出仍在转换工作目录,后续打包将两张 DLA 放入模型资源 `models/1024c/`,板端同样从 `models/1024c/` 加载.底层命令不包含重新准备、量化或评价步骤.
