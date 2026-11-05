# RTMPose Body2d

人体 WholeBody 姿态估计，输入为 256×192 RGB 人体裁剪图，输出 133 个关节点。正式上游为 OpenMMLab MMPose v1.3.2；来源见[模型卡](model_card.md)。

量化方式: INT8 训练后量化 (PTQ),采用逐输出通道权重量化.

## 执行范围

先在已有编译容器执行 `bash deploy/compile.sh`,完成转换、量化、编译与上传后结束.随后登录开发板,在部署目录执行 `bash compile.sh`,做少量通路检查并保存 README 示例到 `examples/output/`.该阶段计算并打印子集任务精度,不发布正式报告,不统计性能和峰值 RSS.

正式评价使用独立的 `evaluate.sh`: 容器端只计算已有浮点模型的同协议参考及必要评价输入,资源单独放入已有部署包后,板端运行 `bash evaluate.sh`,统计精度、性能和推理进程峰值 RSS.正式结果只写入 `results/summary.json`,不重新量化或编译.板端预测、日志及内部指标统一位于本模型部署目录的 build/work/,每次开始推理时覆盖上次中间结果.详见[板端规范](../../../../docs/genio_720.md).

下文已归档结果保留原始模型版本和评价范围,本次入口调整不代表重新实测.

## 第一步: 编译并上传

模型、数据集和部署路径见 [config.yaml](config.yaml),工具链与 NAS 根目录按[仓库说明](../../../../README.md)配置.

在编译主机的本模型目录运行:

```bash
bash deploy/compile.sh
```

入口在当前容器完成量化、DLA 编译和上传。

## 第二步: 独立评价

上传完成后,请登录开发板,在部署目录执行 `bash compile.sh` 完成通路检查和示例生成.正式评价前,按[板端规范](../../../../docs/genio_720.md)准备同协议参考及必要输入,在已部署目录执行:

```bash
bash evaluate.sh
```

精度、性能和峰值 RSS 统一保存到 results/summary.json,板端过程数据位于本模型部署目录的 build/work/,每次开始推理时覆盖,不重新量化或编译.

## 数据与精度评测

参考评价使用 `config.yaml` 指定的数据集,与板端保持相同样本、标注和预后处理协议.

数据目录要求: COCO WholeBody: images/、annotations/coco_wholebody_val_v1.0.json 和 person_detection_results/COCO_val2017_detections_AP_H_56_person.json.

人体检测文件使用官方标准 JSON 数组.板端 Python 直接读取,无需修改数据文件; 检测框编号和排序保持不变.

精度变化以百分点表示,正数为改善,负数为下降.

Docker 需要 ONNX Runtime、NumPy、OpenCV、tqdm 和 xtcocotools。
板端需要 NumPy、OpenCV 和 xtcocotools,用于可视化和计算 WholeBody AP.

## G720 实测结果

正式评价覆盖 COCO-WholeBody val2017。主指标计分 104,125 个人体框，运行编号 `20261103_080139_1918290`。[正式报告](results/summary.json)

| 指标 | 浮点参考 | G720 | 损失 |
| --- | ---: | ---: | ---: | ---: |
| WholeBody AP | 57.037% | 53.221% | +3.815 个百分点 |

板端性能为 NPU 平均 3.892 ms，P95 4.051 ms，Python 进程峰值 RSS 为 163.08 MiB。

编译子集主指标计分 527 个人体框，运行编号 `20261030_080341_1720309`。[子集报告](results/summary_test.json)

| 指标 | 浮点参考 | G720 | 损失 |
| --- | ---: | ---: | ---: | ---: |
| WholeBody AP | 54.004% | 50.055% | +3.948 个百分点 |

损失为正表示退化。子集报告中的输出误差仅覆盖下表所列调用。

| 输出 | 调用次数 | CosSim | RMSE | NRMSE |
| --- | ---: | ---: | ---: | ---: |
| `pred_x` | 5 | 0.980235 | 0.0480014 | 20.114% |
| `pred_y` | 5 | 0.979896 | 0.046601 | 20.493% |

## 效果示例

原图叠加板端 WholeBody 关节点和人体、手部骨架.

输入已放在 `examples/input/`,编译时从配置的数据集自动生成来源与样本清单,随部署上传,不纳入 Git.

compile.sh 复用固定少量样本的板端预测,生成展示文件到部署目录的 `examples/output/`,不依赖全量评价.
以下输出来自本次全量测试的板端预测,展示 WholeBody 133 个关节点.示例目录仅保留三组输入和输出图片:

```text
examples/
├── input/
│   ├── sample_1.jpg
│   ├── sample_2.jpg
│   └── sample_3.jpg
└── output/
    ├── sample_1_keypoints.jpg
    ├── sample_2_keypoints.jpg
    └── sample_3_keypoints.jpg
```

### 示例 1: 骑车姿态

![骑车姿态输入](examples/input/sample_1.jpg)

![骑车姿态 WholeBody 输出](examples/output/sample_1_keypoints.jpg)

### 示例 2: 站立姿态

![站立姿态输入](examples/input/sample_2.jpg)

![站立姿态 WholeBody 输出](examples/output/sample_2_keypoints.jpg)

### 示例 3: 街头多人

![街头多人输入](examples/input/sample_3.jpg)

![街头多人 WholeBody 输出](examples/output/sample_3_keypoints.jpg)

## 板端部署结构

第一步上传到 `BOARD_DEPLOY_DIR` 后的布局统一为:

```text
部署目录/
├── compile.sh
├── evaluate.sh          # 独立精度、性能与 RSS 评价.
├── config.json          # 不含连接地址的模型配置.
├── models/              # 模型与推理所需参数.
├── board/               # 板端程序、评测代码及必要依赖.
├── examples/            # 少量输入、来源清单和展示输出.
└── results/summary.json  # 唯一正式评价报告.
```

评价结果只保存在部署目录的 `results/summary.json`.板端预测、日志和中间指标统一保存在本模型部署目录的 `build/work/`,每次开始推理时覆盖;展示示例保存在 `examples/output/`.取回唯一报告后据实更新 README.
## 文件结构

- `deploy/compile.sh`: 编译部署与展示示例入口;精度、性能和 RSS 使用独立的 `deploy/evaluate.sh`.
- `deploy/host/`: 编译主机使用的导出、转换与辅助工具.
- `deploy/board/`: 板端程序源码、预处理和评测代码.
- `models/`: 原始权重、转换产物与来源说明; 必要的上游源码放在 `models/upstream/`.
- `examples/`: 少量固定输入与实际板端效果输出.
- `results/summary.json`: 上传后的最新测试汇总.

示例图片来源: 示例 1: [原图](http://farm7.staticflickr.com/6199/6207968507_effbfb5a1f_z.jpg) / [图片许可](http://creativecommons.org/licenses/by/2.0/); 示例 2: [原图](http://farm6.staticflickr.com/5328/8789640387_14e10562cb_z.jpg) / [图片许可](http://creativecommons.org/licenses/by/2.0/); 示例 3: [原图](http://farm4.staticflickr.com/3446/3232237447_13d84bd0a1_z.jpg) / [图片许可](http://creativecommons.org/licenses/by/2.0/).

### 输出误差统计

compile.sh 独立对固定少量输入比较浮点与 NPU 输出,在终端显示 CosSim、RMSE、NRMSE 和实际覆盖范围.普通模型最多选择 5 次输入,不减少原有 compile 通路检查样本数;LLM 比较示例首个 Prefill 块的有效位置,BEVFusion 比较单帧的 NPU 子图.这些结果用于数值检查,不能代替任务精度.全量 evaluate.sh 只计算任务精度、性能和 RSS,不保存或比较全量浮点张量.

参考为普通 NPZ 数组文件,保存在编译机器本地日期临时目录并上传板端 build/reference/output_reference.npz,由下次 compile 覆盖,不使用数据库,不写入 NAS.板端中间结果仍使用本模型 build/work/.NRMSE 定义为 100 × RMSE / RMS(浮点参考),多输出分别统计.详见[输出误差统计](../../../../docs/output_metrics.md).

### compile 子集精度检查

compile.sh 使用 至少 500 个人体实例,保留选中图片的全部人体框,同时打印实际图片数和实例数,复用正式评价协议.主机执行子集浮点推理,板端执行同一子集 NPU 推理,在终端打印精度和损失,并保存 results/summary_test.json.正的损失表示 NPU 变差,负值表示改善;Top-1、AP、AR 等使用浮点减 NPU 的百分点,WER 使用 NPU 减浮点的百分点,PPL 使用增量及相对增幅.

普通样本跨完整划分按固定间隔抽取,保留展示及数值检查的必要输入;LFW、ESC-50 和 MMLU 保持折、类别和学科覆盖.同一子集两端通过样本指纹核对;MMLU 固定 seed=17 分层抽样.该报告不代表全量成绩,不改写 results/summary.json.中间数据覆盖保存在本模型 build/work/,CosSim、RMSE、NRMSE 仍只比较固定少量输入.详见[输出误差统计](../../../../docs/output_metrics.md).

## 量化参数与编译命令

以下为当前 [config.yaml](config.yaml)、[量化实现](deploy/host/convert_int8.py) 和 [编译实现](deploy/host/compile_model.py) 中的配置.未显式覆盖的 Converter 属性沿用当前 SDK 默认值,实际 scale/zero point 由校准产物读取,不能写死为通用常量.

### 量化设置

| 参数 | 当前设置 |
| --- | --- |
| 转换入口 | `deploy/host/convert_int8.py`,MTK `OnnxConverter` |
| 精度 | W8A8 PTQ,逐输出通道权重量化 |
| 校准 | COCO-WholeBody 的有效 person 框,按既有 `load_person_samples` 顺序取 100 个框;`offset=0` |
| 输入 | FP32 `[1,3,256,192]`,按人体框仿射裁剪后的 RGB [0,255];mean/std 在导出图内 |
| 显式设置 | `quantize=True`,`use_per_output_channel_quantization=True`;其余属性沿用当前 SDK 默认值 |
| 量化输入/结果 | `rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192_mtk_compatible.onnx` → `build/model_int8.tflite` |

### 特殊 ONNX 导出处理

[export_model.py](deploy/host/export_model.py) 将官方 MMPose 模型包装成含 RGB mean/std 的主干 + SimCC 双输出图,固定 opset 17、133 个关键点.兼容处理合并外部权重,将 IR 限制到 8、标准 opset 上限限制到 18;仅在确认不改变当前 reshape 语义时移除 `allowzero`.GAU 中不受支持的双轴 Mul 广播通过常量扩展与 Concat 改写为同形状乘法.保留原始 ONNX 和兼容 ONNX,量化只读取后者.

### 完整 NCC 编译命令

在仓库根目录的已配置编译容器中,对已生成的 TFLite 执行.路径通过本地配置加载,命令不触发重新量化:

```bash
source tools/storage.sh mtk_models/interaction/pose_detection/rtmpose_body2d
env LD_LIBRARY_PATH="${NCC_ROOT}/lib" \
    "${NCC_ROOT}/bin/ncc-tflite" \
    --arch=mdla5.3 --suppress-output --disallow-bridge \
    "${MODEL_ASSETS_DIR}/build/model_int8.tflite" \
    -o "${MODEL_ASSETS_DIR}/model_int8.dla"
```
