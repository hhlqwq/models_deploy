# 参考来源

下列开源 skill 于 2026-10-21 查阅.本技能结合本仓库独立编写,不复制厂商实现或混用厂商编译参数.后续使用时重新核对来源版本.

| 来源 | 采用的思路 |
| --- | --- |
| [NVIDIA DeepStream import skill](https://github.com/NVIDIA/DeepStream/blob/main/skills/deepstream-import-vision-model/SKILL.md) | 任务分流、张量和预后处理契约、以真实运行及语义证据判定就绪 |
| [NVIDIA TAO RT-DETR skill](https://github.com/NVIDIA/skills/blob/main/skills/tao-train-rtdetr/SKILL.md) | 通用入口与特定部署说明分开,参考按阶段读取 |
| [Hugging Face evaluation skill](https://github.com/huggingface/skills/blob/main/skills/huggingface-community-evals/SKILL.md) | 小样本先行、明确后端及评价协议;本项目另设独立全量入口 |
| [Hugging Face skills](https://github.com/huggingface/skills) | `SKILL.md` 发现入口与按需参考 |
| [MediaTek EDPA/EDMA 说明](https://genio-community.mediatek.com/t/edpa-in-genio520-720/1902) | Genio 520/720 目标差异,结合本项目编译与 Runtime 诊断 |

MediaTek 工具、系统发布和算子支持资料优先沿仓库 `docs/environment.md`、`docs/genio_720.md` 与 `docs_study/` 中官方入口查询.页面要求 MOL/NDA 登录时标为需授权,不把搜索摘要当作完整支持列表.

新增模型选型与评测协议参考以下官方入口,使用时继续查看具体型号的模型卡、精度表和评价代码,目录或演示支持列表本身不证明评测协议:

| 厂商 | 官方入口 |
| --- | --- |
| 高通 | [Qualcomm AI Hub 模型目录](https://aihub.qualcomm.com/models) |
| 地瓜 | [RDK Model Zoo](https://d-robotics.github.io/rdk_model_zoo/) |
| 地平线 | [Open Explorer 模型性能基准](https://developer.d-robotics.cc/api/v1/fileData/horizon_j5_open_explorer_en_doc/oe_mapper/source/appendix/model_benchmark.html),注意文档与工具链版本 |
| 英伟达 | [TAO 模型入口](https://developer.nvidia.com/transfer-learning-toolkit),查看具体模型的评测说明 |
| 英特尔 | [OpenVINO Open Model Zoo](https://docs.openvino.ai/2023.3/omz_models_group_public.html),注意文档与模型版本 |

核对的是已有权重的精度评测集及其划分,不是为本仓库规划训练.页面只写了数据集名称、支持 COCO 格式或给出训练信息时,不能据此认定具体精度使用了 COCO val2017;未公开的评测细节保持未核实.厂商精度用于选型参考,各目标平台指标仍需本仓库同协议实测.

项目事实以当前 skills、平台配置、模型总览、注册表及模型源码/来源/结果为准,不要求恢复已删除的 AGENTS.md.MTK 公共工具为 tools/deploy.py、tools/board.py;S100 为 tools/rdk_config.py、tools/rdk_deploy.py;共用评价接口为 tools/evaluation.py,其他汇总与展示复用现有实现.现有 C++ 实现待迁移,不作为新模型的推荐实现.采用外部 skill 的分阶段思路时必须遵守本项目 Python 推理、compile.sh/evaluate.sh 双入口和summary_test.json / summary.json 双报告约定.

README 数字和注册状态仅用于定位已有证据,不能证明当前设备状态或新运行成功.正式移植来源不明确时保持 pending,厂商选型、评测协议及交付参考不能替代官方模型源.

S100 先核对 [RDK 文档入口](https://d-robotics.github.io/rdk_s_doc/)中对应型号和 SDK 版本.已归档 SDK 的配方、加载代码和依赖声明可作为适配依据,不把内部资源路径或凭据复制到公开文档.示例权重型号、目标芯片和厂商公开成绩须分别核对: 网络适配存在不等于原始权重已成功加载,编译成功不等于板端运行,只有 W8 benchmark 不代表禁止 W4.原始权重仍来自作者,厂商量化包/预编译 HBM 只作部署参考.
