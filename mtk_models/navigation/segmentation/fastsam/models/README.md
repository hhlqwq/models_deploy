# FastSAM 模型产物

模型二进制保存在外部资源目录,不进入普通 Git.本目录只保留资源布局说明与公开来源记录.

| 外部资源目录内的路径 | 内容 |
| --- | --- |
| FastSAM-s.pt | 官方原始权重 |
| FastSAM-s.onnx | 唯一一份 FP32 ONNX,opset 13,静态 640,10 个原始输出 |
| build/model_int8.tflite | MTK INT8 校准产物 |
| build/model_int8.json | 量化参数、输出索引语义映射、校准图片清单 |
| build/runtime_config.csv | Python 板端代码使用的输入输出张量契约 |
| model_int8.dla | MDLA 5.3 无桥接编译产物 |
| build/pytorch_reference.npz | 当前导出检查图片的 FP32 输入及原始输出 |
| build/export_manifest.json | 导出实现、图片、检查结果与最终 ONNX 路径及 SHA-256 |
| build/fastsam_board | 交叉编译的板端程序 |
| build/samples.json | 示例清单 |
| build/onnx_reference.json | 完整 5000 张同协议 ONNX 评测报告,成功评测后生成 |
| build/config.json | 上传时生成的板端配置快照 |
| build/cache/、build/tmp/ | 缓存与临时文件 |
| build/source_url.txt | 原始模型和依赖的公开来源记录 |

资源根目录只保留原始 PT 权重、最终 ONNX、最终 DLA 和 build 目录.build 中间文件平铺,子目录只保留 cache 和 tmp.评测临时文件先保存到 tmp,报告校验通过后发布为 onnx_reference.json; 缺失时上传直接停止,不生成占位报告.
