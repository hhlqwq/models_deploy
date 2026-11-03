# 模型资源

仓库中的本目录保存公开来源记录 [source_url.txt](source_url.txt) 和 [source_manifest.json](source_manifest.json),不保存权重、量化图或编译产物.官方模型为 [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B),固定 revision 与许可见 [模型卡](../model_card.md).

实际资源位于本地存储配置指定的 NAS 对应模型目录,路径由 [config.yaml](../config.yaml) 解析:

| NAS 相对目录 | 用途 |
| --- | --- |
| `original/` | 官方原始权重、配置、分词器和许可证,保持源文件不变 |
| `build/native_hessian16/` | 当前选定的转换副本、量化 TFLite、编译中间产物和配方记录 |
| `models/` | 打包后的多文件运行资源,包含双 DLA、embedding、分词器和张量契约 |

当前 `models/` 包内两个 DLA 的相对路径为:

- `1024c/MiniCPM5-2B_asym4W_sym16A_42layer_prefill_128t_1024c.dla`
- `1024c/MiniCPM5-2B_asym4W_sym16A_42layer_decode_1t_1024c.dla`

实际加载路径由包内 `manifest.json` 的 `model_files` 指定.这是原生 LLM 静态图链路,不要求额外生成一个同名 ONNX 文件.当前入口复用已选产物,缺失时停止,不会自动下载资源或重新量化.

板端多文件运行资源同样放在本模型 `models/`,代码放在 `board/`,必要评价输入和浮点参考放在 `build/reference/`,运行过程文件覆盖 `build/work/`.主机评价过程数据使用本地日期临时目录,不写入 NAS.最终子集与全量报告分别为 `results/summary_test.json`、`results/summary.json`,示例放在 `examples/output/`;使用方式见 [README](../README.md).
