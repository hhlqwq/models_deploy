# 示例音频

音频来自已有 LibriSpeech test-clean 数据集,与固定子集的前三条数值参考及真实板端转写对应.

| 文件 | 官方音频标识 |
| --- | --- |
| sample_1.flac | 1089-134686-0008 |
| sample_2.flac | 1089-134691-0023 |
| sample_3.flac | 1188-133604-0000 |

主机 compile 上传对应原始音频到板端 examples/input/.板端 compile 保存原始标注、FP32 转写与真实 BPU 转写到 examples/output/transcripts.json.模型 README 中直接展示识别结果及播放链接.

来源为 https://www.openslr.org/12/ .
