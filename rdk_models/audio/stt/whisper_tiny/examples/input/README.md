# 示例音频

输入来自已有 LibriSpeech test-clean 数据集.主机 compile 自动上传本次固定 50 条子集,其中前三条同时用于 Encoder 数值比较和展示转写:

- 1089-134686-0008
- 1089-134691-0023
- 1188-133604-0000

板端原始音频位于本地 S100 配置指定的公共数据集目录,精度参考记录对应相对路径.转写保存在 examples/output/transcripts.json,包含原始标注、FP32 识别文本及真实 BPU 识别文本.

公开来源为 https://www.openslr.org/12/ .
