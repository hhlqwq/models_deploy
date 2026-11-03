# Qwen3-8B G720 Benchmark

## 1. 创建模型软链接

```bash
mkdir -p /usr/share/llm

ln -sfn \
$(pwd) \
/usr/share/llm/qwen3-8b
```

## 2. 开启 Performance Mode

```bash

chmod +x scripts/set_performance.sh
./scripts/set_performance.sh
```

## 3. 运行 Benchmark

```bash
llm_cmdline_tool \
  scripts/config-yocto_np8-qwen3-8b.yaml \
  -i scripts/prompts/sample_prompt-introduction.txt \
  -i scripts/prompts/p01.txt \
  -i scripts/prompts/p02.txt \
  -i scripts/prompts/p03.txt \
  -i scripts/prompts/p04.txt \
  -i scripts/prompts/p05.txt \
  -i scripts/prompts/p06.txt \
  -i scripts/prompts/p07.txt \
  -i scripts/prompts/p08.txt \
  -i scripts/prompts/p09.txt \
  --preformatter Qwen3NoInput \
  -m 384 \
  2>&1 | tee benchmark_qwen3-8b.log
```

## 4. 查看结果

```bash
grep -A3 "Average Performance" benchmark_qwen3-8b.log
```

重点记录：

```text
Prompt Mode: xxx tok/s
Generative Mode: xxx tok/s
```
