# Phi-3-mini-4k-instruct

## 1. 创建软链接

```bash
mkdir -p /usr/share/llm

ln -sfn \
$(pwd) \
/usr/share/llm/phi-3-mini-4k-instruct
```

## 2. 开启 Performance Mode

```bash
bash scripts/set_performance.sh
```

## 3. 运行 10 Prompts Benchmark

```bash
llm_cmdline_tool \
  scripts/config-yocto_phi4_mini_instruct.yaml \
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
  --preformatter Phi3NoInput \
  -m 300 \
  2>&1 | tee benchmark_phi-3-mini-4k-instruct.log
```

## 4. 查看平均性能

```bash
grep -E "Average.*Prompt|Average.*Generative|Prompt Mode|Generative Mode" \
  benchmark_phi-3-mini-4k-instruct.log | tail -20
```
