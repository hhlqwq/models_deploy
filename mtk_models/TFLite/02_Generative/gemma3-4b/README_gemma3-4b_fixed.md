# Gemma3-4B

## 1. 创建软链接

```bash
mkdir -p /usr/share/llm

ln -sfn \
$(pwd) \
/usr/share/llm/gemma3-4b
```

## 2. 修正 YAML 中的 DLA 文件名

```bash

sed -i \
's/Overall_128t1024c_0\.dla/Overall_hessian_128t1024c_0.dla/' \
scripts/config-yocto_gemma3_4b.yaml

sed -i \
's/Overall_1t1024c_0\.dla/Overall_hessian_1t1024c_0.dla/' \
scripts/config-yocto_gemma3_4b.yaml
```

确认：

```bash
grep -A5 -E "dlaPromptPaths|dlaGenPaths" \
scripts/config-yocto_gemma3_4b.yaml
```

## 3. 开启 Performance Mode

```bash
bash scripts/set_performance.sh
```

## 4. 运行 10 Prompts Benchmark

```bash
llm_cmdline_tool \
  scripts/config-yocto_gemma3_4b.yaml \
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
  --preformatter GemmaNoInput \
  -m 250 \
  2>&1 | tee benchmark_gemma3-4b.log
```

## 5. 查看性能结果

```bash
grep -E "Average.*Prompt|Average.*Generative|Prompt Mode|Generative Mode" \
benchmark_gemma3-4b.log | tail -20
```
