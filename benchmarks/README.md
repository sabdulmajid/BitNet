# BitNet Retrofit Benchmark Plan

This directory contains reusable benchmark harnesses for comparing:

1. original Hugging Face FP checkpoints,
2. QAT/distilled W1.58A8 ternary checkpoints,
3. future naive-PTQ and GGUF/CPU-runtime baselines.

## Rules

- Use the same tokenizer and packed text blocks for every model in a comparison.
- Report checkpoint path, git commit, hardware, dtype, sequence length, block count, and wall-clock throughput.
- Treat prompt generation as a sanity check only. Publishable claims require heldout perplexity and task accuracy.
- Do not compare PyTorch ternary throughput to `bitnet.cpp` throughput as a final CPU speed claim; PyTorch simulates the math but does not use packed TL/I2 kernels.

## Benchmark Tiers

### Tier 0: Local Smoke

Fast correctness check for loader/export regressions.

```bash
python benchmarks/run_perplexity.py \
  --model-kind hf \
  --model Qwen/Qwen2.5-0.5B \
  --output-json benchmark_results/perplexity/qwen05b_fp_wikitext_smoke.json \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --dataset-split test \
  --max-blocks 2 \
  --max-seq-len 128 \
  --device cpu \
  --dtype fp32

python benchmarks/run_perplexity.py \
  --model-kind ternary \
  --checkpoint-dir checkpoints/qwen2.5-0.5b-fineweb-edu-12/step-1000 \
  --output-json benchmark_results/perplexity/qwen05b_ternary_step1000_wikitext_smoke.json \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --dataset-split test \
  --max-blocks 2 \
  --max-seq-len 128 \
  --device cpu \
  --dtype fp32
```

### Tier 1: Heldout Perplexity

Use enough blocks to stabilize NLL.

- WikiText-2 test: `--max-blocks 256 --max-seq-len 512`
- FineWeb-Edu heldout stream: `--dataset-name HuggingFaceFW/fineweb-edu --dataset-config sample-10BT --dataset-split train --text-column text --skip-rows 25000 --max-blocks 256 --max-seq-len 1024`

FineWeb skip rows must be beyond the packed training prefix. Job 9730 packed 19,968 rows, so `--skip-rows 25000` avoids direct train-prefix reuse.

### Tier 2: Fixed Generation Suite

```bash
python benchmarks/run_generation_suite.py \
  --checkpoint-dir checkpoints/qwen2.5-1.5b-fineweb-edu/step-5000 \
  --output-jsonl benchmark_results/generation/qwen15b_step5000_core_cpu.jsonl \
  --device cpu \
  --dtype fp32 \
  --max-new-tokens 32
```

Generation outputs should be reviewed for repetition, instruction following, arithmetic, and code structure. They are not enough for a paper claim.

### Tier 3: lm-eval Accuracy

Install EleutherAI `lm-eval` in a controlled environment and run at least:

- `hellaswag`
- `piqa`
- `arc_easy`
- `arc_challenge`
- `winogrande`

Report exact package version and task settings. If wrapping ternary PyTorch models for `lm-eval`, validate that the wrapper uses `StaticTernaryLinear`, not dense fallback.

### Tier 4: CPU Runtime

After GGUF/TL2/I2_S conversion, compare against llama.cpp baselines:

- FP or BF16 HF reference quality
- llama.cpp Q8_0
- llama.cpp Q4_K_M
- BitNet TL2/I2_S repaired ternary

Measure prompt throughput, decode throughput, RSS, model file size, and perplexity on the same text blocks.

## Publishability Gate

A credible claim needs all of:

- complete ternary export for the model under test,
- heldout PPL and task accuracy against FP and quantized baselines,
- actual packed CPU runtime numbers, not only PyTorch simulation,
- ablations showing QAT/distillation beats naive PTQ,
- reproducible scripts, logs, and checkpoint hashes.
