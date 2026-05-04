# BitNet Retrofit Benchmark Plan

This directory contains reusable benchmark harnesses for comparing:

1. original Hugging Face FP checkpoints,
2. QAT/distilled W1.58A8 ternary checkpoints,
3. naive-PTQ W1.58A8 ternary checkpoints,
4. future GGUF/CPU-runtime baselines.

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

### Naive PTQ Baselines

Naive PTQ performs no calibration, QAT, or distillation. It is a destructive
control condition for testing whether training under ternary forward constraints
is doing useful recovery work.

```bash
python benchmarks/export_naive_ptq.py \
  --model Qwen/Qwen2.5-0.5B \
  --output-dir checkpoints/qwen2.5-0.5b-naive-ptq-tensor \
  --dtype bf16 \
  --scale-mode tensor \
  --expect-ternary-keys 168

MODEL=Qwen/Qwen2.5-1.5B \
OUTPUT_DIR=checkpoints/qwen2.5-1.5b-naive-ptq-tensor \
EXPECT_TERNARY_KEYS=196 \
sbatch slurm_export_naive_ptq.sh
```

Then run PTQ-only quality evals:

```bash
RUN_FP=false RUN_QAT=false RUN_PTQ=true \
OUT_DIR=benchmark_results/quality-ptq-qwen05b \
QWEN15_PTQ=checkpoints/does-not-exist \
sbatch slurm_benchmark_quality.sh
```

For single-model ablation evaluation in `slurm_benchmark_quality.sh` or
`slurm_benchmark_mc.sh`, use `RUN_QWEN05=false` or `RUN_QWEN15=false` rather
than passing dummy checkpoint paths.

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
For the Slurm wrapper, set `LIMIT=0` to run full tasks instead of capped
debug slices.

To produce a selected-metric comparison table from lm-eval JSON outputs:

```bash
python benchmarks/compare_lm_eval.py \
  --run FP=benchmark_results/lm-eval-qwen15b-core5-full/qwen15b_fp.json \
  --run naive_PTQ=benchmark_results/lm-eval-qwen15b-core5-full/qwen15b_naive_ptq.json \
  --run QAT=benchmark_results/lm-eval-qwen15b-core5-full/qwen15b_qat_ternary.json \
  --metric arc_challenge=acc_norm \
  --metric arc_easy=acc_norm \
  --metric hellaswag=acc_norm \
  --metric piqa=acc_norm \
  --metric winogrande=acc
```

For paired deltas with 95% confidence intervals, use the logged samples from
the same lm-eval runs:

```bash
python benchmarks/paired_lm_eval_delta.py \
  --a naive_PTQ=benchmark_results/lm-eval-qwen15b-core5-full/qwen15b_naive_ptq.json \
  --b QAT=benchmark_results/lm-eval-qwen15b-core5-full/qwen15b_qat_ternary.json \
  --metric arc_challenge=acc_norm \
  --metric arc_easy=acc_norm \
  --metric hellaswag=acc_norm \
  --metric piqa=acc_norm \
  --metric winogrande=acc \
  --output-md benchmark_results/lm-eval-qwen15b-core5-full/paired_qat_minus_ptq.md
```

### Tier 4: CPU Runtime

After GGUF/TL2/I2_S conversion, compare against llama.cpp baselines:

- FP or BF16 HF reference quality
- llama.cpp Q8_0
- llama.cpp Q4_K_M
- BitNet TL2/I2_S repaired ternary

Measure prompt throughput, decode throughput, RSS, model file size, and perplexity on the same text blocks.

The reusable GGUF suite runner consumes a manifest and writes raw logs plus
machine-readable summaries:

```bash
python benchmarks/run_gguf_suite.py \
  --models-json benchmarks/gguf_qwen15b_manifest.json \
  --out-dir benchmark_results/gguf-qwen15b-suite \
  --perplexity-file benchmark_results/gguf-ppl/wikitext2_test_excerpt.txt \
  --threads 12 \
  --prompt-tokens 512 \
  --gen-tokens 128 \
  --ppl-chunks 16
```

For QAT checkpoints, do not convert `model.safetensors` directly and treat it
as the trained ternary artifact. The validated bridge is:

```bash
python benchmarks/materialize_static_ternary_hf.py \
  --checkpoint-dir checkpoints/qwen2.5-1.5b-fineweb-edu/step-5000 \
  --output-dir checkpoints/qwen2.5-1.5b-fineweb-edu/step-5000-static-ternary-dense-f16 \
  --dtype float16 \
  --expect-ternary-keys 197

python 3rdparty/llama.cpp/convert_hf_to_gguf.py \
  checkpoints/qwen2.5-1.5b-fineweb-edu/step-5000-static-ternary-dense-f16 \
  --outfile models/qwen2.5-1.5b-static-ternary-dense/qwen15b_static_ternary_dense_f16.gguf \
  --outtype f16
```

Current finding: this materialized static-ternary F16 GGUF recovers the QAT
quality signal, and `TQ2_0` preserves it as a packed ternary GGUF. Blind
`TQ2_0` on the original dense Qwen checkpoint still fails, so this is not a
post-training retrofit result. `I2_S` also preserves the static-ternary quality
when quantized single-threaded:

```bash
build/bin/llama-quantize \
  models/qwen2.5-1.5b-static-ternary-dense/qwen15b_static_ternary_dense_f16.gguf \
  models/qwen2.5-1.5b-static-ternary-dense/qwen15b_static_ternary_dense_i2_s_t1.gguf \
  I2_S \
  1
```

The multi-thread I2_S writer path produced a corrupted artifact for this layout,
so the production gate is a writer/chunking fix plus a rerun without forcing
`nthreads=1`.

## Publishability Gate

A credible claim needs all of:

- complete ternary export for the model under test,
- heldout PPL and task accuracy against FP and quantized baselines,
- actual packed CPU runtime numbers, not only PyTorch simulation,
- ablations showing QAT/distillation beats naive PTQ,
- reproducible scripts, logs, and checkpoint hashes.

Useful ablation launch knobs in `slurm_distill.sh` include `SCALE_MODE=row`,
`HIDDEN_MSE_WEIGHT=0`, `KL_WEIGHT`, `MAX_STEPS`, `MAX_PACKED_BLOCKS`, and
`NPROC_PER_NODE`. Keep dataset, sequence length, optimizer, and evaluation
settings fixed across an ablation table unless the changed variable is the
thing being tested.
