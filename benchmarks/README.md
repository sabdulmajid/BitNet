# BitNet Retrofit Benchmark Plan

This directory contains reusable benchmark harnesses for comparing:

1. original Hugging Face FP checkpoints,
2. QAT/distilled W1.58A8 ternary checkpoints,
3. naive-PTQ W1.58A8 ternary checkpoints,
4. GGUF/CPU-runtime baselines.

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

For `slurm_benchmark_mc.sh`, prefer `MC_LIMIT=200` over the legacy `LIMIT=200`
environment variable. `LIMIT` is still accepted for older commands, but it is
generic enough to be accidentally inherited by Slurm submissions.

When submitting Slurm jobs with comma-valued variables such as `TASKS`, set the
variable in the `sbatch` process environment and use `--export=ALL`. Do not put
`TASKS=a,b,c` inside Slurm's comma-separated `--export=ALL,...` argument unless
your Slurm version has been explicitly tested with escaped commas.

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

If a full task suite is split across Slurm jobs, merge task-keyed lm-eval JSON
files before comparing. This preserves `samples` for paired analysis:

```bash
OUT=benchmark_results/lm-eval-qwen15b-full10
mkdir -p "$OUT"

python benchmarks/merge_lm_eval_results.py \
  --input benchmark_results/lm-eval-qwen15b-core5-full/qwen15b_fp.json \
  --input benchmark_results/lm-eval-qwen15b-rest5-full-v2/qwen15b_fp.json \
  --output-json "$OUT/qwen15b_fp.json"

python benchmarks/merge_lm_eval_results.py \
  --input benchmark_results/lm-eval-qwen15b-core5-full/qwen15b_naive_ptq.json \
  --input benchmark_results/lm-eval-qwen15b-rest5-full-v2/qwen15b_naive_ptq.json \
  --output-json "$OUT/qwen15b_naive_ptq.json"

python benchmarks/merge_lm_eval_results.py \
  --input benchmark_results/lm-eval-qwen15b-core5-full/qwen15b_qat_ternary.json \
  --input benchmark_results/lm-eval-qwen15b-rest5-full-v2/qwen15b_qat_ternary.json \
  --output-json "$OUT/qwen15b_qat_ternary.json"

python benchmarks/compare_lm_eval.py \
  --run FP="$OUT/qwen15b_fp.json" \
  --run naive_PTQ="$OUT/qwen15b_naive_ptq.json" \
  --run QAT="$OUT/qwen15b_qat_ternary.json" \
  --output-md "$OUT/selected_metrics.md"
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

Before copying any result into a public report, run the artifact audit against
the exact files being cited. It verifies checkpoint ternary key counts, scale
layout, optional `tie_word_embeddings` expectations, lm-eval task/sample
presence, perplexity JSON structure, multiple-choice JSON structure, runtime
probe JSON structure, GGUF CPU summary rows, and PTQ math JSON outputs:

```bash
python experiments/math_viability_test.py \
  --out-features 2048 \
  --in-features 2048 \
  --batch 128 \
  --seed 0 \
  --trials 10 \
  --output-json benchmark_results/math_viability/qwen_dense_2048_trials10_seed0.json

python benchmarks/audit_evidence.py \
  --checkpoint qwen15b=checkpoints/qwen2.5-1.5b-fineweb-edu/step-5000/ternary_state_dict.pt:197:197:scalar:tie_false \
  --lm-eval qwen15b_qat=benchmark_results/lm-eval-qwen15b-full10/qwen15b_qat_ternary.json \
  --perplexity qwen15b_qat_wikitext=benchmark_results/quality-9735/qwen15b_ternary_wikitext.json \
  --mc qwen05b_ternary_piqa=benchmark_results/mc-qwen05b-klonly-notiehead-1000/qwen05b_ternary_piqa.json \
  --runtime qwen05b_runtime=benchmark_results/runtime-qwen05b-klonly-notiehead-512x32/qwen05b_klonly_notiehead.json \
  --gguf-summary qwen15b_cpu=benchmark_results/gguf-qwen15b-suite-v2/summary.json:8 \
  --math dense_gaussian_2048=benchmark_results/math_viability/qwen_dense_2048_trials10_seed0.json \
  --output-md benchmark_results/evidence_audit/current.md
```

For the Qwen2.5-1.5B row-scale dense-head ablation, the postprocess wrapper
audits the final checkpoint/eval artifacts and writes paired lm-eval comparison
tables after the quality, MC, and full ten-task jobs complete:

```bash
sbatch --dependency=afterok:<quality_job>:<mc_job>:<lm_eval_job> \
  slurm_postprocess_row_densehead.sh
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

For the stronger Qwen2.5-1.5B KL-only static-ternary checkpoint, use
`benchmarks/gguf_qwen15b_klonly_manifest.json` and write to a separate output
directory, for example `benchmark_results/gguf-qwen15b-klonly-suite`.

For one-off static-ternary checkpoint families, the Slurm wrapper below
materializes `ternary_state_dict.pt`, converts the materialized checkpoint to
F16 GGUF, packs TQ2_0 and safe single-thread-written I2_S artifacts, runs the
standard GGUF suite against FP/Q8/Q4 controls, and audits the resulting summary:

```bash
CHECKPOINT_DIR=checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-notiehead-5000/step-5000 \
EXPECT_TERNARY_KEYS=196 \
RUN_LABEL=qwen15b_klonly_notie_static_ternary \
OUT_MODEL_DIR=models/qwen2.5-1.5b-klonly-notie-static-ternary-dense \
RESULTS_DIR=benchmark_results/gguf-qwen15b-klonly-notiehead-suite \
sbatch --export=ALL slurm_gguf_static_ternary_suite.sh
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
python benchmarks/quantize_gguf_safe.py \
  --input models/qwen2.5-1.5b-static-ternary-dense/qwen15b_static_ternary_dense_f16.gguf \
  --output models/qwen2.5-1.5b-static-ternary-dense/qwen15b_static_ternary_dense_i2_s_t1.gguf \
  --type I2_S \
  --threads 12
```

The original multi-thread I2_S writer path produced a corrupted artifact for
this layout, so `quantize_gguf_safe.py` still forces I2_S to one writer thread
unless `--allow-unsafe-i2s-multithread` is passed. A validated submodule patch
is included at `patches/llama-i2s-threaded-quantization.patch`; apply it with:

```bash
git -C 3rdparty/llama.cpp apply ../../patches/llama-i2s-threaded-quantization.patch
cmake --build build --target llama-quantize -j 4
```

After applying that patch locally, 12-thread I2_S quantization of the
Qwen2.5-1.5B KL-only static-ternary GGUF matched the single-thread PPL
(`54.7366`) and produced a sensible smoke completion. Keep the safe wrapper's
single-thread default until the llama.cpp submodule itself is advanced to a
commit containing the fix.

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

For the full FineWeb/Qwen2.5-1.5B launcher, `slurm_distill_full.sh` exposes the
same loss and quantization knobs while preserving the original defaults. The
current 1.5B KL-only transfer test was launched as:

```bash
HIDDEN_MSE_WEIGHT=0 \
HIDDEN_STATE_LAYERS=none \
OUTPUT_DIR=checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-5000 \
sbatch --job-name=bitnet-kl15-full --export=ALL slurm_distill_full.sh
```
