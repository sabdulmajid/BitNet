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

### BitDistill Reproduction

The BitDistill reproduction harness is centered on `train_bitdistill.py`.
It is separate from the older generic `train_distill.py` path and implements:

- SubLN insertion before Qwen attention output and MLP down projections.
- Stage-2 causal-LM continued pretraining with CE.
- Stage-3 GLUE task fine-tuning with CE, logits KL, and Q/K/V
  attention-relation distillation.
- Sequence-classification and causal prompt-scoring GLUE formats.
- Tensor-scale and row-scale ternary `BitLinear` students.

Primary GLUE3 sequence-classification wave:

```bash
bash benchmarks/submit_bitdistill_glue_seqcls_wave.sh

python benchmarks/summarize_bitdistill_glue.py \
  --root checkpoints/bitdistill-glue-seqcls \
  --tasks mnli qnli sst2 \
  --output-json benchmark_results/bitdistill_seqcls_glue3_primary_summary_2026-05-14.json \
  --output-md benchmarks/results/bitdistill_seqcls_glue3_primary_summary_2026-05-14.md
```

Current status: Qwen2.5-0.5B short-budget BitDistill does not reproduce the
paper-level target. The first completed GLUE3 sequence-classification wave used
the older tau-squared KD convention and is now treated as a diagnostic rather
than a paper-faithful logits-KL reproduction. Its accuracy is:

| task | FP16-SFT | BitNet-SFT | BitDistill tensor | BitDistill row |
| --- | ---: | ---: | ---: | ---: |
| MNLI | `0.807641` | `0.487621` | `0.525217` | `0.516556` |
| QNLI | `0.898957` | `0.596925` | `0.596925` | `0.618525` |
| SST2 | `0.925459` | `0.770642` | `0.815367` | `0.808486` |

The known reproduction gap is training budget and search, not a contradiction
of the paper. The completed Stage-2 warm-up is 40.96M effective token
presentations, while the paper reports 10B continued-pretraining tokens. New
runs use `--logit-kd-temperature-scale none`, matching the paper equations.
Teacher-head initialization, attention-layer sweep, CE-only ablation, and
longer warm-up jobs are tracked in
`benchmarks/results/bitdistill_reproduction_status_2026-05-14.md`.
The completed and initially queued downstream BitDistill jobs use
`ATTENTION_KD_WEIGHT=100`; the paper text reports `gamma=1e5` for
classification. A separate long-warm-up `papergamma` branch uses that stricter
hyperparameter, and the reproduction gate treats that branch as the paper
candidate.

The corrected MNLI diagnostic wave plus attention-layer sweep improved the best
short-budget BitDistill result to `0.535711` against FP16-SFT `0.807641`.
CE-only ablations stay near `0.492-0.498`, so distillation helps but the
short-warm-up recipe remains far from reproduced. See
`benchmarks/results/bitdistill_seqcls_mnli_diagnostic_variant_summary_2026-05-14.md`.

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

To audit the full user-facing objective rather than only artifact structure,
generate the current prompt-to-artifact completion report:

```bash
python benchmarks/audit_objective_completion.py
```

The current report is
`benchmarks/results/objective_completion_audit_2026-05-14.md`. It is expected
to report `Objective achieved: False` until the remaining TL2 row-scale and
MoE/Kimi claims are either backed by quality-valid runtime artifacts or removed
from the product scope.

To classify the current publication/product scope from artifacts:

```bash
python benchmarks/audit_product_scope.py
```

The current scope gate is
`benchmarks/results/product_scope_gate_2026-05-13.md`. It separates the
supported dense-Qwen negative result plus row-scale recovery path from
unsupported one-click, TL2, default-runtime, and MoE/Kimi claims.

To monitor the active BitDistill long-warm-up and dependent downstream jobs
without relying on memory or chat history:

```bash
python benchmarks/monitor_bitdistill_jobs.py
```

The monitor reads the latest
`benchmark_results/bitdistill_longwarmup_downstream_*.tsv`, parses the warm-up
log, queries `squeue` when available, and reports whether downstream metrics
have materialized.

To gate the exact MNLI/QNLI/SST2 reproduction target against the FP16-SFT
baselines:

```bash
python benchmarks/gate_bitdistill_reproduction.py
```

The gate reports whether the paper-style tensor candidate and the row-scale
candidate are complete and whether each is within the configured FP16-SFT
accuracy gap. Missing long-warm-up outputs are treated as pending failures.
The strict tensor candidate is read from
`checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma` by default.

To audit whether the local setup is actually comparable to the BitDistill paper
before interpreting results as a reproduction:

```bash
python benchmarks/audit_bitdistill_paper_alignment.py
```

This report separates implemented mechanisms from unmatched or pending paper
assumptions such as the 10B-token warm-up, `gamma=1e5`, Qwen3 scale ladder, and
learning-rate/epoch search.

To inspect whether the paper's classification attention-KD coefficient is on
the same numerical scale as this implementation's relation loss:

```bash
python benchmarks/audit_bitdistill_loss_scales.py
```

The report projects saved gamma-100 runs to `gamma=1e5` and includes the local
strict-gamma smoke result when present.

To measure PyTorch CPU task runtime for saved GLUE sequence-classification
checkpoints:

```bash
python benchmarks/benchmark_bitdistill_glue_cpu.py \
  --tasks mnli qnli sst2 \
  --runs short:fp16_sft-tensor-layer-1 short:bitnet_sft-tensor-layer-1 short:bitdistill-tensor-layer-1 short:bitdistill-row-layer-1 longwarmup:bitdistill-longwarmup-tensor-layer-8 papergamma:bitdistill-longwarmup-tensor-layer-8 \
  --max-eval-samples 128 \
  --threads 12
```

This benchmark reconstructs the saved checkpoint in an isolated child process,
including SubLN and `BitLinear` modules when present. It reports CPU accuracy,
examples/sec, batch latency, and RSS. It is a task-runtime probe for the
sequence-classification checkpoints, not a packed `I2_SR`/llama.cpp inference
claim.
For the active long-warm-up chain, `slurm_bitdistill_cpu_benchmark.sh` runs the
same benchmark after all downstream jobs finish and includes the strict
`papergamma` family.

To gate the CPU task-runtime artifact before citing it:

```bash
python benchmarks/gate_bitdistill_cpu_benchmark.py
```

This checks that the critical FP16-SFT, BitNet-SFT, BitDistill tensor,
BitDistill row, long-warmup, and `papergamma` task rows exist and include
accuracy, examples/sec, RSS, max RSS, and eval-example counts. Timeout or
failed rows are blockers, not partial successes.

To export row-scale BitDistill checkpoints through the stable packed `I2_SR`
GGUF path, the checkpoint must be a causal-LM architecture such as
`Qwen2ForCausalLM`:

```bash
python benchmarks/export_bitdistill_i2sr_suite.py \
  --root checkpoints/bitdistill-glue \
  --tasks mnli \
  --scales row \
  --layer -1 \
  --skip-existing
```

The active paper-style GLUE reproduction uses
`Qwen2ForSequenceClassification`. Those checkpoints are intentionally not
claimed as llama.cpp / `I2_SR` task-runtime artifacts. The exporter records this
as an unsupported architecture when
`--skip-unsupported-architecture` is passed:

```bash
python benchmarks/export_bitdistill_i2sr_suite.py \
  --root checkpoints/bitdistill-glue-seqcls \
  --tasks mnli qnli sst2 \
  --scales row \
  --layer -1 \
  --skip-unsupported-architecture
```

To get packed CPU task inference for the sequence-classification reproduction,
either train/evaluate the task as causal prompt scoring or add explicit
classifier-head support to the GGUF mapping and runtime.

For the active Slurm pipeline, queue `slurm_bitdistill_postprocess.sh` with an
`afterany` dependency on the downstream jobs. It refreshes the monitor,
reproduction gate, variant summary, and objective audit from the materialized
artifacts.

For future long Stage-2 warm-ups, set `SAVE_EVERY_STEPS=1000` or another
interval when submitting `slurm_bitdistill_glue.sh`. This writes restartable
`checkpoint-<step>/custom_state_dict.pt` snapshots under the run directory.

To consolidate the inputs required before the remaining open claims can advance:

```bash
python benchmarks/audit_unblock_requirements.py
```

The current report is `benchmarks/results/unblock_requirements_2026-05-13.md`.
It checks the writable `llama.cpp` fork requirement, local Kimi/Qwen2MoE
artifacts, MoE packing readiness, and whether more local benchmarking can
continue productively without new input.

To verify the current MoE packing blocker with synthetic merged expert tensors:

```bash
python benchmarks/audit_moe_packing_contract.py
```

The current report is
`benchmarks/results/moe_packing_contract_2026-05-14.md`. It shows that the TL2,
direct `I2_S`, and direct `I2_SR` synthetic 3D packing contracts now accept
merged `[experts, out, in]` tensors. This is still only a packing contract until
runtime and real-checkpoint validation exist.

To verify the deeper TL2 runtime contract for MoE expert tensors:

```bash
python benchmarks/audit_moe_tl2_runtime_contract.py
```

The current report is
`benchmarks/results/moe_tl2_runtime_contract_2026-05-14.md`. It checks the
Python TL2 preprocessor, the active `ggml_nbytes` TL2 size contract, and whether
`ggml_mul_mat_id` routes TL2 expert matmuls through the BitNet LUT path.

To verify whether `I2_SR` is actually active in the committed submodule state
rather than only available as a patch:

```bash
python benchmarks/audit_i2sr_submodule_promotion.py
```

To include the current Git blocker in the audit, probe the configured upstream
for non-mutating write access and probe the intended writable fork URL:

```bash
python benchmarks/audit_i2sr_submodule_promotion.py \
  --check-remote-write \
  --candidate-fork-url https://github.com/sabdulmajid/llama.cpp.git
```

The current report is
`benchmarks/results/i2sr_submodule_promotion_audit_2026-05-13.md`; it is
expected to fail until the submodule points at a pushed branch containing the
row-scale qtype/runtime changes. The current failure is specific: the upstream
submodule remote returns 403 for this user, and the expected
`sabdulmajid/llama.cpp` fork URL is not reachable from this environment.

The monolithic candidate patch can be split into promotion-ready pieces with:

```bash
python benchmarks/split_i2sr_patch.py
git apply --check patches/bitnet-i2sr-root-runtime.patch
git -C 3rdparty/llama.cpp apply --check ../../patches/llama-i2sr-row-scale-qtype.submodule.patch
```

The generated split patches are tracked as
`patches/bitnet-i2sr-root-runtime.patch` and
`patches/llama-i2sr-row-scale-qtype.submodule.patch`.

To make the remaining Git handoff reproducible before a writable fork exists,
generate the non-mutating promotion handoff report:

```bash
python benchmarks/prepare_i2sr_promotion_handoff.py \
  --fork-url https://github.com/sabdulmajid/llama.cpp.git
```

The report is
`benchmarks/results/i2sr_promotion_handoff_2026-05-13.md`. It records local
patch applicability, the fork reachability probe, exact commands for creating
and pushing the submodule branch, the superproject pointer-update commands, and
the post-promotion gates. After the fork exists, rerun it with
`--prepare-worktree`; add `--push` only when the fork URL is confirmed writable.

To rehearse promotion with those split patches, including patched build,
productization gate, patch reversal, and optional unpatched rebuild:

```bash
python benchmarks/run_i2sr_promotion_rehearsal.py --restore-build
```

The current rehearsal report is
`benchmarks/results/row_scale_qtype_productization_gate_i2sr_promotion_rehearsal_2026-05-13.md`.

Large checkpoints, model conversions, and CPU benchmark byproducts are ignored
by Git. To make pruning reproducible, generate the dry-run artifact plan before
deleting anything:

```bash
TMPDIR=/mnt/slurm_nfs/a6abdulm/tmp python benchmarks/plan_artifact_pruning.py
```

The current plan is
`benchmarks/results/artifact_prune_plan_2026-05-13.md`. It protects the hashed
evidence manifest files, benchmark-manifest model paths, and final checkpoints,
then separates intermediate checkpoint cleanup from rebuildable caches and model
artifacts.

Use `benchmarks/apply_artifact_prune_plan.py` for a guarded dry run before
deleting anything:

```bash
python benchmarks/apply_artifact_prune_plan.py
```

The default group is `prune_intermediate_checkpoints`. Destructive cleanup
requires an explicit `--execute` and a clean Git worktree.

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

Current Qwen status: I2_S and TQ2_0 are benchmarked through the static-ternary
materialization bridge. TL2 is not yet a validated Qwen path. `llama-quantize`
does not expose TL2 as an allowed output type, and the BitNet-specific TL2 path
requires exact model-shape code generation plus a matching `--kernel-config`
and TL2-enabled runtime build. The Qwen2.5-0.5B TL2 probe is mechanically
convertible but quality-invalid (`NaN` PPL). For the stronger Qwen2.5-1.5B
row-scale checkpoint, the current one-scale TL2 format has expected relative
linear output RMS error `1.904230`; exact fp16 row scales reduce that design
error to `0.000197` with only `1.230 MiB` of scale metadata, but no generated
TL2 runtime here indexes those row scales yet. Treat TL2 as pending until there
is row/group-scale metadata support and a TL2-enabled build has passed smoke,
PPL, throughput, and RSS audits.

The TL2 scale-design audit is reproducible with:

```bash
python benchmarks/audit_tl2_row_scale_design.py \
  --state qwen15b_tensor_scale=checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-5000/step-5000/ternary_state_dict.pt \
  --state qwen15b_row_scale=checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-row-notiehead-5000/step-5000/ternary_state_dict.pt \
  --output-json benchmark_results/tl2_row_scale_design_2026-05-13.json \
  --output-md benchmarks/results/tl2_row_scale_design_2026-05-13.md
```

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

For Slurm jobs that may move across CPU families, do not assume the repo-root
`build/bin` binaries are portable. A build configured with `GGML_NATIVE=ON` on
an AVX-512 Intel node can fail with `SIGILL` on an AVX2-only AMD node. The
`slurm_gguf_static_ternary_suite.sh` wrapper therefore defaults to
`build-portable-avx2/bin` and configures it with `GGML_NATIVE=OFF`,
`GGML_AVX2=ON`, `GGML_FMA=ON`, and `GGML_F16C=ON` before running
`llama-quantize`, `llama-cli`, `llama-bench`, and `llama-perplexity`.

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

For dense GGUF export only, `convert_static_ternary_to_gguf.py` skips the
intermediate Hugging Face materialization directory and streams effective
dense tensors from `ternary_state_dict.pt` into the llama.cpp GGUF writer:

```bash
python benchmarks/convert_static_ternary_to_gguf.py \
  --checkpoint-dir checkpoints/qwen2.5-0.5b-fineweb-edu-klonly-notiehead-1000/step-1000 \
  --outfile models/qwen2.5-0.5b-direct-static-ternary/qwen05b_klonly_notie_direct_f16.gguf \
  --outtype f16 \
  --expect-ternary-keys 168 \
  --validate-codes
```

This is not the final packed `I2_S` writer. Quantized Python-converter outtypes
are blocked by default because unsupported Qwen shapes can fall back to F16.
Use `benchmarks/audit_direct_packed_gguf_support.py` to check the current
direct packed-writer gates; the current audit reports dense direct GGUF support
as present and direct packed row-scale support as absent.

For scalar-scale static ternary checkpoints, `convert_static_ternary_to_i2s_gguf.py`
can pack `ternary_state_dict.pt` directly into the existing tensor-scale
`I2_S` GGUF layout without modifying the vendored `llama.cpp` Python constants:

```bash
python benchmarks/convert_static_ternary_to_i2s_gguf.py \
  --checkpoint-dir checkpoints/qwen2.5-0.5b-fineweb-edu-klonly-1000/step-1000 \
  --outfile models/qwen2.5-0.5b-direct-static-ternary/qwen05b_klonly_direct_i2_s_x86act.gguf \
  --expect-ternary-keys 169 \
  --validate-codes \
  --summary-json benchmark_results/direct-i2s-qwen05b-klonly-x86act-2026-05-13/conversion_summary.json
```

By default this path intentionally rejects row-scale checkpoints. On the
Qwen2.5-0.5B KL-only scalar checkpoint it produced a loadable `I2_S` GGUF with
`168` packed projection tensors. The fixed x86 `ACT_PARALLEL` artifact is
finite but weak on the fixed-excerpt CPU benchmark: PPL `423.4528`,
`540.97` prompt tok/s, and `40.04` decode tok/s versus FP16 PPL `18.0986`.
Treat it as a scalar-format/runtime probe, not as the row-scale product path. See
`benchmarks/results/direct_i2s_scalar_gguf_2026-05-13.md`.

For debugging only, `--row-scale-prototype` writes one float32 scale per output
row after the packed active x86 `I2_S` codes. This requires a matching
experimental runtime layout and is not a stable GGUF contract. The
Qwen2.5-0.5B row-scale control is a historical negative control from before the
x86 `ACT_PARALLEL` packing fix and failed quality badly:
materialized-F16 PPL `578.4833`, direct row-scale `I2_S` PPL `59401.5449`,
materialized-then-`I2_S` PPL `NaN`, and materialized-then-`TQ2_0` PPL
`5118527.5782`. See
`benchmarks/results/direct_row_i2s_qwen05b_2026-05-13.md`.

The quality-valid direct row-scale packed evidence is the fixed candidate
`I2_SR` path on Qwen2.5-1.5B: PPL `38.8477`, `211.67` prompt tok/s, and
`19.07` decode tok/s with `patches/llama-i2sr-row-scale-qtype.patch` applied.
That qtype is still downstream, not active by default.

To reproduce the active-patch productization gate without leaving the source
tree patched:

```bash
python benchmarks/run_i2sr_active_patch_gate.py --restore-build
```

The harness checks that the patch applies cleanly, applies it, rebuilds
`llama-cli`, `llama-bench`, `llama-perplexity`, and `llama-quantize`, runs
`benchmarks/audit_row_scale_qtype_productization.py` in the patched-source
scenario, reverses the patch, and optionally rebuilds the default binaries.
The current generated proof is
`benchmarks/results/row_scale_qtype_productization_gate_i2sr_active_patch_2026-05-13.md`.

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

Row-scale checkpoints need a separate layout fix. Default `I2_S` stores one
tensor-level scale and therefore cannot preserve per-output-row ternary scales.
The prototype patch at `patches/llama-i2s-row-scale.patch` stores one scale per
output row and updates the CPU matmul/get_rows paths:

```bash
git apply patches/llama-i2s-row-scale.patch
cmake --build build-portable-avx2 --target llama-quantize llama-cli llama-perplexity -j 12
```

With that patch, the Qwen2.5-1.5B KL-only row-scale dense-`lm_head` artifact
recovered fixed-excerpt PPL `38.8832 +/- 1.97093` as `I2_S`, compared with
`1,197,135.5848` for the default `I2_S` layout. This patch changes the binary
layout of `I2_S`; regenerate existing `I2_S` GGUF files before comparing.

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
