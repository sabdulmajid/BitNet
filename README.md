# BitNet Retrofit Research Fork

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This fork investigates whether pretrained FP16/BF16 language models can be
retrofitted into BitNet-style W1.58A8 / ternary CPU inference without training
from scratch.

The short answer from the current evidence is:

**No, blind post-training ternarization is not a lossless or acceptable retrofit
path for arbitrary pretrained models.** The useful path is **QAT/distillation
under ternary forward constraints**, plus a runtime format that preserves the
scale semantics the distilled model learned.

This repository is therefore a research fork, not a claim that arbitrary Qwen,
Kimi, or other open-weight models can be converted to 1.58-bit form with no
quality loss.

## What This Fork Adds

The upstream Microsoft BitNet project provides optimized inference machinery
for native BitNet-style models. This fork uses that codebase to test a harder
question: can we take standard pretrained models and make them CPU-native after
the fact?

Added work in this fork includes:

- Qwen2.5 dense-model retrofit experiments with BitLinear replacement.
- FSDP-compatible ternary checkpoint export repair.
- Post-training ternarization, QAT, KL-only distillation, dense-head, and
  row-scale ablations.
- Ten-task lm-eval, WikiText, FineWeb-heldout, prompt sanity, paired-delta,
  CPU throughput, file-size, and RSS/context scaling reports.
- Direct static-ternary GGUF export for dense Qwen checkpoints.
- A stable row-scale `I2_SR` llama.cpp qtype/file type for preserving row-wise
  ternary scales in packed CPU inference.
- A BitDistill smoke contract that now validates PyTorch QAT, tensor-scale
  `I2_S` GGUF export, row-scale `I2_SR` GGUF export, and SubLN key remapping.
- MoE/Kimi feasibility audits that separate generic routing support from real
  Kimi/MoE benchmark evidence.
- A tiny random Qwen2MoE FP16 GGUF runtime fixture proving generic converter
  and CPU execution plumbing, without claiming Kimi or ternary MoE quality.

The llama.cpp submodule now points at the writable fork:

`https://github.com/sabdulmajid/llama.cpp`

with the active `i2sr-row-scale-runtime` branch.

## Current Verdict

| claim | status | evidence |
| --- | --- | --- |
| Arbitrary FP16/BF16 to ternary conversion is lossless | **No** | Qwen2.5-1.5B naive PTQ collapses from ten-task mean `0.644169` to `0.348671`; WikiText PPL jumps from `13.901` to `3,813,121.803`. |
| Distillation/QAT can recover useful signal | **Yes, partially** | Best row-scale dense-Qwen run reaches ten-task mean `0.499459`, well above naive PTQ but below FP. |
| Stable CPU row-scale packed inference exists for dense Qwen | **Yes, for the audited path** | `I2_SR` productization gate passes `9/9`; Xeon I2_SR PPL `38.8477`, prompt `211.67 tok/s`, decode `19.07 tok/s`. |
| BitDistill paper-level GLUE reproduction is achieved here | **No, not yet** | Qwen2.5-0.5B short-budget GLUE3 sequence-classification runs remain 11.0-30.2 accuracy points below FP16-SFT. |
| TL2 is ready for the best row-scale checkpoint | **No** | Runtime contract gate fails: current TL2 one-scale error is `1.904230` relative output RMS; exact fp16 row scales would be `0.000197` with only `1.230469` MiB of scales, but converter/runtime/kernel metadata do not carry them. |
| Kimi/MoE retrofit is proven | **No** | A tiny random Qwen2MoE FP16 fixture now passes converter/runtime smoke. A Kimi-K2 config audit shows the real target needs Kimi/DeepSeekV3 loading, MLA metadata conversion, shared-expert mapping, block-FP8 import, and MoE-aware I2_SR validation before quality or speed claims are defensible. |

## BitDistill Reproduction Status

Microsoft's BitDistill paper changes the answer from "PTQ is enough" to
"task-specific QAT/distillation may work if the training recipe is strong
enough." That does not contradict the negative PTQ result in this fork.

This fork now implements the key BitDistill components for Qwen-style models:
SubLN insertion, Stage-2 continued pretraining, Stage-3 CE + logits KL +
Q/K/V attention-relation distillation, attention-layer sweep support, and both
paper-style tensor-scale and experimental row-scale ternary students. New
BitDistill jobs default to paper-style logits KL scaling and paper-style
summation over Q/K/V relation losses.

The implementation smoke gate currently passes `40/40` checks, including
tensor-scale GGUF export and row-scale `I2_SR` GGUF export for a tiny
causal-LM BitDistill checkpoint. Those export checks use a smoke-only synthetic
tokenizer stub, so they prove tensor packing and metadata wiring, not text
generation quality.

The BitDistill claim-control reports now enforce full GLUE validation counts:
MNLI `9,815`, QNLI `5,463`, and SST2 `872` examples. Aggregate reproduction
rows, paired prediction traces, and CPU runtime rows cannot pass as quality
evidence unless their stored full-validation metrics match those counts. CPU
runtime sampling remains allowed for speed measurement, but the gate labels it
as sampled runtime and separately requires the full task-quality metric.
Packed causal-LM exports are also gated for provenance: the `I2_S`/`I2_SR`
GGUF file, converter summary, benchmark suite, RSS probe, qtype, SubLN mapping,
and manifest path must all agree.

Current completed Qwen2.5-0.5B GLUE sequence-classification results:

| task | FP16-SFT | BitNet-SFT | BitDistill tensor | BitDistill row |
| --- | ---: | ---: | ---: | ---: |
| MNLI | `0.807641` | `0.487621` | `0.525217` | `0.516556` |
| QNLI | `0.898957` | `0.596925` | `0.596925` | `0.618525` |
| SST2 | `0.925459` | `0.770642` | `0.815367` | `0.808486` |

The paired prediction audit now covers BitNet-SFT versus FP16-SFT on the full
GLUE validation splits. BitNet-SFT trails FP16-SFT by `31.89` accuracy points
on MNLI, `29.95` on QNLI, and `14.79` on SST2, with paired confidence
intervals and exact McNemar tests recorded in the evidence bundle.

These runs do **not** reproduce the paper target of being within 0.5-1.0
accuracy point of FP16-SFT. They are also now labeled as a short-budget
diagnostic rather than a fully paper-faithful reproduction, because the first
completed wave used the common KD convention of multiplying logits KL by
`temperature**2`; the BitDistill equations do not include that multiplier.
The completed wave also used a legacy Q/K/V mean for attention relation KD;
the current code defaults new jobs to the paper-style Q/K/V sum and records
that setting in each metrics file.

The strongest remaining known gap is training budget:
the completed Stage-2 warm-up used `40.96M` effective token presentations,
while the paper reports `10B` continued-pretraining tokens. The result should
therefore be read as a failure boundary for direct or short-warm-up retrofit,
not as a disproof of BitDistill.

Active follow-ups are probing teacher-head initialization, attention-layer
selection, paper-style logits KL scaling, CE-only ablations, a longer
warm-up pilot, and a strict paper-hyperparameter branch with classification
attention KD weight `1e5`. The active long-warmup queue also includes MNLI
gamma probes at `1e3` and `1e4` because the local relation-loss scale audit
showed the paper's `1e5` coefficient can dominate CE by orders of magnitude.
The earlier completed BitDistill runs use attention KD weight `100`; those are
useful diagnostics but are not a strict match to the paper's reported
classification setting. Until those gates close, the public claim remains
conservative:
**BitDistill is the right class of method, but this fork has not yet reproduced
paper-level task quality.**

Focused MNLI diagnostics after fixing logits-KL scaling and sweeping the
attention-distillation layer improved the best short-budget BitDistill result
to `0.535711` versus FP16-SFT `0.807641`. CE-only ablations stay near
`0.492-0.498`, so distillation helps, but the run is still far from
reproduction-quality.

Runtime boundary: the active paper-style GLUE reproduction uses
`Qwen2ForSequenceClassification`. Those checkpoints can be evaluated on CPU
with the PyTorch task benchmark in this fork, but they are **not** valid packed
llama.cpp / `I2_SR` exports today because the runtime path does not implement a
Qwen sequence-classification head. The stable `I2_SR` exporter is valid for
causal-LM BitDistill checkpoints; packed task inference requires either
causal prompt-scoring checkpoints or new classifier-head support in the
runtime. A causal prompt-scoring long-warmup branch is queued under
`checkpoints/bitdistill-glue-causal-longwarmup-densehead` to test that
exportable path without conflating it with the sequence-classification
reproduction gate.

## Key Dense-Qwen Results

Quality measurements below are Qwen2.5-1.5B unless noted.

| run | WikiText PPL | FineWeb PPL | ten-task mean |
| --- | ---: | ---: | ---: |
| FP reference | `13.901` | `10.269` | `0.644169` |
| naive PTQ ternary | `3,813,121.803` | `9,582,923.269` | `0.348671` |
| QAT hidden-MSE | `86.414` | `40.398` | `0.464809` |
| QAT KL-only | `50.595` | `26.599` | `0.483438` |
| QAT KL-only, dense tied `lm_head` | `43.372` | `22.759` | `0.484378` |
| QAT KL-only, row-scale, dense tied `lm_head` | `38.580` | `21.333` | `0.499459` |

Paired ten-task deltas:

- Row-scale QAT minus naive PTQ: `+0.150788`, 95% CI `[+0.053427, +0.248149]`.
- Row-scale QAT minus FP: `-0.144710`, 95% CI `[-0.185756, -0.103664]`.
- Row-scale QAT minus tensor-scale dense-head QAT: `+0.015081`, 95% CI `[+0.009028, +0.021134]`.

Interpretation: row-scale distillation is a real recovery path, but it does not
close the gap to FP quality.

## Xeon CPU Runtime Snapshot

These are fixed-excerpt llama.cpp CPU runs on an Intel Xeon Silver 4116
portable-AVX2 build. Throughput should be compared only within this hardware
and build context.

| artifact | file MiB | PPL | prompt tok/s | decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| FP F16 | `2950.4` | `12.2808` | `114.47` | `5.56` |
| FP Q8_0 | `1570.3` | `12.3056` | `124.86` | `10.13` |
| FP Q4_K_M | `940.4` | `12.8112` | `92.08` | `16.01` |
| blind FP-to-I2_S | `766.1` | catastrophic | `204.57` | `18.34` |
| row-scale ternary TQ2_0 | `1218.6` | `38.8224` | `169.46` | `18.68` |
| row-scale ternary I2_S prototype | `1211.3` | `38.8832` | `218.17` | `18.97` |
| row-scale ternary I2_SR | `1211.3` | `38.8477` | `211.67` | `19.07` |

`I2_SR` is the stable row-scale path in this fork. It preserves the row-scale
QAT checkpoint quality while keeping the packed ternary CPU throughput benefit.
It does not make blind PTQ viable.

## What Is Proven

1. Naive post-training ternarization destroys dense pretrained model quality in
   the tested Qwen artifacts.
2. Training or distillation under ternary constraints recovers measurable
   quality.
3. Row-wise scales matter; collapsing row-wise scale information into one
   tensor scale breaks the best checkpoint.
4. A stable packed CPU runtime can preserve row-scale quality when the file
   format and kernels represent the same scale semantics.
5. The current dense-Qwen work is useful as a research MVP and evaluator, not as
   a universal model converter.
6. The current short-budget BitDistill reproduction fails GLUE3, reinforcing
   that continued pretraining and distillation budget are core parts of the
   method rather than implementation details.

## What Is Not Proven

- No evidence supports a one-click arbitrary FP16/BF16-to-ternary product.
- No evidence shows FP-quality 1.58-bit Qwen from this retrofit recipe.
- No completed local run yet reproduces BitDistill paper-level GLUE quality.
- No packed `I2_SR` path currently runs the sequence-classification GLUE heads.
- No evidence validates Kimi or a trained MoE model in this ternary runtime.
- The Kimi-K2 config path is specifically blocked on Kimi/DeepSeekV3 mapping,
  MLA/Q-LoRA attention metadata, shared experts, FP8 block import, and trained
  MoE quality/locality benchmarks.
- TL2 remains an engineering probe for row-scale Qwen, not a supported product
  path for the strongest checkpoint.

## Repository Map

| path | purpose |
| --- | --- |
| `benchmarks/` | benchmark, audit, conversion, and report-generation scripts |
| `benchmarks/results/` | public Markdown reports with parsed results and verdicts |
| `benchmark_results/` | raw JSON summaries and benchmark artifacts |
| `experiments/` | small mathematical audits and viability probes |
| `patches/` | historical/runtime patch records used by the audits |
| `utils/` | Hugging Face conversion and preprocessing utilities |
| `3rdparty/llama.cpp` | llama.cpp fork with active `I2_SR` support |
| `src/ggml-bitnet-mad.cpp` | BitNet CPU quantization/runtime integration |

## Reproducing The Current Gates

The main reports are generated from checked-in benchmark artifacts:

```bash
python benchmarks/audit_i2sr_submodule_promotion.py \
  --check-remote-write \
  --candidate-fork-url https://github.com/sabdulmajid/llama.cpp.git \
  --output-json benchmark_results/i2sr_submodule_promotion_audit_2026-05-13.json \
  --output-md benchmarks/results/i2sr_submodule_promotion_audit_2026-05-13.md

python benchmarks/audit_product_scope.py \
  --output-json benchmark_results/product_scope_gate_2026-05-13.json \
  --output-md benchmarks/results/product_scope_gate_2026-05-13.md

python benchmarks/audit_objective_completion.py \
  --output-json benchmark_results/objective_completion_audit_2026-05-14.json \
  --output-md benchmarks/results/objective_completion_audit_2026-05-14.md

python benchmarks/monitor_bitdistill_jobs.py \
  --output-json benchmark_results/bitdistill_job_monitor_2026-05-14.json \
  --output-md benchmarks/results/bitdistill_job_monitor_2026-05-14.md

python benchmarks/gate_bitdistill_reproduction.py \
  --output-json benchmark_results/bitdistill_reproduction_gate_2026-05-14.json \
  --output-md benchmarks/results/bitdistill_reproduction_gate_2026-05-14.md

python benchmarks/audit_bitdistill_paired_predictions.py \
  --output-json benchmark_results/bitdistill_paired_predictions_2026-05-14.json \
  --output-md benchmarks/results/bitdistill_paired_predictions_2026-05-14.md

python benchmarks/audit_bitdistill_paper_alignment.py \
  --output-json benchmark_results/bitdistill_paper_alignment_2026-05-14.json \
  --output-md benchmarks/results/bitdistill_paper_alignment_2026-05-14.md

python benchmarks/audit_bitdistill_loss_scales.py \
  --output-json benchmark_results/bitdistill_loss_scale_audit_2026-05-14.json \
  --output-md benchmarks/results/bitdistill_loss_scale_audit_2026-05-14.md

python benchmarks/benchmark_bitdistill_glue_cpu.py \
  --tasks mnli qnli sst2 \
  --runs short:fp16_sft-tensor-layer-1 short:bitnet_sft-tensor-layer-1 short:bitdistill-tensor-layer-1 short:bitdistill-row-layer-1 \
  --max-eval-samples 128 \
  --threads 12 \
  --output-json benchmark_results/bitdistill_glue_cpu_2026-05-14.json \
  --output-md benchmarks/results/bitdistill_glue_cpu_2026-05-14.md

python benchmarks/gate_bitdistill_cpu_benchmark.py \
  --input-json benchmark_results/bitdistill_glue_cpu_latest.json \
  --output-json benchmark_results/bitdistill_glue_cpu_gate_2026-05-14.json \
  --output-md benchmarks/results/bitdistill_glue_cpu_gate_2026-05-14.md

python benchmarks/gate_bitdistill_i2sr_export.py \
  --results-dir benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14 \
  --output-json benchmark_results/bitdistill_i2sr_export_gate_2026-05-14.json \
  --output-md benchmarks/results/bitdistill_i2sr_export_gate_2026-05-14.md

python benchmarks/submit_bitdistill_afterany_postprocess.py

python benchmarks/submit_bitdistill_warmup_finalizer.py

python benchmarks/audit_bitdistill_postprocess_dependencies.py \
  --postprocess-job-name bitdistill-postprocess-any \
  --postprocess-job-id <afterany-job-id> \
  --output-json benchmark_results/bitdistill_afterany_postprocess_dependency_2026-05-14.json \
  --output-md benchmarks/results/bitdistill_afterany_postprocess_dependency_2026-05-14.md

MAX_EVAL_SAMPLES=512 sbatch --dependency=afterany:<downstream-job-ids> \
  slurm_bitdistill_cpu_benchmark.sh

python benchmarks/build_qwen_side_by_side.py \
  --output-md benchmarks/results/qwen_side_by_side_2026-05-14.md

python benchmarks/run_tiny_qwen2moe_fixture.py --skip-existing
```

Build smoke used for the active runtime:

```bash
cmake --build build-portable-avx2 --target llama-cli llama-bench llama-perplexity llama-quantize -j 12
./build-portable-avx2/bin/llama-quantize --help | rg "I2_SR|I2_S"
```

## Primary Reports

- [Qwen side-by-side summary](benchmarks/results/qwen_side_by_side_2026-05-14.md)
- [BitDistill reproduction status](benchmarks/results/bitdistill_reproduction_status_2026-05-14.md)
- [BitDistill active job monitor](benchmarks/results/bitdistill_job_monitor_2026-05-14.md)
- [BitDistill dependency graph audit](benchmarks/results/bitdistill_dependency_graph_2026-05-14.md)
- [BitDistill warm-up health audit](benchmarks/results/bitdistill_warmup_health_2026-05-14.md)
- [BitDistill active goal audit](benchmarks/results/bitdistill_active_goal_audit_2026-05-14.md)
- [BitDistill reproduction gate](benchmarks/results/bitdistill_reproduction_gate_2026-05-14.md)
- [BitDistill paper alignment audit](benchmarks/results/bitdistill_paper_alignment_2026-05-14.md)
- [BitDistill loss-scale audit](benchmarks/results/bitdistill_loss_scale_audit_2026-05-14.md)
- [BitDistill GLUE CPU gate](benchmarks/results/bitdistill_glue_cpu_gate_2026-05-14.md)
- [BitDistill causal I2_SR export gate](benchmarks/results/bitdistill_i2sr_export_gate_2026-05-14.md)
- [BitDistill GLUE3 primary summary](benchmarks/results/bitdistill_seqcls_glue3_primary_summary_2026-05-14.md)
- [BitDistill MNLI diagnostic variants](benchmarks/results/bitdistill_seqcls_mnli_diagnostic_variant_summary_2026-05-14.md)
- [Objective completion audit](benchmarks/results/objective_completion_audit_2026-05-14.md)
- [Product scope gate](benchmarks/results/product_scope_gate_2026-05-14.md)
- [Benchmark coverage gate](benchmarks/results/benchmark_coverage_gate_2026-05-14.md)
- [I2_SR submodule promotion audit](benchmarks/results/i2sr_submodule_promotion_audit_2026-05-13.md)
- [Row-scale qtype productization gate](benchmarks/results/row_scale_qtype_productization_gate_2026-05-13.md)
- [Direct packed GGUF support audit](benchmarks/results/direct_packed_gguf_support_2026-05-13.md)
- [TL2 row-scale design audit](benchmarks/results/tl2_row_scale_design_2026-05-13.md)
- [TL2 row-scale runtime contract](benchmarks/results/tl2_row_scale_runtime_contract_2026-05-14.md)
- [MoE support audit](benchmarks/results/moe_support_audit_2026-05-14.md)
- [Tiny Qwen2MoE runtime fixture](benchmarks/results/tiny_qwen2moe_fixture_2026-05-14.md)
- [Unblock requirements audit](benchmarks/results/unblock_requirements_2026-05-14.md)
- [Evidence manifest](benchmarks/results/evidence_manifest_2026-05-14.md)

## Product Direction

The credible product is not "convert any model to 1.58-bit." The credible
product is a CPU-first retrofit evaluator and distillation pipeline:

1. Ingest a dense Hugging Face checkpoint.
2. Replace selected linear projections with BitLinear-compatible modules.
3. Distill with KL loss under ternary forward constraints.
4. Preserve learned scale semantics in GGUF using `I2_SR`.
5. Report quality, size, throughput, RSS, and paired statistical deltas before
   any model is advertised as usable.

MoE/Kimi should be treated as the next research milestone, requiring licensed
artifacts, router/expert distillation, ternary expert runtime validation,
expert-locality measurement, and CPU quality/throughput/RSS benchmarks. The
tiny Qwen2MoE fixture only proves generic FP16 converter/runtime plumbing.

## Upstream Attribution

This fork is based on Microsoft's BitNet / bitnet.cpp work and uses llama.cpp
as the CPU inference substrate. Upstream BitNet release claims apply to native
BitNet models, not to the arbitrary-retrofit experiments reported here.
