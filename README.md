# BitNet Retrofit Research Fork

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This fork investigates whether pretrained FP16/BF16 language models can be
retrofitted into BitNet-style W1.58A8 / ternary CPU inference.

**Binary answer from the current evidence: no, blind post-training
ternarization is not a viable general retrofit path for arbitrary pretrained
models.** The useful direction is task-specific QAT/distillation under ternary
forward constraints, plus a CPU runtime format that preserves the scale
semantics learned during training.

This repository is therefore a research and evaluation fork. It does not claim
that arbitrary Qwen, Kimi, or other open-weight models can be converted to
1.58-bit form with no quality loss.

## Current Thesis

Extreme ternary quantization is not just a storage format. It changes the model
family:

```text
FP model:       W in R^(m x n)
ternary model:  W ~= scale * T, where T in {-1, 0, +1}^(m x n)
```

Blind PTQ asks whether an FP solution can be projected into that constrained
family after training. The tested answer is no. QAT and distillation ask whether
the solution can be moved into the ternary family while preserving task
behavior. The tested answer is partially, but not yet at paper-level quality in
this fork.

The strongest original systems result here is that row-scale ternary students
need a matching row-scale packed runtime contract. Collapsing learned row scales
to one tensor scale breaks outputs; preserving them through `I2_SR` keeps the
packed CPU path faithful to the trained checkpoint.

## Claim Ledger

| claim | status | evidence |
| --- | --- | --- |
| Arbitrary FP16/BF16 to ternary conversion is lossless | **No** | Qwen2.5-1.5B naive PTQ drops ten-task mean from `0.644169` to `0.348671`; WikiText PPL jumps from `13.901` to `3,813,121.803`. |
| QAT/distillation recovers useful signal | **Yes, partially** | Best row-scale dense-Qwen run reaches ten-task mean `0.499459`, improving over naive PTQ by `+0.150788` paired mean accuracy. |
| Row-scale packed CPU inference is viable for compatible dense causal artifacts | **Yes, for the audited path** | `I2_SR` row-scale Qwen2.5-1.5B Xeon run: PPL `38.8477`, prompt `211.67 tok/s`, decode `19.07 tok/s`, file `1211.3 MiB`. |
| BitDistill paper-level GLUE reproduction is achieved | **No, not yet** | Local FP16-SFT MNLI is close to the paper anchor (`0.807641` vs `0.799100`), and BitNet-SFT now clears its paper anchor (`0.628935` vs `0.608000`) after more budget. This is CE-only BitNet-SFT, not BitDistill or FP16-level recovery. |
| Row-scale `I2_SR` is standard BitNet | **No** | It is a fork-specific retrofit variant for row-scale students, not the upstream per-tensor BitNet format. |
| Kimi/MoE retrofit is proven | **No** | Tiny Qwen2MoE fixtures prove converter/runtime plumbing only. No Kimi-specific mapping, trained MoE quality, or real expert-locality benchmark is proven. |

## Evidence Labels

| label | meaning |
| --- | --- |
| `paper-reproduction` | Same task family, model class, quantization semantics, and recipe target as the BitDistill paper. These rows must use full validation splits and close the FP16 gap. |
| `paper-inspired` | Uses BitDistill-like ingredients but changes budget, backbone, scale granularity, loss normalization, or task formulation. These rows are diagnostics, not reproduction claims. |
| `retrofit-variant` | Fork-specific extensions such as row-scale ternary and `I2_SR`. These are evaluated as original systems variants, not as standard BitNet or paper-equivalent results. |

## Key Results

### Dense Qwen Retrofit

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

Interpretation: row-scale QAT is a real recovery path, but it does not close the
gap to FP quality.

### BitDistill Reproduction Status

The strict local reproduction branch uses `Qwen2ForSequenceClassification` and
full GLUE validation counts: MNLI `9815`, QNLI `5463`, SST2 `872`.

| result | local | paper anchor / target |
| --- | ---: | ---: |
| Qwen2.5-0.5B MNLI FP16-SFT | `0.807641` | `0.799100` |
| Qwen2.5-0.5B MNLI BitNet-SFT default | `0.487621` | `0.608000` |
| Qwen2.5-0.5B MNLI BitNet-SFT best completed budget row | `0.628935` | `0.608000` |
| Qwen2.5-0.5B MNLI best current long-warmup row-scale diagnostic | `0.653591` | `retrofit-variant`; not a paper-reproduction row |

The earlier BitNet-SFT failure was substantially a budget/schedule issue:
`1000`- and `3000`-step rows were undertrained, while the first `10000`-step row
clears the paper's BitNet-SFT anchor. The current blocker has shifted from
baseline viability to BitDistill-style recovery toward the FP16 task model.

Current BitNet-SFT controls:

- Expected ternary decoder projection tensors: `168/168`.
- Classifier head remains dense.
- Weights-only/no-A8 control: `0.493734`, only `+0.006113` over W1.58A8.
- SubLN-only local control: `0.350280`, so current SubLN insertion by itself
  worsens the local baseline.
- SubLN activation audit on a local Qwen2.5-0.5B checkpoint inserts `48`
  modules and shows untrained architecture surgery is not identity-preserving:
  last-token logit relative RMS drift `0.768044`, cosine `0.698252`, top-1
  agreement `0.000000`.
- Best completed budget row: `0.628935` at `10000` steps, `2e-5`, exceeding
  the paper BitNet-SFT anchor by `0.020935` but still `0.178706` below the
  local FP16-SFT row.
- Paired prediction audit against the FP16 trace gives candidate minus FP16
  `-0.179215`, 95% CI `[-0.189580, -0.168851]`, McNemar
  `p=3.438389e-240`.

The completed `3000`-step sweep improves over the best `1000`-step row, and the
first `10000`-step row improves again. This does not prove BitDistill; it proves
that the short-budget BitNet-SFT baseline was not a sufficient reproduction
budget.

### Xeon CPU Runtime

Fixed-excerpt llama.cpp CPU runs on Intel Xeon Silver 4116, portable AVX2 build.
Compare throughput only within this hardware/build context.

| artifact | file MiB | PPL | prompt tok/s | decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| FP F16 | `2950.4` | `12.2808` | `114.47` | `5.56` |
| FP Q8_0 | `1570.3` | `12.3056` | `124.86` | `10.13` |
| FP Q4_K_M | `940.4` | `12.8112` | `92.08` | `16.01` |
| blind FP-to-I2_S | `766.1` | catastrophic | `204.57` | `18.34` |
| row-scale ternary TQ2_0 | `1218.6` | `38.8224` | `169.46` | `18.68` |
| row-scale ternary I2_SR | `1211.3` | `38.8477` | `211.67` | `19.07` |

`I2_SR` proves that a row-scale ternary checkpoint can be represented in a
packed CPU format and run faster than FP16 decode while preserving that
checkpoint's scale semantics. It does not make blind PTQ viable, and it does
not beat Q4_K_M on quality for the current Qwen2.5-1.5B artifact.

Normalized against FP Q4_K_M on the same Xeon run, row-scale `I2_SR` is
`1.288133x` the file size, `1.268574x` the RSS at context 512, `2.298818x` the
prefill throughput, `1.190617x` the decode throughput, and `3.032323x` the PPL.
This is a speed/runtime-semantics result, not a mature Q4 quality/storage win.
Using the recorded `llama-bench` standard deviations across three repetitions,
the Q4-normalized `I2_SR` speedup intervals remain above `1.0`: prefill
`[2.257867, 2.340511]`, decode `[1.186108, 1.195143]`.

## What This Fork Adds

- Mathematical and empirical PTQ audits showing why blind FP/BF16 to ternary
  projection collapses tested dense-Qwen checkpoints.
- Qwen2.5 dense-model retrofit experiments with BitLinear replacement,
  ternary checkpoint export repair, QAT, KL distillation, dense-head, tensor
  scale, and row-scale variants.
- Full lm-eval, WikiText, FineWeb-heldout, paired-delta, CPU throughput,
  file-size, and RSS/context-scaling reports.
- Direct static-ternary GGUF export for dense Qwen checkpoints.
- A stable row-scale `I2_SR` GGUF/qtype path for preserving row-wise ternary
  scales in packed CPU inference.
- BitDistill-style training components for Qwen-style models: SubLN insertion,
  Stage-2 continued pretraining, Stage-3 CE + logits KL + Q/K/V
  attention-relation distillation, attention-layer sweep support, and strict
  task-formulation gates.
- MoE/Kimi feasibility audits that separate generic routing support from real
  Kimi or trained-MoE evidence.

The llama.cpp submodule points at the writable fork:

```text
https://github.com/sabdulmajid/llama.cpp
```

with the active `i2sr-row-scale-runtime` branch.

## Not Yet Proven

- One-click universal FP/BF16-to-ternary conversion.
- Paper-level BitDistill reproduction.
- FP-quality 1.58-bit Qwen from this retrofit recipe.
- Packed llama.cpp support for `Qwen2ForSequenceClassification` heads.
- General-LM quality for task-distilled causal prompt-scoring exports.
- Kimi or trained MoE quality, speed, memory, routing locality, or CPU
  product viability.
- TL2 support for row-scale Qwen; current TL2 one-scale error is `1.904230`
  relative output RMS, while exact FP16 row scales would be `0.000197`.

## Canonical Next Matrix

The next experiments are intentionally narrow:

| axis | setting |
| --- | --- |
| Base model | Qwen2.5-0.5B first; Qwen3-0.6B after baseline alignment |
| Task | MNLI first, then QNLI and SST2 |
| Formulation | `Qwen2ForSequenceClassification`, not causal prompt scoring |
| Baselines | FP16-SFT, BitNet-SFT, BitDistill |
| Quantization | paper-style tensor scale first; row-scale only as `retrofit-variant` |
| Budget curve | `1000`, `3000`, `10000` downstream steps now; larger Stage-2 token budgets if the curve justifies it |
| Success gate | full validation accuracy within about `0.5-1.0` point of FP16-SFT, with paired traces |

Decision rule:

- The first `10000`-step BitNet-SFT row clears the paper BitNet-SFT anchor, so
  the short-run failure was mostly undertraining/budget.
- The next interpretation gate is BitDistill/FP16 recovery, not the weaker
  CE-only BitNet-SFT anchor.
- Row-scale results should be reported as a separate runtime/retrofit
  contribution, not as a BitDistill reproduction.

## Product Direction

The credible product is not "convert any model to 1.58-bit." The credible
product is a CPU-first retrofit evaluator and distillation pipeline that tells
users whether a specific model-task pair is viable:

1. Ingest a dense Hugging Face checkpoint.
2. Replace selected projection layers with BitLinear-compatible modules.
3. Train or distill under ternary forward constraints.
4. Preserve learned scale semantics in GGUF using `I2_SR` when row-scale is
   selected.
5. Report quality delta, paired confidence intervals, PPL, file size, RSS,
   prompt/decode throughput, and a safe claim label.

This is useful even when the answer is negative, because it prevents
overclaiming and identifies the failure mode.

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

## Reproduce The Evidence Snapshot

The current public reports use `BITNET_REPORT_DATE=2026-05-15`.

```bash
export BITNET_REPORT_DATE=2026-05-15

python benchmarks/audit_bitnet_sft_budget_sweep.py
python benchmarks/audit_benchmark_coverage.py
python benchmarks/audit_product_scope.py
python benchmarks/build_evidence_manifest.py \
  --output-json benchmarks/results/evidence_manifest_2026-05-15.json \
  --output-md benchmarks/results/evidence_manifest_2026-05-15.md
```

Build smoke for the active packed runtime:

```bash
cmake --build build-portable-avx2 --target llama-cli llama-bench llama-perplexity llama-quantize -j 12
./build-portable-avx2/bin/llama-quantize --help | rg "I2_SR|I2_S"
```

## Primary Reports

- [Research redirect and next plan](benchmarks/results/research_redirect_2026-05-15.md)
- [Qwen side-by-side summary](benchmarks/results/qwen_side_by_side_2026-05-15.md)
- [BitDistill reproduction gap analysis](benchmarks/results/bitdistill_reproduction_gap_analysis_2026-05-15.md)
- [BitNet-SFT baseline audit](benchmarks/results/bitnet_sft_baseline_audit_2026-05-15.md)
- [BitNet-SFT recipe alignment audit](benchmarks/results/bitnet_sft_recipe_alignment_2026-05-15.md)
- [BitNet-SFT budget sweep audit](benchmarks/results/bitnet_sft_budget_sweep_2026-05-15.md)
- [BitNet-SFT budget paired audit](benchmarks/results/bitnet_sft_budget_paired_2026-05-15.md)
- [SubLN activation variance audit](benchmarks/results/subln_activation_variance_2026-05-15.md)
- [BitDistill paper alignment audit](benchmarks/results/bitdistill_paper_alignment_2026-05-15.md)
- [Task formulation audit](benchmarks/results/bitdistill_task_formulation_audit_2026-05-15.md)
- [Causal I2_SR export gate](benchmarks/results/bitdistill_i2sr_export_gate_2026-05-15.md)
- [CPU tradeoff frontier audit](benchmarks/results/cpu_tradeoff_frontier_2026-05-15.md)
- [CPU speed uncertainty audit](benchmarks/results/cpu_speed_uncertainty_2026-05-15.md)
- [Benchmark coverage gate](benchmarks/results/benchmark_coverage_gate_2026-05-15.md)
- [Product scope gate](benchmarks/results/product_scope_gate_2026-05-15.md)
- [Evidence manifest](benchmarks/results/evidence_manifest_2026-05-15.md)
- [TL2 row-scale runtime contract](benchmarks/results/tl2_row_scale_runtime_contract_2026-05-15.md)
- [Kimi config feasibility audit](benchmarks/results/kimi_config_feasibility_2026-05-15.md)

## Upstream Attribution

This fork is based on Microsoft's BitNet / bitnet.cpp work and uses llama.cpp
as the CPU inference substrate. Upstream BitNet release claims apply to native
BitNet models, not to the arbitrary-retrofit experiments reported here.
