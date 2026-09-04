# Ternary Retrofit Evaluator and CPU Runtime-Contract Tester

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This fork investigates whether pretrained FP16/BF16 language models can be
adapted into BitNet-style W1.58A8 ternary models for commodity CPU inference.

The current answer is deliberately narrow:

> Extreme ternary quantization is not a file-format conversion problem. It is a
> representation-learning problem plus a runtime-contract problem.

This repository is not a universal BitNet converter. The evidence so far
supports a CPU-first evaluation stack: test whether a model-task pair survives
ternary training/distillation, then verify that the packed runtime preserves the
same scale semantics the trained checkpoint learned.

## Claim Ledger

The frozen baseline bundle and current decision reports are:

- [canonical_evidence_bundle_2026-05-20.md](benchmarks/results/canonical_evidence_bundle_2026-05-20.md)
- [canonical_evidence_bundle_2026-05-20.json](benchmarks/results/canonical_evidence_bundle_2026-05-20.json)
- [bitdistill_benchmark_scoreboard_2026-05-23.md](benchmarks/results/bitdistill_benchmark_scoreboard_2026-05-23.md)
- [bitdistill_benchmark_scoreboard_2026-05-23.json](benchmarks/results/bitdistill_benchmark_scoreboard_2026-05-23.json)
- [seqcls_native_cpu_matrix_2026-09-04.md](benchmarks/results/seqcls_native_cpu_matrix_2026-09-04.md)
- [seqcls_native_cpu_repeated_inplace_2026-09-04.md](benchmarks/results/seqcls_native_cpu_repeated_inplace_2026-09-04.md)
- [seqcls_i2sr_runtime_ab_2026-09-04.md](benchmarks/results/seqcls_i2sr_runtime_ab_2026-09-04.md)
- [i2_kernel_profile_2026-09-04.md](benchmarks/results/i2_kernel_profile_2026-09-04.md)

| Claim | Status | Evidence | Caveat |
| --- | --- | --- | --- |
| Blind FP/BF16 to ternary PTQ works as a general retrofit | **No: strong negative result in the tested setup** | Qwen2.5-1.5B FP WikiText PPL `13.901`; naive ternary PTQ PPL `3,813,121.803`. FP ten-task mean `0.644169`; naive PTQ mean `0.348671`. | Dense Qwen2.5-1.5B tested setup; do not generalize as a theorem for every architecture. |
| QAT/distillation recovers signal | **Partial recovery, not FP quality** | Best row-scale QAT ten-task mean `0.499459`, a `+0.150788` recovery over naive PTQ and still `-0.144710` below FP. | Row-scale QAT is this fork's retrofit variant, not standard BitDistill. |
| BitDistill paper-level GLUE reproduction is complete | **No** | Qwen2.5-0.5B local FP16-SFT MNLI is `0.808151`. Fixed-gamma Stage-2 reaches `0.729903` at `655.36M`, delta `-0.078248`, CI `[-0.086720, -0.069775]`; the best completed loss-balanced tensor run is `0.738462`. | The loss-balanced run remains `-0.069689` below FP16 with paired CI `[-0.078431, -0.060947]`; the paper-level recovery target is not met. |
| The `655.36M` Stage-2 checkpoint is usable | **Yes, with a verified manifest** | [stage2_manifest_655m_2026-05-23.md](benchmarks/results/stage2_manifest_655m_2026-05-23.md) records job `10250`, four complete snapshots, final CE `3.426713`, the state-dict SHA-256, and downstream job `10260`. | This was a `327.68M` continuation with a fresh optimizer/scheduler segment, not one uninterrupted 80k-step run. |
| Paper gamma can be copied literally into this implementation | **No, not without matching loss normalization** | Historical telemetry measures attention/CE gradient ratio `221.384986` for the paper-gamma path versus `0.346044` at gamma 60. A controlled A4500 screen measures median ratio `69.2248` for cosine split-1 at fixed gamma `100,000`, versus `0.273975` with adaptive balancing. | This is not a task-quality result. The source-pinned dualcard replication remains required before a paper-aligned claim; the active 10k runs are explicitly cross-environment. |
| The paper defines one unambiguous attention-relation objective | **No** | [attention_relation_equivalence_2026-09-04.md](benchmarks/results/attention_relation_equivalence_2026-09-04.md) proves that Equation 12 scaled-dot relations and Algorithm 1 normalized-cosine relations are not generally equivalent. In a deterministic probe their gradient-norm ratio is `18.7073` and gradient cosine is `0.2437`. | This is a mathematical contract result, not downstream quality evidence. For Qwen's 14:2 grouped-query attention, KV repetition leaves cosine relations invariant but multiplies scaled-dot logits by `sqrt(7)`. |
| The local GLUE formulation is paper-exact | **Unresolved** | [bitdistill_task_formulation_audit_2026-09-04.md](benchmarks/results/bitdistill_task_formulation_audit_2026-09-04.md) separates sequence-classification from causal answer-token results. | Token-level CE and decoding language favor the causal interpretation, but no authoritative released templates or training code establish equivalence. |
| Row-scale semantics matter at runtime | **Yes: strong systems result** | TL2 one-scale relative output RMS error `1.904230`; exact FP16 row scales reduce it to `0.000197`. | Row scales are part of the learned function. TL2 row-scale support is not implemented. |
| `I2_SR` packed CPU inference works | **Yes, for compatible causal artifacts** | Xeon Silver 4116: row-scale `I2_SR` file `1211.3 MiB`, PPL `38.8477`, prompt `211.67 tok/s`, decode `19.07 tok/s`. | It does **not** beat Q4_K_M on quality or file size. Q4_K_M is `940.4 MiB` with PPL `12.8112`. |
| Native packed sequence classification preserves task quality | **Yes, for one audited artifact; not product-ready** | Full MNLI native sequence-isolated path: `0.652165` versus PyTorch `0.653591`, paired delta `-0.001426`, 95% CI `[-0.004193, 0.001341]`, exact McNemar `p=0.348`; `7.456204` examples/s and RSS `960.15 MiB`. | Exact prediction agreement is `0.976668`, the 0.5-point non-inferiority gate is retrospective, multi-prompt batching remains excluded, and the underlying model quality is weak. |
| Mixed `I2_SR` plus Q8 embedding improves deployed storage | **Yes, on one Qwen2.5 classifier** | The packed artifact is `230.90 MiB`: `4.106x` smaller than FP16 and `1.527x` smaller than the F16-embedding I2_SR artifact. On 512 MNLI examples it changes accuracy by `-0.001953`, paired CI `[-0.011719, 0.007812]`, with `0.982422` prediction agreement. | This is a same-student format comparison on a fixed, non-random sample; it is not evidence of FP-quality recovery. |
| Removing I2 output staging accelerates the classifier runtime | **Yes, for both audited I2_SR artifacts** | Old/new binaries, four rotated pinned pairs: base I2_SR speed ratio `1.4619`, 95% CI `[1.3686, 1.5616]`; mixed I2_SR+Q8 `1.4358`, CI `[1.2857, 1.6035]`. Logits are bit-identical. | `ggml.c` is the only fingerprinted source difference. This is a local implementation effect, not model-quality recovery. |
| Packed ternary accelerates native sequence classification on the Xeon 4116 | **No, even after the runtime optimization** | Four interleaved pinned runs: I2_SR/FP16 geometric throughput ratio `0.650`, 95% CI `[0.646, 0.653]`; mixed I2_SR+Q8 ratio `0.605`, CI `[0.603, 0.607]`. | This does not contradict the causal decode result. Kernel benefit is workload-, shape-, and execution-path-dependent. |
| A8 activation quantization is the main remaining I2 projection bottleneck | **No: packed I2 arithmetic dominates** | Seven pinned one-core profiles at Qwen2.5-0.5B shapes put A8 quantization at `5.49%` of weighted projection cost, CI `[5.29%, 5.69%]`, versus `94.51%` for I2 GEMM. Making A8 quantization free has an ideal `1.0581x` upper bound. | This is a scalar-verified kernel decomposition, not end-to-end attribution. |
| Kimi/MoE support is proven | **No: not supported** | Only tiny Qwen2MoE fixture/plumbing exists. | No trained Kimi quality, MLA/shared-expert mapping, routed expert locality, or CPU product result is proven. |

## Current Reproduction Gap

The latest focused gap report is:

- [bitdistill_reproduction_gap_2026-05-23.md](benchmarks/results/bitdistill_reproduction_gap_2026-05-23.md)
- [bitdistill_reproduction_gap_2026-05-23.json](benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json)

The current objective snapshot is:

- [current_goal_status_2026-05-23.md](benchmarks/results/current_goal_status_2026-05-23.md)
- [current_goal_status_2026-05-23.json](benchmarks/results/current_goal_status_2026-05-23.json)

For external technical review or a deep-research agent, use:

- [EXPERIMENTS.md](EXPERIMENTS.md)
- [deep_research_handoff_2026-05-23.md](benchmarks/results/deep_research_handoff_2026-05-23.md)
- [deep_research_handoff_2026-05-23.json](benchmarks/results/deep_research_handoff_2026-05-23.json)
- [bitdistill_goal_traceability_2026-05-23.md](benchmarks/results/bitdistill_goal_traceability_2026-05-23.md)
- [bitdistill_goal_traceability_2026-05-23.json](benchmarks/results/bitdistill_goal_traceability_2026-05-23.json)
- [bitdistill_paper_alignment_2026-05-23.md](benchmarks/results/bitdistill_paper_alignment_2026-05-23.md)
- [bitdistill_paper_alignment_2026-05-23.json](benchmarks/results/bitdistill_paper_alignment_2026-05-23.json)
- [bitdistill_publication_product_plan_2026-05-23.md](benchmarks/results/bitdistill_publication_product_plan_2026-05-23.md)
- [bitdistill_publication_product_plan_2026-05-23.json](benchmarks/results/bitdistill_publication_product_plan_2026-05-23.json)

The short default BitNet-SFT row was undertrained: `0.487621` improves to
`0.628935` at 10k steps, `+0.020935` above the paper's Qwen2.5-0.5B
BitNet-SFT anchor. This resolves the earlier baseline-sanity concern, but does
not reproduce BitDistill.

## Completed 655M Gate

Job `10250` completed a controlled continuation from `327.68M` to
`655.36M` cumulative Stage-2 token presentations. Downstream job `10260`
then evaluated all `9,815` MNLI validation examples with a paired prediction
trace:

| Stage-2 tokens | MNLI | Gain vs prior row | Delta vs FP16 |
| ---: | ---: | ---: | ---: |
| `40.96M` | `0.616607` | - | `-0.191544` |
| `163.84M` | `0.691187` | `+0.074580` | `-0.116964` |
| `327.68M` | `0.720020` | `+0.028833` | `-0.088130` |
| `655.36M` | `0.729903` | `+0.009883` | `-0.078248` |

The result is useful but not a reproduction: doubling the latest Stage-2
budget bought less than one accuracy point, and the paired confidence interval
still excludes the FP16 recovery gate by a wide margin.

A conditional saturation audit now quantifies why the unchanged fixed-gamma
recipe should not simply be scaled to 10B token presentations. Fitting the last
two exact budget doublings gives a gain contraction of `0.342756`, a projected
10B accuracy of `0.734981`, and a paired-bootstrap 95% interval
`[0.723750, 0.755819]`. The interval remains below the pre-registered
`0.798151` recovery target. Even repeating the latest gain without further
decay projects `0.768758` with bootstrap interval `[0.741530, 0.795733]`;
recovery would require the average future gain per doubling to be `1.756x`
the latest observed gain. This is evidence against budget-only scaling for this
fixed local recipe, not evidence against BitDistill generally:
[bitdistill_stage2_saturation_2026-09-04.md](benchmarks/results/bitdistill_stage2_saturation_2026-09-04.md).

## Completed Loss-Scale Quality Control

A previously completed 10,000-step run pair provides a matched historical
quality test of the loss-scale hypothesis at the `163.84M` Stage-2 checkpoint.
The run declaring local attention-KD coefficient `60` raises MNLI over the run
declaring `100,000` from
`0.691187` to `0.738462`: delta `+0.047275`, paired 95% CI
`[0.039256, 0.055293]`, exact McNemar `p=9.07e-31`. Despite using one quarter
of the Stage-2 budget, it also exceeds the `655.36M` fixed-gamma result by
`+0.008558`, CI `[0.000919, 0.016197]`. It remains `-0.069689` behind FP16.

Every available serialized training field matches, and step-1 CE, logits-KD,
and attention-KD values are exactly identical, fingerprinting the same initial
state and first batch. This is strong local evidence that objective-scale
alignment is more valuable than additional Stage-2 compute under the old fixed
recipe. It does not validate gamma `60` across implementations; the historical
metrics predate source-revision and seed serialization, so the source-pinned
adaptive replications remain necessary. See
[bitdistill_gamma60_quality_2026-09-04.md](benchmarks/results/bitdistill_gamma60_quality_2026-09-04.md).

Evidence and decision artifacts:

- [stage2_655m_ingestion_2026-05-23.md](benchmarks/results/stage2_655m_ingestion_2026-05-23.md)
- [bitdistill_controlled_curve_2026-05-23.md](benchmarks/results/bitdistill_controlled_curve_2026-05-23.md)
- [bitdistill_stage2_saturation_2026-09-04.md](benchmarks/results/bitdistill_stage2_saturation_2026-09-04.md)
- [bitdistill_gamma60_quality_2026-09-04.md](benchmarks/results/bitdistill_gamma60_quality_2026-09-04.md)
- [gamma60_gradient_balance_2026-05-23.md](benchmarks/results/gamma60_gradient_balance_2026-05-23.md)
- [bitdistill_next_decision_2026-05-23.md](benchmarks/results/bitdistill_next_decision_2026-05-23.md)
- [bitdistill_next_experiment_blueprint_2026-05-23.md](benchmarks/results/bitdistill_next_experiment_blueprint_2026-05-23.md)
- [bitdistill_next_experiment_blueprint_2026-05-23.json](benchmarks/results/bitdistill_next_experiment_blueprint_2026-05-23.json)
- [active_gate_watchdog_2026-05-23.md](benchmarks/results/active_gate_watchdog_2026-05-23.md)
- [active_gate_watchdog_2026-05-23.json](benchmarks/results/active_gate_watchdog_2026-05-23.json)

## Packed Classifier Quality Contract

The native `I2_SR` sequence-classification path now has a paired full-split
quality audit. On the same 9,815 MNLI labels, the saved GPU-BF16 PyTorch trace
is correct on `6,415` examples and the sequence-isolated CPU artifact is correct
on `6,401`. The runtime wins `89` discordant examples and loses `103`, yielding
delta `-0.001426`, paired bootstrap CI `[-0.004177, 0.001325]`, and exact
McNemar `p=0.348`. This passes a clearly labeled retrospective 0.5-accuracy-point
non-inferiority criterion.

This closes task-accuracy preservation for that artifact, not exact numerical
parity: predictions agree on `0.976668` of examples. It also does not repair the
checkpoint's weak absolute accuracy or validate multi-prompt batching. See
[seqcls_runtime_quality_equivalence_2026-09-04.md](benchmarks/results/seqcls_runtime_quality_equivalence_2026-09-04.md).

## Controlled Xeon Deployment Matrix

The native runtime now supports both Qwen2 sequence-classification heads and
the row-scale BitNet-Qwen classifier. Classifier graphs stop at the dense task
head instead of computing the unused `151,936`-token language-model logits.
All four artifacts below were evaluated with the same binary and linked-library
hashes, first 512 MNLI `validation_matched` examples, token IDs, sequence
isolation, 12 threads, and CPU affinity `0-11` on the Xeon Silver 4116.

| Artifact | Pre-deployment function | MNLI | GGUF MiB | Size vs FP16 |
| --- | --- | ---: | ---: | ---: |
| FP16 teacher | FP16-SFT teacher | `0.789062` | `948.11` | `1.000x` |
| Q4_0 teacher | Same teacher, Q4_0 format | `0.675781` | `335.84` | `2.823x` smaller |
| I2_SR student | Row-scale QAT student | `0.669922` | `352.62` | `2.689x` smaller |
| I2_SR + Q8 embedding | Same student, mixed format | `0.667969` | `230.90` | `4.106x` smaller |

Q4_0 versus FP16 isolates a same-model format effect: accuracy delta
`-0.113281`, paired CI `[-0.154297, -0.074219]`, exact McNemar
`p=4.842e-08`. I2_SR versus FP16 is instead a deployed-model comparison that
includes the student training gap. The cleanest new format result is mixed
I2_SR+Q8 versus base I2_SR: one net correct prediction is lost (`3` wins,
`4` losses), delta `-0.001953`, CI `[-0.011719, 0.007812]`, while storage
falls another `34.52%`.

Throughput comes from a separate four-repetition interleaved benchmark over a
fixed 128-example subset. Each run used the same 12 pinned physical cores and
predictions were stable across repetitions:

| Artifact | Mean tok/s | Geometric speed / FP16 | Paired 95% CI |
| --- | ---: | ---: | ---: |
| FP16 teacher | `440.733` | `1.000` | `[1.000, 1.000]` |
| Q4_0 teacher | `389.748` | `0.884` | `[0.878, 0.890]` |
| I2_SR student | `286.300` | `0.650` | `[0.646, 0.653]` |
| I2_SR + Q8 embedding | `266.500` | `0.605` | `[0.603, 0.607]` |

This is a useful negative systems result: the custom ternary path does not
accelerate short, sequence-isolated classification on this CPU. The causal
decode path can still benefit because it has a different matrix-shape and
memory-access regime. The high-vocabulary embedding also dominates small-Qwen
storage unless it is compressed separately; Q8 embedding export closes that
storage gap but does not improve this workload's throughput.

The same protocol also isolated and fixed one runtime defect. The old I2 path
allocated a temporary output for every matrix multiply, post-scaled it, copied
it into the graph output, and freed it. The new path writes raw accumulators to
the final destination and post-scales in place. A separate old/new binary A/B
holds models, prompts, flags, cores, and affinity fixed; only the fingerprinted
`ggml.c` differs. Base I2_SR improves `1.4619x`, CI `[1.3686, 1.5616]`, and
mixed I2_SR+Q8 improves `1.4358x`, CI `[1.2857, 1.6035]`, with zero logit
difference. This is a real kernel improvement, although it is not enough to
overtake FP16 on the classifier workload.

A scalar-verified kernel profile now localizes the remaining projection cost.
Across seven process repetitions on one pinned core, activation quantization is
`5.49%` of the Qwen2.5-0.5B projection-weighted total, CI
`[5.29%, 5.69%]`; packed I2 GEMM is `94.51%`. By Amdahl's law, deleting A8
quantization entirely could improve this isolated mix by at most `1.0581x`.
The next runtime work must therefore change packed-dot arithmetic or data
layout, not merely vectorize the activation prepass. Every raw accumulator in
the profile matches a scalar decoder exactly. See
[i2_kernel_profile_2026-09-04.md](benchmarks/results/i2_kernel_profile_2026-09-04.md).

## Active Method-Equivalence Gate

The earlier fixed-gamma-60 proposal is superseded by a more fundamental audit:
the paper's Equation 12 and Algorithm 1 specify different attention-relation
objectives, and the historical local path also used eight relation heads where
the pseudocode recommends one. Comparing gamma values before fixing those
contracts would confound loss definition, head partitioning, and coefficient
scale.

The code now exposes both published definitions and records them in every run:

```text
cosine:    softmax(normalize(A) normalize(A)^T / temperature)
scaled_dot: softmax(A A^T / (sqrt(d_r) * temperature))
```

It also provides optional gradient-norm EMA balancing. At each balance update,
the attention coefficient is estimated from the Q/K/V projection gradients:

```text
gamma* = target_ratio * ||grad(CE)|| / (||grad(attention_KD)|| + epsilon)
```

Six pre-registered 120-step MNLI pilots compare cosine versus scaled-dot,
one versus eight relation heads, fixed versus adaptive weighting, and local
sequence-classification versus causal answer-token training. They all start
from the verified 655M Stage-2 manifest. The launcher targets the only GPU node
currently verified to see the shared filesystem and refuses to run if its
expected Git revision differs from the checkout. Submission history and
infrastructure probes are recorded in
[bitdistill_method_parity_submission_2026-09-04.md](benchmarks/results/bitdistill_method_parity_submission_2026-09-04.md).

An exploratory cross-environment screen has completed on an RTX A4500. It is
not the reference-environment result, but it is sufficient to reject unstable
contracts. With sequence classification, fixed `gamma=100,000` produced median
attention/CE gradient ratios of `69.22` for cosine split-1, `119.82` for cosine
split-8, and `61,810.61` for scaled-dot split-1. Gradient-norm EMA balancing
reduced the cosine split-1 median ratio to `0.274` with median effective gamma
`146.27`. The scaled-dot arm was also `-0.142578` behind cosine split-1 on the
paired 512-example diagnostic, CI `[-0.199489, -0.085667]`, exact McNemar
`p=1.79e-6`. This is contract-selection evidence, not a downstream benchmark.
The pinned dualcard replication remains pending and is required before making
a paper-aligned reproduction claim:
[bitdistill_method_parity_midcard_exploratory_2026-09-04.md](benchmarks/results/bitdistill_method_parity_midcard_exploratory_2026-09-04.md).

The screen also exposed and fixed a PyTorch-version-dependent SubLN dtype
contract: older supported PyTorch versions can promote BF16 RMSNorm inputs to
FP32, which then fails at BF16 projections. `SubLNLinear` now casts normalized
activations back to the incoming activation dtype, with a regression test that
forces the promotion path.

To use otherwise idle compute while the dualcard queue is blocked, the
surviving cosine split-1 adaptive contract is being tested in a separately
labeled, pre-registered cross-environment MNLI gate: 10,000 steps, all 392,702
available training examples, all 9,815 matched validation examples, and seeds
`1234`, `1235`, and `1236`. Jobs `10392`, `10395`, and `10396` run serially on
the A4500; only the first saves model artifacts. Success requires a
statistically positive paired delta over the fixed-gamma 655M baseline and
three-seed mean accuracy within one point of local FP16. Passing would support
the adaptive method, not by itself establish paper-exact reproduction. See
[bitdistill_adaptive_full_submission_2026-09-04.md](benchmarks/results/bitdistill_adaptive_full_submission_2026-09-04.md).

The six 120-step pilots are method diagnostics only. The active 10k-step,
three-seed series is the cross-environment quality gate and must evaluate all
`9,815` MNLI examples with paired prediction traces. Any passing A4500 result
still requires confirmation in the pinned reference environment before a
reproduction claim. QNLI/SST2, row/group-scale sweeps, more Stage-2 tokens, and
MoE remain blocked behind that gate.

A matched fixed-`gamma=60` three-seed control is now queued as jobs `10399`,
`10400`, and `10401` behind the adaptive series. This closes an important
causal gap: the historical gamma-60 result changed checkpoint budget and
relation-head partitioning, so adaptive balancing could not be credited from
that comparison. The new control holds the 655M checkpoint, one-head cosine
objective, 10k-step schedule, full validation set, and seeds fixed. Its method
selection rule, practical-effect threshold, immutable asset hashes, and batch
hashes were recorded before adaptive full-validation quality was observed:
[bitdistill_adaptive_vs_fixed_submission_2026-09-04.md](benchmarks/results/bitdistill_adaptive_vs_fixed_submission_2026-09-04.md).
Fail-closed audit job `10402` runs after the final control even when an upstream
job fails, preventing a broken dependency chain from being mistaken for a
successful experiment.

## What This Fork Adds

- An independent open BitDistill-style training implementation. The audited
  Microsoft upstream `main` revision `0b341e58` provides inference/conversion
  tooling and training descriptions but no training/distillation/QAT
  entrypoint. The upstream request for BitDistill code remains open in
  [microsoft/BitNet#344](https://github.com/microsoft/BitNet/issues/344); see
  [upstream_training_surface_2026-09-04.md](benchmarks/results/upstream_training_surface_2026-09-04.md).
- Mathematical and empirical audits showing why blind ternary PTQ collapses on
  tested dense-Qwen checkpoints.
- BitDistill-style training components for Qwen-family models: SubLN, Stage-2
  continued pretraining, Stage-3 CE + logits KL + Q/K/V attention-relation
  distillation, explicit Equation-12/Algorithm-1 relation modes, adaptive
  gradient balancing, objective-gradient cosine diagnostics, layer sweeps, and
  training telemetry.
- Deterministic invariance and gradient audits that expose method-definition
  mismatches before spending GPU time on downstream sweeps.
- Row-scale ternary retrofit experiments and paired statistical audits.
- A llama.cpp fork with packed `I2_SR`, Qwen2 classifier-head execution, and
  graph-level elimination of unused language-model logits for classifiers.
- Mixed `I2_SR` plus Q8 embedding export, which exposes and reduces the
  high-vocabulary embedding floor in small-model GGUF storage.
- Interleaved, affinity-pinned CPU benchmarks that distinguish repeatable
  workload throughput from noisy one-shot timing.
- An allocation-free I2 matrix-output path with bit-identical logits and a
  controlled `1.44-1.46x` local runtime improvement on two classifier formats.
- Manifest-based checkpoint handoff for long Stage-2 jobs, so downstream runs
  consume the actual snapshot state dict instead of guessed paths.
- Pre-training run contracts that record the resolved recipe, source state,
  software/hardware environment, Slurm context, and optional content hashes for
  local model and checkpoint inputs.
- Fail-closed evidence reporting: missing artifacts and `0/0` reports are
  treated as incomplete rather than successful.

The llama.cpp submodule points at:

```text
https://github.com/sabdulmajid/llama.cpp
```

with active row-scale runtime work on `i2sr-row-scale-runtime`.

## Repository Map

| Path | Purpose |
| --- | --- |
| [CLAIMS.md](CLAIMS.md) | Current claim boundaries and evidence status. |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Reproducible commands for manifests, evidence bundles, and reruns. |
| [RUNTIME_CONTRACT.md](RUNTIME_CONTRACT.md) | Why row-scale checkpoints require a matching packed runtime contract. |
| [REPORTING.md](REPORTING.md) | Rules for public reports and fail-closed validation. |
| `benchmarks/` | Benchmark, audit, conversion, validation, and report scripts. |
| `benchmark_results/` | Raw JSON summaries and benchmark artifacts. |
| `benchmarks/results/` | Public Markdown reports and canonical evidence bundles. |
| `experiments/` | Small mathematical probes. |
| `utils/` | Hugging Face conversion and preprocessing utilities. |
| `3rdparty/llama.cpp` | llama.cpp fork with `I2_SR` row-scale runtime work. |
| `src/ggml-bitnet-mad.cpp` | BitNet CPU quantization/runtime integration. |

## Validate The Current Evidence

```bash
python benchmarks/run_active_gate_watchdog.py
python benchmarks/validate_public_docs.py
```

The watchdog rebuilds the completed-gate status graph and fails if required
artifacts are absent or inconsistent. `EXPERIMENTS.md` contains the exact
manifest, Slurm, and historical rerun commands.

## Current Research Direction

Do not position this as a one-click converter. The credible direction is:

1. Explain why blind ternary PTQ fails.
2. Reproduce BitDistill-style task recovery with controlled token-budget curves.
3. Separate paper-style tensor-scale BitDistill from row-scale retrofit variants.
4. Preserve learned scale semantics in packed CPU formats such as `I2_SR`.
5. Report quality, memory, RSS, and speed as separate gates.
6. Complete the pre-registered adaptive-versus-fixed, three-seed MNLI gate.
7. If that gate fails, stop scaling the current objective and test
   Hessian-aware ternary initialization plus group/row-scale hybrids.
8. Extend only surviving methods to QNLI and SST-2, then optimize the measured
   classifier bottleneck before making a CPU-speed claim.
9. Keep MoE/Kimi as future work until dense models are solved.

## Upstream Projects

This work extends [microsoft/BitNet](https://github.com/microsoft/BitNet) and a
[llama.cpp fork](https://github.com/sabdulmajid/llama.cpp). The BitDistill
method belongs to Microsoft Research; this fork's research scope is independent
reproduction, retrofit boundary analysis, and row-scale packed CPU execution.
