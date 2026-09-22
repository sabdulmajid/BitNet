# Research Status

Last updated: 2026-09-22

## Executive Assessment

The original product thesis was a universal converter: take an arbitrary
FP16/BF16 checkpoint, round its linear weights to `{-1, 0, +1}`, and obtain a
small, fast CPU model without material quality loss. The tested implementation
rejects that thesis. The useful project that remains is a **ternary retrofit
evaluation and deployment stack**: determine whether a model can be adapted or
calibrated successfully, then prove that its packed runtime preserves the exact
mathematical representation that was evaluated.

The work has produced two durable findings:

1. Naive absmean ternarization is catastrophically destructive for the tested
   dense Qwen checkpoint.
2. A trained row-scale ternary representation can be packed and executed
   faithfully, but fidelity does not guarantee competitive quality or speed.

The project is a credible research artifact. It is not yet a complete research
paper demonstrating a new algorithmic state of the art.

## What Was Built

| Area | Implemented capability |
| --- | --- |
| PTQ audit | Exact repository quantizer probes, normally distributed matrix tests, held-out PPL, and ten-task `lm-eval` comparisons |
| QAT/distillation | Ternary `BitLinear`, STE, tensor or row scales, teacher logits KL, hidden losses, and checkpoint export |
| BitDistill-style training | Qwen SubLN surgery, Stage-2 causal CE warm-up, Stage-3 task CE + logits KD + Q/K/V relation KD, relation-layer sweeps, sequence-classification and causal formulations |
| Training diagnostics | Per-component losses, gradient norms and cosines, activation clipping, ternary flip rates, scale drift, and adaptive loss weighting |
| Statistical evaluation | Aligned prediction traces, paired bootstrap intervals, exact McNemar tests, seed-level intervals, preregistered gates, and fail-closed audits |
| Packed runtime | Row-scale `I2_SR`, Qwen2 sequence-classification heads, mixed Q8 embedding export, and unused LM-logit elimination |
| Runtime optimization | In-place I2 output scaling, layout fingerprints, exact accumulator checks, pinned interleaved CPU benchmarking, and kernel cost attribution |
| Experimental layout | `TL2_SR` generated kernels with per-row scale semantics, shape/layout guards, and three tile configurations |
| MoE exploration | Tiny Qwen2MoE conversion/routing fixtures only; no production MoE claim |

The initial jobs `9730` and `9734` established that the first pipeline could
replace Qwen linear layers, train under ternary forward constraints, and export
checkpoints. They were engineering proofs, not quality proofs. Later work added
held-out evaluation, strict BitDistill components, controlled baselines, and a
native runtime so claims no longer depend on training loss alone.

## Result 1: Naive PTQ Fails

For Qwen2.5-1.5B, direct tensor-absmean ternarization changes WikiText
perplexity from `13.901` to `3,813,121.803` and the mean over ten zero-shot
tasks from `0.644169` to `0.348671`.

The operation is approximately

```text
alpha = mean(abs(W))
T = clip(round(W / alpha), -1, 1)
Wq = alpha * T
```

This projection maps every continuous weight to one of three reconstruction
values. For a generic pretrained matrix, it discards within-bin magnitude
information and optimizes neither layer-output error nor end-to-end behavior.
The empirical collapse establishes that storage conversion alone is not a
usable model conversion in this setup.

This does **not** prove that every post-training ternary method fails. Modern
PTQ methods optimize asymmetric levels, thresholds, rotations, or output error;
those are materially different algorithms and are now the primary missing
baseline.

## Result 2: Training Recovers Signal, Not FP Quality

The best Qwen2.5-1.5B row-scale QAT checkpoint reaches WikiText PPL `38.580`
and ten-task mean `0.499459`. Relative to naive PTQ, the ten-task gain is
`+0.150788`; relative to FP, the gap remains `-0.144710`.

This proves that optimization under the ternary forward constraint matters. It
does not establish an acceptable general-language retrofit. Task-specific and
general-language quality must remain separate endpoints.

The simple least-squares and diagonal-Hessian initializers also produced an
important negative result. Although diagonal-Hessian weighting reduced
synthetic layer-output RMS error in all tested trials, MNLI accuracy fell from
the matched absmean baseline `0.628935` to `0.361895` for LS and `0.350993` for
diagonal LS. A local reconstruction objective was not a sufficient proxy for
task behavior.

## Result 3: Local BitDistill Remains Below FP16

The implementation now contains the major published stages:

1. SubLN before attention output and FFN down projections.
2. Continued causal pretraining with cross entropy.
3. Downstream CE, temperature-scaled logits KL, and Q/K/V relation
   distillation with explicit layer and head partitioning.

The controlled Stage-2 curve improved MNLI as token presentations increased:

| Cumulative Stage-2 presentations | MNLI accuracy | Gain from prior row |
| ---: | ---: | ---: |
| `40.96M` | `0.616607` | - |
| `163.84M` | `0.691187` | `+0.074580` |
| `327.68M` | `0.720020` | `+0.028833` |
| `655.36M` | `0.729903` | `+0.009883` |

The diminishing gains reject simply scaling the unchanged local recipe as the
best next use of compute. A historical controlled pair also showed that loss
normalization matters: local `gamma=60` beat a literal `gamma=100,000` run by
`+0.047275` MNLI, paired 95% CI `[0.039256, 0.055293]`. Coefficients are not
portable unless all KL reductions and normalizations match.

### Final matched controller test

Three adaptive runs were compared with three fixed-60 runs using the same
Stage-2 state, seeds, data, objective, and 10,000-step schedule.

| Arm | Seed 1234 | Seed 1235 | Seed 1236 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Adaptive | `0.755782` | `0.756903` | `0.753337` | `0.755340` |
| Fixed 60 | `0.758635` | `0.754457` | `0.758023` | `0.757039` |

Adaptive minus fixed is `-0.001698`; the primary seed-level paired 95% CI is
`[-0.010898, 0.007502]`. No seed-level or example-level comparison establishes
adaptive superiority. Both means are more than five points below FP16
`0.808151` and fail the paper-recovery floor `0.798151`.

The correct conclusion is not that BitDistill is false. This is not a
paper-exact reproduction: model version, corpus, total continued-pretraining
budget, task details, and unreleased reference implementation remain material
differences. The correct conclusion is that the local recipe has not reproduced
the paper and the adaptive controller has not earned further promotion.

## Result 4: Scale Semantics Are a Runtime Contract

The strongest systems finding is the row-scale contract. The trained model
represents each projection approximately as

```text
W[row, :] = scale[row] * ternary_codes[row, :]
```

Replacing those scales with one tensor scalar gives relative output RMS error
`1.904230`. Preserving FP16 row scales gives `0.000197`. The scale vector is
therefore part of the learned function, not optional metadata.

`I2_SR` carries that vector through GGUF packing and CPU execution. On the
causal Qwen2.5-1.5B artifact it reaches `211.67` prompt tok/s and `19.07`
decode tok/s on the Xeon Silver 4116. Its PPL is `38.8477` and file size is
`1211.3 MiB`, versus Q4_K_M PPL `12.8112` and `940.4 MiB`. It is a valid
runtime proof, not a superior Pareto point.

`TL2_SR` also preserves the tested classifier: all `18/18` deterministic kernel
cases pass and full MNLI differs from `I2_SR` by only `+0.001426`, paired CI
`[-0.000917, 0.003872]`. Projection storage falls `12.862%`, but whole-file
storage falls only `3.154%`, and all three tile variants are slower than
`I2_SR`. This path should remain an experimental negative result.

## Result 5: CPU Benefit Is Workload Dependent

The classifier runtime closes a previous product gap by executing the actual
Qwen2 sequence-classification head without computing unused vocabulary logits.
On all `9,815` MNLI examples, one audited `I2_SR` artifact scores `0.652165`
versus its PyTorch checkpoint's `0.653591`, paired delta `-0.001426`, CI
`[-0.004193, 0.001341]`. Packing preserves that weak student's task behavior.

It does not accelerate the target workload. Four interleaved 12-core runs give
`I2_SR/FP16 = 0.650x`, CI `[0.646, 0.653]`; the smaller mixed I2/Q8 model is
`0.605x`, CI `[0.603, 0.607]`. Kernel profiling puts `94.51%` of projection
time in packed I2 arithmetic and `5.49%` in activation quantization. The next
speed optimization must change arithmetic or layout, not only the A8 prepass.

The same runtime can be faster for causal decode because its shapes and memory
traffic differ. No universal CPU speedup should be claimed.

## What Is Solved

- The naive universal-converter hypothesis has a decisive negative control.
- The earlier weak BitNet-SFT baseline was diagnosed as undertraining; a
  10,000-step sanity row reached `0.628935`.
- Loss-coefficient portability was identified as a normalization problem, not
  a number to copy blindly from a paper.
- Adaptive weighting now has a matched control and no longer blocks decisions.
- Row-scale checkpoint semantics have a packed GGUF representation and tested
  CPU execution path.
- Native classifier quality preservation, storage, repeated throughput, and
  kernel attribution are measured separately.
- Public evidence contains aligned predictions and hashes rather than only
  prose summaries.

## What Is Not Solved

- No local method preserves general-language FP quality at ternary precision.
- BitDistill paper-level GLUE quality has not been reproduced.
- The best quality experiment and best causal runtime experiment are not one
  common production artifact.
- No ternary classifier beats FP16 or mature Q4 on the measured CPU workload.
- Advanced PTQ methods released in 2026 have not yet been independently run in
  this harness.
- Kimi, MLA, shared experts, expert paging, and real routed MoE quality are not
  implemented or benchmarked.
- Energy measurements and cross-CPU generalization are absent.

## Impact and Publishability

### Publishable now

The repository can support a careful technical report or systems workshop
artifact on:

- a reproducible negative study of naive ternary retrofit;
- the mismatch between local reconstruction error and downstream quality;
- row-scale training-to-runtime contract preservation;
- workload-dependent CPU behavior and rejected lookup-table optimizations;
- a public evidence methodology for extreme quantization claims.

Those claims are useful, but they are not a state-of-the-art algorithm paper.
The negative result is also narrower after CAT-Q, PT2-LLM, TWLA, and
ScaleQ-1.58: it rules out naive rounding, not optimized PTQ.

### Stronger paper opportunity

A stronger paper becomes possible if the next phase contributes one of these:

1. An independent cross-method study that explains when advanced ternary PTQ,
   QAT, or distillation wins under equal calibration and effective-bit budgets.
2. A runtime representation that supports a quality-surviving asymmetric or
   group-scale PTQ method and beats Q4 at a measured Pareto point.
3. A principled predictor of retrofit success from activation, Hessian,
   outlier, and ternary-code statistics.
4. A dense-to-MoE boundary study with real expert routing, memory locality, and
   CPU throughput.

The minimum standard is a common artifact evaluated for both quality and CPU
performance, multiple seeds where training is involved, independent baselines,
and confidence intervals for every speed or accuracy comparison.

## Product Direction

The credible product is not a one-click converter. It is a **ternary
feasibility compiler** that accepts a model and deployment target, runs staged
calibration and evaluation, and returns:

- quality retention versus FP and Q4;
- effective bits per weight and total file/RSS cost;
- PPL and task deltas with uncertainty;
- hardware-specific prompt/decode throughput;
- the runtime representation required by the learned scales;
- a fail/pass recommendation with an auditable manifest.

A nearer-term product can target embeddings, rerankers, or fixed classifiers,
where task-specific adaptation is acceptable and the output contract is
narrow. General chat conversion and Kimi-class MoE serving should remain later
milestones.

## Evidence

- [Current evidence snapshot](../benchmarks/results/current_evidence_2026-09-22.md)
- [Matched adaptive vs fixed audit](../benchmarks/results/bitdistill_adaptive_vs_fixed_matched_audit_2026-09-22.md)
- [Matched prediction bundle](../benchmarks/results/bitdistill_matched_prediction_bundle_2026-09-22.md)
- [TL2_SR audit](../benchmarks/results/tl2sr_evidence_audit_2026-09-04.md)
- [CPU classifier matrix](../benchmarks/results/seqcls_native_cpu_matrix_2026-09-04.md)
- [Repeated CPU benchmark](../benchmarks/results/seqcls_native_cpu_repeated_inplace_2026-09-04.md)
- [Kernel profile](../benchmarks/results/i2_kernel_profile_2026-09-04.md)
