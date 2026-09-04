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

| Claim | Status | Evidence | Caveat |
| --- | --- | --- | --- |
| Blind FP/BF16 to ternary PTQ works as a general retrofit | **No: strong negative result in the tested setup** | Qwen2.5-1.5B FP WikiText PPL `13.901`; naive ternary PTQ PPL `3,813,121.803`. FP ten-task mean `0.644169`; naive PTQ mean `0.348671`. | Dense Qwen2.5-1.5B tested setup; do not generalize as a theorem for every architecture. |
| QAT/distillation recovers signal | **Partial recovery, not FP quality** | Best row-scale QAT ten-task mean `0.499459`, a `+0.150788` recovery over naive PTQ and still `-0.144710` below FP. | Row-scale QAT is this fork's retrofit variant, not standard BitDistill. |
| BitDistill paper-level GLUE reproduction is complete | **No** | Qwen2.5-0.5B local FP16-SFT MNLI is `0.808151`. Controlled Stage-2 rows are `0.616607` at `40.96M`, `0.691187` at `163.84M`, `0.720020` at `327.68M`, and `0.729903` at `655.36M` token presentations. | The 655M row remains `-0.078248` below FP16 with paired CI `[-0.086720, -0.069775]`; the paper-level recovery target is not met. |
| The `655.36M` Stage-2 checkpoint is usable | **Yes, with a verified manifest** | [stage2_manifest_655m_2026-05-23.md](benchmarks/results/stage2_manifest_655m_2026-05-23.md) records job `10250`, four complete snapshots, final CE `3.426713`, the state-dict SHA-256, and downstream job `10260`. | This was a `327.68M` continuation with a fresh optimizer/scheduler segment, not one uninterrupted 80k-step run. |
| Paper gamma can be copied literally into this implementation | **No, not without matching loss normalization** | Historical telemetry measures attention/CE gradient ratio `221.384986` for the paper-gamma path versus `0.346044` at gamma 60. A controlled A4500 screen measures median ratio `69.2248` for cosine split-1 at fixed gamma `100,000`, versus `0.273975` with adaptive balancing. | This is not a task-quality result. The source-pinned dualcard replication remains required before a paper-aligned claim; the active 10k runs are explicitly cross-environment. |
| The paper defines one unambiguous attention-relation objective | **No** | [attention_relation_equivalence_2026-09-04.md](benchmarks/results/attention_relation_equivalence_2026-09-04.md) proves that Equation 12 scaled-dot relations and Algorithm 1 normalized-cosine relations are not generally equivalent. In a deterministic probe their gradient-norm ratio is `18.7073` and gradient cosine is `0.2437`. | This is a mathematical contract result, not downstream quality evidence. For Qwen's 14:2 grouped-query attention, KV repetition leaves cosine relations invariant but multiplies scaled-dot logits by `sqrt(7)`. |
| The local GLUE formulation is paper-exact | **Unresolved** | [bitdistill_task_formulation_audit_2026-09-04.md](benchmarks/results/bitdistill_task_formulation_audit_2026-09-04.md) separates sequence-classification from causal answer-token results. | Token-level CE and decoding language favor the causal interpretation, but no authoritative released templates or training code establish equivalence. |
| Row-scale semantics matter at runtime | **Yes: strong systems result** | TL2 one-scale relative output RMS error `1.904230`; exact FP16 row scales reduce it to `0.000197`. | Row scales are part of the learned function. TL2 row-scale support is not implemented. |
| `I2_SR` packed CPU inference works | **Yes, for compatible causal artifacts** | Xeon Silver 4116: row-scale `I2_SR` file `1211.3 MiB`, PPL `38.8477`, prompt `211.67 tok/s`, decode `19.07 tok/s`. | It does **not** beat Q4_K_M on quality or file size. Q4_K_M is `940.4 MiB` with PPL `12.8112`. |
| Native packed sequence classification is product-ready | **No: research demo only** | Full MNLI native sequence-isolated path: accuracy `0.652165`, PyTorch agreement `0.976668`, `7.456204` examples/s, RSS `960.15 MiB`. | Agreement is below the `0.99` product gate and the model quality is weak. |
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

Evidence and decision artifacts:

- [stage2_655m_ingestion_2026-05-23.md](benchmarks/results/stage2_655m_ingestion_2026-05-23.md)
- [bitdistill_controlled_curve_2026-05-23.md](benchmarks/results/bitdistill_controlled_curve_2026-05-23.md)
- [bitdistill_stage2_saturation_2026-09-04.md](benchmarks/results/bitdistill_stage2_saturation_2026-09-04.md)
- [gamma60_gradient_balance_2026-05-23.md](benchmarks/results/gamma60_gradient_balance_2026-05-23.md)
- [bitdistill_next_decision_2026-05-23.md](benchmarks/results/bitdistill_next_decision_2026-05-23.md)
- [bitdistill_next_experiment_blueprint_2026-05-23.md](benchmarks/results/bitdistill_next_experiment_blueprint_2026-05-23.md)
- [bitdistill_next_experiment_blueprint_2026-05-23.json](benchmarks/results/bitdistill_next_experiment_blueprint_2026-05-23.json)
- [active_gate_watchdog_2026-05-23.md](benchmarks/results/active_gate_watchdog_2026-05-23.md)
- [active_gate_watchdog_2026-05-23.json](benchmarks/results/active_gate_watchdog_2026-05-23.json)

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
surviving cosine split-1 adaptive contract is running a separately labeled,
pre-registered cross-environment MNLI gate: 10,000 steps, all 392,702 available
training examples, all 9,815 matched validation examples, and seeds `1234`,
`1235`, and `1236`. Jobs `10392`, `10395`, and `10396` run serially on the
A4500; only the first saves model artifacts. Success requires a statistically
positive paired delta over the fixed-gamma 655M baseline and three-seed mean
accuracy within one point of local FP16. Passing would support the adaptive
method, not by itself establish paper-exact reproduction. See
[bitdistill_adaptive_full_submission_2026-09-04.md](benchmarks/results/bitdistill_adaptive_full_submission_2026-09-04.md).

The six 120-step pilots are method diagnostics only. The active 10k-step,
three-seed series is the cross-environment quality gate and must evaluate all
`9,815` MNLI examples with paired prediction traces. Any passing A4500 result
still requires confirmation in the pinned reference environment before a
reproduction claim. QNLI/SST2, row/group-scale sweeps, more Stage-2 tokens, and
MoE remain blocked behind that gate.

## What This Fork Adds

- An independent open BitDistill-style training implementation. The audited
  Microsoft upstream `main` revision `0b341e58` provides inference/conversion
  tooling and training descriptions but no training/distillation/QAT
  entrypoint; see
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
- A llama.cpp fork with a packed `I2_SR` row-scale CPU runtime path.
- Manifest-based checkpoint handoff for long Stage-2 jobs, so downstream runs
  consume the actual snapshot state dict instead of guessed paths.
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
6. Keep MoE/Kimi as future work until dense models are solved.

## Upstream Projects

This work extends [microsoft/BitNet](https://github.com/microsoft/BitNet) and a
[llama.cpp fork](https://github.com/sabdulmajid/llama.cpp). The BitDistill
method belongs to Microsoft Research; this fork's research scope is independent
reproduction, retrofit boundary analysis, and row-scale packed CPU execution.
