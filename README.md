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

The current canonical evidence bundle is:

- [canonical_evidence_bundle_2026-05-20.md](benchmarks/results/canonical_evidence_bundle_2026-05-20.md)
- [canonical_evidence_bundle_2026-05-20.json](benchmarks/results/canonical_evidence_bundle_2026-05-20.json)
- [bitdistill_benchmark_scoreboard_2026-05-23.md](benchmarks/results/bitdistill_benchmark_scoreboard_2026-05-23.md)
- [bitdistill_benchmark_scoreboard_2026-05-23.json](benchmarks/results/bitdistill_benchmark_scoreboard_2026-05-23.json)

| Claim | Status | Evidence | Caveat |
| --- | --- | --- | --- |
| Blind FP/BF16 to ternary PTQ works as a general retrofit | **No: strong negative result in the tested setup** | Qwen2.5-1.5B FP WikiText PPL `13.901`; naive ternary PTQ PPL `3,813,121.803`. FP ten-task mean `0.644169`; naive PTQ mean `0.348671`. | Dense Qwen2.5-1.5B tested setup; do not generalize as a theorem for every architecture. |
| QAT/distillation recovers signal | **Partial recovery, not FP quality** | Best row-scale QAT ten-task mean `0.499459`, a `+0.150788` recovery over naive PTQ and still `-0.144710` below FP. | Row-scale QAT is this fork's retrofit variant, not standard BitDistill. |
| BitDistill paper-level GLUE reproduction is complete | **No** | Qwen2.5-0.5B local FP16-SFT MNLI is about `0.808151`. Controlled MNLI rows: `40.96M` Stage-2 token presentations gives `0.616607`; `163.84M` gives `0.691187`; `327.68M` gives `0.720020`. | The largest completed row improves over shorter warm-ups but still trails FP by paired delta `-0.088130` with CI `[-0.096749, -0.079511]`. |
| The `327.68M` Stage-2 checkpoint is usable | **Yes, as a producer checkpoint** | [stage2_manifest_2026-05-20.md](benchmarks/results/stage2_manifest_2026-05-20.md) records job `10070`, `40000` steps, `327,680,000` token presentations, final CE `3.784057`, rerun job `10169`, and the exact state dict path. | Job `10071` failed before quality evaluation because it expected a root `custom_state_dict.pt`; corrected rerun `10169` completed and produced the `0.720020` MNLI row. |
| Paper gamma can be copied literally into this implementation | **No, not without matching loss normalization** | Local gamma-60 diagnostic MNLI `0.738462`, still `-0.069689` below FP. Telemetry shows paper-gamma attention KD can dominate CE under local reductions. | This is a local normalization mismatch, not evidence that the paper coefficient is wrong. |
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

The short default BitNet-SFT row was undertrained: the default row is
`0.487621`, while the best 10k-step BitNet-SFT budget row reaches `0.628935`,
which is `+0.020935` above the paper's Qwen2.5-0.5B MNLI BitNet-SFT anchor.
That does **not** reproduce BitDistill. The completed `327.68M` Stage-2
BitDistill row reaches `0.720020`, which is still `-0.088130` below the local
FP16-SFT MNLI reference.

Active next gate:

- [stage2_655m_submission_2026-05-23.md](benchmarks/results/stage2_655m_submission_2026-05-23.md)
- [gamma60_telemetry_submission_2026-05-23.md](benchmarks/results/gamma60_telemetry_submission_2026-05-23.md)
- [stage2_655m_afterany_submission_2026-05-23.md](benchmarks/results/stage2_655m_afterany_submission_2026-05-23.md)
- [stage2_655m_afterany_submission_2026-05-23.json](benchmarks/results/stage2_655m_afterany_submission_2026-05-23.json)
- [stage2_655m_ingestion_2026-05-23.md](benchmarks/results/stage2_655m_ingestion_2026-05-23.md)
- [stage2_655m_ingestion_2026-05-23.json](benchmarks/results/stage2_655m_ingestion_2026-05-23.json)
- [stage2_snapshot_salvage_2026-05-23.md](benchmarks/results/stage2_snapshot_salvage_2026-05-23.md)
- [stage2_snapshot_salvage_2026-05-23.json](benchmarks/results/stage2_snapshot_salvage_2026-05-23.json)
- [active_gate_watchdog_2026-05-23.md](benchmarks/results/active_gate_watchdog_2026-05-23.md)
- [active_gate_watchdog_2026-05-23.json](benchmarks/results/active_gate_watchdog_2026-05-23.json)
- [gamma60_gradient_balance_2026-05-23.md](benchmarks/results/gamma60_gradient_balance_2026-05-23.md)
- [bitdistill_next_decision_2026-05-23.md](benchmarks/results/bitdistill_next_decision_2026-05-23.md)
- [bitdistill_decision_scenarios_2026-05-23.md](benchmarks/results/bitdistill_decision_scenarios_2026-05-23.md)
- [bitdistill_next_experiment_blueprint_2026-05-23.md](benchmarks/results/bitdistill_next_experiment_blueprint_2026-05-23.md)
- [bitdistill_next_experiment_blueprint_2026-05-23.json](benchmarks/results/bitdistill_next_experiment_blueprint_2026-05-23.json)

Job `10250` is a cumulative continuation from the verified `327.68M` checkpoint
to `655.36M` token presentations. It is explicitly labeled as a continuation
with a fresh optimizer/scheduler segment, not an uninterrupted 80k-step run.
A dependent handoff job, `10255`, is queued with `afterok:10250` to build the
655M manifest and submit the matched downstream MNLI evaluation if Stage-2
finishes successfully. It also queues a postprocess job after downstream MNLI
terminates so the controlled curve and reproduction-gap reports can be rebuilt
from the actual metrics and prediction trace. The postprocess also rebuilds the
next-decision report so the repository records whether to extend Stage-2,
switch to loss-normalization debugging, or replicate a successful recovery row.
The decision-scenario matrix documents these thresholds before the 655M result
arrives; it is policy documentation, not benchmark evidence.
The next-experiment blueprint maps each decision status to a bounded command
template and claim boundary; in the current pending state it only permits
watchdog and ingestion checks, not new quality runs.

Job `10257` is a dependent gamma-60 component-gradient diagnostic; it is not a quality benchmark.
It replaces pending job `10256` so the stored Slurm script also generates the
post-run gamma-balance report and refreshes the next-decision report.
Its role is to compare a lower attention-KD coefficient against the existing
paper-gamma telemetry, where attention KD dominates CE under this
implementation's current loss reductions.

## What This Fork Adds

- Mathematical and empirical audits showing why blind ternary PTQ collapses on
  tested dense-Qwen checkpoints.
- BitDistill-style training components for Qwen-family models: SubLN, Stage-2
  continued pretraining, Stage-3 CE + logits KL + Q/K/V attention-relation
  distillation, layer sweeps, and training telemetry.
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

## Reproduce The Current Evidence Bundle

```bash
python benchmarks/build_stage2_manifest.py \
  --output-json benchmarks/results/stage2_manifest_2026-05-20.json \
  --output-md benchmarks/results/stage2_manifest_2026-05-20.md

python benchmarks/validate_stage2_manifest.py \
  benchmarks/results/stage2_manifest_2026-05-20.json

python benchmarks/build_canonical_evidence_bundle.py \
  --stage2-manifest benchmarks/results/stage2_manifest_2026-05-20.json \
  --output-json benchmarks/results/canonical_evidence_bundle_2026-05-20.json \
  --output-md benchmarks/results/canonical_evidence_bundle_2026-05-20.md

python benchmarks/build_reproduction_gap_report.py
```

Fail-closed report validation example:

```bash
python benchmarks/validate_reports_fail_closed.py \
  benchmark_results/bitdistill_controlled_curve_2026-05-17.json \
  benchmarks/results/bitdistill_controlled_curve_2026-05-17.md
```

That command is expected to fail for the stale 2026-05-17 controlled-curve
files because they summarize `0/0` rows without an explicit empty-report reason.
Validate the public docs against the canonical JSON bundle:

```bash
python benchmarks/validate_public_docs.py
```

## Correct 327.68M Downstream Rerun

The completed Stage-2 producer checkpoint is recorded in:

```text
benchmarks/results/stage2_manifest_2026-05-20.json
```

Downstream jobs should set:

```bash
INIT_STATE_MANIFEST=benchmarks/results/stage2_manifest_2026-05-20.json
```

or use the resolved state dict directly:

```text
checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-40k/checkpoint-40000/custom_state_dict.pt
```

The previous job `10071` failed before evaluation because it looked for a
root-level `custom_state_dict.pt` that was never written under the chosen
snapshot-only save mode. Corrected rerun job `10169` was submitted with
`INIT_STATE_MANIFEST` and completed with MNLI accuracy `0.720020`.

## Current Research Direction

Do not position this as a one-click converter. The credible direction is:

1. Explain why blind ternary PTQ fails.
2. Reproduce BitDistill-style task recovery with controlled token-budget curves.
3. Separate paper-style tensor-scale BitDistill from row-scale retrofit variants.
4. Preserve learned scale semantics in packed CPU formats such as `I2_SR`.
5. Report quality, memory, RSS, and speed as separate gates.
6. Keep MoE/Kimi as future work until dense models are solved.
