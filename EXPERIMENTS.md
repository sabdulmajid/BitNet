# Experiments

This document records the active experiment workflow. It is intentionally
narrow: use manifests, keep paper-style tensor-scale rows separate from
row-scale retrofit variants, and mark missing downstream results as pending.

## Stage-2 Manifest

The completed 327.68M-token Stage-2 producer is job `10070`.

```bash
python benchmarks/build_stage2_manifest.py \
  --output-json benchmarks/results/stage2_manifest_2026-05-20.json \
  --output-md benchmarks/results/stage2_manifest_2026-05-20.md

python benchmarks/validate_stage2_manifest.py \
  benchmarks/results/stage2_manifest_2026-05-20.json
```

The manifest records:

- model: `Qwen/Qwen2.5-0.5B`
- steps: `40000`
- token presentations: `327,680,000`
- final CE: `3.784057`
- state dict:
  `checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-40k/checkpoint-40000/custom_state_dict.pt`

## Correct 327.68M MNLI Rerun

Use `INIT_STATE_MANIFEST` so the Slurm wrapper resolves the snapshot state dict:

```bash
MODEL=Qwen/Qwen2.5-0.5B \
STAGE=task_sft \
METHOD=bitdistill \
TASK_NAME=mnli \
TASK_FORMAT=sequence_classification \
LABEL_SCHEME=letters \
CANDIDATE_SCORE=mean \
TEACHER_MODEL=checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1 \
INIT_STATE_MANIFEST=benchmarks/results/stage2_manifest_2026-05-20.json \
SCALE_MODE=tensor \
EXCLUDE_LINEAR_REGEX='score|classifier' \
DISTILL_LAYER=-1 \
ATTENTION_SPLIT_HEADS=8 \
ACTIVATION_QUANTIZATION=1 \
USE_SUBLN=1 \
LOGIT_KD_WEIGHT=10 \
ATTENTION_KD_WEIGHT=100000 \
LOGIT_TEMPERATURE=5.0 \
LOGIT_KD_TEMPERATURE_SCALE=none \
ATTENTION_TEMPERATURE=1.0 \
INIT_OUTPUT_HEAD_FROM_TEACHER=1 \
MAX_SEQ_LEN=512 \
MAX_STEPS=10000 \
PER_DEVICE_BATCH_SIZE=4 \
GRAD_ACCUM_STEPS=4 \
LR=2e-5 \
SAVE_EVERY_STEPS=0 \
SAVE_MODEL_ARTIFACTS=0 \
OUTPUT_DIR=checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-40kwarmup-steps10000-lr2em5-papergamma-headinit-rerun \
sbatch --partition=midcard slurm_bitdistill_glue.sh
```

Corrected rerun job `10169` completed with full metrics and prediction traces.
The measured MNLI accuracy is `0.720020`, paired delta `-0.088130` versus the
local FP16-SFT reference, CI `[-0.096749, -0.079511]`.

## Canonical Evidence Bundle

```bash
python benchmarks/build_canonical_evidence_bundle.py \
  --stage2-manifest benchmarks/results/stage2_manifest_2026-05-20.json \
  --output-json benchmarks/results/canonical_evidence_bundle_2026-05-20.json \
  --output-md benchmarks/results/canonical_evidence_bundle_2026-05-20.md
```

The bundle fails if a required artifact is missing. It does not discover
reports by date.

## Reproduction Gap Report

The reproduction-gap report is manifest/artifact based and intentionally
separates the BitNet-SFT baseline from the remaining BitDistill recovery gap.

```bash
python benchmarks/audit_bitnet_sft_budget_sweep.py \
  --output-json benchmarks/results/bitnet_sft_budget_sweep_2026-05-23.json \
  --output-md benchmarks/results/bitnet_sft_budget_sweep_2026-05-23.md

python benchmarks/audit_bitdistill_training_dynamics.py \
  --output-json benchmarks/results/bitdistill_training_dynamics_2026-05-23.json \
  --output-md benchmarks/results/bitdistill_training_dynamics_2026-05-23.md

python benchmarks/build_reproduction_gap_report.py
```

Current result: the best 10k-step BitNet-SFT budget row reaches MNLI
`0.628935`, clearing the paper BitNet-SFT anchor by `+0.020935`, but the
completed `327.68M` BitDistill row is still `-0.088130` below local FP16-SFT.

## 655.36M Stage-2 Continuation

The next Stage-2 point has been submitted as a cumulative continuation from the
verified `327.68M` checkpoint:

- `benchmarks/results/stage2_655m_submission_2026-05-23.json`
- `benchmarks/results/stage2_655m_submission_2026-05-23.md`

Job `10250` extends the 327.68M checkpoint by another 327.68M token
presentations, for a cumulative `655,360,000` token presentations. This is a
continuation with a fresh optimizer/scheduler segment, not an uninterrupted
80k-step Stage-2 run.

After completion, build the manifest with:

```bash
python benchmarks/build_stage2_manifest.py \
  --output-dir checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m \
  --parent-manifest benchmarks/results/stage2_manifest_2026-05-20.json \
  --run-id qwen25-05b-bitdistill-tensor-stage2-655m-from327m-job10250 \
  --job-id 10250 \
  --downstream-status pending_submission \
  --downstream-failed-job-id "" \
  --downstream-failure-mode "" \
  --output-json benchmarks/results/stage2_manifest_655m_2026-05-23.json \
  --output-md benchmarks/results/stage2_manifest_655m_2026-05-23.md
```

A dependent handoff has also been queued:

- `benchmarks/results/stage2_655m_handoff_submission_2026-05-23.json`
- `benchmarks/results/stage2_655m_handoff_submission_2026-05-23.md`

Job `10259` has dependency `afterok:10250`. If `10250` succeeds, it will build
and validate the `655.36M` manifest, submit the matched downstream MNLI
BitDistill job, and queue `slurm_stage2_655m_postprocess.sh` after the
downstream job terminates.

After the downstream job submitted by `10259` produces both `metrics.json` and
`eval_predictions.jsonl`, the queued postprocess job should rebuild the
controlled curve with the same fixed recipe. The manual command is:

```bash
BITNET_REPORT_DATE=2026-05-20 python benchmarks/audit_bitdistill_controlled_curve.py \
  --submission-json benchmark_results/bitdistill_stage2_curve_submission_2026-05-15.json \
  --recovery-submission-json benchmark_results/bitdistill_recovery_submission_2026-05-15.json \
  --output-json benchmarks/results/bitdistill_controlled_curve_2026-05-20.json \
  --output-md benchmarks/results/bitdistill_controlled_curve_2026-05-20.md

python benchmarks/build_reproduction_gap_report.py
```

The audit auto-loads the verified `327.68M` manifest and the expected
`655.36M` manifest path. If the 655M downstream result is still pending, the row
must remain pending and public quality claims must not change.

The downstream postprocess also writes:

```bash
benchmarks/results/bitdistill_next_decision_2026-05-23.md
benchmarks/results/bitdistill_next_decision_2026-05-23.json
benchmarks/results/bitdistill_decision_scenarios_2026-05-23.md
benchmarks/results/bitdistill_decision_scenarios_2026-05-23.json
benchmarks/results/bitdistill_next_experiment_blueprint_2026-05-23.md
benchmarks/results/bitdistill_next_experiment_blueprint_2026-05-23.json
```

That report is decision support, not a new benchmark. It remains pending until
the `655.36M` downstream row exists, then records whether the next step should
be a larger Stage-2 point, a loss-normalization ablation, or a replication run.
The scenario matrix applies the same thresholds to representative hypothetical
655M outcomes so the decision policy can be reviewed before the result arrives.
The next-experiment blueprint consumes the decision report and records the
allowed command template plus claim boundary for each possible decision state.
While the status is `pending_655m_downstream`, it only allows status refresh and
ingestion audit commands.

## Gamma-Balanced Telemetry

The loss-normalization gate now has a queued equalized-gamma telemetry job:

- `benchmarks/results/gamma60_telemetry_submission_2026-05-23.json`
- `benchmarks/results/gamma60_telemetry_submission_2026-05-23.md`

Job `10257` is dependency-blocked on `afterok:10250`. It is a short 200-step
component-gradient telemetry diagnostic with `ATTENTION_KD_WEIGHT=60`; it is
not a task-quality benchmark. It replaces pending job `10256` so the stored
Slurm script also writes the post-run gamma-balance audit and refreshes the
next-decision report.

After it materializes, rebuild the training-dynamics audit:

```bash
python benchmarks/audit_bitdistill_training_dynamics.py \
  --output-json benchmarks/results/bitdistill_training_dynamics_2026-05-23.json \
  --output-md benchmarks/results/bitdistill_training_dynamics_2026-05-23.md
```

The Slurm script now performs this rebuild automatically and also writes:

```bash
benchmarks/results/gamma60_gradient_balance_2026-05-23.md
benchmarks/results/gamma60_gradient_balance_2026-05-23.json
benchmarks/results/bitdistill_next_decision_2026-05-23.md
benchmarks/results/bitdistill_next_decision_2026-05-23.json
```

The audit includes both paper-gamma telemetry directories and the queued
`bitdistill-glue-seqcls-telemetry-gamma60` directory. Compare component-gradient
ratios, not task accuracy, for this diagnostic.

## Active Gate Monitor

Use the watchdog while `10250`, `10259`, and `10257` are still live:

```bash
python benchmarks/run_active_gate_watchdog.py
```

The watchdog refreshes and validates these status artifacts in one pass:

- `benchmarks/results/active_stage2_extension_monitor_2026-05-23.{json,md}`
- `benchmarks/results/stage2_655m_ingestion_2026-05-23.{json,md}`
- `benchmarks/results/active_slurm_batch_scripts_2026-05-23.{json,md}`
- `benchmarks/results/current_goal_status_2026-05-23.{json,md}`
- `benchmarks/results/deep_research_handoff_2026-05-23.{json,md}`
- `benchmarks/results/bitdistill_goal_traceability_2026-05-23.{json,md}`
- `benchmarks/results/bitdistill_paper_alignment_2026-05-23.{json,md}`
- `benchmarks/results/bitdistill_publication_product_plan_2026-05-23.{json,md}`
- `benchmarks/results/bitdistill_next_decision_2026-05-23.{json,md}`
- `benchmarks/results/bitdistill_next_experiment_blueprint_2026-05-23.{json,md}`

The watchdog's own report is:

```text
benchmarks/results/active_gate_watchdog_2026-05-23.md
benchmarks/results/active_gate_watchdog_2026-05-23.json
```

It is status evidence only. `quality_claim` must remain `none`.

To inspect only the live Stage-2 job and artifact paths:

```bash
python benchmarks/monitor_active_stage2_extension.py
```

The monitor report is status-only and must not be used as quality evidence.

To audit the 655M downstream-result ingestion gate:

```bash
python benchmarks/audit_stage2_655m_ingestion.py
```

The ingestion audit remains `pending_handoff`, `downstream_pending_or_running`,
or `downstream_incomplete` until the downstream `metrics.json`,
`eval_predictions.jsonl`, controlled-curve row, reproduction-gap report, and
next-decision report are mutually consistent. It is the receipt that prevents a
completed Slurm job from being treated as a quality result without paired
prediction evidence.

To generate a reviewer-facing snapshot of the current objective state:

```bash
python benchmarks/build_current_goal_status.py
python benchmarks/build_deep_research_handoff.py
python benchmarks/build_goal_traceability_audit.py
python benchmarks/build_bitdistill_paper_alignment_audit.py
python benchmarks/build_publication_product_plan.py
```

These reports are status ledgers, not completion declarations. They read the
canonical evidence bundle, reproduction-gap report, and active 655M monitor.

After changing or resubmitting queued Slurm scripts, verify the stored batch
script contents:

```bash
python benchmarks/audit_active_slurm_batch_scripts.py
```

## Reviewer Reproduction Checklist

Use this sequence for an external technical review of the current state:

```bash
python benchmarks/run_active_gate_watchdog.py
python benchmarks/validate_public_docs.py
python benchmarks/validate_reports_fail_closed.py \
  benchmarks/results/active_gate_watchdog_2026-05-23.json \
  benchmarks/results/active_gate_watchdog_2026-05-23.md \
  benchmarks/results/bitdistill_benchmark_scoreboard_2026-05-23.json \
  benchmarks/results/bitdistill_goal_traceability_2026-05-23.json \
  benchmarks/results/bitdistill_paper_alignment_2026-05-23.json \
  benchmarks/results/stage2_655m_ingestion_2026-05-23.json \
  benchmarks/results/bitdistill_next_decision_2026-05-23.json \
  benchmarks/results/bitdistill_next_experiment_blueprint_2026-05-23.json
python -m py_compile train_bitdistill.py train_distill.py benchmarks/*.py
bash -n slurm_gamma60_telemetry.sh \
  slurm_stage2_655m_handoff.sh \
  slurm_stage2_655m_postprocess.sh
```

Interpretation rules for the current state:

- `active_gate_watchdog.status == passed` means the status pipeline is healthy,
  not that BitDistill quality has recovered.
- `stage2_655m_ingestion.status == pending_handoff` is expected while job
  `10250` is still running.
- `bitdistill_paper_alignment.status == not_exact_reproduction` is intentional:
  the active experiment differs from the paper in Stage-2 budget, corpus,
  hardware, effective batch size, and unfinished QNLI/SST2/CNNDM coverage.
- `bitdistill_next_decision.status == pending_655m_downstream` must remain
  until the 655M downstream prediction trace exists.
- `bitdistill_next_experiment_blueprint.status == pending_655m_downstream`
  means the only runnable action is watchdog/ingestion status refresh.
- Do not update public quality claims until the ingestion audit is
  `ingested_reports_rebuilt`.

## Validation

```bash
python -m py_compile train_bitdistill.py benchmarks/*.py
python benchmarks/validate_stage2_manifest.py benchmarks/results/stage2_manifest_2026-05-20.json
python benchmarks/validate_reports_fail_closed.py <reports-to-check>
python benchmarks/validate_public_docs.py
```

The report validator must reject silent `0/0` reports unless the report has an
explicit `empty_expected_reason` or equivalent text.
