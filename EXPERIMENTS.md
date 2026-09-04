# Experiments

This document records the active experiment workflow. It is intentionally
narrow: use manifests, keep paper-style tensor-scale rows separate from
row-scale retrofit variants, and mark missing downstream results as pending.

## Run Provenance

`train_bitdistill.py` writes `run_contract.json` before loading models or
starting optimization. It records the resolved CLI arguments, Git revision and
tracked-dirty state, Python/package versions, visible accelerator hardware, and
allowlisted Slurm variables. Final metrics contain the contract path and
SHA-256.

For publication runs with local model and checkpoint paths, enable immutable
input fingerprints:

```bash
HASH_INPUT_ARTIFACTS=1 sbatch ... slurm_bitdistill_glue.sh
```

This hashes every file in local model directories and can add startup I/O. A
remote Hugging Face identifier is recorded but cannot be content-hashed; stage
the exact revision locally or use a separately pinned asset manifest when an
immutable input claim is required.

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
HASH_INPUT_ARTIFACTS=1 \
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
completed `655.36M` BitDistill row reaches `0.729903` and remains
`-0.078248` below local FP16-SFT.

## Completed 655.36M Stage-2 Continuation

Job `10250` completed the cumulative continuation from the verified
`327.68M` checkpoint:

- `benchmarks/results/stage2_655m_submission_2026-05-23.json`
- `benchmarks/results/stage2_655m_submission_2026-05-23.md`
- `benchmarks/results/stage2_manifest_655m_2026-05-23.json`
- `benchmarks/results/stage2_manifest_655m_2026-05-23.md`

The segment adds `327,680,000` token presentations for a cumulative
`655,360,000`. It ran `40,000` steps, wrote four complete snapshots, and
ended at CE `3.426713`. This is a continuation with a fresh
optimizer/scheduler segment, not an uninterrupted 80k-step run.

Validate the materialized manifest with:

```bash
python benchmarks/validate_stage2_manifest.py \
  benchmarks/results/stage2_manifest_655m_2026-05-23.json
```

Handoff job `10259` submitted downstream MNLI job `10260`, which completed
the fixed paper-gamma 10k-step recipe over all `9,815` validation examples.
Postprocess job `10261` rebuilt the curve and paired evidence:

| Stage-2 tokens | MNLI | Gain vs prior row | Delta vs FP16 |
| ---: | ---: | ---: | ---: |
| `40.96M` | `0.616607` | - | `-0.191544` |
| `163.84M` | `0.691187` | `+0.074580` | `-0.116964` |
| `327.68M` | `0.720020` | `+0.028833` | `-0.088130` |
| `655.36M` | `0.729903` | `+0.009883` | `-0.078248` |

The latest paired 95% confidence interval is
`[-0.086720, -0.069775]`. The ingestion audit is
`ingested_reports_rebuilt`: the result has metrics, all 9,815 prediction rows,
a paired comparison, and mutually consistent downstream reports.

## Gamma-Balanced Telemetry

Job `10257` completed the 200-step gamma-60 component-gradient diagnostic:

- `benchmarks/results/gamma60_telemetry_submission_2026-05-23.json`
- `benchmarks/results/gamma60_telemetry_submission_2026-05-23.md`
- `benchmarks/results/gamma60_gradient_balance_2026-05-23.json`
- `benchmarks/results/gamma60_gradient_balance_2026-05-23.md`

The measured attention/CE gradient ratio drops from `221.384986` on the
paper-gamma path to `0.346044` at gamma 60, a `639.759x` reduction. This is
not a quality result. It justifies one matched 10k-step quality ablation from
the 655M checkpoint. The exact command is generated in
`bitdistill_next_experiment_blueprint_2026-05-23.md`; its status is
`run_gamma_balanced_downstream`.

## Evidence Refresh

Use the watchdog to rebuild and validate the completed-gate reports:

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

To inspect the Stage-2 job and artifact paths:

```bash
python benchmarks/monitor_active_stage2_extension.py
```

The monitor report is status-only and must not be used as quality evidence.

To audit the 655M downstream-result ingestion gate:

```bash
python benchmarks/audit_stage2_655m_ingestion.py
```

The current expected state is
`stage2_655m_ingestion.status == ingested_reports_rebuilt`. This receipt
prevents a completed Slurm job from being treated as a quality result without
the matching metrics, prediction trace, paired comparison, and rebuilt reports.

To generate a reviewer-facing snapshot of the current objective state:

```bash
python benchmarks/build_current_goal_status.py
python benchmarks/build_deep_research_handoff.py
python benchmarks/build_goal_traceability_audit.py
python benchmarks/build_bitdistill_paper_alignment_audit.py
python benchmarks/build_publication_product_plan.py
```

These reports are status ledgers, not completion declarations. They read the
canonical evidence bundle, reproduction-gap report, and completed 655M monitor.

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
- `stage2_655m_ingestion.status == ingested_reports_rebuilt` means the completed
  655M result has a consistent paired evidence chain.
- `bitdistill_paper_alignment.status == not_exact_reproduction` is intentional:
  the active experiment differs from the paper in Stage-2 budget, corpus,
  hardware, effective batch size, and unfinished QNLI/SST2/CNNDM coverage.
- `bitdistill_next_decision.status == run_gamma_balanced_downstream` records
  the bounded next experiment; it is decision support, not new quality evidence.
- `bitdistill_next_experiment_blueprint.status == run_gamma_balanced_downstream`
  permits only the matched gamma-60 MNLI quality run.
- Do not update quality claims from gamma-60 until its 10k-step run produces a
  complete paired prediction trace.

## Validation

```bash
python -m py_compile train_bitdistill.py benchmarks/*.py
python benchmarks/validate_stage2_manifest.py benchmarks/results/stage2_manifest_2026-05-20.json
python benchmarks/validate_reports_fail_closed.py <reports-to-check>
python benchmarks/validate_public_docs.py
```

The report validator must reject silent `0/0` reports unless the report has an
explicit `empty_expected_reason` or equivalent text.
