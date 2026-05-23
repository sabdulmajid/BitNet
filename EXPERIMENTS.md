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

## Validation

```bash
python -m py_compile train_bitdistill.py benchmarks/*.py
python benchmarks/validate_stage2_manifest.py benchmarks/results/stage2_manifest_2026-05-20.json
python benchmarks/validate_reports_fail_closed.py <reports-to-check>
python benchmarks/validate_public_docs.py
```

The report validator must reject silent `0/0` reports unless the report has an
explicit `empty_expected_reason` or equivalent text.
