#!/bin/bash
#SBATCH --job-name=bd-655m-handoff
#SBATCH --partition=midcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs benchmark_results benchmarks/results

DATE="${BITNET_REPORT_DATE:-2026-05-23}"
STAGE2_JOB_ID="${STAGE2_JOB_ID:-10250}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
PARENT_MANIFEST="${PARENT_MANIFEST:-benchmarks/results/stage2_manifest_2026-05-20.json}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m}"
MANIFEST_JSON="${MANIFEST_JSON:-benchmarks/results/stage2_manifest_655m_${DATE}.json}"
MANIFEST_MD="${MANIFEST_MD:-benchmarks/results/stage2_manifest_655m_${DATE}.md}"
RUN_ID="${RUN_ID:-qwen25-05b-bitdistill-tensor-stage2-655m-from327m-job10250}"
DOWNSTREAM_OUTPUT_DIR="${DOWNSTREAM_OUTPUT_DIR:-checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit}"
HANDOFF_JSON="${HANDOFF_JSON:-benchmarks/results/stage2_655m_handoff_${DATE}.json}"
HANDOFF_MD="${HANDOFF_MD:-benchmarks/results/stage2_655m_handoff_${DATE}.md}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "STAGE2_JOB_ID=$STAGE2_JOB_ID"
echo "PARENT_MANIFEST=$PARENT_MANIFEST"
echo "STAGE2_OUTPUT_DIR=$STAGE2_OUTPUT_DIR"
echo "MANIFEST_JSON=$MANIFEST_JSON"
echo "DOWNSTREAM_OUTPUT_DIR=$DOWNSTREAM_OUTPUT_DIR"

python benchmarks/build_stage2_manifest.py \
  --output-dir "$STAGE2_OUTPUT_DIR" \
  --parent-manifest "$PARENT_MANIFEST" \
  --run-id "$RUN_ID" \
  --job-id "$STAGE2_JOB_ID" \
  --downstream-status pending_submission \
  --output-json "$MANIFEST_JSON" \
  --output-md "$MANIFEST_MD"

python benchmarks/validate_stage2_manifest.py "$MANIFEST_JSON"

DOWNSTREAM_JOB_ID="$(
  MODEL="$MODEL" \
  STAGE=task_sft \
  METHOD=bitdistill \
  TASK_NAME=mnli \
  TASK_FORMAT=sequence_classification \
  LABEL_SCHEME=letters \
  CANDIDATE_SCORE=mean \
  TEACHER_MODEL=checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1 \
  INIT_STATE_MANIFEST="$MANIFEST_JSON" \
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
  LR_SCHEDULER=cosine \
  SAVE_EVERY_STEPS=0 \
  SAVE_MODEL_ARTIFACTS=0 \
  OUTPUT_DIR="$DOWNSTREAM_OUTPUT_DIR" \
  sbatch --parsable --partition=midcard --job-name=bd-mnli-655m slurm_bitdistill_glue.sh
)"

python benchmarks/build_stage2_manifest.py \
  --output-dir "$STAGE2_OUTPUT_DIR" \
  --parent-manifest "$PARENT_MANIFEST" \
  --run-id "$RUN_ID" \
  --job-id "$STAGE2_JOB_ID" \
  --downstream-status submitted_downstream \
  --downstream-rerun-job-id "$DOWNSTREAM_JOB_ID" \
  --downstream-output-dir "$DOWNSTREAM_OUTPUT_DIR" \
  --output-json "$MANIFEST_JSON" \
  --output-md "$MANIFEST_MD"

python benchmarks/validate_stage2_manifest.py "$MANIFEST_JSON"

python - <<PY
import json
from pathlib import Path

data = {
    "schema": "bitnet-stage2-extension-handoff-v1",
    "status": "submitted_downstream",
    "stage2_job_id": "$STAGE2_JOB_ID",
    "handoff_job_id": "${SLURM_JOB_ID:-local}",
    "downstream_job_id": "$DOWNSTREAM_JOB_ID",
    "manifest_json": "$MANIFEST_JSON",
    "manifest_md": "$MANIFEST_MD",
    "downstream_output_dir": "$DOWNSTREAM_OUTPUT_DIR",
    "next_after_downstream": [
        "Run benchmarks/audit_bitdistill_controlled_curve.py with the 655M manifest included.",
        "Update canonical reports only after downstream metrics.json and eval_predictions.jsonl exist."
    ],
}
Path("$HANDOFF_JSON").write_text(json.dumps(data, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
Path("$HANDOFF_MD").write_text(
    "\\n\\n".join([
        "# Stage-2 655.36M Handoff",
        "Status: **submitted_downstream**.",
        "| field | value |\\n| --- | --- |\\n"
        f"| stage2_job_id | `{data['stage2_job_id']}` |\\n"
        f"| handoff_job_id | `{data['handoff_job_id']}` |\\n"
        f"| downstream_job_id | `{data['downstream_job_id']}` |\\n"
        f"| manifest_json | `{data['manifest_json']}` |\\n"
        f"| downstream_output_dir | `{data['downstream_output_dir']}` |",
        "Do not update quality claims until the downstream directory has both `metrics.json` and `eval_predictions.jsonl`."
    ]) + "\\n",
    encoding="utf-8",
)
PY

echo "DOWNSTREAM_JOB_ID=$DOWNSTREAM_JOB_ID"
