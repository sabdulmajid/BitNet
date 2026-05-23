#!/bin/bash
#SBATCH --job-name=bd-g60-telemetry
#SBATCH --partition=midcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs benchmarks/results

DATE="${BITNET_REPORT_DATE:-2026-05-23}"
STATUS_JSON="${STATUS_JSON:-benchmarks/results/gamma60_telemetry_status_${DATE}.json}"
STATUS_MD="${STATUS_MD:-benchmarks/results/gamma60_telemetry_status_${DATE}.md}"
DYNAMICS_JSON="${DYNAMICS_JSON:-benchmarks/results/bitdistill_training_dynamics_${DATE}.json}"
DYNAMICS_MD="${DYNAMICS_MD:-benchmarks/results/bitdistill_training_dynamics_${DATE}.md}"
BALANCE_JSON="${BALANCE_JSON:-benchmarks/results/gamma60_gradient_balance_${DATE}.json}"
BALANCE_MD="${BALANCE_MD:-benchmarks/results/gamma60_gradient_balance_${DATE}.md}"
DECISION_JSON="${DECISION_JSON:-benchmarks/results/bitdistill_next_decision_${DATE}.json}"
DECISION_MD="${DECISION_MD:-benchmarks/results/bitdistill_next_decision_${DATE}.md}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/bitdistill-glue-seqcls-telemetry-gamma60/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-gamma60-headinit-steps200}"

write_status_report() {
  local status="$1"
  local exit_code="$2"
  python - <<PY
import json
from pathlib import Path

data = {
    "schema": "bitdistill-gamma60-telemetry-status-v1",
    "status": "$status",
    "job_id": "${SLURM_JOB_ID:-local}",
    "exit_code": int("$exit_code"),
    "output_dir": "$OUTPUT_DIR",
    "telemetry_path": "$OUTPUT_DIR/telemetry.jsonl",
    "metrics_path": "$OUTPUT_DIR/metrics.json",
    "attention_kd_weight": 60,
    "max_steps": 200,
    "quality_claim": "none",
    "caveat": "This is a component-gradient diagnostic, not a quality benchmark.",
}
Path("$STATUS_JSON").write_text(json.dumps(data, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
Path("$STATUS_MD").write_text(
    "\\n\\n".join([
        "# Gamma-60 Telemetry Status",
        f"Status: **{data['status']}**.",
        "Quality claim: **none**.",
        "| field | value |\\n| --- | --- |\\n"
        f"| job_id | `{data['job_id']}` |\\n"
        f"| exit_code | `{data['exit_code']}` |\\n"
        f"| output_dir | `{data['output_dir']}` |\\n"
        f"| telemetry_path | `{data['telemetry_path']}` |\\n"
        f"| metrics_path | `{data['metrics_path']}` |\\n"
        f"| attention_kd_weight | `{data['attention_kd_weight']}` |\\n"
        f"| max_steps | `{data['max_steps']}` |",
        data["caveat"],
    ]) + "\\n",
    encoding="utf-8",
)
PY
}

trap 'status=$?; trap - ERR; write_status_report failed "$status"; exit "$status"' ERR

export MODEL="Qwen/Qwen2.5-0.5B"
export STAGE=task_sft
export METHOD=bitdistill
export TASK_NAME=mnli
export TASK_FORMAT=sequence_classification
export LABEL_SCHEME=letters
export CANDIDATE_SCORE=mean
export TEACHER_MODEL="checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1"
export INIT_STATE_DICT="checkpoints/bitdistill-glue-longwarmup/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-20k/custom_state_dict.pt"
export SCALE_MODE=tensor
export EXCLUDE_LINEAR_REGEX='score|classifier'
export DISTILL_LAYER=-1
export ATTENTION_SPLIT_HEADS=8
export ACTIVATION_QUANTIZATION=1
export USE_SUBLN=1
export LOGIT_KD_WEIGHT=10
export ATTENTION_KD_WEIGHT=60
export LOGIT_TEMPERATURE=5.0
export LOGIT_KD_TEMPERATURE_SCALE=none
export ATTENTION_TEMPERATURE=1.0
export INIT_OUTPUT_HEAD_FROM_TEACHER=1
export MAX_SEQ_LEN=512
export MAX_STEPS=200
export PER_DEVICE_BATCH_SIZE=4
export GRAD_ACCUM_STEPS=4
export LR=2e-5
export LR_SCHEDULER=cosine
export TELEMETRY_EVERY_STEPS=25
export TELEMETRY_COMPONENT_GRAD_NORMS=1
export SAVE_EVERY_STEPS=0
export SAVE_MODEL_ARTIFACTS=0
export OUTPUT_DIR

write_status_report running 0
bash slurm_bitdistill_glue.sh
write_status_report complete 0
python benchmarks/audit_bitdistill_training_dynamics.py \
  --output-json "$DYNAMICS_JSON" \
  --output-md "$DYNAMICS_MD"
python benchmarks/audit_bitdistill_gamma_balance.py \
  --job-id "${SLURM_JOB_ID:-local}" \
  --gamma-status "$STATUS_JSON" \
  --gamma-telemetry "$OUTPUT_DIR/telemetry.jsonl" \
  --output-json "$BALANCE_JSON" \
  --output-md "$BALANCE_MD"
python benchmarks/build_bitdistill_next_decision.py \
  --gamma-balance "$BALANCE_JSON" \
  --output-json "$DECISION_JSON" \
  --output-md "$DECISION_MD"
python benchmarks/validate_reports_fail_closed.py \
  "$STATUS_JSON" "$STATUS_MD" \
  "$DYNAMICS_JSON" "$DYNAMICS_MD" \
  "$BALANCE_JSON" "$BALANCE_MD" \
  "$DECISION_JSON" "$DECISION_MD"
trap - ERR
