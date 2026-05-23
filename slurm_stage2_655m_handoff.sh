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
POSTPROCESS_JSON="${POSTPROCESS_JSON:-benchmarks/results/stage2_655m_postprocess_${DATE}.json}"
POSTPROCESS_MD="${POSTPROCESS_MD:-benchmarks/results/stage2_655m_postprocess_${DATE}.md}"
PRODUCER_BITNET_COMMIT="${PRODUCER_BITNET_COMMIT:-10341701e5104c66d18fc9779ab9799bf2190c9a}"
PRODUCER_LLAMA_CPP_COMMIT="${PRODUCER_LLAMA_CPP_COMMIT:-dc0bc5ee0423a2202d6284a4fc2d78d1e39905d7}"
PRODUCER_BITNET_COMMIT_NOTE="${PRODUCER_BITNET_COMMIT_NOTE:-inferred from the commit that captured the LR_SCHEDULER wrapper patch used by Stage-2 job 10250; the job log confirms LR_SCHEDULER=constant}"
PRODUCER_LLAMA_CPP_COMMIT_NOTE="${PRODUCER_LLAMA_CPP_COMMIT_NOTE:-recorded in the Stage-2 submission report for job 10250}"

write_failure_report() {
  local exit_code="$1"
  local line_no="$2"
  python - <<PY
import json
from pathlib import Path

data = {
    "schema": "bitnet-stage2-extension-handoff-v1",
    "status": "failed",
    "stage2_job_id": "$STAGE2_JOB_ID",
    "handoff_job_id": "${SLURM_JOB_ID:-local}",
    "exit_code": int("$exit_code"),
    "line": "$line_no",
    "manifest_json": "$MANIFEST_JSON",
    "manifest_md": "$MANIFEST_MD",
    "stage2_output_dir": "$STAGE2_OUTPUT_DIR",
    "downstream_output_dir": "$DOWNSTREAM_OUTPUT_DIR",
    "postprocess_json": "$POSTPROCESS_JSON",
    "postprocess_md": "$POSTPROCESS_MD",
    "producer_bitnet_commit": "$PRODUCER_BITNET_COMMIT",
    "producer_llama_cpp_commit": "$PRODUCER_LLAMA_CPP_COMMIT",
    "caveat": "The handoff did not submit or validate downstream quality evidence.",
}
Path("$HANDOFF_JSON").write_text(json.dumps(data, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
Path("$HANDOFF_MD").write_text(
    "\\n\\n".join([
        "# Stage-2 655.36M Handoff",
        "Status: **failed**.",
        "| field | value |\\n| --- | --- |\\n"
        f"| stage2_job_id | `{data['stage2_job_id']}` |\\n"
        f"| handoff_job_id | `{data['handoff_job_id']}` |\\n"
        f"| exit_code | `{data['exit_code']}` |\\n"
        f"| line | `{data['line']}` |\\n"
        f"| manifest_json | `{data['manifest_json']}` |\\n"
        f"| downstream_output_dir | `{data['downstream_output_dir']}` |\\n"
        f"| postprocess_json | `{data['postprocess_json']}` |",
        "No quality claim should be updated from this failed handoff."
    ]) + "\\n",
    encoding="utf-8",
)
PY
}

trap 'status=$?; trap - ERR; write_failure_report "$status" "${BASH_LINENO[0]:-unknown}"; exit "$status"' ERR

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "STAGE2_JOB_ID=$STAGE2_JOB_ID"
echo "PARENT_MANIFEST=$PARENT_MANIFEST"
echo "STAGE2_OUTPUT_DIR=$STAGE2_OUTPUT_DIR"
echo "MANIFEST_JSON=$MANIFEST_JSON"
echo "DOWNSTREAM_OUTPUT_DIR=$DOWNSTREAM_OUTPUT_DIR"
echo "PRODUCER_BITNET_COMMIT=$PRODUCER_BITNET_COMMIT"
echo "PRODUCER_LLAMA_CPP_COMMIT=$PRODUCER_LLAMA_CPP_COMMIT"

python benchmarks/build_stage2_manifest.py \
  --output-dir "$STAGE2_OUTPUT_DIR" \
  --parent-manifest "$PARENT_MANIFEST" \
  --producer-bitnet-commit "$PRODUCER_BITNET_COMMIT" \
  --producer-llama-cpp-commit "$PRODUCER_LLAMA_CPP_COMMIT" \
  --producer-bitnet-commit-note "$PRODUCER_BITNET_COMMIT_NOTE" \
  --producer-llama-cpp-commit-note "$PRODUCER_LLAMA_CPP_COMMIT_NOTE" \
  --run-id "$RUN_ID" \
  --job-id "$STAGE2_JOB_ID" \
  --downstream-status pending_submission \
  --downstream-failed-job-id "" \
  --downstream-failure-mode "" \
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

POSTPROCESS_JOB_ID="$(
  DOWNSTREAM_JOB_ID="$DOWNSTREAM_JOB_ID" \
  DOWNSTREAM_OUTPUT_DIR="$DOWNSTREAM_OUTPUT_DIR" \
  POSTPROCESS_JSON="$POSTPROCESS_JSON" \
  POSTPROCESS_MD="$POSTPROCESS_MD" \
  sbatch --parsable --partition=midcard --dependency=afterany:"$DOWNSTREAM_JOB_ID" \
    --job-name=bd-655m-post slurm_stage2_655m_postprocess.sh
)"

python benchmarks/build_stage2_manifest.py \
  --output-dir "$STAGE2_OUTPUT_DIR" \
  --parent-manifest "$PARENT_MANIFEST" \
  --producer-bitnet-commit "$PRODUCER_BITNET_COMMIT" \
  --producer-llama-cpp-commit "$PRODUCER_LLAMA_CPP_COMMIT" \
  --producer-bitnet-commit-note "$PRODUCER_BITNET_COMMIT_NOTE" \
  --producer-llama-cpp-commit-note "$PRODUCER_LLAMA_CPP_COMMIT_NOTE" \
  --run-id "$RUN_ID" \
  --job-id "$STAGE2_JOB_ID" \
  --downstream-status submitted_downstream \
  --downstream-failed-job-id "" \
  --downstream-failure-mode "" \
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
    "postprocess_job_id": "$POSTPROCESS_JOB_ID",
    "manifest_json": "$MANIFEST_JSON",
    "manifest_md": "$MANIFEST_MD",
    "downstream_output_dir": "$DOWNSTREAM_OUTPUT_DIR",
    "postprocess_json": "$POSTPROCESS_JSON",
    "postprocess_md": "$POSTPROCESS_MD",
    "producer_bitnet_commit": "$PRODUCER_BITNET_COMMIT",
    "producer_llama_cpp_commit": "$PRODUCER_LLAMA_CPP_COMMIT",
    "next_after_downstream": [
        "Wait for the postprocess job to rebuild the controlled curve and reproduction-gap reports.",
        "Update canonical/public claims only after downstream metrics.json and eval_predictions.jsonl exist."
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
        f"| postprocess_job_id | `{data['postprocess_job_id']}` |\\n"
        f"| manifest_json | `{data['manifest_json']}` |\\n"
        f"| downstream_output_dir | `{data['downstream_output_dir']}` |\\n"
        f"| postprocess_json | `{data['postprocess_json']}` |",
        "Do not update quality claims until the downstream directory has both `metrics.json` and `eval_predictions.jsonl`."
    ]) + "\\n",
    encoding="utf-8",
)
PY

trap - ERR
echo "DOWNSTREAM_JOB_ID=$DOWNSTREAM_JOB_ID"
echo "POSTPROCESS_JOB_ID=$POSTPROCESS_JOB_ID"
