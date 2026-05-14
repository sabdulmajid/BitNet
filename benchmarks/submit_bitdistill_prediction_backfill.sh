#!/bin/bash
# Submit max_steps=0 evaluation jobs that backfill eval_predictions.jsonl for
# already trained BitDistill/FP16 checkpoints.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
MODEL_SLUG="${MODEL//\//-}"
SOURCE_ROOT="${SOURCE_ROOT:-checkpoints/bitdistill-glue-seqcls}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/bitdistill-glue-seqcls-predtrace}"
TASKS=(${TASKS:-mnli qnli sst2})
METHODS=(${METHODS:-fp16_sft})
SBATCH_PARTITION="${SBATCH_PARTITION:-dualcard}"
SBATCH_MEM="${SBATCH_MEM:-24G}"
DEPENDENCY="${DEPENDENCY:-}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
TASK_FORMAT="${TASK_FORMAT:-sequence_classification}"
LABEL_SCHEME="${LABEL_SCHEME:-letters}"

mkdir -p benchmark_results
JOB_TABLE="${JOB_TABLE:-benchmark_results/bitdistill_prediction_backfill_$(date -u +%Y%m%d_%H%M%S)_$$_${RANDOM}.tsv}"
printf "phase\ttask\tmethod\tjob_id\tdependency\tsource_state\toutput_dir\tmax_seq_len\teval_batch_size\n" > "$JOB_TABLE"

for task in "${TASKS[@]}"; do
  for method in "${METHODS[@]}"; do
    source_dir="$SOURCE_ROOT/$MODEL_SLUG/$task/${method}-tensor-layer-1"
    source_state="$source_dir/custom_state_dict.pt"
    output_dir="$OUTPUT_ROOT/$MODEL_SLUG/$task/${method}-tensor-layer-1"
    if [ ! -s "$source_state" ]; then
      echo "missing source state: $source_state" >&2
      exit 1
    fi

    sbatch_args=(--parsable --job-name=bitdistill-predtrace --partition="$SBATCH_PARTITION" --mem="$SBATCH_MEM")
    if [ -n "$DEPENDENCY" ]; then
      sbatch_args+=(--dependency="$DEPENDENCY")
    fi

    job_id="$(
      env \
        MODEL="$MODEL" \
        STAGE=task_sft \
        METHOD="$method" \
        TASK_NAME="$task" \
        TASK_FORMAT="$TASK_FORMAT" \
        LABEL_SCHEME="$LABEL_SCHEME" \
        INIT_STATE_DICT="$source_state" \
        OUTPUT_DIR="$output_dir" \
        MAX_STEPS=0 \
        MAX_SEQ_LEN="$MAX_SEQ_LEN" \
        MAX_TRAIN_SAMPLES=1 \
        MAX_EVAL_SAMPLES=0 \
        PER_DEVICE_BATCH_SIZE=1 \
        EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
        GRAD_ACCUM_STEPS=1 \
        SAVE_MODEL_ARTIFACTS=0 \
        SAVE_EVERY_STEPS=0 \
        sbatch "${sbatch_args[@]}" slurm_bitdistill_glue.sh
    )"
    printf "prediction_backfill\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$task" "$method" "$job_id" "${DEPENDENCY:-none}" "$source_state" "$output_dir" "$MAX_SEQ_LEN" "$EVAL_BATCH_SIZE" | tee -a "$JOB_TABLE"
  done
done

echo "wrote $JOB_TABLE"
