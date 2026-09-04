#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=${LOCAL_ROOT:-/local/${USER:?}/bitnet-b7fc773}
PARTITION=${PARTITION:-midcard}
NODE=${NODE:-ece-nebula12}
OUTPUT_TSV=${OUTPUT_TSV:-benchmark_results/bitdistill_fixed60_resubmission_2026-09-04.tsv}
AUDIT_SCRIPT=${AUDIT_SCRIPT:-$LOCAL_ROOT/audit-bundle/benchmarks/audit_bitdistill_adaptive_vs_fixed_resumed.py}

COMMON_EXPORTS=(
  "SLURM_SUBMIT_DIR=$LOCAL_ROOT/source"
  "HF_HOME=$LOCAL_ROOT/hf_cache"
  "HF_DATASETS_CACHE=$LOCAL_ROOT/hf_cache/datasets"
  "HF_DATASETS_OFFLINE=1"
  "TRANSFORMERS_OFFLINE=1"
  "PYTHON_BIN=$LOCAL_ROOT/venv/bin/python"
  "MODEL=$LOCAL_ROOT/assets/base_model"
  "TEACHER_MODEL=$LOCAL_ROOT/assets/seqcls_teacher"
  "INIT_STATE_DICT=$LOCAL_ROOT/assets/stage2.pt"
  "STAGE=task_sft"
  "METHOD=bitdistill"
  "TASK_NAME=mnli"
  "TASK_FORMAT=sequence_classification"
  "LABEL_SCHEME=letters"
  "CANDIDATE_SCORE=mean"
  "SCALE_MODE=tensor"
  "EXCLUDE_LINEAR_REGEX=score|classifier"
  "DISTILL_LAYER=-1"
  "ATTENTION_SPLIT_HEADS=1"
  "ACTIVATION_QUANTIZATION=1"
  "USE_SUBLN=1"
  "LOGIT_KD_WEIGHT=10"
  "ATTENTION_KD_WEIGHT=60"
  "LOGIT_TEMPERATURE=5.0"
  "LOGIT_KD_TEMPERATURE_SCALE=none"
  "ATTENTION_TEMPERATURE=1.0"
  "ATTENTION_RELATION_MODE=cosine"
  "ATTENTION_KD_BALANCE=fixed"
  "ATTENTION_BALANCE_TARGET_RATIO=1.0"
  "ATTENTION_BALANCE_BETA=0.9"
  "ATTENTION_BALANCE_EVERY_STEPS=20"
  "ATTENTION_BALANCE_MIN_WEIGHT=0.001"
  "ATTENTION_BALANCE_MAX_WEIGHT=100000"
  "TELEMETRY_EVERY_STEPS=500"
  "TELEMETRY_COMPONENT_GRAD_NORMS=1"
  "TELEMETRY_GRADIENT_COSINES=0"
  "TELEMETRY_MAX_ELEMENTS_PER_LAYER=65536"
  "INIT_OUTPUT_HEAD_FROM_TEACHER=1"
  "MAX_SEQ_LEN=512"
  "MAX_STEPS=10000"
  "PER_DEVICE_BATCH_SIZE=4"
  "EVAL_BATCH_SIZE=16"
  "GRAD_ACCUM_STEPS=4"
  "LR=2e-5"
  "LR_SCHEDULER=cosine"
  "WARMUP_STEPS=100"
  "MIN_LR_RATIO=0.1"
  "MAX_TRAIN_SAMPLES=0"
  "MAX_EVAL_SAMPLES=0"
  "SAVE_EVERY_STEPS=0"
  "SAVE_MODEL_ARTIFACTS=0"
)

join_exports() {
  local IFS=,
  printf '%s' "${*}"
}

submit_seed() {
  local seed=$1
  local dependency=${2:-}
  local output_dir="$LOCAL_ROOT/runs-fixed60/mnli-seqcls-cosine-s1-fixed60-seed$seed"
  local exports
  exports=$(join_exports ALL "${COMMON_EXPORTS[@]}" "SEED=$seed" "OUTPUT_DIR=$output_dir")
  local args=(
    --parsable
    --partition="$PARTITION"
    --nodelist="$NODE"
    --gres=gpu:1
    --cpus-per-task=8
    --mem=24G
    --time=08:00:00
    --chdir="$LOCAL_ROOT/source"
    --job-name="bdm-mnli-fixed60-10k-s$seed"
    --output="$LOCAL_ROOT/logs/bdm-mnli-fixed60-10k-s$seed-%j.out"
    --error="$LOCAL_ROOT/logs/bdm-mnli-fixed60-10k-s$seed-%j.err"
    --export="$exports"
  )
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="afterok:$dependency")
  fi
  sbatch "${args[@]}" \
    --wrap="export SLURM_SUBMIT_DIR=$LOCAL_ROOT/source; exec bash $LOCAL_ROOT/source/slurm_bitdistill_glue.sh"
}

job_1234=$(submit_seed 1234)
job_1235=$(submit_seed 1235 "$job_1234")
job_1236=$(submit_seed 1236 "$job_1235")

audit_job=$(sbatch \
  --parsable \
  --partition="$PARTITION" \
  --nodelist="$NODE" \
  --cpus-per-task=2 \
  --mem=4G \
  --time=00:20:00 \
  --chdir="$LOCAL_ROOT" \
  --dependency="afterany:$job_1236" \
  --job-name=bdm-adapt-vs-fixed-audit-resumed \
  --output="$LOCAL_ROOT/logs/bdm-adapt-vs-fixed-audit-resumed-%j.out" \
  --error="$LOCAL_ROOT/logs/bdm-adapt-vs-fixed-audit-resumed-%j.err" \
  --wrap="$LOCAL_ROOT/venv/bin/python $AUDIT_SCRIPT --fixed-job-ids $job_1234 $job_1235 $job_1236")

mkdir -p "$(dirname "$OUTPUT_TSV")"
printf 'arm\tseed\tjob_id\tdependency\n' > "$OUTPUT_TSV"
printf 'fixed60\t1234\t%s\t\n' "$job_1234" >> "$OUTPUT_TSV"
printf 'fixed60\t1235\t%s\tafterok:%s\n' "$job_1235" "$job_1234" >> "$OUTPUT_TSV"
printf 'fixed60\t1236\t%s\tafterok:%s\n' "$job_1236" "$job_1235" >> "$OUTPUT_TSV"
printf 'audit\t-\t%s\tafterany:%s\n' "$audit_job" "$job_1236" >> "$OUTPUT_TSV"
cat "$OUTPUT_TSV"
