#!/bin/bash
#SBATCH --job-name=bitnet-qat-distill
#SBATCH --partition=midcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=28G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs/bitnet-distill

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
STUDENT_INIT_MODEL="${STUDENT_INIT_MODEL:-$MODEL}"
DATASET_NAME="${DATASET_NAME:-HuggingFaceFW/fineweb-edu}"
DATASET_CONFIG="${DATASET_CONFIG:-sample-10BT}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
TEXT_COLUMN="${TEXT_COLUMN:-text}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
MAX_STEPS="${MAX_STEPS:-1000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
LR="${LR:-2e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/bitnet-distill/qwen2.5-1.5b}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "MODEL=$MODEL"
echo "DATASET_NAME=$DATASET_NAME"
echo "OUTPUT_DIR=$OUTPUT_DIR"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  train_distill.py \
  --teacher-model "$MODEL" \
  --student-init-model "$STUDENT_INIT_MODEL" \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  --dataset-split "$DATASET_SPLIT" \
  --text-column "$TEXT_COLUMN" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --max-steps "$MAX_STEPS" \
  --learning-rate "$LR" \
  --temperature 2.0 \
  --kl-weight 1.0 \
  --hidden-mse-weight 1.0 \
  --hidden-state-layers last \
  --model-dtype bf16 \
  --master-weight-dtype fp32 \
  --scale-mode tensor \
  --gradient-checkpointing \
  --use-fsdp \
  --fsdp-mixed-precision \
  --fsdp-cpu-offload \
  --fsdp-wrap-class-names Qwen2DecoderLayer \
  --output-dir "$OUTPUT_DIR" \
  --save-every-steps 250 \
  --save-final \
  --log-every-steps 1
