#!/bin/bash
#SBATCH --job-name=bitnet-ptq-export
#SBATCH --partition=dualcard
#SBATCH --nodelist=ece-nebula10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs checkpoints

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/qwen2.5-1.5b-naive-ptq-tensor}"
DTYPE="${DTYPE:-bf16}"
SCALE_MODE="${SCALE_MODE:-tensor}"
EXPECT_TERNARY_KEYS="${EXPECT_TERNARY_KEYS:-196}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "MODEL=$MODEL"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "DTYPE=$DTYPE"
echo "SCALE_MODE=$SCALE_MODE"
echo "EXPECT_TERNARY_KEYS=$EXPECT_TERNARY_KEYS"
echo "HF_HOME=$HF_HOME"

python benchmarks/export_naive_ptq.py \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --dtype "$DTYPE" \
  --scale-mode "$SCALE_MODE" \
  --expect-ternary-keys "$EXPECT_TERNARY_KEYS"
