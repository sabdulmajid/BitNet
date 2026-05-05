#!/bin/bash
#SBATCH --job-name=bitnet-row-postprocess
#SBATCH --partition=dualcard
#SBATCH --nodelist=ece-nebula10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p benchmark_results/evidence_audit

ROW_CKPT="${ROW_CKPT:-checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-row-notiehead-5000/step-5000}"
ROW_QUALITY="${ROW_QUALITY:-benchmark_results/quality-qwen15b-klonly-row-notiehead-5000}"
ROW_MC="${ROW_MC:-benchmark_results/mc-qwen15b-klonly-row-notiehead-5000-200}"
ROW_LMEVAL="${ROW_LMEVAL:-benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10}"

echo "ROW_CKPT=$ROW_CKPT"
echo "ROW_QUALITY=$ROW_QUALITY"
echo "ROW_MC=$ROW_MC"
echo "ROW_LMEVAL=$ROW_LMEVAL"

python benchmarks/audit_evidence.py \
  --checkpoint qwen15b_row_notie_step5000="$ROW_CKPT/ternary_state_dict.pt:196:196:row:tie_true" \
  --lm-eval qwen15b_row_notie_full10="$ROW_LMEVAL/qwen15b_qat_ternary.json" \
  --perplexity qwen15b_row_notie_wikitext="$ROW_QUALITY/qwen15b_ternary_wikitext.json" \
  --perplexity qwen15b_row_notie_fineweb="$ROW_QUALITY/qwen15b_ternary_fineweb_heldout.json" \
  --mc qwen15b_row_notie_piqa="$ROW_MC/qwen15b_ternary_piqa.json" \
  --mc qwen15b_row_notie_arc_easy="$ROW_MC/qwen15b_ternary_arc_easy.json" \
  --mc qwen15b_row_notie_arc_challenge="$ROW_MC/qwen15b_ternary_arc_challenge.json" \
  --mc qwen15b_row_notie_hellaswag="$ROW_MC/qwen15b_ternary_hellaswag.json" \
  --output-md benchmark_results/evidence_audit/qwen15b_row_notie_5000.md

python benchmarks/compare_lm_eval.py \
  --run FP=benchmark_results/lm-eval-qwen15b-full10/qwen15b_fp.json \
  --run naive_PTQ=benchmark_results/lm-eval-qwen15b-full10/qwen15b_naive_ptq.json \
  --run QAT_hiddenMSE=benchmark_results/lm-eval-qwen15b-full10/qwen15b_qat_ternary.json \
  --run QAT_KLonly=benchmark_results/lm-eval-qwen15b-klonly-full10/qwen15b_qat_ternary.json \
  --run QAT_KLonly_denseHead=benchmark_results/lm-eval-qwen15b-klonly-notiehead-full10/qwen15b_qat_ternary.json \
  --run QAT_KLonly_rowDenseHead="$ROW_LMEVAL/qwen15b_qat_ternary.json" \
  --output-md "$ROW_LMEVAL/selected_metrics_with_baselines.md"

python benchmarks/paired_lm_eval_delta.py \
  --a QAT_KLonly_denseHead=benchmark_results/lm-eval-qwen15b-klonly-notiehead-full10/qwen15b_qat_ternary.json \
  --b QAT_KLonly_rowDenseHead="$ROW_LMEVAL/qwen15b_qat_ternary.json" \
  --output-md "$ROW_LMEVAL/paired_row_densehead_minus_tensor_densehead.md"

python benchmarks/paired_lm_eval_delta.py \
  --a QAT_KLonly=benchmark_results/lm-eval-qwen15b-klonly-full10/qwen15b_qat_ternary.json \
  --b QAT_KLonly_rowDenseHead="$ROW_LMEVAL/qwen15b_qat_ternary.json" \
  --output-md "$ROW_LMEVAL/paired_row_densehead_minus_klonly.md"

echo "wrote benchmark_results/evidence_audit/qwen15b_row_notie_5000.md"
echo "wrote $ROW_LMEVAL/selected_metrics_with_baselines.md"
echo "wrote $ROW_LMEVAL/paired_row_densehead_minus_tensor_densehead.md"
echo "wrote $ROW_LMEVAL/paired_row_densehead_minus_klonly.md"
