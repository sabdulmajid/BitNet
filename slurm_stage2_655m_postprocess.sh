#!/bin/bash
#SBATCH --job-name=bd-655m-post
#SBATCH --partition=midcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs benchmark_results benchmarks/results

DATE="${BITNET_REPORT_DATE:-2026-05-23}"
DOWNSTREAM_JOB_ID="${DOWNSTREAM_JOB_ID:-}"
DOWNSTREAM_OUTPUT_DIR="${DOWNSTREAM_OUTPUT_DIR:-checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit}"
METRICS_JSON="$DOWNSTREAM_OUTPUT_DIR/metrics.json"
PREDICTIONS_JSONL="$DOWNSTREAM_OUTPUT_DIR/eval_predictions.jsonl"
CONTROLLED_JSON="${CONTROLLED_JSON:-benchmarks/results/bitdistill_controlled_curve_${DATE}.json}"
CONTROLLED_MD="${CONTROLLED_MD:-benchmarks/results/bitdistill_controlled_curve_${DATE}.md}"
GAP_JSON="${GAP_JSON:-benchmarks/results/bitdistill_reproduction_gap_${DATE}.json}"
GAP_MD="${GAP_MD:-benchmarks/results/bitdistill_reproduction_gap_${DATE}.md}"
POSTPROCESS_JSON="${POSTPROCESS_JSON:-benchmarks/results/stage2_655m_postprocess_${DATE}.json}"
POSTPROCESS_MD="${POSTPROCESS_MD:-benchmarks/results/stage2_655m_postprocess_${DATE}.md}"
DECISION_JSON="${DECISION_JSON:-benchmarks/results/bitdistill_next_decision_${DATE}.json}"
DECISION_MD="${DECISION_MD:-benchmarks/results/bitdistill_next_decision_${DATE}.md}"

write_postprocess_report() {
  local status="$1"
  local caveat="$2"
  python - <<PY
import json
from pathlib import Path

data = {
    "schema": "bitnet-stage2-extension-postprocess-v1",
    "status": "$status",
    "postprocess_job_id": "${SLURM_JOB_ID:-local}",
    "downstream_job_id": "$DOWNSTREAM_JOB_ID",
    "downstream_output_dir": "$DOWNSTREAM_OUTPUT_DIR",
    "metrics_json": "$METRICS_JSON",
    "metrics_exists": Path("$METRICS_JSON").exists(),
    "predictions_jsonl": "$PREDICTIONS_JSONL",
    "predictions_exists": Path("$PREDICTIONS_JSONL").exists(),
    "controlled_curve_json": "$CONTROLLED_JSON",
    "controlled_curve_md": "$CONTROLLED_MD",
    "reproduction_gap_json": "$GAP_JSON",
    "reproduction_gap_md": "$GAP_MD",
    "next_decision_json": "$DECISION_JSON",
    "next_decision_md": "$DECISION_MD",
    "caveat": "$caveat",
}
Path("$POSTPROCESS_JSON").write_text(json.dumps(data, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
Path("$POSTPROCESS_MD").write_text(
    "\\n\\n".join([
        "# Stage-2 655.36M Postprocess",
        f"Status: **{data['status']}**.",
        "| field | value |\\n| --- | --- |\\n"
        f"| postprocess_job_id | `{data['postprocess_job_id']}` |\\n"
        f"| downstream_job_id | `{data['downstream_job_id']}` |\\n"
        f"| metrics_exists | `{data['metrics_exists']}` |\\n"
        f"| predictions_exists | `{data['predictions_exists']}` |\\n"
        f"| controlled_curve_json | `{data['controlled_curve_json']}` |\\n"
        f"| reproduction_gap_json | `{data['reproduction_gap_json']}` |\\n"
        f"| next_decision_json | `{data['next_decision_json']}` |",
        data["caveat"],
    ]) + "\\n",
    encoding="utf-8",
)
PY
}

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "DOWNSTREAM_JOB_ID=$DOWNSTREAM_JOB_ID"
echo "DOWNSTREAM_OUTPUT_DIR=$DOWNSTREAM_OUTPUT_DIR"

if [[ ! -s "$METRICS_JSON" || ! -s "$PREDICTIONS_JSONL" ]]; then
  python benchmarks/monitor_active_stage2_extension.py
  python benchmarks/build_current_goal_status.py
  python benchmarks/build_deep_research_handoff.py
  python benchmarks/build_bitdistill_next_decision.py \
    --output-json "$DECISION_JSON" \
    --output-md "$DECISION_MD"
  write_postprocess_report \
    "downstream_incomplete" \
    "Downstream metrics or prediction traces are missing. No quality reports were rebuilt."
  exit 0
fi

BITNET_REPORT_DATE="$DATE" python benchmarks/audit_bitdistill_controlled_curve.py \
  --output-json "$CONTROLLED_JSON" \
  --output-md "$CONTROLLED_MD"

python benchmarks/build_reproduction_gap_report.py \
  --controlled-curve "$CONTROLLED_JSON" \
  --output-json "$GAP_JSON" \
  --output-md "$GAP_MD"

python benchmarks/monitor_active_stage2_extension.py
python benchmarks/build_current_goal_status.py
python benchmarks/build_deep_research_handoff.py
python benchmarks/build_bitdistill_next_decision.py \
  --reproduction-gap "$GAP_JSON" \
  --controlled-curve "$CONTROLLED_JSON" \
  --output-json "$DECISION_JSON" \
  --output-md "$DECISION_MD"

python benchmarks/validate_reports_fail_closed.py \
  "$CONTROLLED_JSON" \
  "$CONTROLLED_MD" \
  "$GAP_JSON" \
  "$GAP_MD" \
  "$DECISION_JSON" \
  "$DECISION_MD" \
  benchmarks/results/current_goal_status_2026-05-23.json \
  benchmarks/results/current_goal_status_2026-05-23.md \
  benchmarks/results/deep_research_handoff_2026-05-23.json \
  benchmarks/results/deep_research_handoff_2026-05-23.md

write_postprocess_report \
  "reports_rebuilt" \
  "The controlled curve and reproduction-gap reports were rebuilt from completed downstream metrics and prediction traces."
