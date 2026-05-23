#!/bin/bash
#SBATCH --job-name=bd-655m-afterany
#SBATCH --partition=midcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -u -o pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs benchmarks/results

DATE="${BITNET_REPORT_DATE:-2026-05-23}"
STAGE2_JOB_ID="${STAGE2_JOB_ID:-10250}"
DEPENDENCY="${DEPENDENCY:-afterany:${STAGE2_JOB_ID}}"
REPORT_JSON="${REPORT_JSON:-benchmarks/results/stage2_655m_afterany_audit_${DATE}.json}"
REPORT_MD="${REPORT_MD:-benchmarks/results/stage2_655m_afterany_audit_${DATE}.md}"
SALVAGE_JSON="${SALVAGE_JSON:-benchmarks/results/stage2_snapshot_salvage_${DATE}.json}"
INGESTION_JSON="${INGESTION_JSON:-benchmarks/results/stage2_655m_ingestion_${DATE}.json}"
WATCHDOG_JSON="${WATCHDOG_JSON:-benchmarks/results/active_gate_watchdog_${DATE}.json}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "STAGE2_JOB_ID=$STAGE2_JOB_ID"
echo "DEPENDENCY=$DEPENDENCY"
echo "REPORT_JSON=$REPORT_JSON"

python benchmarks/audit_stage2_snapshot_salvage.py
SALVAGE_RC=$?

python benchmarks/audit_stage2_655m_ingestion.py
INGESTION_RC=$?

python benchmarks/run_active_gate_watchdog.py
WATCHDOG_RC=$?

if [[ "$SALVAGE_RC" -eq 0 && "$INGESTION_RC" -eq 0 && "$WATCHDOG_RC" -eq 0 ]]; then
  STATUS="completed"
  EXIT_CODE=0
else
  STATUS="failed"
  EXIT_CODE=1
fi

python - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

data = {
    "schema": "bitnet-stage2-afterany-audit-v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "quality_claim": "none",
    "status": "$STATUS",
    "stage2_job_id": "$STAGE2_JOB_ID",
    "afterany_job_id": "${SLURM_JOB_ID:-local}",
    "dependency": "$DEPENDENCY",
    "returncodes": {
        "snapshot_salvage": int("$SALVAGE_RC"),
        "ingestion": int("$INGESTION_RC"),
        "watchdog": int("$WATCHDOG_RC"),
    },
    "source_paths": {
        "snapshot_salvage": "$SALVAGE_JSON",
        "ingestion": "$INGESTION_JSON",
        "watchdog": "$WATCHDOG_JSON",
    },
    "caveat": "This afterany audit refreshes postmortem/salvage status only. It does not create downstream quality evidence.",
}
Path("$REPORT_JSON").write_text(json.dumps(data, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
Path("$REPORT_MD").write_text(
    "\\n\\n".join([
        "# Stage-2 655.36M Afterany Audit",
        f"Generated: `{data['created_utc']}`",
        f"Status: **{data['status']}**.",
        "Quality claim: **none**.",
        data["caveat"],
        "| field | value |\\n| --- | --- |\\n"
        f"| stage2_job_id | `{data['stage2_job_id']}` |\\n"
        f"| afterany_job_id | `{data['afterany_job_id']}` |\\n"
        f"| dependency | `{data['dependency']}` |\\n"
        f"| snapshot_salvage_rc | `{data['returncodes']['snapshot_salvage']}` |\\n"
        f"| ingestion_rc | `{data['returncodes']['ingestion']}` |\\n"
        f"| watchdog_rc | `{data['returncodes']['watchdog']}` |",
    ]) + "\\n",
    encoding="utf-8",
)
PY

exit "$EXIT_CODE"
