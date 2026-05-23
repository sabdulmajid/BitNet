#!/usr/bin/env python3
"""Audit queued Slurm batch scripts for active BitDistill handoff jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def squeue_state(job_id: str) -> dict[str, str]:
    result = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%i\t%T\t%M\t%R\t%j"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"job_id": job_id, "state": "not_in_squeue"}
    parts = result.stdout.strip().split("\t", 4)
    return {
        "job_id": parts[0] if len(parts) > 0 else job_id,
        "state": parts[1] if len(parts) > 1 else "unknown",
        "time": parts[2] if len(parts) > 2 else "",
        "reason": parts[3] if len(parts) > 3 else "",
        "name": parts[4] if len(parts) > 4 else "",
    }


def slurm_batch_script(job_id: str) -> tuple[str, str]:
    with tempfile.TemporaryDirectory(prefix="bitnet-slurm-script-") as tmpdir:
        script_path = Path(tmpdir) / f"{job_id}.sh"
        result = subprocess.run(
            ["scontrol", "write", "batch_script", job_id, str(script_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        if result.returncode != 0:
            return "", stderr or stdout
        if not script_path.exists():
            return "", stderr or stdout or "scontrol did not write a batch script"
        return script_path.read_text(encoding="utf-8", errors="replace"), stderr or stdout


def check_snippets(script: str, required: list[str]) -> list[dict[str, Any]]:
    return [{"snippet": snippet, "present": snippet in script} for snippet in required]


def audit_handoff(submission: dict[str, Any]) -> dict[str, Any]:
    job_id = str(submission.get("handoff_job_id", ""))
    script, message = slurm_batch_script(job_id)
    required = [
        "write_failure_report()",
        "trap 'status=$?; trap - ERR; write_failure_report",
        "--downstream-failed-job-id \"\"",
        "--downstream-failure-mode \"\"",
        "slurm_stage2_655m_postprocess.sh",
        "POSTPROCESS_JOB_ID",
        "INIT_STATE_MANIFEST=\"$MANIFEST_JSON\"",
        "SCALE_MODE=tensor",
        "TASK_NAME=mnli",
        "TASK_FORMAT=sequence_classification",
        "LABEL_SCHEME=letters",
        "CANDIDATE_SCORE=mean",
        "TEACHER_MODEL=checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1",
        "ATTENTION_KD_WEIGHT=100000",
        "LOGIT_KD_WEIGHT=10",
        "LOGIT_TEMPERATURE=5.0",
        "LOGIT_KD_TEMPERATURE_SCALE=none",
        "ATTENTION_TEMPERATURE=1.0",
        "INIT_OUTPUT_HEAD_FROM_TEACHER=1",
        "MAX_SEQ_LEN=512",
        "MAX_STEPS=10000",
        "PER_DEVICE_BATCH_SIZE=4",
        "GRAD_ACCUM_STEPS=4",
        "LR=2e-5",
        "LR_SCHEDULER=cosine",
        "SAVE_MODEL_ARTIFACTS=0",
        "OUTPUT_DIR=\"$DOWNSTREAM_OUTPUT_DIR\"",
        "sbatch --parsable --partition=midcard --job-name=bd-mnli-655m slurm_bitdistill_glue.sh",
    ]
    checks = check_snippets(script, required)
    passed = bool(script) and all(check["present"] for check in checks)
    return {
        "job_id": job_id,
        "purpose": "655M Stage-2 handoff",
        "slurm": squeue_state(job_id),
        "script_available": bool(script),
        "scontrol_message": message,
        "checks": checks,
        "passed": passed,
        "cancelled_handoff_job_id": submission.get("cancelled_handoff_job_id", ""),
        "cancelled_handoff_reason": submission.get("cancelled_handoff_reason", ""),
    }


def audit_postprocess_script(path: Path = Path("slurm_stage2_655m_postprocess.sh")) -> dict[str, Any]:
    script = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    required = [
        "build_bitdistill_next_decision.py",
        "DECISION_JSON",
        "DECISION_MD",
        "INGESTION_JSON",
        "audit_stage2_655m_ingestion.py",
        "validate_reports_fail_closed.py",
    ]
    checks = check_snippets(script, required)
    passed = bool(script) and all(check["present"] for check in checks)
    return {
        "job_id": "local",
        "purpose": "655M Stage-2 postprocess script",
        "slurm": {"state": "local_file"},
        "script_available": bool(script),
        "scontrol_message": str(path),
        "checks": checks,
        "passed": passed,
    }


def audit_gamma(submission: dict[str, Any]) -> dict[str, Any]:
    job_id = str(submission.get("job_id", ""))
    script, message = slurm_batch_script(job_id)
    required = [
        "write_status_report()",
        "export ATTENTION_KD_WEIGHT=60",
        "export MAX_STEPS=200",
        "export TELEMETRY_EVERY_STEPS=25",
        "export TELEMETRY_COMPONENT_GRAD_NORMS=1",
        "audit_bitdistill_gamma_balance.py",
        "build_bitdistill_next_decision.py",
        "validate_reports_fail_closed.py",
    ]
    checks = check_snippets(script, required)
    passed = bool(script) and all(check["present"] for check in checks)
    return {
        "job_id": job_id,
        "purpose": "gamma-60 gradient telemetry",
        "slurm": squeue_state(job_id),
        "script_available": bool(script),
        "scontrol_message": message,
        "checks": checks,
        "passed": passed,
        "caveat": submission.get("caveat", ""),
    }


def audit_afterany(submission: dict[str, Any]) -> dict[str, Any]:
    job_id = str(submission.get("job_id", ""))
    script, message = slurm_batch_script(job_id)
    local_script_path = Path(str(submission.get("script") or "slurm_stage2_655m_afterany_audit.sh"))
    if not script and local_script_path.exists():
        script = local_script_path.read_text(encoding="utf-8", errors="replace")
        message = (message + "; " if message else "") + f"validated local script fallback {local_script_path}"
    required = [
        "audit_stage2_snapshot_salvage.py",
        "audit_stage2_655m_ingestion.py",
        "run_active_gate_watchdog.py",
        "bitnet-stage2-afterany-audit-v1",
        "quality_claim",
        "This afterany audit refreshes postmortem/salvage status only",
        "exit \"$EXIT_CODE\"",
    ]
    checks = check_snippets(script, required)
    passed = bool(script) and all(check["present"] for check in checks)
    return {
        "job_id": job_id,
        "purpose": "655M Stage-2 afterany audit",
        "slurm": squeue_state(job_id),
        "script_available": bool(script),
        "scontrol_message": message,
        "checks": checks,
        "passed": passed,
        "caveat": submission.get("caveat", ""),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    handoff = audit_handoff(read_json(args.handoff_submission))
    postprocess = audit_postprocess_script()
    gamma = audit_gamma(read_json(args.gamma_submission))
    afterany = audit_afterany(read_json(args.afterany_submission))
    checks = [handoff, postprocess, gamma, afterany]
    return {
        "schema": "bitnet-active-slurm-batch-script-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quality_claim": "none",
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "checks": checks,
    }


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    snippet_rows = []
    for check in report["checks"]:
        rows.append(
            [
                check["purpose"],
                check["job_id"],
                check["slurm"].get("state"),
                check["script_available"],
                check["passed"],
            ]
        )
        for snippet in check["checks"]:
            snippet_rows.append([check["job_id"], snippet["snippet"], snippet["present"]])
    return "\n\n".join(
        [
            "# Active Slurm Batch Script Audit",
            f"Status: **{report['status']}**.",
            "Quality claim: **none**. This validates queued script contents only.",
            md_table(["purpose", "job", "state", "script available", "passed"], rows),
            "## Required Snippets",
            md_table(["job", "snippet", "present"], snippet_rows),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--handoff-submission",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_handoff_submission_2026-05-23.json"),
    )
    parser.add_argument(
        "--gamma-submission",
        type=Path,
        default=Path("benchmarks/results/gamma60_telemetry_submission_2026-05-23.json"),
    )
    parser.add_argument(
        "--afterany-submission",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_afterany_submission_2026-05-23.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/active_slurm_batch_scripts_2026-05-23.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/active_slurm_batch_scripts_2026-05-23.md"),
    )
    args = parser.parse_args()

    report = build(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report).rstrip() + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
