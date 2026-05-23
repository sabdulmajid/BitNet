#!/usr/bin/env python3
"""Refresh and validate active BitDistill gate reports with one command.

The watchdog intentionally does not produce benchmark evidence. It refreshes
status/ingestion reports and validates that pending or completed artifacts are
handled fail-closed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE", "2026-05-23")
MAX_CAPTURE_CHARS = 6000


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def truncate(text: str) -> str:
    if len(text) <= MAX_CAPTURE_CHARS:
        return text
    return text[: MAX_CAPTURE_CHARS // 2] + "\n...[truncated]...\n" + text[-MAX_CAPTURE_CHARS // 2 :]


def run_command(label: str, command: list[str]) -> dict[str, Any]:
    start = time.time()
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed = time.time() - start
    return {
        "label": label,
        "command": command,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "passed": result.returncode == 0,
        "stdout": truncate(result.stdout),
        "stderr": truncate(result.stderr),
    }


def report_status(path: Path) -> Any:
    data = read_json(path)
    return data.get("status") or data.get("completion_status")


def build_summary() -> dict[str, Any]:
    monitor = read_json(Path(f"benchmarks/results/active_stage2_extension_monitor_{DATE}.json"))
    ingestion = read_json(Path(f"benchmarks/results/stage2_655m_ingestion_{DATE}.json"))
    slurm_scripts = read_json(Path(f"benchmarks/results/active_slurm_batch_scripts_{DATE}.json"))
    traceability = read_json(Path(f"benchmarks/results/bitdistill_goal_traceability_{DATE}.json"))
    next_decision = read_json(Path(f"benchmarks/results/bitdistill_next_decision_{DATE}.json"))
    next_blueprint = read_json(Path(f"benchmarks/results/bitdistill_next_experiment_blueprint_{DATE}.json"))
    stage2 = monitor.get("stage2", {}) if isinstance(monitor.get("stage2"), dict) else {}
    latest_step = stage2.get("latest_step", {}) if isinstance(stage2.get("latest_step"), dict) else {}
    time_limit_gate = stage2.get("time_limit_gate", {}) if isinstance(stage2.get("time_limit_gate"), dict) else {}
    log_freshness = stage2.get("log_freshness", {}) if isinstance(stage2.get("log_freshness"), dict) else {}
    return {
        "monitor_status": monitor.get("status"),
        "ingestion_status": ingestion.get("status"),
        "slurm_script_status": slurm_scripts.get("status"),
        "traceability_status": traceability.get("completion_status"),
        "next_decision_status": next_decision.get("status"),
        "next_blueprint_status": next_blueprint.get("status"),
        "next_blueprint_action": (next_blueprint.get("current_action") or {}).get("action")
        if isinstance(next_blueprint.get("current_action"), dict)
        else None,
        "stage2_job_id": stage2.get("job_id"),
        "stage2_latest_step": latest_step.get("step"),
        "stage2_latest_ce": latest_step.get("ce"),
        "stage2_progress": stage2.get("progress"),
        "stage2_log_freshness": log_freshness.get("status"),
        "stage2_time_limit_status": time_limit_gate.get("status"),
        "stage2_time_limit_margin_seconds": time_limit_gate.get("margin_seconds"),
        "downstream_status": (monitor.get("downstream") or {}).get("status")
        if isinstance(monitor.get("downstream"), dict)
        else None,
        "telemetry_state": ((monitor.get("telemetry") or {}).get("slurm") or {}).get("state")
        if isinstance(monitor.get("telemetry"), dict)
        else None,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != 0.0 and (abs(value) >= 10000.0 or abs(value) < 0.0001):
            return f"{value:.3e}"
        return f"{value:.6f}"
    if isinstance(value, list):
        return ", ".join(fmt(item) for item in value) if value else "none"
    if isinstance(value, dict):
        return ", ".join(f"{key}={fmt(val)}" for key, val in value.items()) if value else "none"
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
    command_rows = [
        [
            command["label"],
            command["passed"],
            command["returncode"],
            command["elapsed_seconds"],
        ]
        for command in report["commands"]
    ]
    summary_rows = [[key, value] for key, value in report["summary"].items()]
    source_rows = [[key, value] for key, value in report["source_paths"].items()]
    failures = [command for command in report["commands"] if not command["passed"]]
    failure_sections = []
    for command in failures:
        failure_sections.append(
            "\n".join(
                [
                    f"### {command['label']}",
                    "```text",
                    command["stderr"] or command["stdout"] or "(no output)",
                    "```",
                ]
            )
        )
    return "\n\n".join(
        [
            "# Active BitDistill Gate Watchdog",
            f"Generated: `{report['created_utc']}`",
            f"Status: **{report['status']}**.",
            f"Quality claim: **{report['quality_claim']}**.",
            report["caveat"],
            "## Summary",
            md_table(["field", "value"], summary_rows),
            "## Commands",
            md_table(["label", "passed", "returncode", "elapsed seconds"], command_rows),
            "## Failures",
            "\n\n".join(failure_sections) if failure_sections else "none",
            "## Source Artifacts",
            md_table(["artifact", "path"], source_rows),
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=DATE)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"benchmarks/results/active_gate_watchdog_{DATE}.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"benchmarks/results/active_gate_watchdog_{DATE}.md"),
    )
    args = parser.parse_args()

    python = sys.executable
    fail_closed_paths = [
        f"benchmarks/results/active_stage2_extension_monitor_{args.date}.json",
        f"benchmarks/results/active_stage2_extension_monitor_{args.date}.md",
        f"benchmarks/results/stage2_655m_ingestion_{args.date}.json",
        f"benchmarks/results/stage2_655m_ingestion_{args.date}.md",
        f"benchmarks/results/active_slurm_batch_scripts_{args.date}.json",
        f"benchmarks/results/active_slurm_batch_scripts_{args.date}.md",
        f"benchmarks/results/current_goal_status_{args.date}.json",
        f"benchmarks/results/current_goal_status_{args.date}.md",
        f"benchmarks/results/deep_research_handoff_{args.date}.json",
        f"benchmarks/results/deep_research_handoff_{args.date}.md",
        f"benchmarks/results/bitdistill_goal_traceability_{args.date}.json",
        f"benchmarks/results/bitdistill_goal_traceability_{args.date}.md",
        f"benchmarks/results/bitdistill_paper_alignment_{args.date}.json",
        f"benchmarks/results/bitdistill_paper_alignment_{args.date}.md",
        f"benchmarks/results/bitdistill_publication_product_plan_{args.date}.json",
        f"benchmarks/results/bitdistill_publication_product_plan_{args.date}.md",
        f"benchmarks/results/bitdistill_next_decision_{args.date}.json",
        f"benchmarks/results/bitdistill_next_decision_{args.date}.md",
        f"benchmarks/results/bitdistill_next_experiment_blueprint_{args.date}.json",
        f"benchmarks/results/bitdistill_next_experiment_blueprint_{args.date}.md",
    ]
    commands = [
        ["monitor active Stage-2 extension", [python, "benchmarks/monitor_active_stage2_extension.py"]],
        ["audit 655M ingestion", [python, "benchmarks/audit_stage2_655m_ingestion.py"]],
        ["audit active Slurm batch scripts", [python, "benchmarks/audit_active_slurm_batch_scripts.py"]],
        ["build next decision", [python, "benchmarks/build_bitdistill_next_decision.py"]],
        ["build next experiment blueprint", [python, "benchmarks/build_bitdistill_next_experiment_blueprint.py"]],
        ["build current goal status", [python, "benchmarks/build_current_goal_status.py"]],
        ["build deep research handoff", [python, "benchmarks/build_deep_research_handoff.py"]],
        ["build goal traceability", [python, "benchmarks/build_goal_traceability_audit.py"]],
        ["build paper alignment audit", [python, "benchmarks/build_bitdistill_paper_alignment_audit.py"]],
        ["build publication/product plan", [python, "benchmarks/build_publication_product_plan.py"]],
        ["validate fail-closed reports", [python, "benchmarks/validate_reports_fail_closed.py", *fail_closed_paths]],
        ["compile Python sources", [python, "-m", "compileall", "-q", "train_bitdistill.py", "train_distill.py", "benchmarks"]],
        [
            "check Slurm shell syntax",
            [
                "bash",
                "-n",
                "slurm_gamma60_telemetry.sh",
                "slurm_stage2_655m_handoff.sh",
                "slurm_stage2_655m_postprocess.sh",
            ],
        ],
    ]
    command_results = [run_command(label, command) for label, command in commands]
    passed = all(result["passed"] for result in command_results)
    report = {
        "schema": "bitdistill-active-gate-watchdog-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quality_claim": "none",
        "status": "passed" if passed else "failed",
        "caveat": "This watchdog refreshes status and validates artifacts; it does not create benchmark evidence.",
        "commands": command_results,
        "summary": build_summary(),
        "source_paths": {
            "active_monitor": f"benchmarks/results/active_stage2_extension_monitor_{args.date}.json",
            "ingestion": f"benchmarks/results/stage2_655m_ingestion_{args.date}.json",
            "slurm_script_audit": f"benchmarks/results/active_slurm_batch_scripts_{args.date}.json",
            "traceability": f"benchmarks/results/bitdistill_goal_traceability_{args.date}.json",
            "paper_alignment": f"benchmarks/results/bitdistill_paper_alignment_{args.date}.json",
            "publication_product_plan": f"benchmarks/results/bitdistill_publication_product_plan_{args.date}.json",
            "next_decision": f"benchmarks/results/bitdistill_next_decision_{args.date}.json",
            "next_experiment_blueprint": f"benchmarks/results/bitdistill_next_experiment_blueprint_{args.date}.json",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
