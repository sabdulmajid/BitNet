#!/usr/bin/env python3
"""Run warmup-level BitDistill diagnostics after Stage-2 reaches a terminal state.

The full BitDistill postprocess job waits for downstream task, export, and CPU
jobs. If Stage-2 fails, those downstream ``afterok`` jobs may never release, so a
second finalizer is needed at the warmup boundary. This runner refreshes only
the reports that are meaningful immediately after warmup success or failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()


def command_specs(date: str) -> list[tuple[str, list[str], list[str]]]:
    return [
        (
            "monitor",
            [
                "python",
                "benchmarks/monitor_bitdistill_jobs.py",
                "--output-json",
                f"benchmark_results/bitdistill_job_monitor_{date}.json",
                "--output-md",
                f"benchmarks/results/bitdistill_job_monitor_{date}.md",
            ],
            [
                f"benchmark_results/bitdistill_job_monitor_{date}.json",
                f"benchmarks/results/bitdistill_job_monitor_{date}.md",
            ],
        ),
        (
            "warmup_health",
            [
                "python",
                "benchmarks/audit_bitdistill_warmup_health.py",
                "--output-json",
                f"benchmark_results/bitdistill_warmup_health_{date}.json",
                "--output-md",
                f"benchmarks/results/bitdistill_warmup_health_{date}.md",
            ],
            [
                f"benchmark_results/bitdistill_warmup_health_{date}.json",
                f"benchmarks/results/bitdistill_warmup_health_{date}.md",
            ],
        ),
        (
            "dependency_graph",
            [
                "python",
                "benchmarks/audit_bitdistill_dependency_graph.py",
                "--output-json",
                f"benchmark_results/bitdistill_dependency_graph_{date}.json",
                "--output-md",
                f"benchmarks/results/bitdistill_dependency_graph_{date}.md",
            ],
            [
                f"benchmark_results/bitdistill_dependency_graph_{date}.json",
                f"benchmarks/results/bitdistill_dependency_graph_{date}.md",
            ],
        ),
        (
            "job_matrix",
            [
                "python",
                "benchmarks/audit_bitdistill_job_matrix.py",
                "--monitor-json",
                f"benchmark_results/bitdistill_job_monitor_{date}.json",
                "--output-json",
                f"benchmark_results/bitdistill_job_matrix_audit_{date}.json",
                "--output-md",
                f"benchmarks/results/bitdistill_job_matrix_audit_{date}.md",
            ],
            [
                f"benchmark_results/bitdistill_job_matrix_audit_{date}.json",
                f"benchmarks/results/bitdistill_job_matrix_audit_{date}.md",
            ],
        ),
        (
            "paper_alignment",
            [
                "python",
                "benchmarks/audit_bitdistill_paper_alignment.py",
                "--output-json",
                f"benchmark_results/bitdistill_paper_alignment_{date}.json",
                "--output-md",
                f"benchmarks/results/bitdistill_paper_alignment_{date}.md",
            ],
            [
                f"benchmark_results/bitdistill_paper_alignment_{date}.json",
                f"benchmarks/results/bitdistill_paper_alignment_{date}.md",
            ],
        ),
        (
            "active_goal",
            [
                "python",
                "benchmarks/audit_bitdistill_active_goal.py",
                "--output-json",
                f"benchmark_results/bitdistill_active_goal_audit_{date}.json",
                "--output-md",
                f"benchmarks/results/bitdistill_active_goal_audit_{date}.md",
            ],
            [
                f"benchmark_results/bitdistill_active_goal_audit_{date}.json",
                f"benchmarks/results/bitdistill_active_goal_audit_{date}.md",
            ],
        ),
    ]


def run_command(command: list[str], *, date: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["BITNET_REPORT_DATE"] = date
    return subprocess.run(command, check=False, capture_output=True, text=True, env=env)


def tail(text: str, max_lines: int = 20) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, data: dict[str, Any]) -> None:
    rows = []
    for step in data["steps"]:
        rows.append(
            "| {label} | {status} | {rc} | `{outputs}` |".format(
                label=step["label"],
                status="pass" if step["returncode"] == 0 else "fail",
                rc=step["returncode"],
                outputs=", ".join(step["outputs"]),
            )
        )
    lines = [
        f"# BitDistill Warmup Finalizer, {data['date']}",
        "",
        f"Overall status: `{'pass' if data['passed'] else 'fail'}`.",
        f"Failed steps: `{len(data['failed_steps'])}`.",
        "",
        "| step | status | return code | outputs |",
        "| --- | --- | ---: | --- |",
        *rows,
    ]
    if data["failed_steps"]:
        lines.extend(["", "## Failed Step Output", ""])
        for step in data["steps"]:
            if step["returncode"] == 0:
                continue
            lines.extend(
                [
                    f"### {step['label']}",
                    "",
                    "```text",
                    step["stderr_tail"] or step["stdout_tail"],
                    "```",
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--date", default=DATE)
    parser.add_argument("--fail-on-step-error", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_warmup_finalizer_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_warmup_finalizer_{DATE}.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    steps: list[dict[str, Any]] = []
    for label, command, outputs in command_specs(args.date):
        proc = run_command(command, date=args.date)
        steps.append(
            {
                "label": label,
                "command": command,
                "returncode": proc.returncode,
                "outputs": outputs,
                "stdout_tail": tail(proc.stdout),
                "stderr_tail": tail(proc.stderr),
            }
        )

    failed_steps = [step["label"] for step in steps if step["returncode"] != 0]
    result = {
        "schema": "bitdistill-warmup-finalizer-v1",
        "date": args.date,
        "passed": not failed_steps,
        "failed_steps": failed_steps,
        "steps": steps,
    }
    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    write_json(output_json, result)
    write_markdown(output_md, result)
    print(output_md.read_text(encoding="utf-8"))
    if args.fail_on_step_error and failed_steps:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
