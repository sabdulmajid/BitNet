#!/usr/bin/env python3
"""Gate BitDistill GLUE CPU benchmark artifacts.

The benchmark itself is intentionally a PyTorch CPU task-runtime probe, not a
packed llama.cpp/I2_SR inference claim.  This gate prevents a missing, failed,
or timed-out row from being mistaken for runtime evidence.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = datetime.now(timezone.utc).date().isoformat()
TASKS = ["mnli", "qnli", "sst2"]
CRITICAL_RUNS = [
    ("short", "fp16_sft-tensor-layer-1"),
    ("short", "bitnet_sft-tensor-layer-1"),
    ("short", "bitdistill-tensor-layer-1"),
    ("short", "bitdistill-row-layer-1"),
    ("longwarmup", "bitdistill-longwarmup-tensor-layer-8"),
    ("longwarmup", "bitdistill-longwarmup-row-layer-8"),
    ("papergamma", "bitdistill-longwarmup-tensor-layer-8"),
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row.get("task", "")), str(row.get("family", "")), str(row.get("run", ""))


def row_complete(row: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if row.get("status") != "complete":
        blockers.append(f"status={row.get('status')}")
    for field in ["accuracy", "examples_per_second", "rss_after_load_mib", "maxrss_mib"]:
        if not isinstance(row.get(field), (int, float)):
            blockers.append(f"missing {field}")
    if not isinstance(row.get("eval_examples"), int) or int(row.get("eval_examples", 0)) <= 0:
        blockers.append("missing eval_examples")
    return not blockers, blockers


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    data = read_json(args.input_json)
    if not data:
        critical = [
            {
                "task": task,
                "family": family,
                "run": run,
                "present": False,
                "complete": False,
                "blockers": [f"missing input artifact {args.input_json}"],
            }
            for task in args.tasks
            for family, run in CRITICAL_RUNS
        ]
        return {
            "schema": "bitdistill-glue-cpu-benchmark-gate-v1",
            "date": DATE,
            "input_json": str(args.input_json),
            "input_exists": False,
            "passed": False,
            "critical": critical,
            "rows": [],
            "blockers": [f"missing input artifact {args.input_json}"],
        }

    rows = data.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    by_key = {row_key(row): row for row in rows if isinstance(row, dict)}
    critical: list[dict[str, Any]] = []
    blockers: list[str] = []
    for task in args.tasks:
        for family, run in CRITICAL_RUNS:
            key = (task, family, run)
            row = by_key.get(key)
            if row is None:
                message = f"missing {task}:{family}:{run}"
                blockers.append(message)
                critical.append(
                    {
                        "task": task,
                        "family": family,
                        "run": run,
                        "present": False,
                        "complete": False,
                        "blockers": [message],
                    }
                )
                continue
            complete, row_blockers = row_complete(row)
            if not complete:
                blockers.append(f"{task}:{family}:{run} " + ", ".join(row_blockers))
            critical.append(
                {
                    "task": task,
                    "family": family,
                    "run": run,
                    "present": True,
                    "complete": complete,
                    "blockers": row_blockers,
                    "accuracy": row.get("accuracy"),
                    "examples_per_second": row.get("examples_per_second"),
                    "rss_after_load_mib": row.get("rss_after_load_mib"),
                    "maxrss_mib": row.get("maxrss_mib"),
                    "eval_examples": row.get("eval_examples"),
                    "status": row.get("status"),
                }
            )

    return {
        "schema": "bitdistill-glue-cpu-benchmark-gate-v1",
        "date": DATE,
        "input_json": str(args.input_json),
        "input_exists": args.input_json.exists(),
        "passed": not blockers,
        "critical": critical,
        "rows": rows,
        "blockers": blockers,
        "benchmark_note": data.get("note"),
        "threads": data.get("threads"),
        "batch_size": data.get("batch_size"),
        "max_eval_samples": data.get("max_eval_samples"),
        "child_timeout_seconds": data.get("child_timeout_seconds"),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    critical_rows = [
        [
            row.get("task", "-"),
            row.get("family", "-"),
            row.get("run", "-"),
            fmt(row.get("present")),
            fmt(row.get("complete")),
            fmt(row.get("status")),
            fmt(row.get("accuracy")),
            fmt(row.get("examples_per_second")),
            fmt(row.get("rss_after_load_mib")),
            fmt(row.get("maxrss_mib")),
            "; ".join(row.get("blockers", [])),
        ]
        for row in summary["critical"]
    ]
    sections = [
        f"# BitDistill GLUE CPU Benchmark Gate, {summary['date']}",
        f"Input: `{summary['input_json']}`.",
        f"Passed: `{fmt(summary['passed'])}`.",
        (
            f"Threads: `{fmt(summary.get('threads'))}`. Batch size: `{fmt(summary.get('batch_size'))}`. "
            f"Max eval samples: `{fmt(summary.get('max_eval_samples'))}`. "
            f"Child timeout seconds: `{fmt(summary.get('child_timeout_seconds'))}`."
        ),
        "This gate validates PyTorch CPU task-runtime rows only; it is not a packed llama.cpp/I2_SR runtime gate.",
        "## Critical Rows",
        md_table(
            [
                "task",
                "family",
                "run",
                "present",
                "complete",
                "status",
                "accuracy",
                "examples/s",
                "RSS load MiB",
                "max RSS MiB",
                "blockers",
            ],
            critical_rows,
        ),
    ]
    if summary.get("blockers"):
        sections.extend(["## Blockers", "\n".join(f"- {item}" for item in summary["blockers"])])
    return "\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=Path(f"benchmark_results/bitdistill_glue_cpu_{DATE}.json"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_glue_cpu_gate_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_glue_cpu_gate_{DATE}.md"))
    args = parser.parse_args()

    summary = build_summary(args)
    write_json(args.output_json, summary)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(summary)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
