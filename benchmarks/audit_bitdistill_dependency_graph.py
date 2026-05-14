#!/usr/bin/env python3
"""Audit the queued BitDistill Slurm dependency graph and artifact paths.

This is an operational preflight, not a quality gate.  Its job is to catch
miswired downstream jobs before a long Stage-2 warm-up releases them.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = datetime.now(timezone.utc).date().isoformat()
ENV_RE = re.compile(r"(?P<key>[A-Z_]+)=(?P<value>[^ ]+)")
STEP_RE = re.compile(r"step=(?P<step>\d+) ce=(?P<ce>[-+0-9.eE]+) lr=(?P<lr>[-+0-9.eE]+) elapsed=(?P<elapsed>[-+0-9.eE]+)s")
DEPENDENCY_RE = re.compile(r"after[a-z]*:(\d+)")


def parse_env_lines(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in text.splitlines():
        for match in ENV_RE.finditer(line):
            env[match.group("key")] = match.group("value")
    return env


def parse_warmup_log(path: Path, root: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    text = path.read_text(encoding="utf-8", errors="replace")
    env = parse_env_lines(text)
    steps = [
        {
            "step": int(match.group("step")),
            "ce": float(match.group("ce")),
            "lr": float(match.group("lr")),
            "elapsed_seconds": float(match.group("elapsed")),
        }
        for match in STEP_RE.finditer(text)
    ]
    output_dir = env.get("OUTPUT_DIR", "")
    output_path = Path(output_dir) if output_dir else None
    if output_path is not None and not output_path.is_absolute():
        output_path = root / output_path
    max_steps = int(env.get("MAX_STEPS", "0") or 0)
    latest = steps[-1] if steps else None
    return {
        "exists": True,
        "path": str(path),
        "env": env,
        "output_dir": str(output_path) if output_path is not None else "",
        "expected_state": str(output_path / "custom_state_dict.pt") if output_path is not None else "",
        "max_steps": max_steps,
        "latest_step": latest,
        "progress": (latest["step"] / max_steps) if latest and max_steps > 0 else None,
    }


def discover_tables(root: Path) -> list[Path]:
    return sorted((root / "benchmark_results").glob("bitdistill_longwarmup_downstream_*.tsv"))


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle, delimiter="\t"):
            item = {key: value for key, value in row.items() if key is not None}
            item["job_table"] = str(path)
            rows.append(item)
        return rows


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(read_table(path))
    return rows


def dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_output: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for row in rows:
        key = row.get("output_dir") or row.get("job_id") or f"row-{len(order)}"
        if key not in by_output:
            order.append(key)
        by_output[key] = row
    return [by_output[key] for key in order]


def collect_squeue(job_ids: list[str]) -> dict[str, dict[str, str]]:
    if not job_ids:
        return {}
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i\t%T\t%M\t%l\t%R"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {job_id: {"state": "squeue_unavailable"} for job_id in job_ids}
    statuses = {job_id: {"state": "not_in_squeue"} for job_id in job_ids}
    if result.returncode != 0:
        return {job_id: {"state": "squeue_error", "stderr": result.stderr.strip()} for job_id in job_ids}
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        job_id, state, elapsed, limit, reason = parts[:5]
        statuses[job_id] = {
            "state": state,
            "elapsed": elapsed,
            "time_limit": limit,
            "node_or_reason": reason,
        }
    return statuses


def active_rows(rows: list[dict[str, str]], squeue: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    active_states = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}
    result = []
    for row in rows:
        status = squeue.get(row.get("job_id", ""), {})
        if status.get("state") in active_states:
            result.append(row)
    return result


def rel(path: str | Path, root: Path) -> str:
    value = Path(path)
    try:
        return str(value.relative_to(root))
    except ValueError:
        return str(value)


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    warmup_log = args.warmup_log if args.warmup_log.is_absolute() else root / args.warmup_log
    warmup = parse_warmup_log(warmup_log, root)
    expected_state = warmup.get("expected_state", "")

    table_paths = discover_tables(root)
    raw_rows = read_rows(table_paths)
    latest_rows = dedupe_rows(raw_rows)
    job_ids = sorted({row.get("job_id", "") for row in latest_rows if row.get("job_id")})
    squeue = collect_squeue(job_ids)
    live_rows = active_rows(latest_rows, squeue)

    output_counts = Counter(row.get("output_dir", "") for row in raw_rows if row.get("output_dir"))
    duplicate_outputs = sorted([output for output, count in output_counts.items() if count > 1])
    superseded_by_output: dict[str, list[str]] = defaultdict(list)
    latest_job_by_output = {row.get("output_dir", ""): row.get("job_id", "") for row in latest_rows}
    for row in raw_rows:
        output = row.get("output_dir", "")
        job_id = row.get("job_id", "")
        if output and job_id and latest_job_by_output.get(output) != job_id:
            superseded_by_output[output].append(job_id)

    blockers: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, evidence: str, blocker: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence, "blocker": "" if passed else blocker})
        if not passed and blocker:
            blockers.append(blocker)

    add_check(
        "warm-up log exposes final output directory",
        bool(expected_state),
        f"log={rel(warmup_log, root)}, expected_state={rel(expected_state, root) if expected_state else '-'}",
        "The warm-up log does not expose OUTPUT_DIR, so downstream INIT_STATE_DICT paths cannot be verified.",
    )

    live_warmup_mismatches = [
        row
        for row in live_rows
        if expected_state and str((root / row.get("warmup_state", "")).resolve()) != str(Path(expected_state).resolve())
    ]
    add_check(
        "active downstream jobs point at warm-up final state",
        not live_warmup_mismatches,
        f"active_rows={len(live_rows)}, mismatches={len(live_warmup_mismatches)}",
        "At least one active downstream job points at a different warm-up state than the running Stage-2 job will write.",
    )

    missing_teachers = []
    for row in live_rows:
        teacher = row.get("teacher", "")
        if not teacher:
            missing_teachers.append({"job_id": row.get("job_id", ""), "reason": "empty_teacher"})
            continue
        teacher_path = root / teacher if not Path(teacher).is_absolute() else Path(teacher)
        if not (teacher_path / "metrics.json").exists():
            missing_teachers.append({"job_id": row.get("job_id", ""), "teacher": str(teacher_path), "reason": "missing_metrics"})
    add_check(
        "active downstream jobs have FP16 teacher metrics",
        not missing_teachers,
        f"active_rows={len(live_rows)}, missing={len(missing_teachers)}",
        "At least one active downstream job lacks an FP16 teacher metrics.json.",
    )

    bad_dependencies = []
    warmup_job_id = warmup.get("env", {}).get("SLURM_JOB_ID", "")
    for row in live_rows:
        dependency = row.get("dependency", "")
        deps = DEPENDENCY_RE.findall(dependency)
        if warmup_job_id and warmup_job_id not in deps:
            bad_dependencies.append({"job_id": row.get("job_id", ""), "dependency": dependency})
    add_check(
        "active downstream jobs depend on the running warm-up job",
        not bad_dependencies,
        f"warmup_job={warmup_job_id or '-'}, bad={len(bad_dependencies)}",
        "At least one active downstream job does not depend on the running Stage-2 warm-up job.",
    )

    if duplicate_outputs:
        warnings.append(
            f"{len(duplicate_outputs)} output directories appear in multiple historical submission tables; audit uses the latest row per output directory."
        )
    if live_rows and warmup.get("latest_step") and warmup.get("max_steps"):
        latest_step = int(warmup["latest_step"]["step"])
        if latest_step < int(warmup["max_steps"]) and not Path(expected_state).exists():
            warnings.append("Warm-up final state is expectedly absent until Stage-2 finishes; downstream jobs are correctly dependency-blocked.")

    row_summaries = []
    for row in latest_rows:
        job_id = row.get("job_id", "")
        row_summaries.append(
            {
                "job_id": job_id,
                "state": squeue.get(job_id, {"state": "not_in_squeue"}).get("state"),
                "task": row.get("task"),
                "task_format": row.get("task_format", ""),
                "scale": row.get("scale"),
                "layer": row.get("layer"),
                "dependency": row.get("dependency"),
                "teacher_metrics_exists": bool(row.get("teacher")) and (root / row.get("teacher", "") / "metrics.json").exists(),
                "warmup_state_matches": bool(expected_state)
                and str((root / row.get("warmup_state", "")).resolve()) == str(Path(expected_state).resolve()),
                "output_dir": row.get("output_dir"),
            }
        )

    return {
        "schema": "bitdistill-dependency-graph-audit-v1",
        "date": DATE,
        "repo_root": str(root),
        "ready": not blockers,
        "checks": checks,
        "blockers": blockers,
        "warnings": warnings,
        "warmup": warmup,
        "job_tables": [str(path) for path in table_paths],
        "raw_rows": len(raw_rows),
        "deduped_rows": len(latest_rows),
        "active_rows": len(live_rows),
        "duplicate_output_dirs": duplicate_outputs,
        "superseded_jobs_by_output": dict(superseded_by_output),
        "rows": row_summaries,
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
    root = Path(summary["repo_root"])
    warmup = summary["warmup"]
    latest = warmup.get("latest_step") or {}
    check_rows = [
        [
            check["name"],
            "pass" if check["passed"] else "fail",
            str(check["evidence"]),
            str(check.get("blocker", "")),
        ]
        for check in summary["checks"]
    ]
    row_rows = [
        [
            str(row["job_id"]),
            str(row["state"]),
            str(row["task"]),
            str(row["task_format"] or "-"),
            str(row["scale"]),
            str(row["layer"]),
            fmt(row["teacher_metrics_exists"]),
            fmt(row["warmup_state_matches"]),
            str(row["output_dir"]),
        ]
        for row in summary["rows"]
    ]
    duplicate_rows = [
        [rel(output, root), ", ".join(summary["superseded_jobs_by_output"].get(output, []))]
        for output in summary["duplicate_output_dirs"]
    ]
    warning_rows = [[warning] for warning in summary["warnings"]] or [["none"]]
    blocker_rows = [[blocker] for blocker in summary["blockers"]] or [["none"]]
    return "\n\n".join(
        [
            f"# BitDistill Dependency Graph Audit, {summary['date']}",
            f"Ready for downstream release: `{summary['ready']}`.",
            "## Warm-Up",
            md_table(
                ["log", "job", "step", "max steps", "progress", "expected state exists", "expected state"],
                [
                    [
                        rel(warmup.get("path", ""), root),
                        warmup.get("env", {}).get("SLURM_JOB_ID", "-"),
                        fmt(latest.get("step")),
                        fmt(warmup.get("max_steps")),
                        fmt(warmup.get("progress")),
                        fmt(Path(warmup.get("expected_state", "")).exists() if warmup.get("expected_state") else False),
                        rel(warmup.get("expected_state", ""), root) if warmup.get("expected_state") else "-",
                    ]
                ],
            ),
            "## Checks",
            md_table(["check", "status", "evidence", "blocker"], check_rows),
            "## Warnings",
            md_table(["warning"], warning_rows),
            "## Blockers",
            md_table(["blocker"], blocker_rows),
            "## Submission Rows",
            f"Raw rows: `{summary['raw_rows']}`. Deduped rows: `{summary['deduped_rows']}`. Active rows: `{summary['active_rows']}`.",
            md_table(
                ["job", "state", "task", "format", "scale", "layer", "teacher metrics", "warmup match", "output dir"],
                row_rows,
            ),
            "## Duplicate Historical Output Dirs",
            md_table(["output dir", "superseded job ids"], duplicate_rows or [["none", "-"]]),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--warmup-log", type=Path, default=Path("logs/bitdistill-glue-9894.out"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_dependency_graph_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_dependency_graph_{DATE}.md"))
    args = parser.parse_args()

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
