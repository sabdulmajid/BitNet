#!/usr/bin/env python3
"""Summarize active BitDistill Slurm jobs from repo-local artifacts.

This monitor is intentionally lightweight: it reads the downstream submission
table, parses the Stage-2 warm-up log, optionally queries squeue, and reports
which downstream metrics are already materialized.  It is a status tool, not a
quality gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
STEP_RE = re.compile(r"step=(?P<step>\d+) ce=(?P<ce>[-+0-9.eE]+) lr=(?P<lr>[-+0-9.eE]+) elapsed=(?P<elapsed>[-+0-9.eE]+)s")
ENV_RE = re.compile(r"(?P<key>[A-Z_]+)=(?P<value>[^ ]+)")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def discover_job_tables(root: Path) -> list[Path]:
    return sorted((root / "benchmark_results").glob("bitdistill_longwarmup_downstream_*.tsv"))


def read_job_table(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_job_tables(paths: list[Path]) -> list[dict[str, str]]:
    rows_by_key: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for path in paths:
        for row in read_job_table(path):
            row = dict(row)
            row["job_table"] = str(path)
            key = row.get("output_dir") or row.get("job_id") or f"{path}:{len(order)}"
            if key not in rows_by_key:
                order.append(key)
            rows_by_key[key] = row
    return [rows_by_key[key] for key in order]


def parse_dependency_job_ids(rows: list[dict[str, str]]) -> list[str]:
    ids: set[str] = set()
    for row in rows:
        dependency = row.get("dependency", "")
        match = re.search(r"afterok:(\d+)", dependency)
        if match:
            ids.add(match.group(1))
    return sorted(ids)


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


def parse_env_lines(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in text.splitlines():
        for match in ENV_RE.finditer(line):
            env[match.group("key")] = match.group("value")
    return env


def parse_warmup_log(path: Path, *, root: Path, max_seq_len: int) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    text = path.read_text(encoding="utf-8", errors="replace")
    env = parse_env_lines(text)
    steps = []
    for match in STEP_RE.finditer(text):
        steps.append(
            {
                "step": int(match.group("step")),
                "ce": float(match.group("ce")),
                "lr": float(match.group("lr")),
                "elapsed_seconds": float(match.group("elapsed")),
            }
        )
    max_steps = int(env.get("MAX_STEPS", "0") or 0)
    per_device_batch_size = int(env.get("PER_DEVICE_BATCH_SIZE", "0") or 0)
    grad_accum_steps = int(env.get("GRAD_ACCUM_STEPS", "0") or 0)
    observed_max_seq_len = int(env.get("MAX_SEQ_LEN", str(max_seq_len)) or max_seq_len)
    save_every_steps = int(env.get("SAVE_EVERY_STEPS", "0") or 0)
    output_dir_value = env.get("OUTPUT_DIR", "")
    output_dir = Path(output_dir_value) if output_dir_value else None
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = root / output_dir

    def snapshot_step(snapshot: Path) -> int:
        match = re.search(r"checkpoint-(\d+)$", snapshot.name)
        return int(match.group(1)) if match else -1

    snapshot_dirs = (
        sorted(output_dir.glob("checkpoint-*"), key=snapshot_step)
        if output_dir is not None and output_dir.exists()
        else []
    )
    warnings: list[str] = []
    token_step = per_device_batch_size * grad_accum_steps * observed_max_seq_len
    latest = steps[-1] if steps else None
    first = steps[0] if steps else None
    progress = None
    eta_seconds = None
    if latest and max_steps > 0:
        progress = latest["step"] / max_steps
        if latest["step"] > 0 and latest["elapsed_seconds"] > 0:
            seconds_per_step = latest["elapsed_seconds"] / latest["step"]
            eta_seconds = max(max_steps - latest["step"], 0) * seconds_per_step
    if latest and max_steps > 0 and latest["step"] < max_steps and save_every_steps <= 0:
        warnings.append(
            "Stage-2 warm-up is running without intermediate snapshots; if the job fails, progress before final save is not recoverable."
        )
    if latest and save_every_steps > 0 and not snapshot_dirs and latest["step"] >= save_every_steps:
        warnings.append(
            f"No checkpoint snapshots found even though SAVE_EVERY_STEPS={save_every_steps} and latest step is {latest['step']}."
        )
    return {
        "exists": True,
        "path": str(path),
        "env": env,
        "first_step": first,
        "latest_step": latest,
        "max_steps": max_steps,
        "max_seq_len": observed_max_seq_len,
        "progress": progress,
        "eta_seconds": eta_seconds,
        "tokens_per_step": token_step,
        "effective_token_presentations": (latest["step"] * token_step) if latest else None,
        "target_token_presentations": (max_steps * token_step) if max_steps and token_step else None,
        "step_observations": len(steps),
        "output_dir": str(output_dir) if output_dir is not None else "",
        "save_every_steps": save_every_steps,
        "snapshot_count": len(snapshot_dirs),
        "latest_snapshot": str(snapshot_dirs[-1]) if snapshot_dirs else "",
        "warnings": warnings,
    }


def infer_log_path(root: Path, warmup_job_ids: list[str], override: Path | None) -> Path:
    if override is not None:
        return override if override.is_absolute() else root / override
    if warmup_job_ids:
        return root / "logs" / f"bitdistill-glue-{warmup_job_ids[0]}.out"
    return root / "logs" / "bitdistill-glue-unknown.out"


def downstream_status(root: Path, rows: list[dict[str, str]], squeue: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        output_dir = root / row.get("output_dir", "")
        metrics_path = output_dir / "metrics.json"
        metrics = read_json(metrics_path)
        eval_metrics = metrics.get("eval", {}) if isinstance(metrics.get("eval"), dict) else {}
        job_id = row.get("job_id", "")
        result.append(
            {
                **row,
                "job_status": squeue.get(job_id, {"state": "unknown"}),
                "metrics_path": str(metrics_path),
                "metrics_exists": metrics_path.exists(),
                "accuracy": eval_metrics.get("accuracy"),
                "eval_examples": eval_metrics.get("eval_examples"),
                "steps": metrics.get("steps"),
            }
        )
    return result


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def display_path(value: Any, root: Path) -> str:
    if not value:
        return "-"
    path = Path(str(value))
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def seconds_to_hours(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value) / 3600.0:.2f}h"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    root = Path(summary.get("repo_root", ".")).resolve()
    warmup = summary["warmup"]
    latest = warmup.get("latest_step") or {}
    warmup_rows = [
        [
            display_path(warmup.get("path"), root),
            fmt(latest.get("step")),
            fmt(warmup.get("max_steps")),
            fmt(warmup.get("progress")),
            fmt(latest.get("ce")),
            fmt(warmup.get("effective_token_presentations")),
            fmt(warmup.get("target_token_presentations")),
            fmt(warmup.get("save_every_steps")),
            fmt(warmup.get("snapshot_count")),
            display_path(warmup.get("latest_snapshot"), root),
            seconds_to_hours(warmup.get("eta_seconds")),
        ]
    ]
    downstream_rows = [
        [
            row.get("job_id", "-"),
            row.get("task", "-"),
            row.get("task_format", "-"),
            row.get("scale", "-"),
            row.get("layer", "-"),
            row.get("task_max_steps", "-"),
            row.get("logit_kd_weight", "-"),
            row.get("attention_kd_weight", "-"),
            row.get("logit_kd_temperature_scale", "-"),
            row.get("exclude_linear_regex", "-"),
            row.get("job_status", {}).get("state", "-"),
            row.get("job_status", {}).get("elapsed", "-"),
            row.get("job_status", {}).get("node_or_reason", "-"),
            fmt(row.get("metrics_exists")),
            fmt(row.get("accuracy")),
            display_path(row.get("metrics_path"), root),
        ]
        for row in summary["downstream"]
    ]
    warning_rows = [[warning] for warning in warmup.get("warnings", [])] or [["none"]]
    return "\n\n".join(
        [
            f"# BitDistill Job Monitor, {summary['date']}",
            f"Job tables: `{', '.join(display_path(table, root) for table in summary['job_tables'])}`.",
            "## Stage-2 Warm-Up",
            md_table(
                [
                    "log",
                    "step",
                    "max steps",
                    "progress",
                    "latest CE",
                    "effective tokens",
                    "target tokens",
                    "save every",
                    "snapshots",
                    "latest snapshot",
                    "ETA",
                ],
                warmup_rows,
            ),
            "## Operational Warnings",
            md_table(["warning"], warning_rows),
            "## Downstream Jobs",
            md_table(
                [
                    "job",
                    "task",
                    "format",
                    "scale",
                    "layer",
                    "steps",
                    "logit KD",
                    "attention KD",
                    "logit temp scale",
                    "excluded linears",
                    "state",
                    "elapsed",
                    "node/reason",
                    "metrics",
                    "accuracy",
                    "metrics path",
                ],
                downstream_rows,
            ),
        ]
    ) + "\n"


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    if args.job_table is not None:
        requested_tables = args.job_table if isinstance(args.job_table, list) else [args.job_table]
        tables = [table if table.is_absolute() else root / table for table in requested_tables]
    else:
        tables = discover_job_tables(root)
    rows = read_job_tables(tables)
    warmup_job_ids = parse_dependency_job_ids(rows)
    all_job_ids = sorted(set(warmup_job_ids + [row.get("job_id", "") for row in rows if row.get("job_id")]))
    squeue = collect_squeue(all_job_ids)
    warmup_log = infer_log_path(root, warmup_job_ids, args.warmup_log)
    warmup = parse_warmup_log(warmup_log, root=root, max_seq_len=args.max_seq_len)
    return {
        "schema": "bitdistill-job-monitor-v1",
        "date": DATE,
        "repo_root": str(root),
        "job_table": str(tables[-1]) if tables else "",
        "job_tables": [str(table) for table in tables],
        "warmup_job_ids": warmup_job_ids,
        "squeue": squeue,
        "warmup": warmup,
        "downstream": downstream_status(root, rows, squeue),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--job-table", type=Path, nargs="+")
    parser.add_argument("--warmup-log", type=Path)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_job_monitor_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_job_monitor_{DATE}.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    summary = build_summary(args)
    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
