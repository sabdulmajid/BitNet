#!/usr/bin/env python3
"""Audit BitDistill Stage-2 warm-up health from the live Slurm log."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = datetime.now(timezone.utc).date().isoformat()
STEP_RE = re.compile(
    r"step=(?P<step>\d+) ce=(?P<ce>[-+0-9.eE]+) lr=(?P<lr>[-+0-9.eE]+) elapsed=(?P<elapsed>[-+0-9.eE]+)s"
)
ENV_RE = re.compile(r"(?P<key>[A-Z_]+)=(?P<value>[^ ]+)")
DEFAULT_LOG_PATH = Path("logs/bitdistill-glue-9894.out")


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def parse_env(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in text.splitlines():
        for match in ENV_RE.finditer(line):
            env[match.group("key")] = match.group("value")
    return env


def parse_steps(text: str) -> list[dict[str, Any]]:
    rows = []
    for match in STEP_RE.finditer(text):
        rows.append(
            {
                "step": int(match.group("step")),
                "ce": float(match.group("ce")),
                "lr": float(match.group("lr")),
                "elapsed_seconds": float(match.group("elapsed")),
            }
        )
    return rows


def load_monitor_warmup(path: Path, root: Path) -> tuple[dict[str, Any], str]:
    monitor_path = path if path.is_absolute() else root / path
    if not monitor_path.exists():
        return {}, f"monitor JSON missing: {rel(monitor_path, root)}"
    try:
        data = json.loads(monitor_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, f"monitor JSON parse error: {rel(monitor_path, root)}: {exc}"
    warmup = data.get("warmup")
    if not isinstance(warmup, dict):
        return {}, f"monitor JSON has no warmup object: {rel(monitor_path, root)}"
    return warmup, ""


def resolve_log_path(args: argparse.Namespace, root: Path) -> tuple[Path, dict[str, Any], str]:
    monitor_warmup, monitor_warning = load_monitor_warmup(args.monitor_json, root)
    if args.log_path is not None:
        path = args.log_path
    else:
        monitor_path = monitor_warmup.get("path") if monitor_warmup else ""
        path = Path(monitor_path) if monitor_path else DEFAULT_LOG_PATH
    if not path.is_absolute():
        path = root / path
    return path, monitor_warmup, monitor_warning


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def collect_squeue(job_id: str) -> dict[str, str]:
    if not job_id:
        return {}
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%i\t%T\t%M\t%l\t%R"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {"state": "squeue_unavailable"}
    if result.returncode != 0:
        return {"state": "squeue_error", "stderr": result.stderr.strip()}
    line = result.stdout.strip()
    if not line:
        return {"state": "not_in_squeue"}
    parts = line.split("\t")
    if len(parts) < 5:
        return {"state": "squeue_parse_error", "raw": line}
    return {
        "job_id": parts[0],
        "state": parts[1],
        "elapsed": parts[2],
        "time_limit": parts[3],
        "node_or_reason": parts[4],
    }


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, evidence: str, blocker: str = "") -> None:
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence, "blocker": "" if passed else blocker})


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    log_path, monitor_warmup, monitor_warning = resolve_log_path(args, root)
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    if monitor_warning:
        warnings.append(monitor_warning)
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    env = parse_env(text)
    rows = parse_steps(text)
    latest = rows[-1] if rows else None
    first = rows[0] if rows else None
    max_steps = int(env.get("MAX_STEPS", "0") or 0)
    save_every_steps = int(env.get("SAVE_EVERY_STEPS", "0") or 0)
    output_dir_text = env.get("OUTPUT_DIR", "")
    output_dir = Path(output_dir_text) if output_dir_text else None
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = root / output_dir
    final_state = output_dir / "custom_state_dict.pt" if output_dir is not None else None
    snapshots = sorted(output_dir.glob("checkpoint-*")) if output_dir is not None and output_dir.exists() else []
    now = time.time()
    age_seconds = now - log_path.stat().st_mtime if log_path.exists() else None
    job_id = env.get("SLURM_JOB_ID", "")
    squeue = collect_squeue(job_id)
    job_running = squeue.get("state") in {"RUNNING", "CONFIGURING", "COMPLETING"}
    final_exists = final_state.exists() if final_state is not None else False
    monitor_env = monitor_warmup.get("env") if isinstance(monitor_warmup.get("env"), dict) else {}
    monitor_job_id = str(monitor_env.get("SLURM_JOB_ID") or "")
    monitor_log_path = monitor_warmup.get("path") if isinstance(monitor_warmup.get("path"), str) else ""

    steps = [row["step"] for row in rows]
    ces = [row["ce"] for row in rows]
    nonfinite_ce = [value for value in ces if not math.isfinite(value)]
    monotonic = all(a < b for a, b in zip(steps, steps[1:]))
    latest_step = latest["step"] if latest else None
    progress = (latest_step / max_steps) if latest_step is not None and max_steps > 0 else None
    seconds_per_step = None
    eta_seconds = None
    if latest and latest["step"] > 0 and latest["elapsed_seconds"] > 0:
        seconds_per_step = latest["elapsed_seconds"] / latest["step"]
        if max_steps > 0:
            eta_seconds = max(max_steps - latest["step"], 0) * seconds_per_step
    first_window = ces[: args.window]
    last_window = ces[-args.window :]
    first_stats = stats(first_window)
    last_stats = stats(last_window)
    ce_delta = None
    if finite_number(first_stats["mean"]) and finite_number(last_stats["mean"]):
        ce_delta = float(last_stats["mean"]) - float(first_stats["mean"])

    add_check(
        checks,
        "warm-up log exists",
        log_path.exists(),
        rel(log_path, root),
        "The warm-up log is missing.",
    )
    add_check(
        checks,
        "warm-up has enough observations",
        len(rows) >= args.min_observations,
        f"observations={len(rows)}, required={args.min_observations}",
        "Too few step observations to judge warm-up health.",
    )
    add_check(
        checks,
        "step numbers are strictly increasing",
        monotonic,
        f"first={first['step'] if first else '-'}, latest={latest_step}, observations={len(rows)}",
        "Step observations are missing or non-monotonic.",
    )
    add_check(
        checks,
        "CE values are finite",
        bool(rows) and not nonfinite_ce,
        f"nonfinite={len(nonfinite_ce)}, latest_ce={latest['ce'] if latest else '-'}",
        "At least one CE value is NaN/Inf or no CE values were parsed.",
    )
    add_check(
        checks,
        "latest progress is within target",
        latest_step is not None and max_steps > 0 and 0 < latest_step <= max_steps,
        f"latest={latest_step}, max_steps={max_steps}, progress={progress}",
        "Latest step is missing or outside the configured target.",
    )
    fresh = age_seconds is not None and (age_seconds <= args.max_log_age_seconds or final_exists or not job_running)
    add_check(
        checks,
        "log is fresh while job is active",
        fresh,
        f"age_seconds={age_seconds:.1f}" if age_seconds is not None else "age_seconds=-",
        "The Slurm log is stale while the job appears active.",
    )
    add_check(
        checks,
        "ETA is finite",
        finite_number(eta_seconds),
        f"seconds_per_step={seconds_per_step}, eta_seconds={eta_seconds}",
        "ETA could not be computed from elapsed time and step count.",
    )
    if monitor_warmup:
        monitor_agrees = (not monitor_job_id or monitor_job_id == job_id) and (
            not monitor_log_path or Path(monitor_log_path).resolve() == log_path.resolve()
        )
        add_check(
            checks,
            "monitor identifies same warm-up job",
            monitor_agrees,
            f"monitor_job={monitor_job_id or '-'}, parsed_job={job_id or '-'}",
            "The monitor JSON points at a different warm-up job or log path.",
        )

    if save_every_steps <= 0 and not final_exists:
        warnings.append("SAVE_EVERY_STEPS is 0 and no final state exists yet; a job failure would lose current warm-up progress.")
    if finite_number(ce_delta) and ce_delta > args.max_ce_mean_increase:
        warnings.append(
            f"Last-window CE mean is {ce_delta:.4f} higher than the first-window mean; inspect for data or optimization drift."
        )
    if snapshots and save_every_steps <= 0:
        warnings.append("Snapshot directories exist even though SAVE_EVERY_STEPS is 0; verify they belong to the current run.")

    failed = [check for check in checks if not check["passed"]]
    return {
        "schema": "bitdistill-warmup-health-v1",
        "date": DATE,
        "passed": not failed,
        "warning_count": len(warnings),
        "checks": checks,
        "failed": failed,
        "warnings": warnings,
        "log_path": rel(log_path, root),
        "monitor_json": rel(args.monitor_json if args.monitor_json.is_absolute() else root / args.monitor_json, root),
        "monitor_warmup_path": monitor_log_path,
        "job_id": job_id,
        "squeue": squeue,
        "env": env,
        "max_steps": max_steps,
        "latest_step": latest,
        "first_step": first,
        "progress": progress,
        "observations": len(rows),
        "log_age_seconds": age_seconds,
        "seconds_per_step": seconds_per_step,
        "eta_seconds": eta_seconds,
        "first_window": {"count": len(first_window), **first_stats},
        "last_window": {"count": len(last_window), **last_stats},
        "last_minus_first_ce_mean": ce_delta,
        "output_dir": rel(output_dir, root) if output_dir is not None else "",
        "final_state": rel(final_state, root) if final_state is not None else "",
        "final_state_exists": final_exists,
        "save_every_steps": save_every_steps,
        "snapshot_count": len(snapshots),
        "latest_snapshot": rel(snapshots[-1], root) if snapshots else "",
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def hours(value: Any) -> str:
    return "-" if not finite_number(value) else f"{float(value) / 3600.0:.2f}h"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    latest = summary.get("latest_step") or {}
    squeue = summary.get("squeue") or {}
    overview = [
        [
            summary.get("log_path", "-"),
            summary.get("job_id", "-"),
            squeue.get("state", "-"),
            fmt(latest.get("step")),
            fmt(summary.get("max_steps")),
            fmt(summary.get("progress")),
            fmt(latest.get("ce")),
            fmt(summary.get("last_window", {}).get("mean")),
            fmt(summary.get("last_minus_first_ce_mean")),
            fmt(summary.get("seconds_per_step")),
            hours(summary.get("eta_seconds")),
            fmt(summary.get("final_state_exists")),
            fmt(summary.get("snapshot_count")),
        ]
    ]
    check_rows = [
        [check["name"], "pass" if check["passed"] else "fail", check["evidence"], check.get("blocker", "")]
        for check in summary["checks"]
    ]
    warning_rows = [[warning] for warning in summary["warnings"]] or [["none"]]
    return "\n\n".join(
        [
            f"# BitDistill Warm-Up Health Audit, {summary['date']}",
            f"Overall status: `{'pass' if summary['passed'] else 'fail'}`.",
            "## Overview",
            md_table(
                [
                    "log",
                    "job",
                    "state",
                    "step",
                    "max steps",
                    "progress",
                    "latest CE",
                    "last CE mean",
                    "last-first CE mean",
                    "sec/step",
                    "ETA",
                    "final state",
                    "snapshots",
                ],
                overview,
            ),
            "## Checks",
            md_table(["check", "status", "evidence", "blocker"], check_rows),
            "## Warnings",
            md_table(["warning"], warning_rows),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--monitor-json", type=Path, default=Path(f"benchmark_results/bitdistill_job_monitor_{DATE}.json"))
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--min-observations", type=int, default=10)
    parser.add_argument("--max-log-age-seconds", type=float, default=1800.0)
    parser.add_argument("--max-ce-mean-increase", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_warmup_health_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_warmup_health_{DATE}.md"))
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
