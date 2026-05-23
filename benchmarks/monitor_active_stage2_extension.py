#!/usr/bin/env python3
"""Monitor active Stage-2 extension jobs without making quality claims."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STEP_RE = re.compile(
    r"step=(?P<step>\d+)\s+ce=(?P<ce>[0-9.eE+-]+)\s+lr=(?P<lr>[0-9.eE+-]+)\s+elapsed=(?P<elapsed>[0-9.eE+-]+)s"
)
HEADER_KV_RE = re.compile(r"(?P<key>[A-Z0-9_]+)=(?P<value>\S+)")
FATAL_LOG_PATTERNS = [
    ("traceback", re.compile(r"Traceback", re.IGNORECASE)),
    ("runtime_error", re.compile(r"RuntimeError", re.IGNORECASE)),
    ("cuda_oom", re.compile(r"CUDA out of memory|OutOfMemoryError", re.IGNORECASE)),
    ("exception", re.compile(r"\bException:", re.IGNORECASE)),
    ("nan_token", re.compile(r"\bnan\b", re.IGNORECASE)),
    ("inf_token", re.compile(r"\binf(?:inity|inite)?\b", re.IGNORECASE)),
    ("overflow", re.compile(r"\boverflow\b", re.IGNORECASE)),
]
RUNNING_LOG_STALE_SECONDS = 15 * 60
TIME_LIMIT_TIGHT_MARGIN_SECONDS = 30 * 60
MAX_LOG_HEALTH_EXAMPLES = 20


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def squeue_rows(job_ids: list[str]) -> dict[str, dict[str, str]]:
    if not job_ids:
        return {}
    command = ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i\t%T\t%M\t%l\t%R\t%j"]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    rows: dict[str, dict[str, str]] = {}
    if result.returncode != 0:
        return rows
    for line in result.stdout.splitlines():
        parts = line.split("\t", 5)
        if len(parts) != 6:
            continue
        job_id, state, time_used, time_limit, reason, name = parts
        rows[job_id] = {
            "job_id": job_id,
            "state": state,
            "time": time_used,
            "time_limit": time_limit,
            "reason": reason,
            "name": name,
        }
    return rows


def parse_slurm_duration(value: str | None) -> int | None:
    if not value:
        return None
    text = value.strip()
    if text in {"UNLIMITED", "NOT_SET", "N/A", "INVALID"}:
        return None
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        if not day_text.isdigit():
            return None
        days = int(day_text)
    parts = text.split(":")
    if not all(part.isdigit() for part in parts):
        return None
    if len(parts) == 3:
        hours, minutes, seconds = (int(part) for part in parts)
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = (int(part) for part in parts)
    elif len(parts) == 1:
        hours = 0
        minutes = 0
        seconds = int(parts[0])
    else:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def parse_step_rows(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(log_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        match = STEP_RE.search(line)
        if not match:
            continue
        row = {
            "log_exists": True,
            "path": str(log_path),
            "line_no": line_no,
            "step": int(match.group("step")),
            "ce": float(match.group("ce")),
            "lr": float(match.group("lr")),
            "elapsed_seconds": float(match.group("elapsed")),
        }
        rows.append(row)
    return rows


def parse_latest_step(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"log_exists": False}
    latest: dict[str, Any] = {"log_exists": True, "path": str(log_path)}
    rows = parse_step_rows(log_path)
    for row in rows:
        latest = row
    if rows:
        recent = rows[-20:]
        recent_ce = [float(row["ce"]) for row in recent]
        latest.update(
            {
                "parsed_log_rows": len(rows),
                "first_step": rows[0]["step"],
                "first_elapsed_seconds": rows[0]["elapsed_seconds"],
                "recent_window_rows": len(recent),
                "recent_ce_mean": sum(recent_ce) / len(recent_ce),
                "recent_ce_min": min(recent_ce),
                "recent_ce_max": max(recent_ce),
            }
        )
    return latest


def producer_log_health(log_path: Path, stage2_config: dict[str, Any]) -> dict[str, Any]:
    if not log_path.exists():
        return {
            "status": "missing_log",
            "path": str(log_path),
            "parsed_step_rows": 0,
            "issues": [],
            "fatal_matches": [],
            "checks": {},
            "caveat": "This checks producer log structure and fatal patterns; it is not quality evidence.",
        }
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows = parse_step_rows(log_path)
    fatal_matches: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for label, pattern in FATAL_LOG_PATTERNS:
            if pattern.search(line):
                fatal_matches.append(
                    {
                        "line_no": line_no,
                        "pattern": label,
                        "line": line[:300],
                    }
                )
                break
        if len(fatal_matches) >= MAX_LOG_HEALTH_EXAMPLES:
            break

    issues: list[dict[str, Any]] = []
    if not rows:
        return {
            "status": "no_steps",
            "path": str(log_path),
            "parsed_step_rows": 0,
            "issues": [{"type": "no_step_rows"}],
            "fatal_matches": fatal_matches,
            "checks": {
                "has_step_rows": False,
                "steps_monotonic": None,
                "elapsed_monotonic": None,
                "finite_numeric_values": None,
                "constant_lr_matches_expected": None,
                "latest_step_within_max_steps": None,
            },
            "caveat": "This checks producer log structure and fatal patterns; it is not quality evidence.",
        }

    steps_monotonic = True
    elapsed_monotonic = True
    finite_numeric_values = True
    constant_lr_matches_expected = True
    max_steps = int(stage2_config["max_steps"])
    expected_lr = float(stage2_config["learning_rate"])
    lr_scheduler = str(stage2_config.get("lr_scheduler") or "")

    previous_step: int | None = None
    previous_elapsed: float | None = None
    for row in rows:
        step = int(row["step"])
        elapsed = float(row["elapsed_seconds"])
        ce = float(row["ce"])
        lr = float(row["lr"])
        if previous_step is not None and step <= previous_step:
            steps_monotonic = False
            issues.append(
                {
                    "type": "non_monotonic_step",
                    "line_no": row["line_no"],
                    "previous": previous_step,
                    "current": step,
                }
            )
            break
        if previous_elapsed is not None and elapsed < previous_elapsed:
            elapsed_monotonic = False
            issues.append(
                {
                    "type": "non_monotonic_elapsed",
                    "line_no": row["line_no"],
                    "previous": previous_elapsed,
                    "current": elapsed,
                }
            )
            break
        if not (math.isfinite(ce) and math.isfinite(lr) and math.isfinite(elapsed)):
            finite_numeric_values = False
            issues.append(
                {
                    "type": "non_finite_numeric",
                    "line_no": row["line_no"],
                    "step": step,
                    "ce": ce,
                    "lr": lr,
                    "elapsed_seconds": elapsed,
                }
            )
            break
        if lr_scheduler == "constant" and abs(lr - expected_lr) > max(abs(expected_lr) * 1e-6, 1e-12):
            constant_lr_matches_expected = False
            issues.append(
                {
                    "type": "lr_mismatch",
                    "line_no": row["line_no"],
                    "step": step,
                    "expected_lr": expected_lr,
                    "actual_lr": lr,
                }
            )
            break
        previous_step = step
        previous_elapsed = elapsed

    latest = rows[-1]
    latest_step_within_max_steps = int(latest["step"]) <= max_steps
    if not latest_step_within_max_steps:
        issues.append(
            {
                "type": "latest_step_exceeds_max_steps",
                "latest_step": latest["step"],
                "max_steps": max_steps,
            }
        )
    status = "healthy" if not issues and not fatal_matches else "unhealthy"
    return {
        "status": status,
        "path": str(log_path),
        "parsed_step_rows": len(rows),
        "first_step": rows[0]["step"],
        "latest_step": latest["step"],
        "latest_ce": latest["ce"],
        "latest_lr": latest["lr"],
        "latest_elapsed_seconds": latest["elapsed_seconds"],
        "recent_window_rows": min(len(rows), 20),
        "recent_ce_min": min(float(row["ce"]) for row in rows[-20:]),
        "recent_ce_max": max(float(row["ce"]) for row in rows[-20:]),
        "recent_ce_mean": sum(float(row["ce"]) for row in rows[-20:]) / min(len(rows), 20),
        "issues": issues[:MAX_LOG_HEALTH_EXAMPLES],
        "fatal_matches": fatal_matches,
        "checks": {
            "has_step_rows": True,
            "steps_monotonic": steps_monotonic,
            "elapsed_monotonic": elapsed_monotonic,
            "finite_numeric_values": finite_numeric_values,
            "constant_lr_matches_expected": constant_lr_matches_expected if lr_scheduler == "constant" else None,
            "latest_step_within_max_steps": latest_step_within_max_steps,
        },
        "caveat": "This checks producer log structure and fatal patterns; it is not quality evidence.",
    }


def parse_log_header(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"exists": False, "path": str(log_path), "values": {}, "line_count": 0}
    values: dict[str, str] = {}
    line_count = 0
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if STEP_RE.search(line):
            break
        line_count += 1
        for match in HEADER_KV_RE.finditer(line):
            values[match.group("key")] = match.group("value")
    return {
        "exists": True,
        "path": str(log_path),
        "values": values,
        "line_count": line_count,
    }


def config_match(key: str, expected: Any, actual: str | None, mode: str = "string") -> dict[str, Any]:
    if actual is None:
        matched = False
    elif mode == "float":
        try:
            matched = abs(float(actual) - float(expected)) <= max(abs(float(expected)) * 1e-9, 1e-12)
        except ValueError:
            matched = False
    else:
        matched = str(actual) == str(expected)
    return {
        "key": key,
        "expected": expected,
        "actual": actual,
        "mode": mode,
        "matched": matched,
    }


def producer_config_gate(
    *,
    log_path: Path,
    stage2_submission: dict[str, Any],
    stage2_job_id: str,
) -> dict[str, Any]:
    header = parse_log_header(log_path)
    values = header.get("values", {})
    if not header["exists"]:
        status = "missing_log"
        checks: list[dict[str, Any]] = []
    elif not values:
        status = "missing_header"
        checks = []
    else:
        run_config = stage2_submission["run_config"]
        parent = stage2_submission["parent_manifest"]
        checks = [
            config_match("SLURM_JOB_ID", stage2_job_id, values.get("SLURM_JOB_ID")),
            config_match("MODEL", stage2_submission["model"], values.get("MODEL")),
            config_match("STAGE", run_config["stage"], values.get("STAGE")),
            config_match("METHOD", run_config["method"], values.get("METHOD")),
            config_match("INIT_STATE_MANIFEST", parent["path"], values.get("INIT_STATE_MANIFEST")),
            config_match("INIT_STATE_DICT", parent["state_dict_path"], values.get("INIT_STATE_DICT")),
            config_match("SCALE_MODE", run_config["scale_mode"], values.get("SCALE_MODE")),
            config_match(
                "ACTIVATION_QUANTIZATION",
                "1" if run_config["activation_quantization"] else "0",
                values.get("ACTIVATION_QUANTIZATION"),
            ),
            config_match("USE_SUBLN", "1" if run_config["use_subln"] else "0", values.get("USE_SUBLN")),
            config_match("MAX_SEQ_LEN", run_config["max_seq_len"], values.get("MAX_SEQ_LEN")),
            config_match("MAX_STEPS", run_config["max_steps"], values.get("MAX_STEPS")),
            config_match(
                "PER_DEVICE_BATCH_SIZE",
                run_config["per_device_batch_size"],
                values.get("PER_DEVICE_BATCH_SIZE"),
            ),
            config_match("GRAD_ACCUM_STEPS", run_config["grad_accum_steps"], values.get("GRAD_ACCUM_STEPS")),
            config_match("LR", run_config["learning_rate"], values.get("LR"), mode="float"),
            config_match("LR_SCHEDULER", run_config["lr_scheduler"], values.get("LR_SCHEDULER")),
            config_match("SAVE_EVERY_STEPS", run_config["save_every_steps"], values.get("SAVE_EVERY_STEPS")),
            config_match(
                "SAVE_MODEL_ARTIFACTS",
                "1" if run_config["save_model_artifacts"] else "0",
                values.get("SAVE_MODEL_ARTIFACTS"),
            ),
            config_match("OUTPUT_DIR", run_config["output_dir"], values.get("OUTPUT_DIR")),
        ]
        status = "matched" if all(check["matched"] for check in checks) else "mismatched"
    mismatches = [check for check in checks if not check["matched"]]
    return {
        "status": status,
        "log_header": header,
        "checks": checks,
        "mismatches": mismatches,
        "caveat": "This validates the producer log header against the submitted Stage-2 configuration.",
    }


def log_freshness(log_path: Path, squeue_row: dict[str, str] | None) -> dict[str, Any]:
    state = (squeue_row or {}).get("state", "")
    is_running = state.lower() == "running"
    now = datetime.now(timezone.utc)
    if not log_path.exists():
        status = "missing_log_running" if is_running else "missing_log_not_running"
        return {
            "status": status,
            "path": str(log_path),
            "exists": False,
            "checked_utc": now.isoformat(),
            "mtime_utc": None,
            "age_seconds": None,
            "stale_after_seconds": RUNNING_LOG_STALE_SECONDS,
            "slurm_state": state,
            "caveat": "Fresh logs are required while the Stage-2 producer is running.",
        }
    mtime = datetime.fromtimestamp(log_path.stat().st_mtime, tz=timezone.utc)
    age_seconds = (now - mtime).total_seconds()
    if not is_running:
        status = "not_running"
    elif age_seconds > RUNNING_LOG_STALE_SECONDS:
        status = "stale_running_log"
    else:
        status = "fresh_running_log"
    return {
        "status": status,
        "path": str(log_path),
        "exists": True,
        "checked_utc": now.isoformat(),
        "mtime_utc": mtime.isoformat(),
        "age_seconds": age_seconds,
        "stale_after_seconds": RUNNING_LOG_STALE_SECONDS,
        "slurm_state": state,
        "caveat": "Fresh logs are required while the Stage-2 producer is running.",
    }


def file_info(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None}


def expected_snapshots(output_dir: Path, max_steps: int, save_every_steps: int) -> list[dict[str, Any]]:
    if save_every_steps <= 0:
        return []
    rows: list[dict[str, Any]] = []
    for step in range(save_every_steps, max_steps + 1, save_every_steps):
        snapshot_dir = output_dir / f"checkpoint-{step}"
        state = snapshot_dir / "custom_state_dict.pt"
        metrics = snapshot_dir / "metrics.json"
        rows.append(
            {
                "step": step,
                "snapshot_dir": str(snapshot_dir),
                "snapshot_exists": snapshot_dir.exists(),
                "state": file_info(state),
                "metrics": file_info(metrics),
                "complete": state.exists() and metrics.exists(),
            }
        )
    return rows


def snapshot_gate(
    *,
    output_dir: Path,
    latest_step: dict[str, Any],
    max_steps: int,
    save_every_steps: int,
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    step = latest_step.get("step")
    first_snapshot_step = save_every_steps if save_every_steps > 0 else None
    complete_steps = [snapshot["step"] for snapshot in snapshots if snapshot["complete"]]
    next_snapshot_step = None
    steps_to_next_snapshot = None
    if isinstance(step, int) and save_every_steps > 0:
        for snapshot_step in range(save_every_steps, max_steps + 1, save_every_steps):
            if snapshot_step > step:
                next_snapshot_step = snapshot_step
                steps_to_next_snapshot = snapshot_step - step
                break
    if not isinstance(step, int):
        status = "log_not_parsed"
        missing_output_dir_is_expected = False
    elif save_every_steps <= 0:
        status = "snapshots_disabled"
        missing_output_dir_is_expected = not output_dir.exists()
    elif step < save_every_steps:
        status = "pre_first_snapshot"
        missing_output_dir_is_expected = not output_dir.exists()
    elif complete_steps:
        status = "snapshots_present"
        missing_output_dir_is_expected = False
    elif step >= save_every_steps:
        status = "snapshot_due_missing"
        missing_output_dir_is_expected = False
    else:
        status = "unknown"
        missing_output_dir_is_expected = False
    return {
        "status": status,
        "output_dir": str(output_dir),
        "output_dir_exists": output_dir.exists(),
        "save_every_steps": save_every_steps,
        "first_snapshot_step": first_snapshot_step,
        "next_snapshot_step": next_snapshot_step,
        "steps_to_next_snapshot": steps_to_next_snapshot,
        "latest_complete_snapshot_step": complete_steps[-1] if complete_steps else None,
        "missing_output_dir_is_expected": missing_output_dir_is_expected,
        "caveat": (
            "A missing output directory is expected before the first snapshot when "
            "save_every_steps has not been reached."
        ),
    }


def estimate_progress(latest_step: dict[str, Any], max_steps: int, segment_tokens: int | None) -> dict[str, Any]:
    step = latest_step.get("step")
    elapsed = latest_step.get("elapsed_seconds")
    if not isinstance(step, int) or step <= 0 or not isinstance(elapsed, (int, float)) or elapsed <= 0:
        return {
            "seconds_per_step": None,
            "steps_per_hour": None,
            "eta_seconds": None,
            "eta_hours": None,
            "estimated_total_seconds": None,
            "estimated_completion_utc": None,
            "segment_token_presentations_per_second": None,
        }
    seconds_per_step = float(elapsed) / float(step)
    remaining_steps = max(max_steps - step, 0)
    eta_seconds = remaining_steps * seconds_per_step
    estimated_total_seconds = max_steps * seconds_per_step
    token_rate = (
        float(segment_tokens) / estimated_total_seconds
        if isinstance(segment_tokens, int) and segment_tokens > 0 and estimated_total_seconds > 0
        else None
    )
    return {
        "seconds_per_step": seconds_per_step,
        "steps_per_hour": 3600.0 / seconds_per_step,
        "eta_seconds": eta_seconds,
        "eta_hours": eta_seconds / 3600.0,
        "estimated_total_seconds": estimated_total_seconds,
        "estimated_completion_utc": datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() + eta_seconds,
            tz=timezone.utc,
        ).isoformat(),
        "segment_token_presentations_per_second": token_rate,
    }


def time_limit_gate(squeue_row: dict[str, str] | None, estimate: dict[str, Any]) -> dict[str, Any]:
    state = (squeue_row or {}).get("state", "")
    elapsed_text = (squeue_row or {}).get("time")
    limit_text = (squeue_row or {}).get("time_limit")
    elapsed_seconds = parse_slurm_duration(elapsed_text)
    limit_seconds = parse_slurm_duration(limit_text)
    eta_seconds = estimate.get("eta_seconds")
    if state.lower() != "running":
        status = "not_running"
    elif elapsed_seconds is None or limit_seconds is None or not isinstance(eta_seconds, (int, float)):
        status = "unknown"
    else:
        remaining_seconds = limit_seconds - elapsed_seconds
        margin_seconds = remaining_seconds - float(eta_seconds)
        if margin_seconds < 0:
            status = "likely_walltime_failure"
        elif margin_seconds < TIME_LIMIT_TIGHT_MARGIN_SECONDS:
            status = "tight_walltime_margin"
        else:
            status = "within_time_limit"
    remaining_seconds = (
        limit_seconds - elapsed_seconds
        if isinstance(elapsed_seconds, int) and isinstance(limit_seconds, int)
        else None
    )
    margin_seconds = (
        remaining_seconds - float(eta_seconds)
        if isinstance(remaining_seconds, int) and isinstance(eta_seconds, (int, float))
        else None
    )
    return {
        "status": status,
        "slurm_state": state,
        "elapsed": elapsed_text,
        "time_limit": limit_text,
        "elapsed_seconds": elapsed_seconds,
        "time_limit_seconds": limit_seconds,
        "eta_seconds": eta_seconds,
        "remaining_seconds": remaining_seconds,
        "margin_seconds": margin_seconds,
        "tight_margin_threshold_seconds": TIME_LIMIT_TIGHT_MARGIN_SECONDS,
        "caveat": "Compares current ETA with Slurm time remaining; it is a runtime-risk signal, not quality evidence.",
    }


def classify_stage2_status(
    *,
    squeue_row: dict[str, str] | None,
    root_metrics: Path,
    final_state: Path,
) -> str:
    if root_metrics.exists() and final_state.exists():
        return "complete_artifacts_present"
    if squeue_row:
        state = squeue_row.get("state", "").lower()
        if state == "running":
            return "running"
        if state == "pending":
            return "pending"
        return f"slurm_{state}"
    return "not_in_squeue_incomplete"


def classify_downstream_status(
    *,
    handoff_report: dict[str, Any] | None,
    metrics: Path,
    predictions: Path,
    slurm_row: dict[str, str] | None,
) -> str:
    if metrics.exists() and predictions.exists():
        return "complete_artifacts_present"
    if handoff_report is None:
        return "waiting_for_handoff"
    handoff_status = str(handoff_report.get("status", ""))
    if handoff_status == "failed":
        return "handoff_failed"
    if slurm_row:
        state = slurm_row.get("state", "").lower()
        if state == "running":
            return "running"
        if state == "pending":
            return "pending"
        return f"slurm_{state}"
    if handoff_status == "submitted_downstream":
        return "submitted_downstream_not_in_squeue_incomplete"
    return "not_submitted_incomplete"


def build(args: argparse.Namespace) -> dict[str, Any]:
    stage2_submission = read_json(args.stage2_submission)
    handoff_submission = read_json(args.handoff_submission)
    telemetry_submission = read_json(args.telemetry_submission)
    stage2_job_id = str(stage2_submission["submitted_job_id"])
    handoff_job_id = str(handoff_submission["handoff_job_id"])
    telemetry_job_id = str(telemetry_submission["job_id"])
    handoff_report_path = Path(handoff_submission["expected_handoff_json"])
    handoff_report = read_optional_json(handoff_report_path)
    downstream_job_id = ""
    postprocess_job_id = ""
    if isinstance(handoff_report, dict):
        downstream_job_id = str(handoff_report.get("downstream_job_id") or "")
        postprocess_job_id = str(handoff_report.get("postprocess_job_id") or "")
    job_ids = [stage2_job_id, handoff_job_id, telemetry_job_id]
    if downstream_job_id:
        job_ids.append(downstream_job_id)
    if postprocess_job_id:
        job_ids.append(postprocess_job_id)
    rows = squeue_rows(job_ids)

    stage2_config = stage2_submission["run_config"]
    stage2_output = Path(stage2_config["output_dir"])
    final_snapshot = stage2_output / f"checkpoint-{stage2_config['max_steps']}"
    root_metrics = stage2_output / "metrics.json"
    final_state = final_snapshot / "custom_state_dict.pt"
    final_snapshot_metrics = final_snapshot / "metrics.json"
    downstream_output_text = str(
        (handoff_report or {}).get(
            "downstream_output_dir",
            handoff_submission.get("expected_downstream_output_dir", ""),
        )
    )
    downstream_output = Path(downstream_output_text) if downstream_output_text else None
    downstream_metrics = downstream_output / "metrics.json" if downstream_output is not None else Path("")
    downstream_predictions = downstream_output / "eval_predictions.jsonl" if downstream_output is not None else Path("")
    postprocess_json_text = str(
        (handoff_report or {}).get(
            "postprocess_json",
            handoff_submission.get("expected_postprocess_json", ""),
        )
    )
    postprocess_md_text = str(
        (handoff_report or {}).get(
            "postprocess_md",
            handoff_submission.get("expected_postprocess_md", ""),
        )
    )
    next_decision_json_text = str(
        (handoff_report or {}).get(
            "next_decision_json",
            handoff_submission.get(
                "expected_next_decision_json",
                "benchmarks/results/bitdistill_next_decision_2026-05-23.json",
            ),
        )
    )
    next_decision_md_text = str(
        (handoff_report or {}).get(
            "next_decision_md",
            handoff_submission.get(
                "expected_next_decision_md",
                "benchmarks/results/bitdistill_next_decision_2026-05-23.md",
            ),
        )
    )
    latest_step = parse_latest_step(args.stage2_log)
    log_health = producer_log_health(args.stage2_log, stage2_config)
    producer_config_status = producer_config_gate(
        log_path=args.stage2_log,
        stage2_submission=stage2_submission,
        stage2_job_id=stage2_job_id,
    )
    max_steps = int(stage2_config["max_steps"])
    save_every_steps = int(stage2_config.get("save_every_steps") or 0)
    step = latest_step.get("step")
    progress_estimate = estimate_progress(
        latest_step,
        max_steps,
        stage2_config.get("segment_token_presentations"),
    )
    stage2_slurm_row = rows.get(stage2_job_id)
    snapshots = expected_snapshots(stage2_output, max_steps, save_every_steps)
    complete_snapshots = [snapshot for snapshot in snapshots if snapshot["complete"]]
    snapshot_status = snapshot_gate(
        output_dir=stage2_output,
        latest_step=latest_step,
        max_steps=max_steps,
        save_every_steps=save_every_steps,
        snapshots=snapshots,
    )
    next_snapshot_steps = snapshot_status.get("steps_to_next_snapshot")
    seconds_per_step = progress_estimate.get("seconds_per_step")
    if isinstance(next_snapshot_steps, int) and isinstance(seconds_per_step, (int, float)):
        next_snapshot_eta_seconds = float(next_snapshot_steps) * float(seconds_per_step)
        snapshot_status["next_snapshot_eta_seconds"] = next_snapshot_eta_seconds
        snapshot_status["next_snapshot_eta_hours"] = next_snapshot_eta_seconds / 3600.0
        snapshot_status["estimated_next_snapshot_utc"] = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() + next_snapshot_eta_seconds,
            tz=timezone.utc,
        ).isoformat()
    else:
        snapshot_status["next_snapshot_eta_seconds"] = None
        snapshot_status["next_snapshot_eta_hours"] = None
        snapshot_status["estimated_next_snapshot_utc"] = None

    return {
        "schema": "bitnet-active-stage2-extension-monitor-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": classify_stage2_status(
            squeue_row=rows.get(stage2_job_id),
            root_metrics=root_metrics,
            final_state=final_state,
        ),
        "quality_claim": "none",
        "stage2": {
            "job_id": stage2_job_id,
            "slurm": rows.get(stage2_job_id, {"job_id": stage2_job_id, "state": "not_in_squeue"}),
            "latest_step": latest_step,
            "max_steps": max_steps,
            "save_every_steps": save_every_steps,
            "progress": (float(step) / float(max_steps)) if isinstance(step, int) and max_steps else None,
            "progress_estimate": progress_estimate,
            "time_limit_gate": time_limit_gate(stage2_slurm_row, progress_estimate),
            "log_freshness": log_freshness(args.stage2_log, rows.get(stage2_job_id)),
            "log_health": log_health,
            "producer_config": producer_config_status,
            "snapshot_status": snapshot_status,
            "expected_snapshots": snapshots,
            "latest_complete_snapshot_step": complete_snapshots[-1]["step"] if complete_snapshots else None,
            "root_metrics": file_info(root_metrics),
            "final_state": file_info(final_state),
            "final_snapshot_metrics": file_info(final_snapshot_metrics),
            "output_dir": str(stage2_output),
            "cumulative_token_presentations": stage2_config["cumulative_token_presentations"],
            "caveat": stage2_submission["caveat"],
        },
        "handoff": {
            "job_id": handoff_job_id,
            "slurm": rows.get(handoff_job_id, {"job_id": handoff_job_id, "state": "not_in_squeue"}),
            "dependency": handoff_submission["dependency"],
            "expected_manifest_json": handoff_submission["expected_manifest_json"],
            "expected_manifest_exists": Path(handoff_submission["expected_manifest_json"]).exists(),
            "expected_handoff_json": handoff_submission["expected_handoff_json"],
            "expected_handoff_exists": handoff_report_path.exists(),
            "handoff_report_status": handoff_report.get("status") if isinstance(handoff_report, dict) else None,
        },
        "downstream": {
            "job_id": downstream_job_id,
            "slurm": rows.get(downstream_job_id, {"job_id": downstream_job_id, "state": "not_submitted"})
            if downstream_job_id
            else {"job_id": "", "state": "not_submitted"},
            "status": classify_downstream_status(
                handoff_report=handoff_report,
                metrics=downstream_metrics,
                predictions=downstream_predictions,
                slurm_row=rows.get(downstream_job_id) if downstream_job_id else None,
            ),
            "handoff_report_exists": handoff_report_path.exists(),
            "handoff_report_status": handoff_report.get("status") if isinstance(handoff_report, dict) else None,
            "output_dir": downstream_output_text,
            "metrics": file_info(downstream_metrics) if downstream_output is not None else {"path": "", "exists": False, "size_bytes": None},
            "predictions": file_info(downstream_predictions) if downstream_output is not None else {"path": "", "exists": False, "size_bytes": None},
            "complete": downstream_metrics.exists() and downstream_predictions.exists() if downstream_output is not None else False,
            "caveat": "This section tracks downstream artifact existence only; it does not compute or claim MNLI accuracy.",
        },
        "postprocess": {
            "job_id": postprocess_job_id,
            "slurm": rows.get(postprocess_job_id, {"job_id": postprocess_job_id, "state": "not_submitted"})
            if postprocess_job_id
            else {"job_id": "", "state": "not_submitted"},
            "expected_json": postprocess_json_text,
            "expected_json_exists": Path(postprocess_json_text).exists() if postprocess_json_text else False,
            "expected_md": postprocess_md_text,
            "expected_md_exists": Path(postprocess_md_text).exists() if postprocess_md_text else False,
            "expected_next_decision_json": next_decision_json_text,
            "expected_next_decision_json_exists": Path(next_decision_json_text).exists() if next_decision_json_text else False,
            "expected_next_decision_md": next_decision_md_text,
            "expected_next_decision_md_exists": Path(next_decision_md_text).exists() if next_decision_md_text else False,
            "caveat": "This section tracks report-regeneration job state only; it is not quality evidence.",
        },
        "telemetry": {
            "job_id": telemetry_job_id,
            "slurm": rows.get(telemetry_job_id, {"job_id": telemetry_job_id, "state": "not_in_squeue"}),
            "dependency": telemetry_submission["dependency"],
            "expected_artifacts": [file_info(Path(path)) for path in telemetry_submission["expected_artifacts"]],
            "caveat": telemetry_submission["caveat"],
        },
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
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
    stage2 = report["stage2"]
    handoff = report["handoff"]
    downstream = report["downstream"]
    postprocess = report["postprocess"]
    telemetry = report["telemetry"]
    latest = stage2["latest_step"]
    estimate = stage2["progress_estimate"]
    time_gate = stage2["time_limit_gate"]
    freshness = stage2["log_freshness"]
    log_health = stage2["log_health"]
    producer_config = stage2["producer_config"]
    snapshot_status = stage2["snapshot_status"]
    artifact_rows = [
        ["stage2 root metrics", stage2["root_metrics"]["exists"], stage2["root_metrics"]["path"]],
        ["stage2 final state", stage2["final_state"]["exists"], stage2["final_state"]["path"]],
        ["stage2 final snapshot metrics", stage2["final_snapshot_metrics"]["exists"], stage2["final_snapshot_metrics"]["path"]],
        ["handoff manifest", handoff["expected_manifest_exists"], handoff["expected_manifest_json"]],
        ["handoff report", handoff["expected_handoff_exists"], handoff["expected_handoff_json"]],
        ["downstream metrics", downstream["metrics"]["exists"], downstream["metrics"]["path"]],
        ["downstream predictions", downstream["predictions"]["exists"], downstream["predictions"]["path"]],
        ["postprocess report", postprocess["expected_json_exists"], postprocess["expected_json"]],
        ["next decision report", postprocess["expected_next_decision_json_exists"], postprocess["expected_next_decision_json"]],
    ]
    artifact_rows.extend(
        [f"telemetry artifact {idx}", artifact["exists"], artifact["path"]]
        for idx, artifact in enumerate(telemetry["expected_artifacts"], start=1)
    )
    snapshot_rows = [
        [
            snapshot["step"],
            snapshot["snapshot_exists"],
            snapshot["state"]["exists"],
            snapshot["metrics"]["exists"],
            snapshot["complete"],
        ]
        for snapshot in stage2["expected_snapshots"]
    ]
    return "\n\n".join(
        [
            "# Active Stage-2 Extension Monitor",
            f"Status: **{report['status']}**.",
            "Quality claim: **none**. This report monitors job/artifact state only.",
            md_table(
                ["job", "id", "slurm state", "time", "reason"],
                [
                    [
                        "stage2",
                        stage2["job_id"],
                        stage2["slurm"].get("state"),
                        stage2["slurm"].get("time", ""),
                        stage2["slurm"].get("reason", ""),
                    ],
                    [
                        "handoff",
                        handoff["job_id"],
                        handoff["slurm"].get("state"),
                        handoff["slurm"].get("time", ""),
                        handoff["slurm"].get("reason", ""),
                    ],
                    [
                        "gamma60 telemetry",
                        telemetry["job_id"],
                        telemetry["slurm"].get("state"),
                        telemetry["slurm"].get("time", ""),
                        telemetry["slurm"].get("reason", ""),
                    ],
                    [
                        "downstream MNLI",
                        downstream["job_id"] or "-",
                        downstream["slurm"].get("state"),
                        downstream["slurm"].get("time", ""),
                        downstream["slurm"].get("reason", ""),
                    ],
                    [
                        "postprocess",
                        postprocess["job_id"] or "-",
                        postprocess["slurm"].get("state"),
                        postprocess["slurm"].get("time", ""),
                        postprocess["slurm"].get("reason", ""),
                    ],
                ],
            ),
            md_table(
                ["stage2 field", "value"],
                [
                    ["latest_step", latest.get("step", "")],
                    ["max_steps", stage2["max_steps"]],
                    ["save_every_steps", stage2["save_every_steps"]],
                    ["snapshot_status", snapshot_status["status"]],
                    ["output_dir_exists", snapshot_status["output_dir_exists"]],
                    ["missing_output_dir_is_expected", snapshot_status["missing_output_dir_is_expected"]],
                    ["first_snapshot_step", snapshot_status["first_snapshot_step"]],
                    ["next_snapshot_step", snapshot_status["next_snapshot_step"]],
                    ["steps_to_next_snapshot", snapshot_status["steps_to_next_snapshot"]],
                    ["next_snapshot_eta_hours", snapshot_status["next_snapshot_eta_hours"]],
                    ["progress", stage2["progress"]],
                    ["latest_ce", latest.get("ce", "")],
                    ["latest_lr", latest.get("lr", "")],
                    ["log_freshness_status", freshness["status"]],
                    ["log_health_status", log_health["status"]],
                    ["producer_config_status", producer_config["status"]],
                    ["log_age_seconds", freshness["age_seconds"]],
                    ["time_limit_status", time_gate["status"]],
                    ["time_limit_margin_seconds", time_gate["margin_seconds"]],
                    ["log_elapsed_seconds", latest.get("elapsed_seconds", "")],
                    ["parsed_log_rows", latest.get("parsed_log_rows", "")],
                    ["recent_window_rows", latest.get("recent_window_rows", "")],
                    ["recent_ce_mean", latest.get("recent_ce_mean", "")],
                    ["recent_ce_min", latest.get("recent_ce_min", "")],
                    ["recent_ce_max", latest.get("recent_ce_max", "")],
                    ["seconds_per_step", estimate.get("seconds_per_step")],
                    ["steps_per_hour", estimate.get("steps_per_hour")],
                    ["eta_hours", estimate.get("eta_hours")],
                    ["estimated_completion_utc", estimate.get("estimated_completion_utc")],
                    ["segment_token_presentations_per_second", estimate.get("segment_token_presentations_per_second")],
                    ["latest_complete_snapshot_step", stage2["latest_complete_snapshot_step"]],
                    ["cumulative_token_presentations", stage2["cumulative_token_presentations"]],
                ],
            ),
            "## Time Limit Gate",
            md_table(
                ["field", "value"],
                [
                    ["status", time_gate["status"]],
                    ["slurm_state", time_gate["slurm_state"]],
                    ["elapsed", time_gate["elapsed"]],
                    ["time_limit", time_gate["time_limit"]],
                    ["elapsed_seconds", time_gate["elapsed_seconds"]],
                    ["time_limit_seconds", time_gate["time_limit_seconds"]],
                    ["eta_seconds", time_gate["eta_seconds"]],
                    ["remaining_seconds", time_gate["remaining_seconds"]],
                    ["margin_seconds", time_gate["margin_seconds"]],
                    ["tight_margin_threshold_seconds", time_gate["tight_margin_threshold_seconds"]],
                    ["caveat", time_gate["caveat"]],
                ],
            ),
            "## Log Freshness",
            md_table(
                ["field", "value"],
                [
                    ["status", freshness["status"]],
                    ["path", freshness["path"]],
                    ["exists", freshness["exists"]],
                    ["checked_utc", freshness["checked_utc"]],
                    ["mtime_utc", freshness["mtime_utc"]],
                    ["age_seconds", freshness["age_seconds"]],
                    ["stale_after_seconds", freshness["stale_after_seconds"]],
                    ["slurm_state", freshness["slurm_state"]],
                    ["caveat", freshness["caveat"]],
                ],
            ),
            "## Producer Log Health",
            md_table(
                ["field", "value"],
                [
                    ["status", log_health["status"]],
                    ["path", log_health["path"]],
                    ["parsed_step_rows", log_health["parsed_step_rows"]],
                    ["first_step", log_health.get("first_step")],
                    ["latest_step", log_health.get("latest_step")],
                    ["latest_ce", log_health.get("latest_ce")],
                    ["latest_lr", log_health.get("latest_lr")],
                    ["latest_elapsed_seconds", log_health.get("latest_elapsed_seconds")],
                    ["recent_window_rows", log_health.get("recent_window_rows")],
                    ["recent_ce_mean", log_health.get("recent_ce_mean")],
                    ["recent_ce_min", log_health.get("recent_ce_min")],
                    ["recent_ce_max", log_health.get("recent_ce_max")],
                    ["issue_count", len(log_health["issues"])],
                    ["fatal_match_count", len(log_health["fatal_matches"])],
                    ["caveat", log_health["caveat"]],
                ],
            ),
            md_table(
                ["check", "value"],
                [[key, value] for key, value in log_health["checks"].items()],
            ),
            "## Producer Config Gate",
            md_table(
                ["field", "value"],
                [
                    ["status", producer_config["status"]],
                    ["log_path", producer_config["log_header"]["path"]],
                    ["header_exists", producer_config["log_header"]["exists"]],
                    ["header_line_count", producer_config["log_header"]["line_count"]],
                    ["mismatch_count", len(producer_config["mismatches"])],
                    ["caveat", producer_config["caveat"]],
                ],
            ),
            md_table(
                ["key", "expected", "actual", "mode", "matched"],
                [
                    [
                        check["key"],
                        check["expected"],
                        check["actual"],
                        check["mode"],
                        check["matched"],
                    ]
                    for check in producer_config["checks"]
                ],
            ),
            "## Snapshot Gate",
            md_table(
                ["field", "value"],
                [
                    ["status", snapshot_status["status"]],
                    ["output_dir", snapshot_status["output_dir"]],
                    ["output_dir_exists", snapshot_status["output_dir_exists"]],
                    ["first_snapshot_step", snapshot_status["first_snapshot_step"]],
                    ["next_snapshot_step", snapshot_status["next_snapshot_step"]],
                    ["steps_to_next_snapshot", snapshot_status["steps_to_next_snapshot"]],
                    ["next_snapshot_eta_hours", snapshot_status["next_snapshot_eta_hours"]],
                    ["estimated_next_snapshot_utc", snapshot_status["estimated_next_snapshot_utc"]],
                    ["latest_complete_snapshot_step", snapshot_status["latest_complete_snapshot_step"]],
                    ["missing_output_dir_is_expected", snapshot_status["missing_output_dir_is_expected"]],
                    ["caveat", snapshot_status["caveat"]],
                ],
            ),
            "## Expected Snapshots",
            md_table(["step", "dir exists", "state", "metrics", "complete"], snapshot_rows),
            "## Artifacts",
            md_table(["artifact", "exists", "path"], artifact_rows),
            "## Downstream",
            md_table(
                ["field", "value"],
                [
                    ["status", downstream["status"]],
                    ["handoff_report_exists", downstream["handoff_report_exists"]],
                    ["handoff_report_status", downstream["handoff_report_status"]],
                    ["output_dir", downstream["output_dir"]],
                    ["complete", downstream["complete"]],
                    ["caveat", downstream["caveat"]],
                ],
            ),
            "## Postprocess",
            md_table(
                ["field", "value"],
                [
                    ["job_id", postprocess["job_id"]],
                    ["slurm_state", postprocess["slurm"].get("state")],
                    ["expected_json", postprocess["expected_json"]],
                    ["expected_json_exists", postprocess["expected_json_exists"]],
                    ["expected_md", postprocess["expected_md"]],
                    ["expected_md_exists", postprocess["expected_md_exists"]],
                    ["expected_next_decision_json", postprocess["expected_next_decision_json"]],
                    ["expected_next_decision_json_exists", postprocess["expected_next_decision_json_exists"]],
                    ["expected_next_decision_md", postprocess["expected_next_decision_md"]],
                    ["expected_next_decision_md_exists", postprocess["expected_next_decision_md_exists"]],
                    ["caveat", postprocess["caveat"]],
                ],
            ),
            "## Caveat",
            stage2["caveat"],
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage2-submission",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_submission_2026-05-23.json"),
    )
    parser.add_argument(
        "--handoff-submission",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_handoff_submission_2026-05-23.json"),
    )
    parser.add_argument(
        "--telemetry-submission",
        type=Path,
        default=Path("benchmarks/results/gamma60_telemetry_submission_2026-05-23.json"),
    )
    parser.add_argument("--stage2-log", type=Path, default=Path("logs/bd-s2-655m-10250.out"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/active_stage2_extension_monitor_2026-05-23.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/active_stage2_extension_monitor_2026-05-23.md"),
    )
    args = parser.parse_args()

    report = build(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report).rstrip() + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
