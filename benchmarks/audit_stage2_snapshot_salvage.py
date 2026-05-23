#!/usr/bin/env python3
"""Audit intermediate Stage-2 snapshots for failover without quality claims."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def squeue_state(job_id: str) -> dict[str, str]:
    if not job_id:
        return {"job_id": "", "state": "not_submitted"}
    result = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%i\t%T\t%M\t%l\t%R\t%j"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"job_id": job_id, "state": "not_in_squeue"}
    parts = result.stdout.strip().split("\t", 5)
    return {
        "job_id": parts[0] if len(parts) > 0 else job_id,
        "state": parts[1] if len(parts) > 1 else "unknown",
        "time": parts[2] if len(parts) > 2 else "",
        "time_limit": parts[3] if len(parts) > 3 else "",
        "reason": parts[4] if len(parts) > 4 else "",
        "name": parts[5] if len(parts) > 5 else "",
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


def snapshot_metrics(path: Path) -> dict[str, Any]:
    data = read_json(path, required=False)
    if not data:
        return {}
    last = data.get("last", {}) if isinstance(data.get("last"), dict) else {}
    snapshot = data.get("snapshot", {}) if isinstance(data.get("snapshot"), dict) else {}
    return {
        "stage": data.get("stage"),
        "method": data.get("method"),
        "scale_mode": data.get("scale_mode"),
        "student_model": data.get("student_model"),
        "steps": data.get("steps"),
        "effective_train_token_presentations": data.get("effective_train_token_presentations"),
        "snapshot_step": snapshot.get("step"),
        "snapshot_complete": snapshot.get("complete"),
        "last_ce": last.get("ce"),
        "last_lr": last.get("lr"),
        "elapsed_seconds": data.get("elapsed_seconds"),
    }


def expected_snapshot_steps(max_steps: int, save_every_steps: int) -> list[int]:
    if save_every_steps <= 0:
        return []
    return list(range(save_every_steps, max_steps + 1, save_every_steps))


def audit_snapshot(
    *,
    output_dir: Path,
    step: int,
    parent_tokens: int,
    expected_model: str,
    expected_stage: str,
    expected_method: str,
    expected_scale_mode: str,
) -> dict[str, Any]:
    snapshot_dir = output_dir / f"checkpoint-{step}"
    state = snapshot_dir / "custom_state_dict.pt"
    metrics_path = snapshot_dir / "metrics.json"
    ternary_state = snapshot_dir / "ternary_state_dict.pt"
    config = snapshot_dir / "config.json"
    tokenizer = snapshot_dir / "tokenizer.json"
    metrics = snapshot_metrics(metrics_path)
    validation_errors: list[str] = []
    if metrics_path.exists():
        if metrics.get("stage") != expected_stage:
            validation_errors.append(f"stage {metrics.get('stage')} != {expected_stage}")
        if metrics.get("method") != expected_method:
            validation_errors.append(f"method {metrics.get('method')} != {expected_method}")
        if metrics.get("scale_mode") != expected_scale_mode:
            validation_errors.append(f"scale_mode {metrics.get('scale_mode')} != {expected_scale_mode}")
        if metrics.get("student_model") != expected_model:
            validation_errors.append(f"student_model {metrics.get('student_model')} != {expected_model}")
        if metrics.get("steps") != step:
            validation_errors.append(f"steps {metrics.get('steps')} != {step}")
        if metrics.get("snapshot_step") != step:
            validation_errors.append(f"snapshot.step {metrics.get('snapshot_step')} != {step}")
    segment_tokens = metrics.get("effective_train_token_presentations")
    cumulative_tokens = parent_tokens + int(segment_tokens) if isinstance(segment_tokens, int) else None
    complete = snapshot_dir.exists() and state.exists() and metrics_path.exists() and not validation_errors
    if complete:
        status = "complete"
    elif snapshot_dir.exists():
        status = "invalid" if validation_errors else "incomplete"
    else:
        status = "missing"
    return {
        "step": step,
        "status": status,
        "complete": complete,
        "snapshot_dir": str(snapshot_dir),
        "state": file_info(state),
        "metrics_file": file_info(metrics_path),
        "ternary_state": file_info(ternary_state),
        "config": file_info(config),
        "tokenizer": file_info(tokenizer),
        "metrics": metrics,
        "segment_token_presentations": segment_tokens,
        "cumulative_token_presentations": cumulative_tokens,
        "validation_errors": validation_errors,
    }


def classify(
    *,
    snapshots: list[dict[str, Any]],
    latest_step: int | None,
    first_snapshot_step: int | None,
    final_step: int,
    slurm_state: str,
) -> str:
    invalid = [snapshot for snapshot in snapshots if snapshot["validation_errors"]]
    complete = [snapshot for snapshot in snapshots if snapshot["complete"]]
    final = next((snapshot for snapshot in snapshots if snapshot["step"] == final_step), None)
    if invalid:
        return "invalid_snapshot_metadata"
    if final and final["complete"]:
        return "final_snapshot_available"
    if complete:
        return "salvage_available"
    if isinstance(latest_step, int) and isinstance(first_snapshot_step, int) and latest_step < first_snapshot_step:
        return "no_snapshot_expected_yet"
    if slurm_state.lower() in {"failed", "cancelled", "timeout", "out_of_memory"}:
        return "failed_no_salvage_snapshot"
    return "waiting_for_snapshot"


def recommendation(status: str, slurm_state: str) -> str:
    if status == "final_snapshot_available":
        return "Use the normal 655M handoff path; this report is only a fallback inventory."
    if status == "salvage_available" and slurm_state.lower() == "running":
        return "Keep the active producer running; use the latest complete snapshot only if the final run fails."
    if status == "salvage_available":
        return "Build a manifest from the latest complete snapshot and run a clearly labeled fallback downstream row."
    if status == "no_snapshot_expected_yet":
        return "Keep watching; no checkpoint is expected before the first save interval."
    if status == "failed_no_salvage_snapshot":
        return "No usable intermediate checkpoint exists; rerun or shorten the save interval."
    if status == "invalid_snapshot_metadata":
        return "Do not use the snapshot until metadata validation errors are resolved."
    return "Keep watching for the first complete snapshot."


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    submission = read_json(args.stage2_submission)
    monitor = read_json(args.active_monitor, required=False)
    config = submission["run_config"]
    parent = submission["parent_manifest"]
    output_dir = Path(config["output_dir"])
    max_steps = int(config["max_steps"])
    save_every_steps = int(config.get("save_every_steps") or 0)
    steps = expected_snapshot_steps(max_steps, save_every_steps)
    stage2 = monitor.get("stage2", {}) if isinstance(monitor.get("stage2"), dict) else {}
    latest = stage2.get("latest_step", {}) if isinstance(stage2.get("latest_step"), dict) else {}
    latest_step = latest.get("step") if isinstance(latest.get("step"), int) else None
    job_id = str(submission.get("submitted_job_id") or "")
    slurm = squeue_state(job_id)
    snapshots = [
        audit_snapshot(
            output_dir=output_dir,
            step=step,
            parent_tokens=int(parent["token_presentations"]),
            expected_model=str(submission["model"]),
            expected_stage=str(config["stage"]),
            expected_method=str(config["method"]),
            expected_scale_mode=str(config["scale_mode"]),
        )
        for step in steps
    ]
    complete = [snapshot for snapshot in snapshots if snapshot["complete"]]
    best = complete[-1] if complete else None
    status = classify(
        snapshots=snapshots,
        latest_step=latest_step,
        first_snapshot_step=steps[0] if steps else None,
        final_step=max_steps,
        slurm_state=slurm.get("state", ""),
    )
    return {
        "schema": "bitdistill-stage2-snapshot-salvage-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quality_claim": "none",
        "status": status,
        "caveat": "This inventories Stage-2 checkpoints for failover only. It does not run downstream evaluation or create quality evidence.",
        "stage2_job_id": job_id,
        "slurm": slurm,
        "output_dir": str(output_dir),
        "latest_logged_step": latest_step,
        "max_steps": max_steps,
        "save_every_steps": save_every_steps,
        "parent_manifest": parent,
        "target_cumulative_token_presentations": config["cumulative_token_presentations"],
        "complete_snapshot_count": len(complete),
        "best_salvage_snapshot": best,
        "snapshots": snapshots,
        "recommendation": recommendation(status, slurm.get("state", "")),
        "source_paths": {
            "stage2_submission": str(args.stage2_submission),
            "active_monitor": str(args.active_monitor),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    best = report["best_salvage_snapshot"] or {}
    snapshot_rows = [
        [
            snapshot["step"],
            snapshot["status"],
            snapshot["state"]["exists"],
            snapshot["metrics_file"]["exists"],
            snapshot["cumulative_token_presentations"],
            snapshot["metrics"].get("last_ce"),
            snapshot["validation_errors"],
        ]
        for snapshot in report["snapshots"]
    ]
    return "\n\n".join(
        [
            "# Stage-2 Snapshot Salvage Audit",
            f"Generated: `{report['created_utc']}`",
            f"Status: **{report['status']}**.",
            f"Quality claim: **{report['quality_claim']}**.",
            report["caveat"],
            "## Current State",
            md_table(
                ["field", "value"],
                [
                    ["stage2_job_id", report["stage2_job_id"]],
                    ["slurm_state", report["slurm"].get("state")],
                    ["slurm_time", report["slurm"].get("time")],
                    ["latest_logged_step", report["latest_logged_step"]],
                    ["max_steps", report["max_steps"]],
                    ["save_every_steps", report["save_every_steps"]],
                    ["complete_snapshot_count", report["complete_snapshot_count"]],
                    ["target_cumulative_token_presentations", report["target_cumulative_token_presentations"]],
                    ["recommendation", report["recommendation"]],
                ],
            ),
            "## Best Salvage Snapshot",
            md_table(
                ["field", "value"],
                [
                    ["step", best.get("step")],
                    ["status", best.get("status")],
                    ["state", (best.get("state") or {}).get("path")],
                    ["metrics", (best.get("metrics_file") or {}).get("path")],
                    ["cumulative_token_presentations", best.get("cumulative_token_presentations")],
                    ["last_ce", (best.get("metrics") or {}).get("last_ce")],
                ],
            ),
            "## Snapshot Inventory",
            md_table(
                ["step", "status", "state", "metrics", "cumulative tokens", "last_ce", "validation errors"],
                snapshot_rows,
            ),
            "## Source Artifacts",
            md_table(["artifact", "path"], [[key, value] for key, value in report["source_paths"].items()]),
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage2-submission",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_submission_2026-05-23.json"),
    )
    parser.add_argument(
        "--active-monitor",
        type=Path,
        default=Path("benchmarks/results/active_stage2_extension_monitor_2026-05-23.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/stage2_snapshot_salvage_2026-05-23.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/stage2_snapshot_salvage_2026-05-23.md"),
    )
    args = parser.parse_args()
    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    return 1 if report["status"] == "invalid_snapshot_metadata" else 0


if __name__ == "__main__":
    raise SystemExit(main())
