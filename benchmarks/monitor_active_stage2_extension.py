#!/usr/bin/env python3
"""Monitor active Stage-2 extension jobs without making quality claims."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STEP_RE = re.compile(
    r"step=(?P<step>\d+)\s+ce=(?P<ce>[0-9.eE+-]+)\s+lr=(?P<lr>[0-9.eE+-]+)\s+elapsed=(?P<elapsed>[0-9.eE+-]+)s"
)


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
    command = ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i\t%T\t%M\t%R\t%j"]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    rows: dict[str, dict[str, str]] = {}
    if result.returncode != 0:
        return rows
    for line in result.stdout.splitlines():
        parts = line.split("\t", 4)
        if len(parts) != 5:
            continue
        job_id, state, time_used, reason, name = parts
        rows[job_id] = {
            "job_id": job_id,
            "state": state,
            "time": time_used,
            "reason": reason,
            "name": name,
        }
    return rows


def parse_latest_step(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"log_exists": False}
    latest: dict[str, Any] = {"log_exists": True, "path": str(log_path)}
    rows: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        row = {
            "log_exists": True,
            "path": str(log_path),
            "step": int(match.group("step")),
            "ce": float(match.group("ce")),
            "lr": float(match.group("lr")),
            "elapsed_seconds": float(match.group("elapsed")),
        }
        rows.append(row)
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
    latest_step = parse_latest_step(args.stage2_log)
    max_steps = int(stage2_config["max_steps"])
    save_every_steps = int(stage2_config.get("save_every_steps") or 0)
    step = latest_step.get("step")
    progress_estimate = estimate_progress(
        latest_step,
        max_steps,
        stage2_config.get("segment_token_presentations"),
    )
    snapshots = expected_snapshots(stage2_output, max_steps, save_every_steps)
    complete_snapshots = [snapshot for snapshot in snapshots if snapshot["complete"]]

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
    artifact_rows = [
        ["stage2 root metrics", stage2["root_metrics"]["exists"], stage2["root_metrics"]["path"]],
        ["stage2 final state", stage2["final_state"]["exists"], stage2["final_state"]["path"]],
        ["stage2 final snapshot metrics", stage2["final_snapshot_metrics"]["exists"], stage2["final_snapshot_metrics"]["path"]],
        ["handoff manifest", handoff["expected_manifest_exists"], handoff["expected_manifest_json"]],
        ["handoff report", handoff["expected_handoff_exists"], handoff["expected_handoff_json"]],
        ["downstream metrics", downstream["metrics"]["exists"], downstream["metrics"]["path"]],
        ["downstream predictions", downstream["predictions"]["exists"], downstream["predictions"]["path"]],
        ["postprocess report", postprocess["expected_json_exists"], postprocess["expected_json"]],
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
                    ["progress", stage2["progress"]],
                    ["latest_ce", latest.get("ce", "")],
                    ["latest_lr", latest.get("lr", "")],
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
