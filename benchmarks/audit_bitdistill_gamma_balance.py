#!/usr/bin/env python3
"""Compare gamma-60 gradient balance against existing paper-gamma telemetry.

This is an interpretation aid for the loss-normalization gate.  It deliberately
does not score task quality.  The output answers one narrow question: did the
short gamma-60 telemetry run materially reduce the attention-KD update magnitude
relative to CE compared with the existing paper-gamma telemetry trace?
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit_bitdistill_training_dynamics import fmt, md_table, summarize_trace


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
BALANCED_MAX_GRAD_ATTENTION_TO_CE = 10.0


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def ratio(numerator: Any, denominator: Any) -> float | None:
    if finite(numerator) and finite(denominator) and abs(float(denominator)) > 0.0:
        return float(numerator) / float(denominator)
    return None


def squeue_state(job_id: str) -> dict[str, str]:
    if not job_id:
        return {"state": "unknown"}
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%i\t%T\t%M\t%R"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {"job_id": job_id, "state": "squeue_unavailable"}
    if result.returncode != 0:
        if "Invalid job id specified" in result.stderr:
            return {"job_id": job_id, "state": "not_in_squeue"}
        return {"job_id": job_id, "state": "squeue_error", "stderr": result.stderr.strip()}
    line = result.stdout.strip()
    if not line:
        return {"job_id": job_id, "state": "not_in_squeue"}
    parts = line.split("\t")
    return {
        "job_id": parts[0] if len(parts) > 0 else job_id,
        "state": parts[1] if len(parts) > 1 else "unknown",
        "elapsed": parts[2] if len(parts) > 2 else "",
        "reason": parts[3] if len(parts) > 3 else "",
    }


def select_paper_trace(dynamics: dict[str, Any]) -> dict[str, Any]:
    traces = dynamics.get("traces")
    if not isinstance(traces, list):
        return {}
    paper_like = [
        trace
        for trace in traces
        if isinstance(trace, dict)
        and finite(trace.get("inferred_attention_kd_weight"))
        and float(trace["inferred_attention_kd_weight"]) >= 99_000.0
    ]
    if not paper_like:
        paper_like = [trace for trace in traces if isinstance(trace, dict)]
    return max(
        paper_like,
        key=lambda trace: float(trace.get("final_grad_attention_to_ce") or 0.0),
        default={},
    )


def trace_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "rows": 0,
            "has_component_grad_norms": False,
            "has_activation": False,
            "has_dynamics": False,
        }
    trace = summarize_trace(path)
    trace["exists"] = True
    return trace


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    dynamics = read_json(args.paper_dynamics)
    paper = select_paper_trace(dynamics)
    gamma = trace_summary(args.gamma_telemetry)
    status_report = read_json(args.gamma_status)

    gamma_ready = (
        gamma.get("exists") is True
        and int(gamma.get("rows") or 0) > 0
        and gamma.get("has_component_grad_norms") is True
    )
    paper_ready = bool(paper) and finite(paper.get("final_grad_attention_to_ce"))
    grad_reduction = (
        ratio(paper.get("final_grad_attention_to_ce"), gamma.get("final_grad_attention_to_ce"))
        if gamma_ready and paper_ready
        else None
    )
    loss_reduction = (
        ratio(paper.get("final_loss_attention_to_ce"), gamma.get("final_loss_attention_to_ce"))
        if gamma_ready and finite(paper.get("final_loss_attention_to_ce"))
        else None
    )
    gamma_balanced = (
        gamma_ready
        and finite(gamma.get("final_grad_attention_to_ce"))
        and float(gamma["final_grad_attention_to_ce"]) <= BALANCED_MAX_GRAD_ATTENTION_TO_CE
    )

    if not paper_ready:
        status = "missing_paper_gamma_reference"
    elif not gamma_ready:
        status = "pending_gamma60_telemetry"
    elif gamma_balanced:
        status = "gamma60_rebalanced_attention_updates"
    else:
        status = "gamma60_still_attention_dominated"

    return {
        "schema": "bitdistill-gamma-balance-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "quality_claim": "none",
        "job_id": args.job_id,
        "squeue": squeue_state(args.job_id),
        "paper_dynamics": str(args.paper_dynamics),
        "gamma_status": str(args.gamma_status),
        "gamma_status_report_exists": args.gamma_status.exists(),
        "gamma_status_report_status": status_report.get("status"),
        "gamma_telemetry": str(args.gamma_telemetry),
        "balanced_max_grad_attention_to_ce": BALANCED_MAX_GRAD_ATTENTION_TO_CE,
        "paper_trace": paper,
        "gamma60_trace": gamma,
        "metrics": {
            "paper_final_grad_attention_to_ce": paper.get("final_grad_attention_to_ce"),
            "paper_final_loss_attention_to_ce": paper.get("final_loss_attention_to_ce"),
            "paper_max_grad_attention_to_ce": paper.get("max_grad_attention_to_ce"),
            "gamma60_final_grad_attention_to_ce": gamma.get("final_grad_attention_to_ce"),
            "gamma60_final_loss_attention_to_ce": gamma.get("final_loss_attention_to_ce"),
            "gamma60_max_grad_attention_to_ce": gamma.get("max_grad_attention_to_ce"),
            "gamma60_attention_grad_reduction_factor": grad_reduction,
            "gamma60_attention_loss_reduction_factor": loss_reduction,
            "gamma60_max_activation_clipped_fraction": gamma.get("max_activation_clipped_fraction"),
            "gamma60_max_activation_edge_fraction": gamma.get("max_activation_edge_fraction"),
            "gamma60_mean_flip_fraction": gamma.get("mean_flip_fraction"),
        },
        "interpretation": (
            "Gamma-60 materially rebalances attention-KD updates under the local reductions."
            if status == "gamma60_rebalanced_attention_updates"
            else (
                "Gamma-60 telemetry is pending; no loss-normalization conclusion is available yet."
                if status == "pending_gamma60_telemetry"
                else "The gamma-60 trace does not yet show a balanced attention-KD/CE update ratio."
            )
        ),
        "caveat": (
            "This audit compares gradient/loss balance only. It is not a task-quality "
            "benchmark and does not update BitDistill reproduction status."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    rows = [
        ["status", report["status"]],
        ["quality claim", report["quality_claim"]],
        ["job id", report["job_id"]],
        ["squeue state", report["squeue"].get("state")],
        ["squeue elapsed", report["squeue"].get("elapsed")],
        ["gamma telemetry exists", report["gamma60_trace"].get("exists")],
        ["gamma telemetry rows", report["gamma60_trace"].get("rows")],
        ["gamma status report", report["gamma_status_report_status"]],
    ]
    metric_rows = [
        ["paper final grad attention/CE", metrics["paper_final_grad_attention_to_ce"]],
        ["gamma60 final grad attention/CE", metrics["gamma60_final_grad_attention_to_ce"]],
        ["attention grad reduction factor", metrics["gamma60_attention_grad_reduction_factor"]],
        ["paper final loss attention/CE", metrics["paper_final_loss_attention_to_ce"]],
        ["gamma60 final loss attention/CE", metrics["gamma60_final_loss_attention_to_ce"]],
        ["attention loss reduction factor", metrics["gamma60_attention_loss_reduction_factor"]],
        ["gamma60 max activation clipped", metrics["gamma60_max_activation_clipped_fraction"]],
        ["gamma60 max activation edge", metrics["gamma60_max_activation_edge_fraction"]],
        ["gamma60 mean ternary flip fraction", metrics["gamma60_mean_flip_fraction"]],
    ]
    return "\n\n".join(
        [
            "# BitDistill Gamma Balance Audit",
            f"Generated: `{report['created_utc']}`",
            report["interpretation"],
            report["caveat"],
            "## Run State",
            md_table(["field", "value"], rows),
            "## Balance Metrics",
            md_table(["metric", "value"], metric_rows),
            "## Paths",
            md_table(
                ["artifact", "path"],
                [
                    ["paper dynamics", report["paper_dynamics"]],
                    ["gamma status", report["gamma_status"]],
                    ["gamma telemetry", report["gamma_telemetry"]],
                ],
            ),
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", default="10254")
    parser.add_argument(
        "--paper-dynamics",
        type=Path,
        default=Path("benchmarks/results/bitdistill_training_dynamics_2026-05-23.json"),
    )
    parser.add_argument(
        "--gamma-status",
        type=Path,
        default=Path("benchmarks/results/gamma60_telemetry_status_2026-05-23.json"),
    )
    parser.add_argument(
        "--gamma-telemetry",
        type=Path,
        default=Path(
            "checkpoints/bitdistill-glue-seqcls-telemetry-gamma60/"
            "Qwen-Qwen2.5-0.5B/mnli/"
            "bitdistill-tensor-20kwarmup-gamma60-headinit-steps200/telemetry.jsonl"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"benchmarks/results/gamma60_gradient_balance_{DATE}.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"benchmarks/results/gamma60_gradient_balance_{DATE}.md"),
    )
    args = parser.parse_args()

    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
