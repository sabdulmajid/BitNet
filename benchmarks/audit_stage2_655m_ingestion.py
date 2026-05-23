#!/usr/bin/env python3
"""Audit ingestion of the active 655.36M BitDistill Stage-2 result.

This report is safe before completion: it remains pending until the handoff,
downstream metrics, prediction trace, controlled-curve row, reproduction-gap
report, and next-decision report all line up. It fails only on inconsistent or
contradictory evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit_bitdistill_recovery_run import EXPECTED_MNLI, compare, finite, read_predictions


TARGET_STAGE2_TOKENS = 655_360_000
SUCCESS_DELTA_FROM_FP = -0.01


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
        ["squeue", "-h", "-j", job_id, "-o", "%i\t%T\t%M\t%R\t%j"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"job_id": job_id, "state": "not_in_squeue"}
    parts = result.stdout.strip().split("\t", 4)
    return {
        "job_id": parts[0] if len(parts) > 0 else job_id,
        "state": parts[1] if len(parts) > 1 else "unknown",
        "time": parts[2] if len(parts) > 2 else "",
        "reason": parts[3] if len(parts) > 3 else "",
        "name": parts[4] if len(parts) > 4 else "",
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
        if not value:
            return "none"
        return ", ".join(fmt(item) for item in value)
    if isinstance(value, dict):
        if not value:
            return "none"
        return ", ".join(f"{key}={fmt(val)}" for key, val in value.items())
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def load_controlled_row(path: Path, target_tokens: int) -> dict[str, Any]:
    report = read_json(path, required=False)
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    for row in rows:
        if isinstance(row, dict) and row.get("stage2_token_presentations") == target_tokens:
            return row
    return {}


def summarize_downstream(metrics_path: Path, predictions_path: Path, reference_predictions: Path) -> dict[str, Any]:
    metrics = read_json(metrics_path, required=False)
    eval_metrics = metrics.get("eval", {}) if isinstance(metrics.get("eval"), dict) else {}
    metric_accuracy = finite(eval_metrics.get("accuracy"))
    metric_examples = finite(eval_metrics.get("eval_examples"))
    prediction_rows, prediction_errors = read_predictions(predictions_path)
    paired = compare(reference_predictions, predictions_path) if predictions_path.exists() else {
        "status": "pending",
        "missing": [str(predictions_path)],
        "errors": [],
        "matched": 0,
    }
    return {
        "metrics": file_info(metrics_path),
        "predictions": file_info(predictions_path),
        "metric_accuracy": metric_accuracy,
        "metric_eval_examples": metric_examples,
        "prediction_rows": len(prediction_rows),
        "prediction_errors": prediction_errors[:10],
        "prediction_error_count": len(prediction_errors),
        "paired": paired,
    }


def validate_consistency(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    downstream = report["downstream"]
    controlled = report["controlled_curve"]
    reproduction_gap = report["reproduction_gap"]
    next_decision = report["next_decision"]
    postprocess = report["postprocess"]

    metrics_exists = downstream["metrics"]["exists"]
    predictions_exists = downstream["predictions"]["exists"]
    if metrics_exists != predictions_exists:
        errors.append("downstream metrics/predictions existence mismatch")
    if metrics_exists:
        if downstream["metric_eval_examples"] != EXPECTED_MNLI:
            errors.append(f"downstream eval_examples {downstream['metric_eval_examples']} != {EXPECTED_MNLI}")
        if downstream["paired"].get("status") != "pass":
            errors.append(f"downstream paired status is {downstream['paired'].get('status')}")
        if downstream["prediction_error_count"]:
            errors.append(f"downstream prediction parser errors: {downstream['prediction_errors'][:3]}")

    target_row = controlled["target_row"]
    target_row_exists = bool(target_row)
    if target_row_exists:
        if target_row.get("metrics_exists") is not True or target_row.get("predictions_exists") is not True:
            errors.append("controlled target row exists without metrics and predictions")
        paired = target_row.get("paired", {}) if isinstance(target_row.get("paired"), dict) else {}
        if paired.get("status") != "pass":
            errors.append(f"controlled target row paired status is {paired.get('status')}")
        if paired.get("matched") != EXPECTED_MNLI:
            errors.append(f"controlled target row matched {paired.get('matched')} != {EXPECTED_MNLI}")
        if not math.isfinite(float(target_row.get("metric_accuracy", float("nan")))):
            errors.append("controlled target row lacks finite metric_accuracy")

    gap_latest_tokens = reproduction_gap["latest_stage2_tokens"]
    if target_row_exists and gap_latest_tokens != TARGET_STAGE2_TOKENS:
        errors.append(f"gap report latest tokens {gap_latest_tokens} != {TARGET_STAGE2_TOKENS}")
    next_status = next_decision["status"]
    if target_row_exists and next_status == "pending_655m_downstream":
        errors.append("next-decision report is stale: still pending_655m_downstream after target row exists")

    post_status = postprocess["status"]
    if post_status == "reports_rebuilt" and not target_row_exists:
        errors.append("postprocess says reports_rebuilt but controlled target row is missing")
    return errors


def classify(report: dict[str, Any], errors: list[str]) -> str:
    if errors:
        return "failed_inconsistent"
    handoff_status = report["handoff"].get("status")
    post_status = report["postprocess"].get("status")
    downstream = report["downstream"]
    target_row = report["controlled_curve"]["target_row"]
    gap_latest_tokens = report["reproduction_gap"]["latest_stage2_tokens"]
    if target_row and gap_latest_tokens == TARGET_STAGE2_TOKENS and post_status == "reports_rebuilt":
        return "ingested_reports_rebuilt"
    if downstream["metrics"]["exists"] and downstream["predictions"]["exists"]:
        return "downstream_complete_pending_report_ingestion"
    if post_status == "downstream_incomplete":
        return "downstream_incomplete"
    if handoff_status == "failed":
        return "handoff_failed"
    if handoff_status == "submitted_downstream":
        return "downstream_pending_or_running"
    return "pending_handoff"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    stage2_submission = read_json(args.stage2_submission)
    handoff_submission = read_json(args.handoff_submission)
    handoff_report = read_json(args.handoff_report, required=False)
    postprocess_report = read_json(args.postprocess_report, required=False)
    reproduction_gap = read_json(args.reproduction_gap, required=False)
    next_decision = read_json(args.next_decision, required=False)

    downstream_dir = Path(
        handoff_report.get("downstream_output_dir")
        or handoff_submission.get("expected_downstream_output_dir")
        or args.downstream_output_dir
    )
    metrics_path = downstream_dir / "metrics.json"
    predictions_path = downstream_dir / "eval_predictions.jsonl"
    downstream = summarize_downstream(metrics_path, predictions_path, args.reference_predictions)
    target_row = load_controlled_row(args.controlled_curve, args.target_stage2_tokens)
    gap_metrics = reproduction_gap.get("metrics", {}) if isinstance(reproduction_gap.get("metrics"), dict) else {}
    report = {
        "schema": "bitdistill-655m-ingestion-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quality_claim": "none_until_complete_downstream_trace",
        "target_stage2_tokens": args.target_stage2_tokens,
        "success_delta_from_fp": SUCCESS_DELTA_FROM_FP,
        "stage2": {
            "job_id": str(stage2_submission.get("submitted_job_id", "")),
            "slurm": squeue_state(str(stage2_submission.get("submitted_job_id", ""))),
            "expected_completion_artifact": stage2_submission.get("expected_completion_artifact"),
        },
        "handoff": {
            "submission": str(args.handoff_submission),
            "report": file_info(args.handoff_report),
            "status": handoff_report.get("status") or handoff_submission.get("status"),
            "job_id": str(handoff_submission.get("handoff_job_id", "")),
            "slurm": squeue_state(str(handoff_submission.get("handoff_job_id", ""))),
            "downstream_job_id": str(handoff_report.get("downstream_job_id", "")),
        },
        "downstream": downstream,
        "postprocess": {
            "report": file_info(args.postprocess_report),
            "status": postprocess_report.get("status"),
            "job_id": str(postprocess_report.get("postprocess_job_id", "")),
            "downstream_job_id": str(postprocess_report.get("downstream_job_id", "")),
        },
        "controlled_curve": {
            "path": str(args.controlled_curve),
            "exists": args.controlled_curve.exists(),
            "target_row_exists": bool(target_row),
            "target_row": target_row,
        },
        "reproduction_gap": {
            "path": str(args.reproduction_gap),
            "exists": args.reproduction_gap.exists(),
            "status": reproduction_gap.get("status"),
            "latest_stage2_tokens": gap_metrics.get("bitdistill_latest_stage2_tokens"),
            "latest_mnli": gap_metrics.get("bitdistill_latest_mnli"),
            "latest_delta_vs_fp16": gap_metrics.get("bitdistill_latest_delta_vs_fp16"),
        },
        "next_decision": {
            "path": str(args.next_decision),
            "exists": args.next_decision.exists(),
            "status": next_decision.get("status"),
            "recommendation": next_decision.get("recommendation"),
        },
        "source_paths": {
            "stage2_submission": str(args.stage2_submission),
            "handoff_submission": str(args.handoff_submission),
            "handoff_report": str(args.handoff_report),
            "postprocess_report": str(args.postprocess_report),
            "controlled_curve": str(args.controlled_curve),
            "reproduction_gap": str(args.reproduction_gap),
            "next_decision": str(args.next_decision),
            "reference_predictions": str(args.reference_predictions),
        },
    }
    errors = validate_consistency(report)
    report["consistency_errors"] = errors
    report["status"] = classify(report, errors)
    report["complete"] = report["status"] == "ingested_reports_rebuilt"
    return report


def render_markdown(report: dict[str, Any]) -> str:
    downstream = report["downstream"]
    target = report["controlled_curve"]["target_row"]
    target_paired = target.get("paired", {}) if isinstance(target, dict) and isinstance(target.get("paired"), dict) else {}
    return "\n\n".join(
        [
            "# Stage-2 655.36M Ingestion Audit",
            f"Generated: `{report['created_utc']}`",
            f"Status: **{report['status']}**.",
            f"Quality claim: **{report['quality_claim']}**.",
            "This report is an ingestion receipt. It does not create a quality claim; it verifies that quality artifacts are present before other reports may use them.",
            "## Slurm State",
            md_table(
                ["job", "id", "state", "time", "reason"],
                [
                    [
                        "stage2",
                        report["stage2"]["job_id"],
                        report["stage2"]["slurm"].get("state"),
                        report["stage2"]["slurm"].get("time", ""),
                        report["stage2"]["slurm"].get("reason", ""),
                    ],
                    [
                        "handoff",
                        report["handoff"]["job_id"],
                        report["handoff"]["slurm"].get("state"),
                        report["handoff"]["slurm"].get("time", ""),
                        report["handoff"]["slurm"].get("reason", ""),
                    ],
                ],
            ),
            "## Downstream Artifacts",
            md_table(
                ["artifact", "exists", "path/value"],
                [
                    ["metrics", downstream["metrics"]["exists"], downstream["metrics"]["path"]],
                    ["predictions", downstream["predictions"]["exists"], downstream["predictions"]["path"]],
                    ["metric_accuracy", downstream["metric_accuracy"] is not None, downstream["metric_accuracy"]],
                    ["metric_eval_examples", downstream["metric_eval_examples"] is not None, downstream["metric_eval_examples"]],
                    ["prediction_rows", bool(downstream["prediction_rows"]), downstream["prediction_rows"]],
                    ["paired_status", downstream["paired"].get("status") is not None, downstream["paired"].get("status")],
                    ["paired_matched", downstream["paired"].get("matched") is not None, downstream["paired"].get("matched")],
                ],
            ),
            "## Report Ingestion",
            md_table(
                ["item", "status/value"],
                [
                    ["postprocess_status", report["postprocess"]["status"]],
                    ["controlled_target_row_exists", report["controlled_curve"]["target_row_exists"]],
                    ["controlled_accuracy", target.get("metric_accuracy") if isinstance(target, dict) else None],
                    ["controlled_delta_vs_fp16", target_paired.get("delta_vs_reference")],
                    ["controlled_paired_status", target_paired.get("status")],
                    ["gap_latest_stage2_tokens", report["reproduction_gap"]["latest_stage2_tokens"]],
                    ["gap_latest_mnli", report["reproduction_gap"]["latest_mnli"]],
                    ["next_decision_status", report["next_decision"]["status"]],
                ],
            ),
            "## Consistency",
            md_table(
                ["field", "value"],
                [
                    ["complete", report["complete"]],
                    ["consistency_errors", report["consistency_errors"]],
                ],
            ),
            "## Source Artifacts",
            md_table(["artifact", "path"], [[key, value] for key, value in report["source_paths"].items()]),
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-submission", type=Path, default=Path("benchmarks/results/stage2_655m_submission_2026-05-23.json"))
    parser.add_argument("--handoff-submission", type=Path, default=Path("benchmarks/results/stage2_655m_handoff_submission_2026-05-23.json"))
    parser.add_argument("--handoff-report", type=Path, default=Path("benchmarks/results/stage2_655m_handoff_2026-05-23.json"))
    parser.add_argument("--postprocess-report", type=Path, default=Path("benchmarks/results/stage2_655m_postprocess_2026-05-23.json"))
    parser.add_argument("--controlled-curve", type=Path, default=Path("benchmarks/results/bitdistill_controlled_curve_2026-05-23.json"))
    parser.add_argument("--reproduction-gap", type=Path, default=Path("benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json"))
    parser.add_argument("--next-decision", type=Path, default=Path("benchmarks/results/bitdistill_next_decision_2026-05-23.json"))
    parser.add_argument("--reference-predictions", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-predtrace/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1/eval_predictions.jsonl"))
    parser.add_argument("--downstream-output-dir", default="checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit")
    parser.add_argument("--target-stage2-tokens", type=int, default=TARGET_STAGE2_TOKENS)
    parser.add_argument("--output-json", type=Path, default=Path("benchmarks/results/stage2_655m_ingestion_2026-05-23.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/results/stage2_655m_ingestion_2026-05-23.md"))
    args = parser.parse_args()

    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    return 1 if report["status"] == "failed_inconsistent" else 0


if __name__ == "__main__":
    raise SystemExit(main())
