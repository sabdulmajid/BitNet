#!/usr/bin/env python3
"""Audit fixed-recipe BitDistill Stage-2 curve recovery rows.

The controlled curve rows are queued to answer one narrow question: with the
Stage-3 recipe held fixed, does a larger Stage-2 continued-pretraining budget
move the ternary student toward the FP16 MNLI teacher? This audit is safe to
run while jobs are still pending; each row remains pending until metrics and
per-example predictions exist.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit_bitdistill_recovery_run import (
    EXPECTED_MNLI,
    SUCCESS_DELTA_FROM_FP,
    compare,
    finite,
    fmt,
    md_table,
    read_json,
)


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()


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
        return {"state": "squeue_unavailable"}
    if result.returncode != 0:
        return {"state": "squeue_error", "stderr": result.stderr.strip()}
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


def summarize_job(job: dict[str, Any], reference_predictions: Path) -> dict[str, Any]:
    output_dir = Path(str(job.get("output_dir", "")))
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "eval_predictions.jsonl"
    metrics = read_json(metrics_path)
    eval_metrics = metrics.get("eval", {}) if isinstance(metrics.get("eval"), dict) else {}
    last = metrics.get("last", {}) if isinstance(metrics.get("last"), dict) else {}
    paired = compare(reference_predictions, predictions_path)
    metric_accuracy = finite(eval_metrics.get("accuracy"))
    metric_examples = finite(eval_metrics.get("eval_examples"))
    if paired["status"] == "pass" and metric_accuracy is not None:
        candidate_accuracy = paired.get("candidate_accuracy")
        if candidate_accuracy is None or abs(candidate_accuracy - metric_accuracy) > 1e-12:
            paired["status"] = "fail"
            paired.setdefault("errors", []).append(
                f"prediction accuracy {candidate_accuracy} disagrees with metrics {metric_accuracy}"
            )
    if paired["status"] == "pass" and metric_examples != EXPECTED_MNLI:
        paired["status"] = "fail"
        paired.setdefault("errors", []).append(f"metric eval_examples={metric_examples} expected={EXPECTED_MNLI}")
    delta = paired.get("delta_vs_reference")
    passed_gate = paired["status"] == "pass" and isinstance(delta, float) and delta >= SUCCESS_DELTA_FROM_FP
    return {
        "job_id": str(job.get("job_id", "")),
        "label": job.get("label", ""),
        "dependency": job.get("dependency", ""),
        "partition": job.get("partition", ""),
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "metrics_exists": metrics_path.exists(),
        "predictions_exists": predictions_path.exists(),
        "stage2_token_presentations": job.get("stage2_token_presentations"),
        "paper_stage2_fraction": (
            float(job["stage2_token_presentations"]) / 10_000_000_000
            if isinstance(job.get("stage2_token_presentations"), (int, float))
            else None
        ),
        "steps": metrics.get("steps"),
        "metric_accuracy": metric_accuracy,
        "metric_eval_examples": metric_examples,
        "last": {
            "ce": finite(last.get("ce")),
            "logit_kd": finite(last.get("logit_kd")),
            "weighted_logit_kd": finite(last.get("weighted_logit_kd")),
            "attention_kd": finite(last.get("attention_kd")),
            "weighted_attention_kd": finite(last.get("weighted_attention_kd")),
        },
        "paired": paired,
        "squeue": squeue_state(str(job.get("job_id", ""))),
        "passed_fp_recovery_gate": passed_gate,
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    submission = read_json(args.submission_json)
    jobs = submission.get("jobs", []) if isinstance(submission.get("jobs"), list) else []
    downstream_jobs = [
        job
        for job in jobs
        if isinstance(job, dict) and "downstream" in str(job.get("label", "")).lower()
    ]
    recovery = read_json(args.recovery_submission_json)
    if recovery:
        downstream_jobs.insert(
            1,
            {
                "job_id": recovery.get("job_id", ""),
                "label": "20k-warmup downstream control",
                "dependency": recovery.get("dependency", ""),
                "partition": recovery.get("partition", ""),
                "init_state_dict": recovery.get("init_state_dict", ""),
                "output_dir": recovery.get("output_dir", ""),
                "max_steps": recovery.get("recipe", {}).get("max_steps") if isinstance(recovery.get("recipe"), dict) else None,
                "stage2_token_presentations": 163_840_000,
            },
        )
    rows = [summarize_job(job, args.reference_predictions) for job in downstream_jobs]
    complete_rows = [row for row in rows if row["paired"]["status"] == "pass"]
    passed_rows = [row for row in complete_rows if row["passed_fp_recovery_gate"]]
    return {
        "schema": "bitdistill-controlled-curve-audit-v1",
        "date": DATE,
        "submission_json": str(args.submission_json),
        "reference_predictions": str(args.reference_predictions),
        "success_delta_from_fp": SUCCESS_DELTA_FROM_FP,
        "rows": rows,
        "complete": len(complete_rows),
        "expected": len(rows),
        "passed_fp_recovery_gate": len(passed_rows),
        "all_complete": len(complete_rows) == len(rows) and bool(rows),
        "all_passed_fp_recovery_gate": len(passed_rows) == len(rows) and bool(rows),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    if summary["all_passed_fp_recovery_gate"]:
        verdict = "All controlled Stage-2 curve rows meet the configured FP16 recovery gate."
    elif summary["all_complete"]:
        verdict = "Controlled Stage-2 curve rows are complete, but at least one row misses the FP16 recovery gate."
    else:
        verdict = "Pending: at least one controlled Stage-2 curve row lacks metrics or prediction traces."
    rows = []
    for row in summary["rows"]:
        paired = row["paired"]
        rows.append(
            [
                row["job_id"],
                row["label"],
                row["squeue"].get("state"),
                row["stage2_token_presentations"],
                row["paper_stage2_fraction"],
                row["metrics_exists"],
                row["predictions_exists"],
                row["metric_accuracy"],
                paired.get("delta_vs_reference"),
                paired.get("paired_ci95"),
                row["passed_fp_recovery_gate"],
                "; ".join(paired.get("errors", [])),
            ]
        )
    loss_rows = []
    for row in summary["rows"]:
        last = row["last"]
        loss_rows.append(
            [
                row["job_id"],
                row["label"],
                last["ce"],
                last["logit_kd"],
                last["weighted_logit_kd"],
                last["attention_kd"],
                last["weighted_attention_kd"],
            ]
        )
    return "\n\n".join(
        [
            f"# BitDistill Controlled Stage-2 Curve Audit, {summary['date']}",
            verdict,
            md_table(
                ["field", "value"],
                [
                    ["reference_predictions", summary["reference_predictions"]],
                    ["complete", f"{summary['complete']}/{summary['expected']}"],
                    ["passed FP recovery gate", f"{summary['passed_fp_recovery_gate']}/{summary['expected']}"],
                    ["success delta from FP16", summary["success_delta_from_fp"]],
                ],
            ),
            "## Rows",
            md_table(
                [
                    "job",
                    "label",
                    "state",
                    "Stage-2 tokens",
                    "paper fraction",
                    "metrics",
                    "predictions",
                    "accuracy",
                    "delta vs FP16",
                    "paired CI95",
                    "passes gate",
                    "errors",
                ],
                rows,
            ),
            "## Loss Components",
            md_table(
                ["job", "label", "CE", "logit KD", "weighted logit KD", "attention KD", "weighted attention KD"],
                loss_rows,
            ),
            "## Interpretation",
            (
                "These rows are the controlled budget test. They should be interpreted only "
                "after full MNLI prediction traces exist for each output directory."
            ),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission-json",
        type=Path,
        default=Path(f"benchmark_results/bitdistill_stage2_curve_submission_{DATE}.json"),
    )
    parser.add_argument(
        "--reference-predictions",
        type=Path,
        default=Path("checkpoints/bitdistill-glue-seqcls-predtrace/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1/eval_predictions.jsonl"),
    )
    parser.add_argument(
        "--recovery-submission-json",
        type=Path,
        default=Path(f"benchmark_results/bitdistill_recovery_submission_{DATE}.json"),
    )
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_controlled_curve_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_controlled_curve_{DATE}.md"))
    args = parser.parse_args()

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
