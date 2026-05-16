#!/usr/bin/env python3
"""Audit the focused BitDistill gamma-60 diagnostic.

The diagnostic tests one narrow hypothesis: the paper's attention-KD gamma
(`100000`) is not numerically portable under the local loss reductions because
it dominates CE by thousands of times.  This audit compares the focused
`gamma=60` run only against the matched 20k-warmup paper-gamma control and the
local FP16-SFT MNLI reference.

It is safe to run while the Slurm job is still active.  Until metrics and
prediction traces exist, the report remains pending and records only live loss
balance from the job log.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
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
LOG_VALUE_RE = re.compile(r"([a-zA-Z_]+)=([-+0-9.eE]+)")


def summarize_values(values: list[float]) -> dict[str, float | None]:
    finite_values = sorted(value for value in values if math.isfinite(value))
    if not finite_values:
        return {"min": None, "p50": None, "p95": None, "max": None, "mean": None}

    def percentile(probability: float) -> float:
        if len(finite_values) == 1:
            return finite_values[0]
        position = probability * (len(finite_values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(finite_values) - 1)
        fraction = position - lower
        return finite_values[lower] * (1.0 - fraction) + finite_values[upper] * fraction

    return {
        "min": finite_values[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": finite_values[-1],
        "mean": sum(finite_values) / len(finite_values),
    }


def parse_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}

    latest: dict[str, float] = {}
    weighted_attention_to_ce: list[float] = []
    weighted_logit_to_ce: list[float] = []
    equalizing_gamma: list[float] = []
    parsed_steps = 0

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("step="):
            continue
        values: dict[str, float] = {}
        for key, raw in LOG_VALUE_RE.findall(line):
            try:
                values[key] = float(raw)
            except ValueError:
                continue
        if "step" not in values:
            continue
        parsed_steps += 1
        latest = values
        ce = values.get("ce")
        attention = values.get("attention_kd")
        weighted_attention = values.get("weighted_attention_kd")
        weighted_logit = values.get("weighted_logit_kd")
        if ce not in (None, 0.0) and weighted_attention is not None:
            weighted_attention_to_ce.append(weighted_attention / ce)
        if ce not in (None, 0.0) and weighted_logit is not None:
            weighted_logit_to_ce.append(weighted_logit / ce)
        if ce not in (None, 0.0) and attention not in (None, 0.0):
            equalizing_gamma.append(ce / attention)

    latest_ce = latest.get("ce")
    latest_weighted_attention = latest.get("weighted_attention_kd")
    latest_weighted_logit = latest.get("weighted_logit_kd")
    return {
        "exists": True,
        "path": str(path),
        "parsed_steps": parsed_steps,
        "latest": latest,
        "latest_step": int(latest["step"]) if "step" in latest else None,
        "latest_weighted_attention_to_ce": (
            latest_weighted_attention / latest_ce
            if latest_ce not in (None, 0.0) and latest_weighted_attention is not None
            else None
        ),
        "latest_weighted_logit_to_ce": (
            latest_weighted_logit / latest_ce
            if latest_ce not in (None, 0.0) and latest_weighted_logit is not None
            else None
        ),
        "weighted_attention_to_ce_summary": summarize_values(weighted_attention_to_ce),
        "weighted_logit_to_ce_summary": summarize_values(weighted_logit_to_ce),
        "ce_attention_equalizing_gamma_summary": summarize_values(equalizing_gamma),
    }


def squeue_state(job_id: str) -> dict[str, str]:
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


def metric_accuracy(metrics_path: Path) -> float | None:
    metrics = read_json(metrics_path)
    eval_metrics = metrics.get("eval", {}) if isinstance(metrics.get("eval"), dict) else {}
    return finite(eval_metrics.get("accuracy"))


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    candidate_metrics = args.candidate_dir / "metrics.json"
    candidate_predictions = args.candidate_dir / "eval_predictions.jsonl"
    baseline_metrics = args.baseline_dir / "metrics.json"
    baseline_predictions = args.baseline_dir / "eval_predictions.jsonl"

    candidate_accuracy = metric_accuracy(candidate_metrics)
    baseline_accuracy = metric_accuracy(baseline_metrics)
    fp_comparison = compare(args.fp_predictions, candidate_predictions)
    baseline_comparison = compare(baseline_predictions, candidate_predictions)

    completed = (
        candidate_metrics.exists()
        and candidate_predictions.exists()
        and fp_comparison["status"] == "pass"
        and baseline_comparison["status"] == "pass"
    )
    delta_vs_fp = fp_comparison.get("delta_vs_reference")
    delta_vs_baseline = baseline_comparison.get("delta_vs_reference")
    improves_over_baseline = isinstance(delta_vs_baseline, float) and delta_vs_baseline > 0.0
    passes_fp_recovery_gate = isinstance(delta_vs_fp, float) and delta_vs_fp >= SUCCESS_DELTA_FROM_FP

    if not completed:
        status = "pending"
    elif passes_fp_recovery_gate:
        status = "completed_fp_gate_pass"
    elif improves_over_baseline:
        status = "completed_baseline_improved"
    else:
        status = "completed_no_quality_gain"

    return {
        "schema": "bitdistill-gamma60-diagnostic-v1",
        "date": DATE,
        "status": status,
        "job_id": args.job_id,
        "purpose": (
            "Test whether reducing attention-KD gamma from 100000 to 60 improves the matched "
            "20k-warmup tensor-scale MNLI BitDistill row under local loss normalization."
        ),
        "candidate_dir": str(args.candidate_dir),
        "baseline_dir": str(args.baseline_dir),
        "fp_predictions": str(args.fp_predictions),
        "candidate_metrics_exists": candidate_metrics.exists(),
        "candidate_predictions_exists": candidate_predictions.exists(),
        "baseline_metrics_exists": baseline_metrics.exists(),
        "baseline_predictions_exists": baseline_predictions.exists(),
        "expected_eval_examples": EXPECTED_MNLI,
        "success_delta_from_fp": SUCCESS_DELTA_FROM_FP,
        "candidate_accuracy": candidate_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "delta_accuracy_vs_baseline_metric": (
            candidate_accuracy - baseline_accuracy
            if isinstance(candidate_accuracy, float) and isinstance(baseline_accuracy, float)
            else None
        ),
        "fp_comparison": fp_comparison,
        "baseline_comparison": baseline_comparison,
        "improves_over_paper_gamma_baseline": improves_over_baseline,
        "passes_fp_recovery_gate": passes_fp_recovery_gate,
        "live_log": parse_log(args.log_path),
        "squeue": squeue_state(args.job_id),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    if summary["status"] == "pending":
        verdict = "Pending: the gamma-60 job has not produced full metrics and prediction traces yet."
    elif summary["status"] == "completed_fp_gate_pass":
        verdict = "Gamma-60 passes the configured FP16 recovery gate; this would be a major loss-normalization result."
    elif summary["status"] == "completed_baseline_improved":
        verdict = "Gamma-60 improves over the matched paper-gamma control but still misses the FP16 recovery gate."
    else:
        verdict = "Gamma-60 does not improve over the matched paper-gamma control."

    live = summary["live_log"]
    wa = live.get("weighted_attention_to_ce_summary", {}) if isinstance(live, dict) else {}
    wl = live.get("weighted_logit_to_ce_summary", {}) if isinstance(live, dict) else {}
    eq = live.get("ce_attention_equalizing_gamma_summary", {}) if isinstance(live, dict) else {}

    lines = [
        f"# BitDistill Gamma-60 Diagnostic Audit, {summary['date']}",
        "",
        verdict,
        "",
        "This is a focused loss-normalization diagnostic, not a broad sweep and not a paper-reproduction claim by itself.",
        "",
        "## Run State",
        "",
        md_table(
            ["field", "value"],
            [
                ["job id", summary["job_id"]],
                ["squeue state", summary["squeue"].get("state")],
                ["squeue elapsed", summary["squeue"].get("elapsed")],
                ["candidate metrics", summary["candidate_metrics_exists"]],
                ["candidate predictions", summary["candidate_predictions_exists"]],
                ["candidate accuracy", summary["candidate_accuracy"]],
                ["matched paper-gamma accuracy", summary["baseline_accuracy"]],
                ["metric delta vs paper-gamma", summary["delta_accuracy_vs_baseline_metric"]],
                ["paired delta vs FP16", summary["fp_comparison"].get("delta_vs_reference")],
                ["paired delta vs paper-gamma", summary["baseline_comparison"].get("delta_vs_reference")],
                ["passes FP recovery gate", summary["passes_fp_recovery_gate"]],
                ["improves over paper-gamma", summary["improves_over_paper_gamma_baseline"]],
            ],
        ),
        "",
        "## Live Loss Balance",
        "",
        md_table(
            ["quantity", "latest/p50/p95"],
            [
                ["latest step", live.get("latest_step")],
                ["latest weighted attention / CE", live.get("latest_weighted_attention_to_ce")],
                ["weighted attention / CE p50", wa.get("p50")],
                ["weighted attention / CE p95", wa.get("p95")],
                ["latest weighted logits / CE", live.get("latest_weighted_logit_to_ce")],
                ["weighted logits / CE p50", wl.get("p50")],
                ["CE/attention equalizing gamma p50", eq.get("p50")],
                ["CE/attention equalizing gamma p95", eq.get("p95")],
            ],
        ),
        "",
        "## Interpretation Gate",
        "",
        (
            "Compare this row only against the matched 20k-warmup paper-gamma control. "
            "If the final paired delta vs paper-gamma is positive, loss normalization is a likely primary blocker. "
            "If it is non-positive, attention-KD dominance is still a measured risk but not sufficient to explain the quality gap."
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--job-id", default="10077")
    parser.add_argument("--candidate-dir", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-steps10000-lr2em5-gamma60-headinit"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-steps10000-lr2em5-papergamma-headinit"))
    parser.add_argument("--fp-predictions", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-predtrace/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1/eval_predictions.jsonl"))
    parser.add_argument("--log-path", type=Path, default=Path("logs/bitdistill-gamma60-10077.out"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_gamma60_diagnostic_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_gamma60_diagnostic_{DATE}.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    for field in ("candidate_dir", "baseline_dir", "fp_predictions", "log_path", "output_json", "output_md"):
        path = getattr(args, field)
        if isinstance(path, Path) and not path.is_absolute():
            setattr(args, field, root / path)

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
