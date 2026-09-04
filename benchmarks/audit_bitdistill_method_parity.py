#!/usr/bin/env python3
"""Summarize the bounded BitDistill formulation/relation telemetry pilots."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
EXPECTED_STEPS = 120
EXPECTED_EVAL_EXAMPLES = 512
EXPECTED_TELEMETRY_STEPS = (1, 20, 40, 60, 80, 100, 120)
EXPECTED_TELEMETRY_ROWS = len(EXPECTED_TELEMETRY_STEPS)

CASES = (
    "seqcls-cosine-s8-fixed",
    "seqcls-cosine-s1-fixed",
    "seqcls-scaled-dot-s1-fixed",
    "seqcls-cosine-s1-adaptive",
    "causal-cosine-s1-fixed",
    "causal-cosine-s1-adaptive",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def summarize_case(root: Path, case: str) -> dict[str, Any]:
    case_dir = root / case
    metrics_path = case_dir / "metrics.json"
    telemetry_path = case_dir / "telemetry.jsonl"
    metrics = load_json(metrics_path)
    telemetry = load_jsonl(telemetry_path)
    gradient_ratios: list[float] = []
    loss_ratios: list[float] = []
    effective_weights: list[float] = []
    for row in telemetry:
        norms = row.get("component_grad_norms_microbatch", {})
        loss = row.get("loss", {})
        if isinstance(norms, dict):
            ce_norm = finite(norms.get("ce"))
            attention_norm = finite(norms.get("weighted_attention_kd"))
            if ce_norm not in (None, 0.0) and attention_norm is not None:
                gradient_ratios.append(attention_norm / ce_norm)
        if isinstance(loss, dict):
            ce = finite(loss.get("ce"))
            weighted_attention = finite(loss.get("weighted_attention_kd"))
            weight = finite(loss.get("effective_attention_kd_weight"))
            if ce not in (None, 0.0) and weighted_attention is not None:
                loss_ratios.append(weighted_attention / ce)
            if weight is not None:
                effective_weights.append(weight)

    eval_metrics = metrics.get("eval", {}) if isinstance(metrics.get("eval"), dict) else {}
    loss_weights = metrics.get("loss_weights", {}) if isinstance(metrics.get("loss_weights"), dict) else {}
    formulation = (
        metrics.get("task_formulation_contract", {})
        if isinstance(metrics.get("task_formulation_contract"), dict)
        else {}
    )
    blockers: list[str] = []
    if int(metrics.get("steps", 0) or 0) != EXPECTED_STEPS:
        blockers.append(f"expected {EXPECTED_STEPS} steps")
    if int(eval_metrics.get("eval_examples", 0) or 0) != EXPECTED_EVAL_EXAMPLES:
        blockers.append(f"expected {EXPECTED_EVAL_EXAMPLES} diagnostic eval examples")
    if len(telemetry) != EXPECTED_TELEMETRY_ROWS:
        blockers.append(f"expected {EXPECTED_TELEMETRY_ROWS} telemetry rows")
    telemetry_steps = [int(row.get("step", -1) or -1) for row in telemetry]
    if telemetry_steps != list(EXPECTED_TELEMETRY_STEPS):
        blockers.append(f"expected telemetry steps {list(EXPECTED_TELEMETRY_STEPS)}")
    if not gradient_ratios:
        blockers.append("missing finite component-gradient ratios")
    if not metrics.get("source_revision"):
        blockers.append("missing source revision")
    return {
        "case": case,
        "status": "complete" if not blockers else "pending_or_invalid",
        "blockers": blockers,
        "metrics_path": str(metrics_path),
        "telemetry_path": str(telemetry_path),
        "source_revision": metrics.get("source_revision"),
        "task_format": metrics.get("task_format"),
        "task_contract": formulation,
        "relation_mode": loss_weights.get("attention_relation_mode"),
        "split_heads": metrics.get("attention_split_heads"),
        "balance_strategy": loss_weights.get("attention_kd_balance"),
        "configured_attention_weight": loss_weights.get("attention_kd_weight"),
        "gradient_attention_to_ce": summarize(gradient_ratios),
        "loss_attention_to_ce": summarize(loss_ratios),
        "effective_attention_weight": summarize(effective_weights),
        "diagnostic_accuracy": finite(eval_metrics.get("accuracy")),
        "diagnostic_eval_examples": int(eval_metrics.get("eval_examples", 0) or 0),
        "telemetry_rows": len(telemetry),
    }


def build_report(root: Path, submission_job_id: str) -> dict[str, Any]:
    rows = [summarize_case(root, case) for case in CASES]
    complete = all(row["status"] == "complete" for row in rows)
    revisions = sorted({str(row["source_revision"]) for row in rows if row.get("source_revision")})
    return {
        "schema": "bitdistill-method-parity-pilots-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_diagnostic" if complete else "pending_or_invalid",
        "quality_claim": "none_diagnostic_subset_only",
        "submission_job_id": submission_job_id,
        "expected": {
            "steps_per_case": EXPECTED_STEPS,
            "eval_examples_per_case": EXPECTED_EVAL_EXAMPLES,
            "telemetry_rows_per_case": EXPECTED_TELEMETRY_ROWS,
            "telemetry_steps_per_case": list(EXPECTED_TELEMETRY_STEPS),
        },
        "source_revisions": revisions,
        "rows": rows,
        "decision_rule": (
            "Use these pilots to reject numerically unstable contracts and verify adaptive balancing. "
            "Do not select a downstream-quality winner from 512 examples or 120 steps. A full run "
            "requires an explicit paper-definition choice, all 9,815 MNLI examples, paired predictions, and replication."
        ),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "none"
    return str(value)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(fmt(value).replace("|", "\\|") for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    rows = [
        [
            row["case"],
            row["status"],
            row["task_format"],
            row["relation_mode"],
            row["split_heads"],
            row["balance_strategy"],
            row["gradient_attention_to_ce"]["median"],
            row["gradient_attention_to_ce"]["max"],
            row["effective_attention_weight"]["median"],
            row["diagnostic_accuracy"],
            row["blockers"],
        ]
        for row in report["rows"]
    ]
    return "\n\n".join(
        [
            "# BitDistill Method-Parity Pilots",
            f"Generated: `{report['created_utc']}`",
            f"Status: **{report['status']}**.",
            f"Quality claim: **{report['quality_claim']}**.",
            "These bounded pilots compare numerical contracts. Their partial evaluation is not a task benchmark.",
            table(
                [
                    "case",
                    "status",
                    "task format",
                    "relation",
                    "split",
                    "balance",
                    "median grad AD/CE",
                    "max grad AD/CE",
                    "median gamma",
                    "diagnostic accuracy",
                    "blockers",
                ],
                rows,
            ),
            "## Decision Rule",
            report["decision_rule"],
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("checkpoints/bitdistill-method-parity"))
    parser.add_argument("--submission-job-id", default="")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"benchmarks/results/bitdistill_method_parity_{DATE}.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"benchmarks/results/bitdistill_method_parity_{DATE}.md"),
    )
    args = parser.parse_args()
    report = build_report(args.root, args.submission_job_id)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0 if report["status"] == "complete_diagnostic" else 3


if __name__ == "__main__":
    raise SystemExit(main())
