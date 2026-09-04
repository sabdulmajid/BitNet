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
EXPECTED_SEED = 1234
Z_95 = 1.959963984540054

CASES = (
    "seqcls-cosine-s8-fixed",
    "seqcls-cosine-s1-fixed",
    "seqcls-scaled-dot-s1-fixed",
    "seqcls-cosine-s1-adaptive",
    "causal-cosine-s1-fixed",
    "causal-cosine-s1-adaptive",
)

PAIRED_COMPARISONS = (
    ("seqcls_split_s1_minus_s8", "seqcls-cosine-s8-fixed", "seqcls-cosine-s1-fixed"),
    ("seqcls_scaled_dot_minus_cosine", "seqcls-cosine-s1-fixed", "seqcls-scaled-dot-s1-fixed"),
    ("seqcls_adaptive_minus_fixed", "seqcls-cosine-s1-fixed", "seqcls-cosine-s1-adaptive"),
    ("causal_adaptive_minus_fixed", "causal-cosine-s1-fixed", "causal-cosine-s1-adaptive"),
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


def load_predictions(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], [f"missing {path}"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_number}: invalid json: {exc}")
            continue
        if not isinstance(row, dict):
            errors.append(f"{path}:{line_number}: expected object")
            continue
        index = row.get("index")
        label = row.get("label")
        prediction = row.get("prediction")
        if not isinstance(index, int) or not isinstance(label, int) or not isinstance(prediction, int):
            errors.append(f"{path}:{line_number}: index, label, and prediction must be integers")
            continue
        if index in seen:
            errors.append(f"{path}:{line_number}: duplicate index {index}")
            continue
        seen.add(index)
        if row.get("correct") is not (label == prediction):
            errors.append(f"{path}:{line_number}: correct flag disagrees with label/prediction")
            continue
        rows.append(row)
    rows.sort(key=lambda row: int(row["index"]))
    for expected_index, row in enumerate(rows):
        if int(row["index"]) != expected_index:
            errors.append(f"{path}: non-contiguous index at row {expected_index}, saw {row['index']}")
            break
    return rows, errors


def paired_ci(values: list[float]) -> list[float] | None:
    if len(values) <= 1:
        return None
    mean = statistics.fmean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half_width = Z_95 * math.sqrt(variance / len(values))
    return [mean - half_width, mean + half_width]


def binomial_tail(n: int, *, lower_k: int | None = None, upper_k: int | None = None) -> float:
    if n <= 0:
        return 1.0
    if lower_k is None and upper_k is None:
        raise ValueError("lower_k or upper_k is required")
    start = 0 if lower_k is not None else int(upper_k)
    end = int(lower_k) if lower_k is not None else n
    if start > end:
        return 0.0
    terms = [
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2.0)
        for k in range(start, end + 1)
    ]
    peak = max(terms)
    return min(1.0, math.exp(peak) * sum(math.exp(term - peak) for term in terms))


def exact_mcnemar_pvalue(candidate_wins: int, reference_wins: int) -> float:
    discordant = candidate_wins + reference_wins
    if discordant == 0:
        return 1.0
    low = min(candidate_wins, reference_wins)
    high = max(candidate_wins, reference_wins)
    return min(
        1.0,
        2.0
        * min(
            binomial_tail(discordant, lower_k=low),
            binomial_tail(discordant, upper_k=high),
        ),
    )


def compare_predictions(root: Path, reference: str, candidate: str) -> dict[str, Any]:
    reference_rows, reference_errors = load_predictions(root / reference / "eval_predictions.jsonl")
    candidate_rows, candidate_errors = load_predictions(root / candidate / "eval_predictions.jsonl")
    errors = reference_errors + candidate_errors
    if not errors and len(reference_rows) != EXPECTED_EVAL_EXAMPLES:
        errors.append(f"reference rows={len(reference_rows)}, expected={EXPECTED_EVAL_EXAMPLES}")
    if not errors and len(candidate_rows) != EXPECTED_EVAL_EXAMPLES:
        errors.append(f"candidate rows={len(candidate_rows)}, expected={EXPECTED_EVAL_EXAMPLES}")

    reference_correct = 0
    candidate_correct = 0
    reference_wins = 0
    candidate_wins = 0
    differences: list[float] = []
    if not errors:
        for reference_row, candidate_row in zip(reference_rows, candidate_rows):
            if reference_row["index"] != candidate_row["index"]:
                errors.append(f"index mismatch at {reference_row['index']} and {candidate_row['index']}")
                break
            if reference_row["label"] != candidate_row["label"]:
                errors.append(f"label mismatch at index {reference_row['index']}")
                break
            reference_ok = bool(reference_row["correct"])
            candidate_ok = bool(candidate_row["correct"])
            reference_correct += int(reference_ok)
            candidate_correct += int(candidate_ok)
            reference_wins += int(reference_ok and not candidate_ok)
            candidate_wins += int(candidate_ok and not reference_ok)
            differences.append(float(candidate_ok) - float(reference_ok))

    matched = len(differences)
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "reference": reference,
        "candidate": candidate,
        "matched": matched,
        "reference_accuracy": reference_correct / matched if matched else None,
        "candidate_accuracy": candidate_correct / matched if matched else None,
        "delta_candidate_minus_reference": statistics.fmean(differences) if differences else None,
        "paired_ci95": paired_ci(differences),
        "candidate_wins": candidate_wins,
        "reference_wins": reference_wins,
        "mcnemar_exact_p": (
            exact_mcnemar_pvalue(candidate_wins, reference_wins) if matched and not errors else None
        ),
    }


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


def summarize_case(root: Path, case: str, *, artifact_root_label: str = "") -> dict[str, Any]:
    case_dir = root / case
    metrics_path = case_dir / "metrics.json"
    telemetry_path = case_dir / "telemetry.jsonl"
    prediction_path = case_dir / "eval_predictions.jsonl"
    metrics = load_json(metrics_path)
    telemetry = load_jsonl(telemetry_path)
    predictions, prediction_errors = load_predictions(prediction_path)
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
    if metrics.get("seed") != EXPECTED_SEED:
        blockers.append(f"expected seed {EXPECTED_SEED}")
    if len(predictions) != EXPECTED_EVAL_EXAMPLES:
        blockers.append(f"expected {EXPECTED_EVAL_EXAMPLES} prediction rows")
    blockers.extend(prediction_errors)
    display_dir = f"{artifact_root_label.rstrip('/')}/{case}" if artifact_root_label else str(case_dir)
    return {
        "case": case,
        "status": "complete" if not blockers else "pending_or_invalid",
        "blockers": blockers,
        "metrics_path": f"{display_dir}/metrics.json",
        "telemetry_path": f"{display_dir}/telemetry.jsonl",
        "prediction_path": f"{display_dir}/eval_predictions.jsonl",
        "source_revision": metrics.get("source_revision"),
        "seed": metrics.get("seed"),
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


def build_report(
    root: Path,
    submission_job_id: str,
    *,
    evidence_scope: str = "reference_environment",
    environment_note: str = "",
    artifact_root_label: str = "",
) -> dict[str, Any]:
    rows = [summarize_case(root, case, artifact_root_label=artifact_root_label) for case in CASES]
    complete = all(row["status"] == "complete" for row in rows)
    revisions = sorted({str(row["source_revision"]) for row in rows if row.get("source_revision")})
    paired = [
        {"comparison": name, **compare_predictions(root, reference, candidate)}
        for name, reference, candidate in PAIRED_COMPARISONS
    ]
    complete = complete and all(item["status"] == "pass" for item in paired)
    return {
        "schema": "bitdistill-method-parity-pilots-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_diagnostic" if complete else "pending_or_invalid",
        "quality_claim": "none_diagnostic_subset_only",
        "evidence_scope": evidence_scope,
        "environment_note": environment_note,
        "artifact_root_label": artifact_root_label,
        "submission_job_id": submission_job_id,
        "expected": {
            "steps_per_case": EXPECTED_STEPS,
            "eval_examples_per_case": EXPECTED_EVAL_EXAMPLES,
            "telemetry_rows_per_case": EXPECTED_TELEMETRY_ROWS,
            "telemetry_steps_per_case": list(EXPECTED_TELEMETRY_STEPS),
        },
        "source_revisions": revisions,
        "rows": rows,
        "paired_comparisons": paired,
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
    paired_rows = [
        [
            item["comparison"],
            item["status"],
            item["matched"],
            item["delta_candidate_minus_reference"],
            item["paired_ci95"],
            item["candidate_wins"],
            item["reference_wins"],
            item["mcnemar_exact_p"],
        ]
        for item in report["paired_comparisons"]
    ]
    sections = [
        "# BitDistill Method-Parity Pilots",
        f"Generated: `{report['created_utc']}`",
        f"Status: **{report['status']}**.",
        f"Quality claim: **{report['quality_claim']}**.",
        f"Evidence scope: **{report['evidence_scope']}**.",
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
        "## Paired Diagnostics",
        table(
            [
                "comparison",
                "status",
                "n",
                "delta",
                "paired 95% CI",
                "candidate wins",
                "reference wins",
                "McNemar p",
            ],
            paired_rows,
        ),
        "## Decision Rule",
        report["decision_rule"],
    ]
    if report.get("environment_note"):
        sections.insert(6, f"Environment note: {report['environment_note']}")
    return "\n\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("checkpoints/bitdistill-method-parity"))
    parser.add_argument("--submission-job-id", default="")
    parser.add_argument("--evidence-scope", default="reference_environment")
    parser.add_argument("--environment-note", default="")
    parser.add_argument("--artifact-root-label", default="")
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
    report = build_report(
        args.root,
        args.submission_job_id,
        evidence_scope=args.evidence_scope,
        environment_note=args.environment_note,
        artifact_root_label=args.artifact_root_label,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0 if report["status"] == "complete_diagnostic" else 3


if __name__ == "__main__":
    raise SystemExit(main())
