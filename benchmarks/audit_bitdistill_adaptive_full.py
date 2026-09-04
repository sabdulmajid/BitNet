#!/usr/bin/env python3
"""Fail-closed audit for the three-seed adaptive BitDistill MNLI gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SEEDS = (1234, 1235, 1236)
EXPECTED_SOURCE_REVISION = "526ede7b2c3f33c6a9638de54bdae91e8afe39c6"
EXPECTED_STEPS = 10_000
EXPECTED_EVAL_EXAMPLES = 9_815
EXPECTED_TELEMETRY_STEPS = (1, *range(500, EXPECTED_STEPS + 1, 500))
FP16_ACCURACY = 0.808151
FIXED_GAMMA_ACCURACY = 0.729903
RECOVERY_FLOOR = FP16_ACCURACY - 0.01
T_95_DF2 = 4.302652729696142
Z_95 = 1.959963984540054


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


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


def read_telemetry_steps(path: Path) -> tuple[list[int], list[str]]:
    if not path.exists():
        return [], [f"missing {path}"]
    steps: list[int] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_number}: invalid json: {exc}")
            continue
        step = row.get("step") if isinstance(row, dict) else None
        if not isinstance(step, int):
            errors.append(f"{path}:{line_number}: missing integer step")
            continue
        steps.append(step)
    return steps, errors


def compare_prediction_files(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference_rows, reference_errors = load_predictions(reference_path)
    candidate_rows, candidate_errors = load_predictions(candidate_path)
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
    delta = statistics.fmean(differences) if differences else None
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "matched": matched,
        "reference_accuracy": reference_correct / matched if matched else None,
        "candidate_accuracy": candidate_correct / matched if matched else None,
        "delta_candidate_minus_reference": delta,
        "paired_ci95": paired_ci(differences),
        "candidate_wins": candidate_wins,
        "reference_wins": reference_wins,
        "mcnemar_exact_p": (
            exact_mcnemar_pvalue(candidate_wins, reference_wins) if matched and not errors else None
        ),
    }


def nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def summarize_run(root: Path, seed: int, fp16_predictions: Path, fixed_predictions: Path) -> dict[str, Any]:
    run_dir = root / f"mnli-seqcls-cosine-s1-adaptive-seed{seed}"
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "eval_predictions.jsonl"
    telemetry_path = run_dir / "telemetry.jsonl"
    metrics = read_json(metrics_path)
    telemetry_steps, telemetry_errors = read_telemetry_steps(telemetry_path)
    predictions, prediction_errors = load_predictions(predictions_path)
    blockers = list(telemetry_errors) + list(prediction_errors)

    expected_fields = {
        "source_revision": (metrics.get("source_revision"), EXPECTED_SOURCE_REVISION),
        "seed": (metrics.get("seed"), seed),
        "steps": (metrics.get("steps"), EXPECTED_STEPS),
        "eval_examples": (nested(metrics, "eval", "eval_examples"), float(EXPECTED_EVAL_EXAMPLES)),
        "task_format": (metrics.get("task_format"), "sequence_classification"),
        "scale_mode": (metrics.get("scale_mode"), "tensor"),
        "attention_relation_mode": (nested(metrics, "loss_weights", "attention_relation_mode"), "cosine"),
        "attention_split_heads": (metrics.get("attention_split_heads"), 1),
        "attention_kd_balance": (nested(metrics, "loss_weights", "attention_kd_balance"), "gradnorm_ema"),
    }
    for field, (actual, expected) in expected_fields.items():
        if actual != expected:
            blockers.append(f"{field}={actual!r}, expected={expected!r}")
    if telemetry_steps != list(EXPECTED_TELEMETRY_STEPS):
        blockers.append(f"telemetry steps={telemetry_steps}, expected={list(EXPECTED_TELEMETRY_STEPS)}")
    if len(predictions) != EXPECTED_EVAL_EXAMPLES:
        blockers.append(f"prediction rows={len(predictions)}, expected={EXPECTED_EVAL_EXAMPLES}")

    accuracy = nested(metrics, "eval", "accuracy")
    return {
        "seed": seed,
        "status": "complete" if not blockers else "pending_or_invalid",
        "blockers": blockers,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "telemetry_path": str(telemetry_path),
        "accuracy": float(accuracy) if isinstance(accuracy, (int, float)) and math.isfinite(accuracy) else None,
        "effective_attention_weight": nested(metrics, "loss_weights", "effective_attention_kd_weight"),
        "vs_fp16": compare_prediction_files(fp16_predictions, predictions_path),
        "vs_fixed_gamma_655m": compare_prediction_files(fixed_predictions, predictions_path),
    }


def seed_mean_ci(values: list[float]) -> list[float] | None:
    if len(values) <= 1:
        return None
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    return [mean - T_95_DF2 * standard_error, mean + T_95_DF2 * standard_error]


def build_report(root: Path, fp16_predictions: Path, fixed_predictions: Path) -> dict[str, Any]:
    runs = [summarize_run(root, seed, fp16_predictions, fixed_predictions) for seed in SEEDS]
    complete = all(run["status"] == "complete" for run in runs)
    accuracies = [run["accuracy"] for run in runs if isinstance(run.get("accuracy"), float)]
    mean_accuracy = statistics.fmean(accuracies) if len(accuracies) == len(SEEDS) else None
    all_fixed_improvements = complete and all(
        isinstance(run["vs_fixed_gamma_655m"].get("paired_ci95"), list)
        and run["vs_fixed_gamma_655m"]["paired_ci95"][0] > 0.0
        for run in runs
    )
    recovery_met = mean_accuracy is not None and mean_accuracy >= RECOVERY_FLOOR
    return {
        "schema": "bitdistill-adaptive-full-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if complete else "pending_or_invalid",
        "claim_boundary": "Three-seed cross-environment MNLI quality gate; not a paper-exact reproduction.",
        "expected": {
            "source_revision": EXPECTED_SOURCE_REVISION,
            "seeds": list(SEEDS),
            "steps": EXPECTED_STEPS,
            "eval_examples": EXPECTED_EVAL_EXAMPLES,
            "telemetry_steps": list(EXPECTED_TELEMETRY_STEPS),
        },
        "references": {
            "fp16_accuracy": FP16_ACCURACY,
            "fixed_gamma_655m_accuracy": FIXED_GAMMA_ACCURACY,
            "recovery_floor": RECOVERY_FLOOR,
        },
        "runs": runs,
        "aggregate": {
            "completed_seeds": len(accuracies),
            "mean_accuracy": mean_accuracy,
            "sample_standard_deviation": statistics.stdev(accuracies) if len(accuracies) > 1 else None,
            "seed_mean_t_ci95": seed_mean_ci(accuracies) if len(accuracies) == len(SEEDS) else None,
        },
        "decisions": {
            "all_runs_complete": complete,
            "all_seeds_significantly_improve_over_fixed_gamma_655m": all_fixed_improvements,
            "mean_within_one_point_of_fp16": recovery_met,
            "quality_improvement_gate": "pass" if all_fixed_improvements else ("fail" if complete else "pending"),
            "paper_level_local_recovery_gate": "pass" if recovery_met else ("fail" if complete else "pending"),
        },
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return "[" + ", ".join(fmt(item) for item in value) + "]"
    return str(value)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(fmt(value).replace("|", "\\|") for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for run in report["runs"]:
        fixed = run["vs_fixed_gamma_655m"]
        fp16 = run["vs_fp16"]
        rows.append(
            [
                run["seed"],
                run["status"],
                run["accuracy"],
                fixed["delta_candidate_minus_reference"],
                fixed["paired_ci95"],
                fixed["mcnemar_exact_p"],
                fp16["delta_candidate_minus_reference"],
                fp16["paired_ci95"],
                run["effective_attention_weight"],
                run["blockers"],
            ]
        )
    aggregate = report["aggregate"]
    decisions = report["decisions"]
    return "\n\n".join(
        [
            "# BitDistill Adaptive Full-Run Audit",
            f"Generated: `{report['created_utc']}`",
            f"Status: **{report['status']}**.",
            report["claim_boundary"],
            table(
                [
                    "seed",
                    "status",
                    "accuracy",
                    "delta vs fixed",
                    "paired CI vs fixed",
                    "McNemar vs fixed",
                    "delta vs FP16",
                    "paired CI vs FP16",
                    "final gamma",
                    "blockers",
                ],
                rows,
            ),
            "## Aggregate",
            table(
                ["completed seeds", "mean accuracy", "sample SD", "seed-mean t CI"],
                [[aggregate["completed_seeds"], aggregate["mean_accuracy"], aggregate["sample_standard_deviation"], aggregate["seed_mean_t_ci95"]]],
            ),
            "## Decisions",
            table(
                ["gate", "result"],
                [
                    ["all runs complete", decisions["all_runs_complete"]],
                    ["all seeds improve over fixed gamma with paired CI > 0", decisions["quality_improvement_gate"]],
                    ["three-seed mean within one point of FP16", decisions["paper_level_local_recovery_gate"]],
                ],
            ),
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("checkpoints/bitdistill-adaptive-full-replications"))
    parser.add_argument(
        "--fp16-predictions",
        type=Path,
        default=Path("checkpoints/bitdistill-glue-seqcls-predtrace/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1/eval_predictions.jsonl"),
    )
    parser.add_argument(
        "--fixed-predictions",
        type=Path,
        default=Path("checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit/eval_predictions.jsonl"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/bitdistill_adaptive_full_audit_2026-09-04.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/bitdistill_adaptive_full_audit_2026-09-04.md"),
    )
    args = parser.parse_args()
    report = build_report(args.root, args.fp16_predictions, args.fixed_predictions)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0 if report["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
