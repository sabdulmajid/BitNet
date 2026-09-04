#!/usr/bin/env python3
"""Fail-closed audit for the three-seed adaptive BitDistill MNLI gate."""

from __future__ import annotations

import argparse
import hashlib
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
HISTORICAL_GAMMA60_ACCURACY = 0.738462
RECOVERY_FLOOR = FP16_ACCURACY - 0.01
T_95_DF2 = 4.302652729696142
Z_95 = 1.959963984540054


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def read_telemetry(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], [f"missing {path}"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
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
        step = row.get("step")
        if not isinstance(step, int):
            errors.append(f"{path}:{line_number}: missing integer step")
            continue
        rows.append(row)
    return rows, errors


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def finite_values(rows: list[dict[str, Any]], *keys: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = nested(row, *keys)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def telemetry_health(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attention_weights = finite_values(rows, "loss", "effective_attention_kd_weight")
    clipped = finite_values(rows, "activation_quantization", "clipped_fraction")
    int8_edge = finite_values(rows, "activation_quantization", "int8_edge_fraction")
    flips = finite_values(rows, "quantization_dynamics", "flip_fraction")
    scale_deltas = finite_values(rows, "quantization_dynamics", "scale_abs_delta_max")
    attention_ratios: list[float] = []
    logit_ratios: list[float] = []
    probe_attention_ratios: list[float] = []
    global_to_last_controller_probe_ratios: list[float] = []
    for row in rows:
        ce_norm = nested(row, "component_grad_norms_microbatch", "ce")
        attention_norm = nested(row, "component_grad_norms_microbatch", "weighted_attention_kd")
        logit_norm = nested(row, "component_grad_norms_microbatch", "weighted_logit_kd")
        probe_ratio = nested(
            row,
            "attention_balance",
            "last",
            "predicted_weighted_attention_to_ce_gradient_ratio",
        )
        if isinstance(ce_norm, (int, float)) and math.isfinite(ce_norm) and ce_norm > 0.0:
            if isinstance(attention_norm, (int, float)) and math.isfinite(attention_norm):
                global_ratio = float(attention_norm / ce_norm)
                attention_ratios.append(global_ratio)
                if isinstance(probe_ratio, (int, float)) and math.isfinite(probe_ratio) and probe_ratio > 0.0:
                    global_to_last_controller_probe_ratios.append(global_ratio / float(probe_ratio))
            if isinstance(logit_norm, (int, float)) and math.isfinite(logit_norm):
                logit_ratios.append(float(logit_norm / ce_norm))
        if isinstance(probe_ratio, (int, float)) and math.isfinite(probe_ratio):
            probe_attention_ratios.append(float(probe_ratio))
    return {
        "points": len(rows),
        "attention_weight": {
            "first": attention_weights[0] if attention_weights else None,
            "final": attention_weights[-1] if attention_weights else None,
            "median": percentile(attention_weights, 0.5),
            "min": min(attention_weights) if attention_weights else None,
            "max": max(attention_weights) if attention_weights else None,
        },
        "weighted_attention_to_ce_gradient_ratio": {
            "final": attention_ratios[-1] if attention_ratios else None,
            "median": percentile(attention_ratios, 0.5),
            "p95": percentile(attention_ratios, 0.95),
            "max": max(attention_ratios) if attention_ratios else None,
        },
        "probe_weighted_attention_to_ce_gradient_ratio": {
            "final": probe_attention_ratios[-1] if probe_attention_ratios else None,
            "median": percentile(probe_attention_ratios, 0.5),
            "min": min(probe_attention_ratios) if probe_attention_ratios else None,
            "max": max(probe_attention_ratios) if probe_attention_ratios else None,
        },
        "global_to_last_controller_probe_ratio": {
            "final": global_to_last_controller_probe_ratios[-1]
            if global_to_last_controller_probe_ratios
            else None,
            "median": percentile(global_to_last_controller_probe_ratios, 0.5),
            "min": min(global_to_last_controller_probe_ratios)
            if global_to_last_controller_probe_ratios
            else None,
            "max": max(global_to_last_controller_probe_ratios)
            if global_to_last_controller_probe_ratios
            else None,
            "comparison_contract": (
                "descriptive_only: global norm uses the telemetry microbatch and all trainable "
                "parameters; probe ratio uses the most recent controller-update microbatch and "
                "selected-layer Q/K/V parameters"
            ),
        },
        "weighted_logit_to_ce_gradient_ratio": {
            "final": logit_ratios[-1] if logit_ratios else None,
            "median": percentile(logit_ratios, 0.5),
            "p95": percentile(logit_ratios, 0.95),
            "max": max(logit_ratios) if logit_ratios else None,
        },
        "max_activation_clipped_fraction": max(clipped) if clipped else None,
        "max_int8_edge_fraction": max(int8_edge) if int8_edge else None,
        "mean_ternary_flip_fraction": statistics.fmean(flips) if flips else None,
        "final_ternary_flip_fraction": flips[-1] if flips else None,
        "max_scale_abs_delta": max(scale_deltas) if scale_deltas else None,
    }


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


def run_contract_errors(metrics: dict[str, Any], seed: int) -> list[str]:
    expected_fields = {
        "source_revision": (metrics.get("source_revision"), EXPECTED_SOURCE_REVISION),
        "seed": (metrics.get("seed"), seed),
        "stage": (metrics.get("stage"), "task_sft"),
        "method": (metrics.get("method"), "bitdistill"),
        "task": (metrics.get("task"), "mnli"),
        "steps": (metrics.get("steps"), EXPECTED_STEPS),
        "eval_examples": (nested(metrics, "eval", "eval_examples"), float(EXPECTED_EVAL_EXAMPLES)),
        "task_format": (metrics.get("task_format"), "sequence_classification"),
        "label_scheme": (metrics.get("label_scheme"), "letters"),
        "candidate_score": (metrics.get("candidate_score"), "mean"),
        "scale_mode": (metrics.get("scale_mode"), "tensor"),
        "exclude_linear_regex": (metrics.get("exclude_linear_regex"), "score|classifier"),
        "distill_layer": (metrics.get("distill_layer"), -1),
        "attention_split_heads": (metrics.get("attention_split_heads"), 1),
        "activation_quantization": (nested(metrics, "preparation", "activation_quantization"), True),
        "bitlinear_replaced": (nested(metrics, "preparation", "bitlinear_replaced"), 168),
        "subln_inserted": (nested(metrics, "preparation", "subln_inserted"), 48),
        "state_loaded": (nested(metrics, "state_load", "loaded"), True),
        "output_head_copied": (nested(metrics, "output_head_init", "copied"), True),
        "max_train_samples": (nested(metrics, "training_budget", "max_train_samples"), 0),
        "max_eval_samples": (nested(metrics, "training_budget", "max_eval_samples"), 0),
        "max_seq_len": (nested(metrics, "training_budget", "max_seq_len"), 512),
        "per_device_batch_size": (nested(metrics, "training_budget", "per_device_batch_size"), 4),
        "grad_accum_steps": (nested(metrics, "training_budget", "grad_accum_steps"), 4),
        "max_steps": (nested(metrics, "training_budget", "max_steps"), EXPECTED_STEPS),
        "logit_kd_weight": (nested(metrics, "loss_weights", "logit_kd_weight"), 10.0),
        "attention_kd_weight": (nested(metrics, "loss_weights", "attention_kd_weight"), 100_000.0),
        "attention_kd_balance": (nested(metrics, "loss_weights", "attention_kd_balance"), "gradnorm_ema"),
        "attention_balance_target_ratio": (
            nested(metrics, "loss_weights", "attention_balance_target_ratio"),
            1.0,
        ),
        "attention_balance_beta": (nested(metrics, "loss_weights", "attention_balance_beta"), 0.9),
        "attention_balance_every_steps": (
            nested(metrics, "loss_weights", "attention_balance_every_steps"),
            20,
        ),
        "logit_temperature": (nested(metrics, "loss_weights", "logit_temperature"), 5.0),
        "logit_kd_temperature_scale": (
            nested(metrics, "loss_weights", "logit_kd_temperature_scale"),
            "none",
        ),
        "attention_temperature": (nested(metrics, "loss_weights", "attention_temperature"), 1.0),
        "attention_relation_mode": (nested(metrics, "loss_weights", "attention_relation_mode"), "cosine"),
        "attention_qkv_reduction": (nested(metrics, "loss_weights", "attention_qkv_reduction"), "sum"),
        "telemetry_every_steps": (nested(metrics, "telemetry", "every_steps"), 500),
        "telemetry_component_grad_norms": (nested(metrics, "telemetry", "component_grad_norms"), True),
        "telemetry_max_elements_per_layer": (
            nested(metrics, "telemetry", "max_elements_per_layer"),
            65_536,
        ),
    }
    errors = [
        f"{field}={actual!r}, expected={expected!r}"
        for field, (actual, expected) in expected_fields.items()
        if actual != expected
    ]
    state_path = nested(metrics, "state_load", "path")
    if not isinstance(state_path, str) or not state_path.endswith("/assets/stage2.pt"):
        errors.append(f"state_load.path={state_path!r}, expected suffix='/assets/stage2.pt'")
    return errors


def summarize_run(
    root: Path,
    seed: int,
    fp16_predictions: Path,
    fixed_predictions: Path,
    gamma60_predictions: Path,
) -> dict[str, Any]:
    run_dir = root / f"mnli-seqcls-cosine-s1-adaptive-seed{seed}"
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "eval_predictions.jsonl"
    telemetry_path = run_dir / "telemetry.jsonl"
    metrics = read_json(metrics_path)
    telemetry_rows, telemetry_errors = read_telemetry(telemetry_path)
    telemetry_steps = [int(row["step"]) for row in telemetry_rows]
    predictions, prediction_errors = load_predictions(predictions_path)
    blockers = list(telemetry_errors) + list(prediction_errors)

    blockers.extend(run_contract_errors(metrics, seed))
    if telemetry_steps != list(EXPECTED_TELEMETRY_STEPS):
        blockers.append(f"telemetry steps={telemetry_steps}, expected={list(EXPECTED_TELEMETRY_STEPS)}")
    if len(predictions) != EXPECTED_EVAL_EXAMPLES:
        blockers.append(f"prediction rows={len(predictions)}, expected={EXPECTED_EVAL_EXAMPLES}")

    vs_fp16 = compare_prediction_files(fp16_predictions, predictions_path)
    vs_fixed = compare_prediction_files(fixed_predictions, predictions_path)
    vs_gamma60 = compare_prediction_files(gamma60_predictions, predictions_path)
    for label, comparison in (
        ("fp16", vs_fp16),
        ("fixed_gamma_655m", vs_fixed),
        ("historical_gamma60_163m", vs_gamma60),
    ):
        if comparison["status"] != "pass":
            blockers.extend(f"{label}: {error}" for error in comparison["errors"])

    accuracy = nested(metrics, "eval", "accuracy")
    candidate_accuracy = vs_fp16.get("candidate_accuracy")
    if (
        isinstance(accuracy, (int, float))
        and isinstance(candidate_accuracy, (int, float))
        and not math.isclose(float(accuracy), float(candidate_accuracy), abs_tol=1e-15)
    ):
        blockers.append(
            f"metrics accuracy={accuracy!r} disagrees with predictions accuracy={candidate_accuracy!r}"
        )
    return {
        "seed": seed,
        "status": "complete" if not blockers else "pending_or_invalid",
        "blockers": blockers,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "telemetry_path": str(telemetry_path),
        "artifact_sha256": {
            "metrics": sha256(metrics_path),
            "predictions": sha256(predictions_path),
            "telemetry": sha256(telemetry_path),
        },
        "accuracy": float(accuracy) if isinstance(accuracy, (int, float)) and math.isfinite(accuracy) else None,
        "effective_attention_weight": nested(metrics, "loss_weights", "effective_attention_kd_weight"),
        "telemetry_health": telemetry_health(telemetry_rows),
        "vs_fp16": vs_fp16,
        "vs_fixed_gamma_655m": vs_fixed,
        "vs_historical_gamma60_163m": vs_gamma60,
    }


def seed_mean_ci(values: list[float]) -> list[float] | None:
    if len(values) <= 1:
        return None
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    return [mean - T_95_DF2 * standard_error, mean + T_95_DF2 * standard_error]


def build_report(
    root: Path,
    fp16_predictions: Path,
    fixed_predictions: Path,
    gamma60_predictions: Path,
) -> dict[str, Any]:
    runs = [
        summarize_run(root, seed, fp16_predictions, fixed_predictions, gamma60_predictions)
        for seed in SEEDS
    ]
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
            "historical_gamma60_163m_accuracy": HISTORICAL_GAMMA60_ACCURACY,
            "recovery_floor": RECOVERY_FLOOR,
            "prediction_sha256": {
                "fp16": sha256(fp16_predictions),
                "fixed_gamma_655m": sha256(fixed_predictions),
                "historical_gamma60_163m": sha256(gamma60_predictions),
            },
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
        gamma60 = run["vs_historical_gamma60_163m"]
        fp16 = run["vs_fp16"]
        rows.append(
            [
                run["seed"],
                run["status"],
                run["accuracy"],
                fixed["delta_candidate_minus_reference"],
                fixed["paired_ci95"],
                fixed["mcnemar_exact_p"],
                gamma60["delta_candidate_minus_reference"],
                gamma60["paired_ci95"],
                fp16["delta_candidate_minus_reference"],
                fp16["paired_ci95"],
                run["effective_attention_weight"],
                nested(run, "telemetry_health", "weighted_attention_to_ce_gradient_ratio", "median"),
                nested(run, "telemetry_health", "probe_weighted_attention_to_ce_gradient_ratio", "median"),
                nested(run, "telemetry_health", "global_to_last_controller_probe_ratio", "median"),
                nested(run, "telemetry_health", "weighted_attention_to_ce_gradient_ratio", "max"),
                nested(run, "telemetry_health", "max_activation_clipped_fraction"),
                nested(run, "telemetry_health", "mean_ternary_flip_fraction"),
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
                    "delta vs gamma60",
                    "paired CI vs gamma60",
                    "delta vs FP16",
                    "paired CI vs FP16",
                    "final gamma",
                    "median grad A/CE",
                    "median probe A/CE",
                    "median global/last probe",
                    "max grad A/CE",
                    "max A8 clipped",
                    "mean ternary flips",
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
            "## Telemetry Boundary",
            (
                "The global attention/CE norm and controller probe are not same-support, "
                "same-microbatch measurements. Their reported ratio is descriptive only: the "
                "global norm covers all trainable parameters on the telemetry microbatch, while "
                "the probe is the most recent controller update on selected-layer Q/K/V parameters."
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
        "--gamma60-predictions",
        type=Path,
        default=Path("checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-steps10000-lr2em5-gamma60-headinit/eval_predictions.jsonl"),
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
    report = build_report(
        args.root,
        args.fp16_predictions,
        args.fixed_predictions,
        args.gamma60_predictions,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0 if report["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
