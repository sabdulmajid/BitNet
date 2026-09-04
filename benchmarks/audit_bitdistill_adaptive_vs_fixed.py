#!/usr/bin/env python3
"""Fail-closed audit for matched adaptive-vs-fixed BitDistill MNLI runs."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.audit_bitdistill_adaptive_full import (
    EXPECTED_EVAL_EXAMPLES,
    EXPECTED_SOURCE_REVISION,
    EXPECTED_STEPS,
    EXPECTED_TELEMETRY_STEPS,
    SEEDS,
    compare_prediction_files,
    load_predictions,
    read_json,
    read_telemetry,
    seed_mean_ci,
    sha256,
    telemetry_health,
)

EXPECTED_STAGE2_SHA256 = "9fc648a7466adb5f170085cf73d2bf4bd90a500f9de4c2a8f6c68b6cc29fa57d"
FP16_ACCURACY = 0.808151
RECOVERY_FLOOR = FP16_ACCURACY - 0.01
MINIMUM_PRACTICAL_DELTA = 0.005

ARM_SPECS: dict[str, dict[str, Any]] = {
    "adaptive": {
        "root_name": "mnli-seqcls-cosine-s1-adaptive-seed{seed}",
        "job_ids": {1234: 10392, 1235: 10395, 1236: 10396},
        "attention_kd_weight": 100_000.0,
        "attention_kd_balance": "gradnorm_ema",
        "save_model_artifacts": {1234: "1", 1235: "0", 1236: "0"},
    },
    "fixed60": {
        "root_name": "mnli-seqcls-cosine-s1-fixed60-seed{seed}",
        "job_ids": {1234: 10399, 1235: 10400, 1236: 10401},
        "attention_kd_weight": 60.0,
        "attention_kd_balance": "fixed",
        "save_model_artifacts": {1234: "0", 1235: "0", 1236: "0"},
    },
}


def nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def finite_float(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def parse_declared_contract(path: Path) -> tuple[dict[str, str], list[str]]:
    if not path.is_file():
        return {}, [f"missing {path}"]
    values: dict[str, str] = {}
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.startswith("step="):
            break
        try:
            tokens = shlex.split(line)
        except ValueError as exc:
            errors.append(f"{path}:{line_number}: invalid shell-like declaration: {exc}")
            continue
        for token in tokens:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key and key.replace("_", "").isalnum():
                values[key] = value
    return values, errors


def declared_contract_errors(values: dict[str, str], *, arm: str, seed: int) -> list[str]:
    spec = ARM_SPECS[arm]
    expected = {
        "SLURM_JOB_ID": str(spec["job_ids"][seed]),
        "MODEL": "/local/a6abdulm/bitnet-b7fc773/assets/base_model",
        "TEACHER_MODEL": "/local/a6abdulm/bitnet-b7fc773/assets/seqcls_teacher",
        "INIT_STATE_DICT": "/local/a6abdulm/bitnet-b7fc773/assets/stage2.pt",
        "STAGE": "task_sft",
        "METHOD": "bitdistill",
        "TASK_NAME": "mnli",
        "TASK_FORMAT": "sequence_classification",
        "LABEL_SCHEME": "letters",
        "CANDIDATE_SCORE": "mean",
        "SCALE_MODE": "tensor",
        "EXCLUDE_LINEAR_REGEX": "score|classifier",
        "DISTILL_LAYER": "-1",
        "ATTENTION_SPLIT_HEADS": "1",
        "ACTIVATION_QUANTIZATION": "1",
        "USE_SUBLN": "1",
        "LOGIT_KD_WEIGHT": "10",
        "ATTENTION_KD_WEIGHT": str(int(spec["attention_kd_weight"])),
        "LOGIT_TEMPERATURE": "5.0",
        "LOGIT_KD_TEMPERATURE_SCALE": "none",
        "ATTENTION_TEMPERATURE": "1.0",
        "ATTENTION_RELATION_MODE": "cosine",
        "ATTENTION_KD_BALANCE": str(spec["attention_kd_balance"]),
        "ATTENTION_BALANCE_TARGET_RATIO": "1.0",
        "ATTENTION_BALANCE_BETA": "0.9",
        "ATTENTION_BALANCE_EVERY_STEPS": "20",
        "ATTENTION_BALANCE_MIN_WEIGHT": "0.001",
        "ATTENTION_BALANCE_MAX_WEIGHT": "100000",
        "TELEMETRY_EVERY_STEPS": "500",
        "TELEMETRY_COMPONENT_GRAD_NORMS": "1",
        "TELEMETRY_MAX_ELEMENTS_PER_LAYER": "65536",
        "INIT_OUTPUT_HEAD_FROM_TEACHER": "1",
        "MAX_SEQ_LEN": "512",
        "MAX_STEPS": "10000",
        "PER_DEVICE_BATCH_SIZE": "4",
        "GRAD_ACCUM_STEPS": "4",
        "LR": "2e-5",
        "LR_SCHEDULER": "cosine",
        "SAVE_EVERY_STEPS": "0",
        "SAVE_MODEL_ARTIFACTS": spec["save_model_artifacts"][seed],
        "SEED": str(seed),
    }
    return [
        f"declared {key}={values.get(key)!r}, expected={expected_value!r}"
        for key, expected_value in expected.items()
        if values.get(key) != expected_value
    ]


def run_contract_errors(metrics: dict[str, Any], *, arm: str, seed: int) -> list[str]:
    spec = ARM_SPECS[arm]
    expected = {
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
        "attention_kd_weight": (
            nested(metrics, "loss_weights", "attention_kd_weight"),
            spec["attention_kd_weight"],
        ),
        "attention_kd_balance": (
            nested(metrics, "loss_weights", "attention_kd_balance"),
            spec["attention_kd_balance"],
        ),
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
        f"{field}={actual!r}, expected={expected_value!r}"
        for field, (actual, expected_value) in expected.items()
        if actual != expected_value
    ]
    state_path = nested(metrics, "state_load", "path")
    if not isinstance(state_path, str) or not state_path.endswith("/assets/stage2.pt"):
        errors.append(f"state_load.path={state_path!r}, expected suffix='/assets/stage2.pt'")
    effective_weight = nested(metrics, "loss_weights", "effective_attention_kd_weight")
    if arm == "fixed60" and effective_weight != 60.0:
        errors.append(f"effective_attention_kd_weight={effective_weight!r}, expected=60.0")
    if arm == "adaptive" and not finite_float(effective_weight):
        errors.append(f"effective_attention_kd_weight={effective_weight!r}, expected finite number")
    return errors


def summarize_run(root: Path, log_root: Path, *, arm: str, seed: int) -> dict[str, Any]:
    spec = ARM_SPECS[arm]
    run_dir = root / spec["root_name"].format(seed=seed)
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "eval_predictions.jsonl"
    telemetry_path = run_dir / "telemetry.jsonl"
    job_id = spec["job_ids"][seed]
    log_path = log_root / f"bdm-mnli-{arm}-10k-s{seed}-{job_id}.out"

    metrics = read_json(metrics_path)
    predictions, prediction_errors = load_predictions(predictions_path)
    telemetry, telemetry_errors = read_telemetry(telemetry_path)
    declarations, declaration_errors = parse_declared_contract(log_path)
    blockers = prediction_errors + telemetry_errors + declaration_errors
    blockers.extend(run_contract_errors(metrics, arm=arm, seed=seed))
    blockers.extend(declared_contract_errors(declarations, arm=arm, seed=seed))
    if len(predictions) != EXPECTED_EVAL_EXAMPLES:
        blockers.append(f"prediction rows={len(predictions)}, expected={EXPECTED_EVAL_EXAMPLES}")
    telemetry_steps = [row.get("step") for row in telemetry]
    if telemetry_steps != list(EXPECTED_TELEMETRY_STEPS):
        blockers.append(f"telemetry steps={telemetry_steps}, expected={list(EXPECTED_TELEMETRY_STEPS)}")

    prediction_accuracy = (
        sum(int(row["correct"]) for row in predictions) / len(predictions) if predictions else None
    )
    metrics_accuracy = nested(metrics, "eval", "accuracy")
    if (
        finite_float(metrics_accuracy)
        and prediction_accuracy is not None
        and not math.isclose(float(metrics_accuracy), prediction_accuracy, abs_tol=1e-15)
    ):
        blockers.append(
            f"metrics accuracy={metrics_accuracy!r} disagrees with predictions={prediction_accuracy!r}"
        )
    return {
        "arm": arm,
        "seed": seed,
        "job_id": job_id,
        "status": "complete" if not blockers else "pending_or_invalid",
        "blockers": blockers,
        "paths": {
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
            "telemetry": str(telemetry_path),
            "stdout": str(log_path),
        },
        "sha256": {
            "metrics": sha256(metrics_path),
            "predictions": sha256(predictions_path),
            "telemetry": sha256(telemetry_path),
            "stdout": sha256(log_path),
        },
        "accuracy": float(metrics_accuracy) if finite_float(metrics_accuracy) else None,
        "telemetry_health": telemetry_health(telemetry),
    }


def conditional_example_ci(
    fixed_paths: list[Path], adaptive_paths: list[Path]
) -> tuple[list[float] | None, list[str]]:
    per_seed: list[list[float]] = []
    errors: list[str] = []
    for fixed_path, adaptive_path in zip(fixed_paths, adaptive_paths):
        fixed, fixed_errors = load_predictions(fixed_path)
        adaptive, adaptive_errors = load_predictions(adaptive_path)
        errors.extend(fixed_errors + adaptive_errors)
        if len(fixed) != EXPECTED_EVAL_EXAMPLES or len(adaptive) != EXPECTED_EVAL_EXAMPLES:
            errors.append(f"cannot build conditional CI from {fixed_path} and {adaptive_path}")
            continue
        values: list[float] = []
        for fixed_row, adaptive_row in zip(fixed, adaptive):
            if fixed_row["index"] != adaptive_row["index"] or fixed_row["label"] != adaptive_row["label"]:
                errors.append(f"prediction alignment mismatch at {fixed_row.get('index')}")
                break
            values.append(float(adaptive_row["correct"]) - float(fixed_row["correct"]))
        if len(values) == EXPECTED_EVAL_EXAMPLES:
            per_seed.append(values)
    if errors or len(per_seed) != len(SEEDS):
        return None, errors
    example_means = [statistics.fmean(values) for values in zip(*per_seed)]
    mean = statistics.fmean(example_means)
    standard_error = statistics.stdev(example_means) / math.sqrt(len(example_means))
    return [mean - 1.959963984540054 * standard_error, mean + 1.959963984540054 * standard_error], []


def decide_method(deltas: list[float], *, complete: bool) -> dict[str, Any]:
    if not complete or len(deltas) != len(SEEDS):
        return {
            "adaptive_superiority_gate": "pending",
            "fixed_simplicity_gate": "pending",
            "recommended_method": "pending",
        }
    mean_delta = statistics.fmean(deltas)
    interval = seed_mean_ci(deltas)
    assert interval is not None
    adaptive_pass = (
        all(delta > 0.0 for delta in deltas)
        and mean_delta >= MINIMUM_PRACTICAL_DELTA
        and interval[0] > 0.0
    )
    fixed_pass = interval[1] < MINIMUM_PRACTICAL_DELTA
    if adaptive_pass:
        recommended = "adaptive"
    elif fixed_pass:
        recommended = "fixed60"
    else:
        recommended = "inconclusive"
    return {
        "adaptive_superiority_gate": "pass" if adaptive_pass else "fail",
        "fixed_simplicity_gate": "pass" if fixed_pass else "fail",
        "recommended_method": recommended,
    }


def build_report(
    adaptive_root: Path,
    fixed_root: Path,
    log_root: Path,
    stage2_path: Path,
) -> dict[str, Any]:
    runs = {
        arm: [summarize_run(root, log_root, arm=arm, seed=seed) for seed in SEEDS]
        for arm, root in (("adaptive", adaptive_root), ("fixed60", fixed_root))
    }
    stage2_digest = sha256(stage2_path)
    stage2_valid = stage2_digest == EXPECTED_STAGE2_SHA256
    all_complete = stage2_valid and all(
        run["status"] == "complete" for arm_runs in runs.values() for run in arm_runs
    )

    paired: list[dict[str, Any]] = []
    deltas: list[float] = []
    fixed_paths: list[Path] = []
    adaptive_paths: list[Path] = []
    for seed in SEEDS:
        fixed_path = fixed_root / ARM_SPECS["fixed60"]["root_name"].format(seed=seed) / "eval_predictions.jsonl"
        adaptive_path = adaptive_root / ARM_SPECS["adaptive"]["root_name"].format(seed=seed) / "eval_predictions.jsonl"
        comparison = compare_prediction_files(fixed_path, adaptive_path)
        comparison["seed"] = seed
        paired.append(comparison)
        fixed_paths.append(fixed_path)
        adaptive_paths.append(adaptive_path)
        delta = comparison.get("delta_candidate_minus_reference")
        if comparison["status"] == "pass" and finite_float(delta):
            deltas.append(float(delta))
    if len(deltas) != len(SEEDS):
        all_complete = False

    conditional_ci, conditional_errors = conditional_example_ci(fixed_paths, adaptive_paths)
    if conditional_errors:
        all_complete = False
    adaptive_accuracies = [run["accuracy"] for run in runs["adaptive"] if finite_float(run["accuracy"])]
    fixed_accuracies = [run["accuracy"] for run in runs["fixed60"] if finite_float(run["accuracy"])]
    adaptive_mean = statistics.fmean(adaptive_accuracies) if len(adaptive_accuracies) == len(SEEDS) else None
    fixed_mean = statistics.fmean(fixed_accuracies) if len(fixed_accuracies) == len(SEEDS) else None
    decisions = decide_method(deltas, complete=all_complete)
    decisions.update(
        {
            "adaptive_paper_recovery_gate": (
                "pass" if adaptive_mean is not None and adaptive_mean >= RECOVERY_FLOOR else "fail"
            )
            if all_complete
            else "pending",
            "fixed60_paper_recovery_gate": (
                "pass" if fixed_mean is not None and fixed_mean >= RECOVERY_FLOOR else "fail"
            )
            if all_complete
            else "pending",
        }
    )
    return {
        "schema": "bitdistill-adaptive-vs-fixed-matched-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if all_complete else "pending_or_invalid",
        "claim_boundary": (
            "Matched three-seed, cross-environment MNLI method comparison. It is not a "
            "paper-exact reproduction and does not establish generalization beyond this setup."
        ),
        "preregistration": {
            "primary_endpoint": "full MNLI matched-validation accuracy",
            "primary_contrast": "adaptive minus fixed60, paired by training seed",
            "minimum_practical_delta": MINIMUM_PRACTICAL_DELTA,
            "adaptive_success_rule": (
                "all three seed deltas > 0, mean delta >= 0.005, and seed-level t-CI lower bound > 0"
            ),
            "fixed_simplicity_rule": (
                "prefer fixed60 when the seed-level t-CI upper bound excludes a +0.005 adaptive gain"
            ),
            "paper_recovery_floor": RECOVERY_FLOOR,
        },
        "provenance": {
            "source_revision": EXPECTED_SOURCE_REVISION,
            "stage2_path": str(stage2_path),
            "stage2_sha256": stage2_digest,
            "expected_stage2_sha256": EXPECTED_STAGE2_SHA256,
            "stage2_valid": stage2_valid,
        },
        "runs": runs,
        "paired_by_seed": paired,
        "aggregate": {
            "adaptive_mean_accuracy": adaptive_mean,
            "adaptive_seed_mean_t_ci95": seed_mean_ci(adaptive_accuracies)
            if len(adaptive_accuracies) == len(SEEDS)
            else None,
            "fixed60_mean_accuracy": fixed_mean,
            "fixed60_seed_mean_t_ci95": seed_mean_ci(fixed_accuracies)
            if len(fixed_accuracies) == len(SEEDS)
            else None,
            "mean_adaptive_minus_fixed60": statistics.fmean(deltas)
            if len(deltas) == len(SEEDS)
            else None,
            "seed_level_paired_t_ci95": seed_mean_ci(deltas) if len(deltas) == len(SEEDS) else None,
            "conditional_example_level_ci95": conditional_ci,
            "conditional_example_level_errors": conditional_errors,
        },
        "decisions": decisions,
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
    run_rows: list[list[Any]] = []
    for arm in ("adaptive", "fixed60"):
        for run in report["runs"][arm]:
            run_rows.append(
                [
                    arm,
                    run["seed"],
                    run["job_id"],
                    run["status"],
                    run["accuracy"],
                    nested(run, "telemetry_health", "attention_weight", "final"),
                    nested(run, "telemetry_health", "weighted_attention_to_ce_gradient_ratio", "median"),
                    run["blockers"],
                ]
            )
    paired_rows = [
        [
            item["seed"],
            item["candidate_accuracy"],
            item["reference_accuracy"],
            item["delta_candidate_minus_reference"],
            item["paired_ci95"],
            item["mcnemar_exact_p"],
            item["status"],
        ]
        for item in report["paired_by_seed"]
    ]
    aggregate = report["aggregate"]
    decisions = report["decisions"]
    return "\n\n".join(
        [
            "# Matched Adaptive vs Fixed-60 BitDistill Audit",
            f"Generated: `{report['created_utc']}`",
            f"Status: **{report['status']}**.",
            report["claim_boundary"],
            "## Runs",
            table(
                ["arm", "seed", "job", "status", "accuracy", "final gamma", "median grad A/CE", "blockers"],
                run_rows,
            ),
            "## Paired Results",
            table(
                ["seed", "adaptive", "fixed60", "delta", "paired CI", "McNemar p", "status"],
                paired_rows,
            ),
            "## Aggregate",
            table(
                ["adaptive mean", "fixed60 mean", "mean delta", "seed t-CI", "conditional example CI"],
                [[
                    aggregate["adaptive_mean_accuracy"],
                    aggregate["fixed60_mean_accuracy"],
                    aggregate["mean_adaptive_minus_fixed60"],
                    aggregate["seed_level_paired_t_ci95"],
                    aggregate["conditional_example_level_ci95"],
                ]],
            ),
            "## Decisions",
            table(
                ["decision", "result"],
                [
                    ["adaptive superiority", decisions["adaptive_superiority_gate"]],
                    ["fixed simplicity", decisions["fixed_simplicity_gate"]],
                    ["recommended method", decisions["recommended_method"]],
                    ["adaptive within one point of FP16", decisions["adaptive_paper_recovery_gate"]],
                    ["fixed60 within one point of FP16", decisions["fixed60_paper_recovery_gate"]],
                ],
            ),
            "## Statistical Boundary",
            (
                "The seed-level paired t-interval is primary and reflects variation across three training seeds; "
                "with n=3 it has low power. Per-seed intervals and McNemar tests condition on trained checkpoints. "
                "The example-level interval averages the three seed contrasts for each shared validation example "
                "and is secondary because it does not replace training-seed uncertainty."
            ),
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive-root", type=Path, default=Path("/local/a6abdulm/bitnet-b7fc773/runs-full"))
    parser.add_argument("--fixed-root", type=Path, default=Path("/local/a6abdulm/bitnet-b7fc773/runs-fixed60"))
    parser.add_argument("--log-root", type=Path, default=Path("/local/a6abdulm/bitnet-b7fc773/logs"))
    parser.add_argument("--stage2-path", type=Path, default=Path("/local/a6abdulm/bitnet-b7fc773/assets/stage2.pt"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/local/a6abdulm/bitnet-b7fc773/audit-bundle/bitdistill_adaptive_vs_fixed_2026-09-04.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("/local/a6abdulm/bitnet-b7fc773/audit-bundle/bitdistill_adaptive_vs_fixed_2026-09-04.md"),
    )
    args = parser.parse_args()
    report = build_report(args.adaptive_root, args.fixed_root, args.log_root, args.stage2_path)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0 if report["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
