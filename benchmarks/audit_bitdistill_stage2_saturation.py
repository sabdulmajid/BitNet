#!/usr/bin/env python3
"""Audit whether the completed fixed-recipe Stage-2 curve is saturating.

The last three controlled MNLI runs differ only in cumulative continued-
pretraining budget and form two exact budget doublings. The audit fits the
minimal geometric-diminishing-returns model to those paired validation traces
and bootstraps examples jointly, preserving prediction correlation across runs.

This is a conditional extrapolation diagnostic, not evidence that all possible
BitDistill recipes saturate at the fitted limit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TRACES = {
    163_840_000: Path(
        "checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/"
        "bitdistill-tensor-20kwarmup-steps10000-lr2em5-papergamma-headinit/"
        "eval_predictions.jsonl"
    ),
    327_680_000: Path(
        "checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/"
        "bitdistill-tensor-40kwarmup-steps10000-lr2em5-papergamma-headinit-rerun/"
        "eval_predictions.jsonl"
    ),
    655_360_000: Path(
        "checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/"
        "bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit/"
        "eval_predictions.jsonl"
    ),
}
DEFAULT_FP_TRACE = Path(
    "checkpoints/bitdistill-glue-seqcls-predtrace/Qwen-Qwen2.5-0.5B/mnli/"
    "fp16_sft-tensor-layer-1/eval_predictions.jsonl"
)


def read_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    labels: list[int] = []
    correct: list[bool] = []
    for expected_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        if row.get("index") != expected_index:
            raise ValueError(f"{path}: expected index {expected_index}, got {row.get('index')}")
        label = row.get("label")
        prediction = row.get("prediction")
        if not isinstance(label, int) or not isinstance(prediction, int):
            raise ValueError(f"{path}:{expected_index + 1}: label and prediction must be integers")
        labels.append(label)
        correct.append(prediction == label)
    if not labels:
        raise ValueError(f"{path}: empty prediction trace")
    return np.asarray(labels, dtype=np.int16), np.asarray(correct, dtype=np.float64)


def geometric_projection(
    accuracies: np.ndarray,
    *,
    current_tokens: int,
    target_tokens: int,
) -> tuple[float, float, float] | None:
    """Return contraction ratio, asymptote, and target-budget projection."""

    first_gain = float(accuracies[1] - accuracies[0])
    second_gain = float(accuracies[2] - accuracies[1])
    if first_gain <= 0.0 or second_gain < 0.0:
        return None
    contraction = second_gain / first_gain
    if not 0.0 <= contraction < 1.0:
        return None
    if contraction == 0.0:
        return contraction, float(accuracies[2]), float(accuracies[2])

    remaining = second_gain * contraction / (1.0 - contraction)
    asymptote = float(accuracies[2] + remaining)
    future_doublings = math.log2(target_tokens / current_tokens)
    projected = asymptote - remaining * contraction**future_doublings
    return contraction, asymptote, projected


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def paired_bootstrap(
    correctness: np.ndarray,
    *,
    samples: int,
    seed: int,
    current_tokens: int,
    target_tokens: int,
    batch_size: int = 128,
) -> dict[str, Any]:
    if correctness.shape[0] != 3:
        raise ValueError("exactly three budget rows are required")
    rng = np.random.default_rng(seed)
    row_count = correctness.shape[1]
    contractions: list[float] = []
    asymptotes: list[float] = []
    projections: list[float] = []
    constant_gain_projections: list[float] = []
    future_doublings = math.log2(target_tokens / current_tokens)
    for offset in range(0, samples, batch_size):
        this_batch = min(batch_size, samples - offset)
        indexes = rng.integers(0, row_count, size=(this_batch, row_count))
        boot_accuracies = correctness[:, indexes].mean(axis=2).T
        for accuracies in boot_accuracies:
            latest_gain = float(accuracies[2] - accuracies[1])
            constant_gain_projections.append(float(accuracies[2] + latest_gain * future_doublings))
            projection = geometric_projection(
                accuracies,
                current_tokens=current_tokens,
                target_tokens=target_tokens,
            )
            if projection is None:
                continue
            contraction, asymptote, target = projection
            contractions.append(contraction)
            asymptotes.append(asymptote)
            projections.append(target)

    valid = len(projections)
    if valid == 0:
        raise RuntimeError("no bootstrap replicate satisfied the monotone-contraction model")
    contraction_array = np.asarray(contractions)
    asymptote_array = np.asarray(asymptotes)
    projection_array = np.asarray(projections)
    constant_gain_array = np.asarray(constant_gain_projections)
    return {
        "requested_samples": samples,
        "valid_samples": valid,
        "valid_fraction": valid / samples,
        "contraction_ci95": percentile_interval(contraction_array),
        "asymptote_ci95": percentile_interval(asymptote_array),
        "target_projection_ci95": percentile_interval(projection_array),
        "constant_gain_target_projection_ci95": percentile_interval(constant_gain_array),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    observed = payload["observed"]
    bootstrap = payload["paired_bootstrap"]
    lines = [
        "# Fixed-Recipe Stage-2 Saturation Audit",
        "",
        "This is a conditional extrapolation from three paired MNLI runs, not a claim about all BitDistill recipes.",
        "",
        "## Observed Curve",
        "",
        "| Stage-2 token presentations | Accuracy | Gain from previous doubling |",
        "| ---: | ---: | ---: |",
    ]
    previous: float | None = None
    for row in observed["rows"]:
        gain = "-" if previous is None else f"{row['accuracy'] - previous:.6f}"
        lines.append(f"| `{row['tokens']:,}` | `{row['accuracy']:.6f}` | `{gain}` |")
        previous = row["accuracy"]
    lines.extend(
        [
            "",
            "## Conditional Projection",
            "",
            f"- Observed gain contraction: `{observed['contraction']:.6f}`.",
            f"- Fitted asymptote: `{observed['asymptote']:.6f}`.",
            f"- Projection at `{payload['target_tokens']:,}` token presentations: "
            f"`{observed['target_projection']:.6f}`.",
            f"- Constant-latest-gain sensitivity projection at the target: "
            f"`{observed['constant_gain_target_projection']:.6f}`.",
            f"- Required average gain per remaining doubling: `{observed['required_gain_per_doubling']:.6f}` "
            f"(`{observed['required_to_latest_gain_ratio']:.3f}x` the latest observed gain).",
            f"- Paired-bootstrap 95% interval for the asymptote: "
            f"`[{bootstrap['asymptote_ci95'][0]:.6f}, {bootstrap['asymptote_ci95'][1]:.6f}]`.",
            f"- Paired-bootstrap 95% interval at the target budget: "
            f"`[{bootstrap['target_projection_ci95'][0]:.6f}, {bootstrap['target_projection_ci95'][1]:.6f}]`.",
            f"- Paired-bootstrap 95% interval for the constant-latest-gain sensitivity: "
            f"`[{bootstrap['constant_gain_target_projection_ci95'][0]:.6f}, "
            f"{bootstrap['constant_gain_target_projection_ci95'][1]:.6f}]`.",
            f"- Valid monotone-contraction bootstrap replicates: `{bootstrap['valid_samples']}` / "
            f"`{bootstrap['requested_samples']}` (`{bootstrap['valid_fraction']:.3%}`).",
            "",
            "## Decision",
            "",
            payload["decision"],
            "",
            "## Limitations",
            "",
            *[f"- {limitation}" for limitation in payload["limitations"]],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-163m", type=Path, default=DEFAULT_TRACES[163_840_000])
    parser.add_argument("--trace-327m", type=Path, default=DEFAULT_TRACES[327_680_000])
    parser.add_argument("--trace-655m", type=Path, default=DEFAULT_TRACES[655_360_000])
    parser.add_argument("--fp-trace", type=Path, default=DEFAULT_FP_TRACE)
    parser.add_argument("--target-tokens", type=int, default=10_000_000_000)
    parser.add_argument("--recovery-margin", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/bitdistill_stage2_saturation_2026-09-04.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/bitdistill_stage2_saturation_2026-09-04.md"),
    )
    args = parser.parse_args()
    if args.target_tokens <= 655_360_000:
        raise ValueError("target token budget must exceed the largest observed budget")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap sample count must be positive")

    trace_paths = {
        163_840_000: args.trace_163m,
        327_680_000: args.trace_327m,
        655_360_000: args.trace_655m,
    }
    labels: np.ndarray | None = None
    correctness_rows: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for tokens, path in trace_paths.items():
        trace_labels, correctness = read_trace(path)
        if labels is None:
            labels = trace_labels
        elif not np.array_equal(labels, trace_labels):
            raise ValueError(f"label mismatch in paired trace {path}")
        correctness_rows.append(correctness)
        rows.append({"tokens": tokens, "accuracy": float(correctness.mean()), "prediction_path": str(path)})

    fp_labels, fp_correctness = read_trace(args.fp_trace)
    if labels is None or not np.array_equal(labels, fp_labels):
        raise ValueError("FP trace labels do not match Stage-2 traces")
    correctness_matrix = np.stack(correctness_rows)
    observed_accuracies = correctness_matrix.mean(axis=1)
    projection = geometric_projection(
        observed_accuracies,
        current_tokens=655_360_000,
        target_tokens=args.target_tokens,
    )
    if projection is None:
        raise RuntimeError("observed curve does not satisfy the monotone-contraction model")
    contraction, asymptote, target_projection = projection
    recovery_target = float(fp_correctness.mean() - args.recovery_margin)
    future_doublings = math.log2(args.target_tokens / 655_360_000)
    latest_gain = float(observed_accuracies[2] - observed_accuracies[1])
    constant_gain_target_projection = float(observed_accuracies[2] + latest_gain * future_doublings)
    required_gain_per_doubling = float((recovery_target - observed_accuracies[2]) / future_doublings)
    bootstrap = paired_bootstrap(
        correctness_matrix,
        samples=args.bootstrap_samples,
        seed=args.seed,
        current_tokens=655_360_000,
        target_tokens=args.target_tokens,
    )
    upper_projection = bootstrap["target_projection_ci95"][1]
    misses_recovery = upper_projection < recovery_target
    constant_gain_upper = bootstrap["constant_gain_target_projection_ci95"][1]
    constant_gain_misses_recovery = constant_gain_upper < recovery_target
    decision = (
        "Under the fitted diminishing-returns model, Stage-2 budget alone does not close the local FP16 recovery "
        f"gap: even the bootstrap upper bound `{upper_projection:.6f}` is below the pre-registered recovery "
        f"target `{recovery_target:.6f}`. Even repeating the latest gain without further decay has bootstrap upper "
        f"bound `{constant_gain_upper:.6f}`. Reaching the gate requires the average future gain per doubling to be "
        f"`{required_gain_per_doubling / latest_gain:.3f}x` the latest observed gain, reversing the measured "
        "diminishing-return trend. Change the training objective or method contract before scaling this fixed recipe."
        if misses_recovery
        else "The extrapolation interval overlaps the recovery target; this curve does not reject budget-only scaling."
    )
    payload = {
        "schema": "bitdistill-stage2-saturation-v1",
        "scope": "conditional_fixed_recipe_extrapolation",
        "target_tokens": args.target_tokens,
        "fp_accuracy": float(fp_correctness.mean()),
        "recovery_margin": args.recovery_margin,
        "recovery_target": recovery_target,
        "observed": {
            "rows": rows,
            "contraction": contraction,
            "asymptote": asymptote,
            "target_projection": target_projection,
            "future_doublings_to_target": future_doublings,
            "latest_gain": latest_gain,
            "constant_gain_target_projection": constant_gain_target_projection,
            "required_gain_per_doubling": required_gain_per_doubling,
            "required_to_latest_gain_ratio": required_gain_per_doubling / latest_gain,
        },
        "paired_bootstrap": bootstrap,
        "misses_recovery_target": misses_recovery,
        "constant_gain_sensitivity_misses_recovery_target": constant_gain_misses_recovery,
        "decision": decision,
        "limitations": [
            "Three observed budgets identify only two doubling gains.",
            "The geometric contraction model is an extrapolation assumption.",
            "The constant-gain sensitivity assumes future gains do not accelerate.",
            "The paired bootstrap measures validation-example uncertainty conditional on fixed checkpoints, "
            "not training-seed uncertainty.",
            "Token presentations are not proven equivalent to the paper's corpus-token accounting.",
            "The result applies to the fixed-gamma tensor-scale local recipe, not adaptive balancing or all BitDistill variants.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
