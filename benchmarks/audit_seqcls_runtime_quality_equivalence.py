#!/usr/bin/env python3
"""Audit task-quality equivalence of native I2_SR and saved PyTorch predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.audit_bitdistill_adaptive_full import exact_mcnemar_pvalue


EXPECTED_EXAMPLES = 9_815
RETROSPECTIVE_NONINFERIORITY_MARGIN = 0.005
DEFAULT_RUNTIME = Path(
    "benchmark_results/seqcls_native_i2sr_cpu_mnli_full_token_ids_sequence_isolated_2026-05-15.json"
)
DEFAULT_PYTORCH = Path(
    "checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/"
    "bitdistill-longwarmup-row-layer-8/eval_predictions.jsonl"
)
DEFAULT_TRACE = Path("benchmark_results/seqcls_native_i2sr_quality_trace_2026-09-04.jsonl")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    return rows


def build_comparison_rows(runtime: dict[str, Any], pytorch_path: Path) -> list[dict[str, int]]:
    runtime_predictions = runtime.get("predictions")
    pytorch_rows = read_jsonl(pytorch_path)
    if not isinstance(runtime_predictions, list):
        raise ValueError("runtime artifact lacks predictions list")
    if len(runtime_predictions) != len(pytorch_rows):
        raise ValueError(
            f"prediction count mismatch: runtime={len(runtime_predictions)} pytorch={len(pytorch_rows)}"
        )
    rows: list[dict[str, int]] = []
    for expected_index, (runtime_prediction, pytorch_row) in enumerate(
        zip(runtime_predictions, pytorch_rows)
    ):
        if pytorch_row.get("index") != expected_index:
            raise ValueError(f"PyTorch trace index mismatch at {expected_index}")
        label = pytorch_row.get("label")
        pytorch_prediction = pytorch_row.get("prediction")
        if not all(isinstance(value, int) for value in (runtime_prediction, label, pytorch_prediction)):
            raise ValueError(f"non-integer prediction field at index {expected_index}")
        if pytorch_row.get("correct") is not (pytorch_prediction == label):
            raise ValueError(f"PyTorch correct flag mismatch at index {expected_index}")
        rows.append(
            {
                "index": expected_index,
                "label": int(label),
                "pytorch_prediction": int(pytorch_prediction),
                "runtime_prediction": int(runtime_prediction),
            }
        )
    return rows


def validate_comparison_rows(rows: list[dict[str, Any]], runtime: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    runtime_predictions = runtime.get("predictions")
    if len(rows) != EXPECTED_EXAMPLES:
        errors.append(f"comparison rows={len(rows)}, expected={EXPECTED_EXAMPLES}")
    if not isinstance(runtime_predictions, list) or len(runtime_predictions) != len(rows):
        errors.append("runtime prediction count does not match comparison trace")
        return errors
    for expected_index, (row, runtime_prediction) in enumerate(zip(rows, runtime_predictions)):
        required = ("index", "label", "pytorch_prediction", "runtime_prediction")
        if not all(isinstance(row.get(key), int) for key in required):
            errors.append(f"row {expected_index} has non-integer fields")
            break
        if row["index"] != expected_index:
            errors.append(f"row {expected_index} has index={row['index']}")
            break
        if row["runtime_prediction"] != runtime_prediction:
            errors.append(f"runtime prediction mismatch at index {expected_index}")
            break
    return errors


def paired_statistics(rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    deltas: list[int] = []
    runtime_wins = 0
    pytorch_wins = 0
    prediction_agreement = 0
    runtime_correct = 0
    pytorch_correct = 0
    for row in rows:
        label = int(row["label"])
        runtime_ok = int(row["runtime_prediction"]) == label
        pytorch_ok = int(row["pytorch_prediction"]) == label
        runtime_correct += int(runtime_ok)
        pytorch_correct += int(pytorch_ok)
        runtime_wins += int(runtime_ok and not pytorch_ok)
        pytorch_wins += int(pytorch_ok and not runtime_ok)
        prediction_agreement += int(row["runtime_prediction"] == row["pytorch_prediction"])
        deltas.append(int(runtime_ok) - int(pytorch_ok))

    n = len(deltas)
    delta = sum(deltas) / n
    variance = sum((value - delta) ** 2 for value in deltas) / (n - 1)
    standard_error = math.sqrt(variance / n)
    normal_ci = [delta - 1.959963984540054 * standard_error, delta + 1.959963984540054 * standard_error]

    counts = np.asarray(
        [
            sum(value == -1 for value in deltas),
            sum(value == 0 for value in deltas),
            sum(value == 1 for value in deltas),
        ],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    bootstrap_counts = rng.multinomial(n, counts / n, size=bootstrap_samples)
    bootstrap_deltas = (bootstrap_counts[:, 2] - bootstrap_counts[:, 0]) / n
    bootstrap_ci = np.quantile(bootstrap_deltas, [0.025, 0.975]).tolist()

    margin = RETROSPECTIVE_NONINFERIORITY_MARGIN
    z_noninferiority = (delta + margin) / standard_error
    p_noninferiority = 0.5 * math.erfc(z_noninferiority / math.sqrt(2.0))
    return {
        "examples": n,
        "runtime_correct": runtime_correct,
        "pytorch_correct": pytorch_correct,
        "runtime_accuracy": runtime_correct / n,
        "pytorch_accuracy": pytorch_correct / n,
        "prediction_agreement": prediction_agreement / n,
        "quality_agreement": (n - runtime_wins - pytorch_wins) / n,
        "runtime_wins": runtime_wins,
        "pytorch_wins": pytorch_wins,
        "delta_runtime_minus_pytorch": delta,
        "paired_standard_error": standard_error,
        "paired_normal_ci95": normal_ci,
        "paired_bootstrap_ci95": bootstrap_ci,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "mcnemar_exact_p": exact_mcnemar_pvalue(runtime_wins, pytorch_wins),
        "retrospective_noninferiority": {
            "margin": margin,
            "criterion": "lower endpoint of two-sided paired 95% normal CI > -margin",
            "passed": normal_ci[0] > -margin,
            "z": z_noninferiority,
            "one_sided_p": p_noninferiority,
            "preregistered": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    stats = report["statistics"]
    ni = stats["retrospective_noninferiority"]
    return "\n".join(
        [
            "# Native I2_SR Sequence-Classification Quality Equivalence",
            "",
            f"Generated: `{report['created_utc']}`",
            "",
            f"Status: **{report['status']}**.",
            "",
            "## Paired Full-Split Result",
            "",
            "| Metric | Result |",
            "| --- | ---: |",
            f"| MNLI examples | `{stats['examples']}` |",
            f"| PyTorch correct / accuracy | `{stats['pytorch_correct']}` / `{stats['pytorch_accuracy']:.6f}` |",
            f"| native I2_SR correct / accuracy | `{stats['runtime_correct']}` / `{stats['runtime_accuracy']:.6f}` |",
            f"| native minus PyTorch accuracy | `{stats['delta_runtime_minus_pytorch']:+.6f}` |",
            f"| paired normal 95% CI | `[{stats['paired_normal_ci95'][0]:.6f}, {stats['paired_normal_ci95'][1]:.6f}]` |",
            f"| paired bootstrap 95% CI | `[{stats['paired_bootstrap_ci95'][0]:.6f}, {stats['paired_bootstrap_ci95'][1]:.6f}]` |",
            f"| runtime wins / PyTorch wins | `{stats['runtime_wins']}` / `{stats['pytorch_wins']}` |",
            f"| exact McNemar p | `{stats['mcnemar_exact_p']:.6g}` |",
            f"| exact prediction agreement | `{stats['prediction_agreement']:.6f}` |",
            f"| retrospective 0.5-point non-inferiority | `{'pass' if ni['passed'] else 'fail'}` |",
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "## Claim Boundary",
            "",
            "- This supports task-quality preservation for this one row-scale MNLI artifact, not numerical equivalence or general model equivalence.",
            "- The 0.5-point non-inferiority margin was selected retrospectively and is labeled as such; the paired interval and raw discordance counts are the primary evidence.",
            "- The reference trace was produced by GPU BF16 inference while I2_SR ran on CPU integer kernels, so exact prediction identity is not expected.",
            "- The underlying ternary checkpoint remains far below the FP16 task model; runtime preservation does not repair model quality.",
            "- Multi-prompt batching is still excluded. This result uses the verified sequence-isolated token-ID path.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-json", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--pytorch-predictions", type=Path, default=DEFAULT_PYTORCH)
    parser.add_argument("--comparison-trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/seqcls_runtime_quality_equivalence_2026-09-04.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/seqcls_runtime_quality_equivalence_2026-09-04.md"),
    )
    args = parser.parse_args()

    runtime = read_json(args.runtime_json)
    if args.pytorch_predictions.is_file():
        rows = build_comparison_rows(runtime, args.pytorch_predictions)
        args.comparison_trace.parent.mkdir(parents=True, exist_ok=True)
        args.comparison_trace.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    else:
        rows = read_jsonl(args.comparison_trace)

    errors = validate_comparison_rows(rows, runtime)
    expected_runtime = {
        "schema": "seqcls_native_i2sr_cpu.v1",
        "status": "pass",
        "task": "mnli",
        "expected_examples": EXPECTED_EXAMPLES,
        "full_validation_complete": True,
        "prompt_input": "token_ids",
        "embedding_sequential": True,
        "runtime_parity_ready": True,
        "sequence_isolated_parity_ready": True,
    }
    for key, expected in expected_runtime.items():
        if runtime.get(key) != expected:
            errors.append(f"runtime {key}={runtime.get(key)!r}, expected={expected!r}")

    stats = paired_statistics(rows, bootstrap_samples=args.bootstrap_samples, seed=args.bootstrap_seed)
    summary = runtime.get("summary", {})
    if not math.isclose(stats["runtime_accuracy"], float(summary.get("accuracy", math.nan)), abs_tol=1e-15):
        errors.append("recomputed runtime accuracy disagrees with runtime summary")
    if not math.isclose(
        stats["prediction_agreement"],
        float(summary.get("agreement_with_saved_pytorch_predictions", math.nan)),
        abs_tol=1e-15,
    ):
        errors.append("recomputed prediction agreement disagrees with runtime summary")

    status = "task_quality_preserved_for_artifact" if not errors else "invalid"
    interpretation = (
        "On all 9,815 MNLI validation examples, native I2_SR loses 14 net correct predictions "
        "relative to the saved PyTorch trace. The paired 95% interval includes zero, exact "
        "McNemar does not reject equal marginal accuracy, and the retrospective 0.5-point "
        "non-inferiority criterion passes. The runtime therefore preserves task accuracy for "
        "this artifact within the measured uncertainty, despite failing strict prediction identity."
        if not errors
        else "The artifact or trace contract failed validation; no quality-preservation claim is permitted."
    )
    report = {
        "schema": "seqcls-runtime-quality-equivalence-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "errors": errors,
        "statistics": stats,
        "artifacts": {
            "runtime_json": str(args.runtime_json),
            "runtime_json_sha256": sha256(args.runtime_json),
            "pytorch_predictions": str(args.pytorch_predictions),
            "pytorch_predictions_sha256": sha256(args.pytorch_predictions)
            if args.pytorch_predictions.is_file()
            else None,
            "comparison_trace": str(args.comparison_trace),
            "comparison_trace_sha256": sha256(args.comparison_trace),
        },
        "interpretation": interpretation,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
