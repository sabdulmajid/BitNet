#!/usr/bin/env python3
"""Audit the controlled BitNet-SFT least-squares initialization run.

The LS-init benchmark changes one axis relative to the matched absmean
BitNet-SFT baseline: the initial ternary fixed point.  This audit is allowed to
be pending while the Slurm job runs; once complete, it requires full MNLI
validation predictions and reports paired statistics against the matched
absmean baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
EXPECTED_MNLI = 9815
Z_95 = 1.959963984540054


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if 0.0 < abs(value) < 0.0001:
            return f"{value:.6e}"
        return f"{value:.6f}"
    return str(value).replace("|", "\\|")


def fmt_ci(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "-"
    return f"[{fmt(float(value[0]))}, {fmt(float(value[1]))}]"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def read_predictions(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], [f"missing {path}"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[int] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{lineno}: invalid json: {exc}")
            continue
        index = row.get("index")
        label = row.get("label")
        prediction = row.get("prediction")
        correct = row.get("correct")
        if not isinstance(index, int):
            errors.append(f"{path}:{lineno}: non-int index")
            continue
        if index in seen:
            errors.append(f"{path}:{lineno}: duplicate index {index}")
            continue
        seen.add(index)
        if not isinstance(label, int) or not isinstance(prediction, int):
            errors.append(f"{path}:{lineno}: non-int label or prediction")
            continue
        if bool(correct) != (label == prediction):
            errors.append(f"{path}:{lineno}: correct flag mismatch")
            continue
        rows.append(row)
    rows.sort(key=lambda item: int(item["index"]))
    for expected, row in enumerate(rows):
        if int(row["index"]) != expected:
            errors.append(f"{path}: non-contiguous index at {expected}, saw {row['index']}")
            break
    return rows, errors


def paired_ci(values: list[float], z: float = Z_95) -> list[float] | None:
    n = len(values)
    if n <= 1:
        return None
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    half_width = z * math.sqrt(variance / n)
    return [mean - half_width, mean + half_width]


def logsumexp(values: list[float]) -> float:
    if not values:
        return float("-inf")
    peak = max(values)
    if not math.isfinite(peak):
        return peak
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def binomial_tail(n: int, *, lower_k: int | None = None, upper_k: int | None = None) -> float:
    if n <= 0:
        return 1.0
    start = 0 if lower_k is not None else int(upper_k)
    end = int(lower_k) if lower_k is not None else n
    if start > end:
        return 0.0
    log_half = math.log(0.5)
    terms = [
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + n * log_half
        for k in range(start, end + 1)
    ]
    return min(1.0, math.exp(logsumexp(terms)))


def exact_mcnemar_pvalue(candidate_wins: int, reference_wins: int) -> float:
    discordant = candidate_wins + reference_wins
    if discordant == 0:
        return 1.0
    low = min(candidate_wins, reference_wins)
    high = max(candidate_wins, reference_wins)
    return min(1.0, 2.0 * min(binomial_tail(discordant, lower_k=low), binomial_tail(discordant, upper_k=high)))


def compare_predictions(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference_rows, reference_errors = read_predictions(reference_path)
    candidate_rows, candidate_errors = read_predictions(candidate_path)
    errors = reference_errors + candidate_errors
    if not errors and len(reference_rows) != len(candidate_rows):
        errors.append(f"row count mismatch: reference={len(reference_rows)}, candidate={len(candidate_rows)}")
    if not errors and len(reference_rows) != EXPECTED_MNLI:
        errors.append(f"reference rows={len(reference_rows)} expected={EXPECTED_MNLI}")
    if not errors and len(candidate_rows) != EXPECTED_MNLI:
        errors.append(f"candidate rows={len(candidate_rows)} expected={EXPECTED_MNLI}")

    matched = 0
    reference_correct = 0
    candidate_correct = 0
    candidate_wins = 0
    reference_wins = 0
    differences: list[float] = []
    if not errors:
        for reference_row, candidate_row in zip(reference_rows, candidate_rows):
            if int(reference_row["index"]) != int(candidate_row["index"]):
                errors.append(f"index mismatch at {reference_row['index']} / {candidate_row['index']}")
                break
            if int(reference_row["label"]) != int(candidate_row["label"]):
                errors.append(f"label mismatch at index {reference_row['index']}")
                break
            ref_ok = bool(reference_row["correct"])
            cand_ok = bool(candidate_row["correct"])
            matched += 1
            reference_correct += int(ref_ok)
            candidate_correct += int(cand_ok)
            candidate_wins += int(cand_ok and not ref_ok)
            reference_wins += int(ref_ok and not cand_ok)
            differences.append(float(cand_ok) - float(ref_ok))

    reference_accuracy = reference_correct / matched if matched else None
    candidate_accuracy = candidate_correct / matched if matched else None
    delta = (
        candidate_accuracy - reference_accuracy
        if reference_accuracy is not None and candidate_accuracy is not None
        else None
    )
    return {
        "status": "fail" if errors else "pass",
        "errors": errors,
        "matched": matched,
        "reference_accuracy": reference_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "delta_vs_reference": delta,
        "paired_ci95": paired_ci(differences),
        "candidate_wins": candidate_wins,
        "reference_wins": reference_wins,
        "mcnemar_exact_p": exact_mcnemar_pvalue(candidate_wins, reference_wins) if matched and not errors else None,
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    submission = read_json(args.submission_json)
    if not submission:
        return {
            "schema": "bitnet-sft-ls-init-audit-v1",
            "date": DATE,
            "status": "missing_submission",
            "quality_proven": False,
            "submission_json": str(args.submission_json),
        }

    candidate_root = Path(str(submission.get("output_dir", "")))
    baseline_root = Path(str(submission.get("baseline_output_dir", "")))
    candidate_metrics_path = candidate_root / "metrics.json"
    baseline_metrics_path = baseline_root / "metrics.json"
    candidate_metrics = read_json(candidate_metrics_path)
    baseline_metrics = read_json(baseline_metrics_path)
    candidate_accuracy = finite_float(nested(candidate_metrics, "eval", "accuracy"))
    baseline_accuracy = finite_float(nested(baseline_metrics, "eval", "accuracy"))
    candidate_examples = finite_float(nested(candidate_metrics, "eval", "eval_examples"))
    baseline_examples = finite_float(nested(baseline_metrics, "eval", "eval_examples"))
    ternary_init = nested(candidate_metrics, "preparation", "ternary_init", default={})

    status = "pending"
    blockers: list[str] = []
    if not baseline_metrics_path.exists():
        status = "blocked_missing_baseline"
        blockers.append(f"missing baseline metrics: {baseline_metrics_path}")
    elif not candidate_metrics_path.exists():
        status = "pending"
        blockers.append(f"missing candidate metrics: {candidate_metrics_path}")
    elif candidate_examples != EXPECTED_MNLI:
        status = "complete_incomplete_eval"
        blockers.append(f"candidate examples={candidate_examples}, expected={EXPECTED_MNLI}")
    elif (
        not isinstance(ternary_init, dict)
        or ternary_init.get("mode") != args.expected_mode
        or ternary_init.get("applied") is not True
    ):
        status = "complete_init_contract_failed"
        blockers.append(f"candidate ternary_init={ternary_init}")
    else:
        status = "complete"

    paired: dict[str, Any] = {}
    if status == "complete":
        paired = compare_predictions(baseline_root / "eval_predictions.jsonl", candidate_root / "eval_predictions.jsonl")
        if paired.get("status") != "pass":
            status = "complete_prediction_contract_failed"
            blockers.extend(str(error) for error in paired.get("errors", []))

    delta_vs_baseline = (
        candidate_accuracy - baseline_accuracy
        if candidate_accuracy is not None and baseline_accuracy is not None
        else None
    )
    return {
        "schema": "bitnet-sft-ternary-init-audit-v1",
        "date": DATE,
        "status": status,
        "quality_proven": status == "complete" and isinstance(delta_vs_baseline, float),
        "submission_json": str(args.submission_json),
        "submission": submission,
        "candidate_root": str(candidate_root),
        "baseline_root": str(baseline_root),
        "candidate_metrics_path": str(candidate_metrics_path),
        "baseline_metrics_path": str(baseline_metrics_path),
        "candidate_accuracy": candidate_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "delta_vs_absmean_baseline": delta_vs_baseline,
        "candidate_eval_examples": candidate_examples,
        "baseline_eval_examples": baseline_examples,
        "candidate_ternary_init": ternary_init,
        "paired": paired,
        "blockers": blockers,
        "verdict": (
            "Pending Slurm output."
            if status == "pending"
            else f"{args.expected_mode} initialization has a complete paired MNLI comparison."
            if status == "complete"
            else f"{args.expected_mode} initialization comparison is not valid yet; see blockers."
        ),
    }


def render_markdown(summary: dict[str, Any], *, title: str) -> str:
    paired = summary.get("paired", {}) if isinstance(summary.get("paired"), dict) else {}
    rows = [
        ["status", summary.get("status")],
        ["quality proven", summary.get("quality_proven")],
        ["baseline accuracy", summary.get("baseline_accuracy")],
        ["candidate accuracy", summary.get("candidate_accuracy")],
        ["delta vs absmean baseline", summary.get("delta_vs_absmean_baseline")],
        ["candidate eval examples", summary.get("candidate_eval_examples")],
        ["paired matched examples", paired.get("matched")],
        ["paired delta", paired.get("delta_vs_reference")],
        ["paired CI95", fmt_ci(paired.get("paired_ci95"))],
        ["McNemar exact p", paired.get("mcnemar_exact_p")],
    ]
    blockers = summary.get("blockers", []) if isinstance(summary.get("blockers"), list) else []
    return "\n\n".join(
        [
            f"# {title}, {summary['date']}",
            str(summary.get("verdict", "")),
            "## Summary",
            md_table(["field", "value"], rows),
            "## Artifacts",
            md_table(
                ["artifact", "path"],
                [
                    ["baseline root", summary.get("baseline_root")],
                    ["candidate root", summary.get("candidate_root")],
                    ["submission", summary.get("submission_json")],
                ],
            ),
            "## Blockers",
            "\n".join(f"- {item}" for item in blockers) if blockers else "None.",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-json", type=Path, default=Path(f"benchmark_results/bitnet_sft_ls_init_submission_{DATE}.json"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitnet_sft_ls_init_audit_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitnet_sft_ls_init_audit_{DATE}.md"))
    parser.add_argument("--expected-mode", default="ls")
    parser.add_argument("--title", default="BitNet-SFT LS-Init Audit")
    args = parser.parse_args()

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(summary, title=args.title), encoding="utf-8")
    print(render_markdown(summary, title=args.title))


if __name__ == "__main__":
    main()
