#!/usr/bin/env python3
"""Paired prediction audit for the BitNet-SFT MNLI budget sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
Z_95 = 1.959963984540054
EXPECTED_MNLI = 9815
PAPER_BITNET_SFT_MNLI = 0.608000


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
            errors.append(f"{path}:{lineno}: correct flag disagrees with label/prediction")
            continue
        rows.append(row)
    rows.sort(key=lambda item: int(item["index"]))
    for expected, row in enumerate(rows):
        if int(row["index"]) != expected:
            errors.append(f"{path}: non-contiguous index at row {expected}, saw {row['index']}")
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
    if lower_k is None and upper_k is None:
        raise ValueError("lower_k or upper_k is required")
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
    return min(
        1.0,
        2.0
        * min(
            binomial_tail(discordant, lower_k=low),
            binomial_tail(discordant, upper_k=high),
        ),
    )


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if 0.0 < abs(value) < 0.0001:
            return f"{value:.6e}"
        return f"{value:.6f}"
    return str(value)


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
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def compare(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
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
    both_correct = 0
    both_wrong = 0
    differences: list[float] = []

    if not errors:
        for reference_row, candidate_row in zip(reference_rows, candidate_rows):
            if int(reference_row["index"]) != int(candidate_row["index"]):
                errors.append(f"index mismatch at reference={reference_row['index']} candidate={candidate_row['index']}")
                break
            if int(reference_row["label"]) != int(candidate_row["label"]):
                errors.append(f"label mismatch at index {reference_row['index']}")
                break
            ref_ok = bool(reference_row["correct"])
            cand_ok = bool(candidate_row["correct"])
            matched += 1
            reference_correct += int(ref_ok)
            candidate_correct += int(cand_ok)
            both_correct += int(ref_ok and cand_ok)
            both_wrong += int((not ref_ok) and (not cand_ok))
            candidate_wins += int(cand_ok and not ref_ok)
            reference_wins += int(ref_ok and not cand_ok)
            differences.append(float(cand_ok) - float(ref_ok))

    reference_acc = reference_correct / matched if matched else None
    candidate_acc = candidate_correct / matched if matched else None
    delta = (candidate_acc - reference_acc) if reference_acc is not None and candidate_acc is not None else None
    return {
        "status": "fail" if errors else "pass",
        "errors": errors,
        "matched": matched,
        "reference_accuracy": reference_acc,
        "candidate_accuracy": candidate_acc,
        "delta_vs_reference": delta,
        "paired_ci95": paired_ci(differences),
        "candidate_wins": candidate_wins,
        "reference_wins": reference_wins,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "mcnemar_exact_p": exact_mcnemar_pvalue(candidate_wins, reference_wins) if matched and not errors else None,
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    sweep = read_json(args.sweep_json)
    reference_path = args.reference_predictions
    runs = []
    for row in sweep.get("runs", []):
        if not isinstance(row, dict):
            continue
        root = row.get("root")
        if not row.get("exists") or not root:
            runs.append(
                {
                    "job_id": row.get("job_id"),
                    "steps": row.get("steps"),
                    "lr": row.get("lr"),
                    "status": "pending",
                    "accuracy": row.get("accuracy"),
                    "reason": "metrics or root missing",
                }
            )
            continue
        candidate_path = Path(root) / "eval_predictions.jsonl"
        stats = compare(reference_path, candidate_path)
        accuracy = stats.get("candidate_accuracy")
        paper_delta = accuracy - PAPER_BITNET_SFT_MNLI if isinstance(accuracy, float) else None
        runs.append(
            {
                "job_id": row.get("job_id"),
                "steps": row.get("steps"),
                "lr": row.get("lr"),
                "metrics_accuracy": row.get("accuracy"),
                "prediction_path": str(candidate_path),
                "paper_anchor": PAPER_BITNET_SFT_MNLI,
                "delta_vs_paper_anchor": paper_delta,
                "clears_paper_anchor": bool(isinstance(paper_delta, float) and paper_delta >= 0.0),
                **stats,
            }
        )

    complete = [row for row in runs if row.get("status") == "pass"]
    best = max(complete, key=lambda row: float(row.get("candidate_accuracy") or float("-inf")), default=None)
    return {
        "schema": "bitnet-sft-budget-paired-v1",
        "date": DATE,
        "reference_predictions": str(reference_path),
        "sweep_json": str(args.sweep_json),
        "paper_bitnet_sft_mnli": PAPER_BITNET_SFT_MNLI,
        "expected_mnli_examples": EXPECTED_MNLI,
        "complete": len(complete),
        "total": len(runs),
        "best": best,
        "runs": runs,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    best = summary.get("best") or {}
    if best:
        verdict = (
            f"Best completed row is `{fmt(best.get('candidate_accuracy'))}` at "
            f"steps=`{best.get('steps')}`, lr=`{best.get('lr')}`. It clears the "
            f"paper BitNet-SFT anchor by `{fmt(best.get('delta_vs_paper_anchor'))}` "
            f"but remains `{fmt(best.get('delta_vs_reference'))}` below the paired FP16 reference."
        )
    else:
        verdict = "No completed paired rows are available."

    rows = []
    for row in summary["runs"]:
        rows.append(
            [
                row.get("job_id", "-"),
                row.get("steps", "-"),
                row.get("lr", "-"),
                row.get("status", "-"),
                fmt(row.get("matched")),
                fmt(row.get("reference_accuracy")),
                fmt(row.get("candidate_accuracy")),
                fmt(row.get("delta_vs_reference")),
                fmt_ci(row.get("paired_ci95")),
                fmt(row.get("delta_vs_paper_anchor")),
                fmt(row.get("candidate_wins")),
                fmt(row.get("reference_wins")),
                fmt(row.get("mcnemar_exact_p")),
            ]
        )

    return "\n".join(
        [
            f"# BitNet-SFT Budget Paired Audit, {summary['date']}",
            "",
            verdict,
            "",
            f"Reference prediction trace: `{summary['reference_predictions']}`.",
            f"Completed paired rows: `{summary['complete']}/{summary['total']}`.",
            "",
            md_table(
                [
                    "job",
                    "steps",
                    "lr",
                    "status",
                    "matched n",
                    "FP16 acc",
                    "candidate acc",
                    "candidate - FP16",
                    "paired 95% CI",
                    "candidate - paper",
                    "candidate wins",
                    "FP16 wins",
                    "McNemar p",
                ],
                rows,
            ),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-json",
        type=Path,
        default=Path(f"benchmark_results/bitnet_sft_budget_sweep_{DATE}.json"),
    )
    parser.add_argument(
        "--reference-predictions",
        type=Path,
        default=Path(
            "checkpoints/bitdistill-glue-seqcls-predtrace/Qwen-Qwen2.5-0.5B/mnli/"
            "fp16_sft-tensor-layer-1/eval_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"benchmark_results/bitnet_sft_budget_paired_{DATE}.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"benchmarks/results/bitnet_sft_budget_paired_{DATE}.md"),
    )
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
