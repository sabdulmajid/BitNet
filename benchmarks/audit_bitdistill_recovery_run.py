#!/usr/bin/env python3
"""Audit the controlled MNLI BitDistill recovery run.

The run is intentionally narrow: Qwen2.5-0.5B, sequence classification,
tensor-scale BitDistill, long tensor Stage-2 warmup, SubLN enabled, FP16
teacher, and paper-scale attention gamma. This script is safe to run while the
Slurm job is still pending; it emits a pending report until metrics and
prediction traces exist.
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
SUCCESS_DELTA_FROM_FP = -0.01
Z_95 = 1.959963984540054


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> float | None:
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
    if isinstance(value, list):
        return "[" + ", ".join(fmt(item) for item in value) + "]"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(item).replace("|", "\\|") for item in row) + " |")
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
            errors.append(f"{path}:{lineno}: index is not int")
            continue
        if index in seen:
            errors.append(f"{path}:{lineno}: duplicate index {index}")
            continue
        seen.add(index)
        if not isinstance(label, int) or not isinstance(prediction, int):
            errors.append(f"{path}:{lineno}: label/prediction is not int")
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
    half = z * math.sqrt(variance / n)
    return [mean - half, mean + half]


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
        raise ValueError("one bound is required")
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
        2.0 * min(binomial_tail(discordant, lower_k=low), binomial_tail(discordant, upper_k=high)),
    )


def compare(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference_rows, reference_errors = read_predictions(reference_path)
    candidate_rows, candidate_errors = read_predictions(candidate_path)
    errors = reference_errors + candidate_errors
    missing = [str(path) for path in (reference_path, candidate_path) if not path.exists()]
    status = "pending" if missing else "pass"
    if not missing and len(reference_rows) != len(candidate_rows):
        errors.append(f"row count mismatch: reference={len(reference_rows)}, candidate={len(candidate_rows)}")
    if not missing and len(reference_rows) != EXPECTED_MNLI:
        errors.append(f"reference rows={len(reference_rows)} expected={EXPECTED_MNLI}")
    if not missing and len(candidate_rows) != EXPECTED_MNLI:
        errors.append(f"candidate rows={len(candidate_rows)} expected={EXPECTED_MNLI}")

    matched = 0
    reference_correct = 0
    candidate_correct = 0
    reference_wins = 0
    candidate_wins = 0
    both_correct = 0
    both_wrong = 0
    diffs: list[float] = []
    if not errors and not missing:
        for reference_row, candidate_row in zip(reference_rows, candidate_rows):
            if int(reference_row["index"]) != int(candidate_row["index"]):
                errors.append(f"index mismatch: reference={reference_row['index']} candidate={candidate_row['index']}")
                break
            if int(reference_row["label"]) != int(candidate_row["label"]):
                errors.append(f"label mismatch at index {reference_row['index']}")
                break
            ref_ok = bool(reference_row["correct"])
            cand_ok = bool(candidate_row["correct"])
            matched += 1
            reference_correct += int(ref_ok)
            candidate_correct += int(cand_ok)
            reference_wins += int(ref_ok and not cand_ok)
            candidate_wins += int(cand_ok and not ref_ok)
            both_correct += int(ref_ok and cand_ok)
            both_wrong += int((not ref_ok) and (not cand_ok))
            diffs.append(float(cand_ok) - float(ref_ok))

    if errors and not missing:
        status = "fail"
    reference_accuracy = reference_correct / matched if matched else None
    candidate_accuracy = candidate_correct / matched if matched else None
    delta = candidate_accuracy - reference_accuracy if reference_accuracy is not None and candidate_accuracy is not None else None
    return {
        "status": status,
        "missing": missing,
        "errors": errors,
        "matched": matched,
        "reference_accuracy": reference_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "delta_vs_reference": delta,
        "paired_ci95": paired_ci(diffs),
        "candidate_wins": candidate_wins,
        "reference_wins": reference_wins,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "mcnemar_exact_p": exact_mcnemar_pvalue(candidate_wins, reference_wins) if matched and not errors else None,
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    submission = read_json(args.submission_json)
    metrics_path = args.candidate_dir / "metrics.json"
    candidate_predictions = args.candidate_dir / "eval_predictions.jsonl"
    metrics = read_json(metrics_path)
    eval_metrics = metrics.get("eval", {}) if isinstance(metrics.get("eval"), dict) else {}
    last = metrics.get("last", {}) if isinstance(metrics.get("last"), dict) else {}
    loss_weights = metrics.get("loss_weights", {}) if isinstance(metrics.get("loss_weights"), dict) else {}

    paired = compare(args.reference_predictions, candidate_predictions)
    metric_accuracy = finite(eval_metrics.get("accuracy"))
    metric_examples = finite(eval_metrics.get("eval_examples"))
    if paired["status"] == "pass" and metric_accuracy is not None:
        pred_accuracy = paired["candidate_accuracy"]
        if pred_accuracy is None or abs(pred_accuracy - metric_accuracy) > 1e-12:
            paired["status"] = "fail"
            paired.setdefault("errors", []).append(f"prediction accuracy {pred_accuracy} disagrees with metrics {metric_accuracy}")
    if paired["status"] == "pass" and metric_examples != EXPECTED_MNLI:
        paired["status"] = "fail"
        paired.setdefault("errors", []).append(f"metric eval_examples={metric_examples} expected={EXPECTED_MNLI}")

    delta = paired.get("delta_vs_reference")
    success = bool(paired["status"] == "pass" and isinstance(delta, float) and delta >= SUCCESS_DELTA_FROM_FP)
    if paired["status"] == "pending":
        verdict = "pending"
    elif success:
        verdict = "success"
    else:
        verdict = "not_recovered_to_fp"

    ce = finite(last.get("ce"))
    attention = finite(last.get("attention_kd"))
    weighted_attention = finite(last.get("weighted_attention_kd"))
    logit_kd = finite(last.get("logit_kd"))
    weighted_logit = finite(last.get("weighted_logit_kd"))
    summary = {
        "schema": "bitdistill-recovery-run-audit-v1",
        "date": DATE,
        "submission_json": str(args.submission_json),
        "submission": submission,
        "reference_predictions": str(args.reference_predictions),
        "candidate_dir": str(args.candidate_dir),
        "candidate_metrics": str(metrics_path),
        "candidate_predictions": str(candidate_predictions),
        "expected_examples": EXPECTED_MNLI,
        "success_delta_from_fp": SUCCESS_DELTA_FROM_FP,
        "metrics_exists": metrics_path.exists(),
        "predictions_exists": candidate_predictions.exists(),
        "metric_accuracy": metric_accuracy,
        "metric_eval_examples": metric_examples,
        "steps": metrics.get("steps"),
        "last_loss_components": {
            "ce": ce,
            "logit_kd": logit_kd,
            "weighted_logit_kd": weighted_logit,
            "attention_kd": attention,
            "weighted_attention_kd": weighted_attention,
            "attention_to_ce": weighted_attention / ce if ce not in (None, 0.0) and weighted_attention is not None else None,
            "logit_to_ce": weighted_logit / ce if ce not in (None, 0.0) and weighted_logit is not None else None,
        },
        "loss_weights": loss_weights,
        "paired": paired,
        "success": success,
        "verdict": verdict,
    }
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    paired = summary["paired"]
    loss = summary["last_loss_components"]
    verdict_text = {
        "pending": "Pending: metrics or prediction traces are not available yet.",
        "success": "Success: the candidate is within the configured FP16 recovery gate.",
        "not_recovered_to_fp": "Not recovered: the candidate does not meet the configured FP16 recovery gate.",
    }[summary["verdict"]]
    return "\n\n".join(
        [
            f"# BitDistill Recovery Run Audit, {summary['date']}",
            verdict_text,
            "## Run",
            md_table(
                ["field", "value"],
                [
                    ["candidate_dir", summary["candidate_dir"]],
                    ["metrics_exists", summary["metrics_exists"]],
                    ["predictions_exists", summary["predictions_exists"]],
                    ["steps", summary["steps"]],
                    ["metric_accuracy", summary["metric_accuracy"]],
                    ["metric_eval_examples", summary["metric_eval_examples"]],
                    ["success_delta_from_fp", summary["success_delta_from_fp"]],
                ],
            ),
            "## Paired FP16 Comparison",
            md_table(
                ["metric", "value"],
                [
                    ["status", paired["status"]],
                    ["matched", paired["matched"]],
                    ["reference_accuracy", paired["reference_accuracy"]],
                    ["candidate_accuracy", paired["candidate_accuracy"]],
                    ["delta_vs_reference", paired["delta_vs_reference"]],
                    ["paired_ci95", paired["paired_ci95"]],
                    ["candidate_wins", paired["candidate_wins"]],
                    ["reference_wins", paired["reference_wins"]],
                    ["mcnemar_exact_p", paired["mcnemar_exact_p"]],
                    ["errors", "; ".join(paired.get("errors", []))],
                    ["missing", "; ".join(paired.get("missing", []))],
                ],
            ),
            "## Loss Components",
            md_table(
                ["component", "value"],
                [
                    ["ce", loss["ce"]],
                    ["logit_kd", loss["logit_kd"]],
                    ["weighted_logit_kd", loss["weighted_logit_kd"]],
                    ["logit_to_ce", loss["logit_to_ce"]],
                    ["attention_kd", loss["attention_kd"]],
                    ["weighted_attention_kd", loss["weighted_attention_kd"]],
                    ["attention_to_ce", loss["attention_to_ce"]],
                ],
            ),
            "## Interpretation",
            (
                "This audit uses the local FP16-SFT prediction trace as the reference. "
                "It does not judge paper reproduction until the candidate has full MNLI predictions, "
                "a paired confidence interval, and the configured delta gate is satisfied."
            ),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission-json",
        type=Path,
        default=Path(f"benchmark_results/bitdistill_recovery_submission_{DATE}.json"),
    )
    parser.add_argument(
        "--reference-predictions",
        type=Path,
        default=Path("checkpoints/bitdistill-glue-seqcls-predtrace/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1/eval_predictions.jsonl"),
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-steps10000-lr2em5-papergamma-headinit"),
    )
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_recovery_audit_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_recovery_audit_{DATE}.md"))
    args = parser.parse_args()

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(summary)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
