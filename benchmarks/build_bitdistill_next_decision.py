#!/usr/bin/env python3
"""Build the next-action decision report for the BitDistill reproduction work.

The report is intentionally conservative.  It distinguishes missing evidence
from negative evidence and only recommends the next experiment after the
controlled Stage-2 curve and gamma-balance diagnostics provide enough data.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_STAGE2_TOKENS = 655_360_000
SUCCESS_DELTA_FROM_FP = -0.01
MEANINGFUL_STAGE2_GAIN = 0.015
SATURATION_STAGE2_GAIN = 0.005
BALANCED_MAX_GRAD_ATTENTION_TO_CE = 10.0
DECISION_EPS = 1e-12


def read_json(path: Path, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def at_least(value: float, threshold: float) -> bool:
    return value >= threshold - DECISION_EPS


def at_most(value: float, threshold: float) -> bool:
    return value <= threshold + DECISION_EPS


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != 0.0 and (abs(value) >= 1000.0 or abs(value) < 0.0001):
            return f"{value:.3e}"
        return f"{value:.6f}"
    if isinstance(value, list):
        return ", ".join(fmt(item) for item in value)
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def complete_rows(controlled: dict[str, Any]) -> list[dict[str, Any]]:
    rows = controlled.get("rows", [])
    if not isinstance(rows, list):
        return []
    complete: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        paired = row.get("paired", {})
        if not isinstance(paired, dict) or paired.get("status") != "pass":
            continue
        if not finite(row.get("stage2_token_presentations")) or not finite(row.get("metric_accuracy")):
            continue
        complete.append(row)
    return sorted(complete, key=lambda row: int(row["stage2_token_presentations"]))


def latest_and_previous(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not rows:
        return None, None
    latest = rows[-1]
    previous = rows[-2] if len(rows) > 1 else None
    return latest, previous


def classify_decision(
    gap: dict[str, Any],
    controlled: dict[str, Any],
    gamma: dict[str, Any],
) -> tuple[str, str, list[str]]:
    rows = complete_rows(controlled)
    latest, previous = latest_and_previous(rows)
    evidence_gaps: list[str] = []
    if latest is None:
        return (
            "pending_no_controlled_rows",
            "Do not launch new broad sweeps; first materialize at least one controlled downstream row.",
            ["controlled curve has no completed rows"],
        )

    latest_tokens = int(latest["stage2_token_presentations"])
    latest_accuracy = float(latest["metric_accuracy"])
    paired = latest.get("paired", {}) if isinstance(latest.get("paired"), dict) else {}
    latest_delta_vs_fp = paired.get("delta_vs_reference")
    if not finite(latest_delta_vs_fp):
        evidence_gaps.append("latest controlled row lacks paired delta vs FP16")

    if latest_tokens < TARGET_STAGE2_TOKENS:
        evidence_gaps.append(f"latest controlled row is {latest_tokens:,} tokens, below 655.36M")
        return (
            "pending_655m_downstream",
            "Wait for the active 655.36M Stage-2 producer, downstream MNLI, and postprocess reports.",
            evidence_gaps,
        )

    if finite(latest_delta_vs_fp) and at_least(float(latest_delta_vs_fp), SUCCESS_DELTA_FROM_FP):
        return (
            "replicate_recovery_gate",
            "Do not broaden yet; replicate the recovered row and then run QNLI/SST2 with the same recipe.",
            evidence_gaps,
        )

    previous_gain = None
    if previous is not None and finite(previous.get("metric_accuracy")):
        previous_gain = latest_accuracy - float(previous["metric_accuracy"])
    else:
        evidence_gaps.append("no previous controlled row for Stage-2 marginal-gain calculation")

    gamma_status = str(gamma.get("status", "missing_gamma_balance"))
    gamma_metrics = gamma.get("metrics", {}) if isinstance(gamma.get("metrics"), dict) else {}
    gamma_grad_ratio = gamma_metrics.get("gamma60_final_grad_attention_to_ce")
    gamma_rebalanced = (
        gamma_status == "gamma60_rebalanced_attention_updates"
        or (finite(gamma_grad_ratio) and at_most(float(gamma_grad_ratio), BALANCED_MAX_GRAD_ATTENTION_TO_CE))
    )
    gamma_pending = gamma_status in {"pending_gamma60_telemetry", "missing_gamma_balance"} or not gamma

    if finite(previous_gain) and at_least(float(previous_gain), MEANINGFUL_STAGE2_GAIN):
        return (
            "extend_stage2_curve",
            (
                "Stage-2 still has meaningful marginal gain. Queue the next controlled point "
                "before changing the recipe, while keeping gamma telemetry as a diagnostic."
            ),
            evidence_gaps,
        )

    if gamma_pending:
        evidence_gaps.append("gamma-60 balance diagnostic is still pending")
        return (
            "hold_for_gamma_balance",
            (
                "The 655M quality row does not provide enough evidence by itself; wait for "
                "gamma-balance telemetry before launching another expensive broad run."
            ),
            evidence_gaps,
        )

    if gamma_rebalanced:
        return (
            "run_gamma_balanced_downstream",
            (
                "The Stage-2 marginal gain is weak and gamma-60 rebalances updates. Run a "
                "matched 10k-step downstream MNLI row with the balanced coefficient before "
                "spending more Stage-2 tokens."
            ),
            evidence_gaps,
        )

    if finite(previous_gain) and at_most(float(previous_gain), SATURATION_STAGE2_GAIN):
        return (
            "pause_broad_stage2_audit_recipe",
            (
                "The Stage-2 curve appears to saturate and gamma telemetry did not resolve "
                "the update imbalance. Stop broad budget scaling and audit recipe alignment."
            ),
            evidence_gaps,
        )

    return (
        "ambiguous_recovery_continue_with_controls",
        (
            "Evidence is mixed. Run one narrow ablation at a time: either the next Stage-2 "
            "point or one gamma-balanced downstream row, but do not expand axes."
        ),
        evidence_gaps,
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    gap = read_json(args.reproduction_gap)
    controlled = read_json(args.controlled_curve)
    gamma = read_json(args.gamma_balance, required=False)
    monitor = read_json(args.active_monitor, required=False)
    if gap.get("schema") != "bitnet-reproduction-gap-report-v1":
        raise RuntimeError(f"unexpected reproduction-gap schema: {gap.get('schema')}")
    if controlled.get("schema") != "bitdistill-controlled-curve-audit-v1":
        raise RuntimeError(f"unexpected controlled-curve schema: {controlled.get('schema')}")
    if gamma and gamma.get("schema") != "bitdistill-gamma-balance-audit-v1":
        raise RuntimeError(f"unexpected gamma-balance schema: {gamma.get('schema')}")

    rows = complete_rows(controlled)
    latest, previous = latest_and_previous(rows)
    decision, recommendation, evidence_gaps = classify_decision(gap, controlled, gamma)
    latest_paired = latest.get("paired", {}) if isinstance(latest, dict) and isinstance(latest.get("paired"), dict) else {}
    previous_gain = (
        float(latest["metric_accuracy"]) - float(previous["metric_accuracy"])
        if latest is not None and previous is not None
        else None
    )
    gamma_metrics = gamma.get("metrics", {}) if isinstance(gamma.get("metrics"), dict) else {}
    return {
        "schema": "bitdistill-next-decision-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": decision,
        "quality_claim": "decision_support_not_new_benchmark",
        "recommendation": recommendation,
        "evidence_gaps": evidence_gaps,
        "thresholds": {
            "target_stage2_tokens": TARGET_STAGE2_TOKENS,
            "success_delta_from_fp": SUCCESS_DELTA_FROM_FP,
            "meaningful_stage2_gain": MEANINGFUL_STAGE2_GAIN,
            "saturation_stage2_gain": SATURATION_STAGE2_GAIN,
            "balanced_max_grad_attention_to_ce": BALANCED_MAX_GRAD_ATTENTION_TO_CE,
            "decision_eps": DECISION_EPS,
        },
        "latest_controlled_row": {
            "stage2_token_presentations": latest.get("stage2_token_presentations") if latest else None,
            "accuracy": latest.get("metric_accuracy") if latest else None,
            "delta_vs_fp16": latest_paired.get("delta_vs_reference"),
            "paired_ci95": latest_paired.get("paired_ci95"),
            "passes_fp_recovery_gate": latest.get("passed_fp_recovery_gate") if latest else None,
        },
        "previous_controlled_row": {
            "stage2_token_presentations": previous.get("stage2_token_presentations") if previous else None,
            "accuracy": previous.get("metric_accuracy") if previous else None,
        },
        "marginal_stage2_gain": previous_gain,
        "gamma_balance": {
            "status": gamma.get("status") if gamma else "missing_gamma_balance",
            "paper_final_grad_attention_to_ce": gamma_metrics.get("paper_final_grad_attention_to_ce"),
            "gamma60_final_grad_attention_to_ce": gamma_metrics.get("gamma60_final_grad_attention_to_ce"),
            "gamma60_attention_grad_reduction_factor": gamma_metrics.get("gamma60_attention_grad_reduction_factor"),
        },
        "active_monitor": {
            "status": monitor.get("status"),
            "stage2_job_id": (monitor.get("stage2") or {}).get("job_id") if isinstance(monitor.get("stage2"), dict) else None,
            "latest_step": (
                ((monitor.get("stage2") or {}).get("latest_step") or {}).get("step")
                if isinstance((monitor.get("stage2") or {}).get("latest_step"), dict)
                else None
            ),
            "downstream_status": (monitor.get("downstream") or {}).get("status")
            if isinstance(monitor.get("downstream"), dict)
            else None,
        },
        "source_paths": {
            "reproduction_gap": str(args.reproduction_gap),
            "controlled_curve": str(args.controlled_curve),
            "gamma_balance": str(args.gamma_balance),
            "active_monitor": str(args.active_monitor),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    latest = report["latest_controlled_row"]
    previous = report["previous_controlled_row"]
    gamma = report["gamma_balance"]
    thresholds = report["thresholds"]
    return "\n\n".join(
        [
            "# BitDistill Next Decision",
            f"Generated: `{report['created_utc']}`",
            f"Status: **{report['status']}**.",
            f"Quality claim: **{report['quality_claim']}**.",
            "## Recommendation",
            report["recommendation"],
            "## Evidence",
            md_table(
                ["field", "value"],
                [
                    ["latest Stage-2 tokens", latest["stage2_token_presentations"]],
                    ["latest accuracy", latest["accuracy"]],
                    ["latest delta vs FP16", latest["delta_vs_fp16"]],
                    ["latest paired CI95", latest["paired_ci95"]],
                    ["latest passes FP recovery gate", latest["passes_fp_recovery_gate"]],
                    ["previous Stage-2 tokens", previous["stage2_token_presentations"]],
                    ["previous accuracy", previous["accuracy"]],
                    ["marginal Stage-2 gain", report["marginal_stage2_gain"]],
                    ["gamma status", gamma["status"]],
                    ["paper grad attention/CE", gamma["paper_final_grad_attention_to_ce"]],
                    ["gamma60 grad attention/CE", gamma["gamma60_final_grad_attention_to_ce"]],
                    ["gamma60 grad reduction factor", gamma["gamma60_attention_grad_reduction_factor"]],
                ],
            ),
            "## Thresholds",
            md_table(
                ["threshold", "value"],
                [
                    ["target Stage-2 tokens", thresholds["target_stage2_tokens"]],
                    ["success delta from FP16", thresholds["success_delta_from_fp"]],
                    ["meaningful Stage-2 gain", thresholds["meaningful_stage2_gain"]],
                    ["saturation Stage-2 gain", thresholds["saturation_stage2_gain"]],
                    ["balanced max grad attention/CE", thresholds["balanced_max_grad_attention_to_ce"]],
                    ["decision epsilon", thresholds["decision_eps"]],
                ],
            ),
            "## Evidence Gaps",
            md_table(
                ["gap"],
                [[gap] for gap in report["evidence_gaps"]] or [["none"]],
            ),
            "## Source Paths",
            md_table(
                ["artifact", "path"],
                [[key, value] for key, value in report["source_paths"].items()],
            ),
            (
                "This report is decision support. It does not create new benchmark evidence "
                "and must not be cited as a quality result without the source reports."
            ),
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reproduction-gap",
        type=Path,
        default=Path("benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json"),
    )
    parser.add_argument(
        "--controlled-curve",
        type=Path,
        default=Path("benchmarks/results/bitdistill_controlled_curve_2026-05-23.json"),
    )
    parser.add_argument(
        "--gamma-balance",
        type=Path,
        default=Path("benchmarks/results/gamma60_gradient_balance_2026-05-23.json"),
    )
    parser.add_argument(
        "--active-monitor",
        type=Path,
        default=Path("benchmarks/results/active_stage2_extension_monitor_2026-05-23.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/bitdistill_next_decision_2026-05-23.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/bitdistill_next_decision_2026-05-23.md"),
    )
    args = parser.parse_args()

    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
