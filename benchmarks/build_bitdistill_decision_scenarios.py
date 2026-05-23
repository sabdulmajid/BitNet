#!/usr/bin/env python3
"""Build an auditable scenario matrix for the next BitDistill decision gate.

The active 655.36M run is not complete yet.  This report makes the decision
logic reviewable before the result arrives by applying the same classifier used
by build_bitdistill_next_decision.py to representative hypothetical outcomes.
It is decision-policy evidence, not benchmark evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_bitdistill_next_decision import (
    BALANCED_MAX_GRAD_ATTENTION_TO_CE,
    DECISION_EPS,
    MEANINGFUL_STAGE2_GAIN,
    SATURATION_STAGE2_GAIN,
    SUCCESS_DELTA_FROM_FP,
    TARGET_STAGE2_TOKENS,
    classify_decision,
    complete_rows,
    fmt,
    md_table,
    read_json,
)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def clone_with_655m_row(controlled: dict[str, Any], accuracy: float, fp16_accuracy: float) -> dict[str, Any]:
    rows = complete_rows(controlled)
    if not rows:
        raise RuntimeError("controlled curve has no completed rows to clone")
    base = copy.deepcopy(rows[-1])
    base["job_id"] = "hypothetical-655m"
    base["label"] = "hypothetical 655.36M downstream control"
    base["stage2_token_presentations"] = TARGET_STAGE2_TOKENS
    base["paper_stage2_fraction"] = TARGET_STAGE2_TOKENS / 10_000_000_000
    base["metric_accuracy"] = accuracy
    base["metrics_exists"] = True
    base["predictions_exists"] = True
    base["passed_fp_recovery_gate"] = (accuracy - fp16_accuracy) >= SUCCESS_DELTA_FROM_FP
    paired = copy.deepcopy(base.get("paired", {})) if isinstance(base.get("paired"), dict) else {}
    paired["status"] = "pass"
    paired["candidate_accuracy"] = accuracy
    paired["reference_accuracy"] = fp16_accuracy
    paired["delta_vs_reference"] = accuracy - fp16_accuracy
    paired["paired_ci95"] = None
    base["paired"] = paired

    simulated = copy.deepcopy(controlled)
    existing = [
        row
        for row in simulated.get("rows", [])
        if not (
            isinstance(row, dict)
            and finite(row.get("stage2_token_presentations"))
            and int(row["stage2_token_presentations"]) == TARGET_STAGE2_TOKENS
        )
    ]
    existing.append(base)
    simulated["rows"] = existing
    simulated["complete"] = len([row for row in existing if isinstance(row, dict)])
    simulated["expected"] = simulated["complete"]
    return simulated


def gamma_state(status: str) -> dict[str, Any]:
    if status == "pending":
        return {
            "schema": "bitdistill-gamma-balance-audit-v1",
            "status": "pending_gamma60_telemetry",
            "metrics": {
                "paper_final_grad_attention_to_ce": 221.3849856028098,
                "gamma60_final_grad_attention_to_ce": None,
            },
        }
    if status == "rebalanced":
        return {
            "schema": "bitdistill-gamma-balance-audit-v1",
            "status": "gamma60_rebalanced_attention_updates",
            "metrics": {
                "paper_final_grad_attention_to_ce": 221.3849856028098,
                "gamma60_final_grad_attention_to_ce": BALANCED_MAX_GRAD_ATTENTION_TO_CE / 2.0,
                "gamma60_attention_grad_reduction_factor": 44.27699712056196,
            },
        }
    if status == "still_dominated":
        return {
            "schema": "bitdistill-gamma-balance-audit-v1",
            "status": "gamma60_still_attention_dominated",
            "metrics": {
                "paper_final_grad_attention_to_ce": 221.3849856028098,
                "gamma60_final_grad_attention_to_ce": BALANCED_MAX_GRAD_ATTENTION_TO_CE * 5.0,
                "gamma60_attention_grad_reduction_factor": 4.427699712056196,
            },
        }
    raise ValueError(status)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    gap = read_json(args.reproduction_gap)
    controlled = read_json(args.controlled_curve)
    if gap.get("schema") != "bitnet-reproduction-gap-report-v1":
        raise RuntimeError(f"unexpected reproduction-gap schema: {gap.get('schema')}")
    if controlled.get("schema") != "bitdistill-controlled-curve-audit-v1":
        raise RuntimeError(f"unexpected controlled-curve schema: {controlled.get('schema')}")

    rows = complete_rows(controlled)
    latest = rows[-1]
    fp16 = float(gap["metrics"]["fp16_sft_mnli"])
    current_accuracy = float(latest["metric_accuracy"])
    pass_accuracy = fp16 + SUCCESS_DELTA_FROM_FP
    scenarios = [
        {
            "label": "flat 655M",
            "accuracy": current_accuracy,
            "gamma": "pending",
            "why": "No downstream improvement and gamma telemetry still absent.",
        },
        {
            "label": "saturated 655M, gamma rebalanced",
            "accuracy": current_accuracy + SATURATION_STAGE2_GAIN,
            "gamma": "rebalanced",
            "why": "Stage-2 gain is weak but gamma-60 fixes update balance.",
        },
        {
            "label": "saturated 655M, gamma still dominated",
            "accuracy": current_accuracy + SATURATION_STAGE2_GAIN,
            "gamma": "still_dominated",
            "why": "Stage-2 gain is weak and gamma-60 does not fix update balance.",
        },
        {
            "label": "ambiguous mid gain",
            "accuracy": current_accuracy + (SATURATION_STAGE2_GAIN + MEANINGFUL_STAGE2_GAIN) / 2.0,
            "gamma": "still_dominated",
            "why": "Improvement exists but is below the meaningful-gain threshold.",
        },
        {
            "label": "meaningful gain",
            "accuracy": current_accuracy + MEANINGFUL_STAGE2_GAIN,
            "gamma": "pending",
            "why": "Stage-2 gain remains large enough to justify another controlled point.",
        },
        {
            "label": "FP recovery gate",
            "accuracy": pass_accuracy,
            "gamma": "pending",
            "why": "The row reaches the configured within-1pt FP16 recovery gate.",
        },
    ]

    decisions = []
    for scenario in scenarios:
        simulated = clone_with_655m_row(controlled, float(scenario["accuracy"]), fp16)
        gamma = gamma_state(str(scenario["gamma"]))
        status, recommendation, gaps = classify_decision(gap, simulated, gamma)
        decisions.append(
            {
                **scenario,
                "delta_vs_327m": float(scenario["accuracy"]) - current_accuracy,
                "delta_vs_fp16": float(scenario["accuracy"]) - fp16,
                "decision_status": status,
                "recommendation": recommendation,
                "evidence_gaps": gaps,
            }
        )

    return {
        "schema": "bitdistill-decision-scenarios-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quality_claim": "decision_policy_not_benchmark",
        "current_completed_accuracy": current_accuracy,
        "fp16_sft_mnli": fp16,
        "thresholds": {
            "target_stage2_tokens": TARGET_STAGE2_TOKENS,
            "success_delta_from_fp": SUCCESS_DELTA_FROM_FP,
            "success_accuracy": pass_accuracy,
            "meaningful_stage2_gain": MEANINGFUL_STAGE2_GAIN,
            "saturation_stage2_gain": SATURATION_STAGE2_GAIN,
            "balanced_max_grad_attention_to_ce": BALANCED_MAX_GRAD_ATTENTION_TO_CE,
            "decision_eps": DECISION_EPS,
        },
        "scenarios": decisions,
        "source_paths": {
            "reproduction_gap": str(args.reproduction_gap),
            "controlled_curve": str(args.controlled_curve),
        },
        "caveat": (
            "This report simulates decision outcomes using existing thresholds. "
            "It does not add benchmark evidence or predict the 655M result."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    scenario_rows = [
        [
            row["label"],
            row["accuracy"],
            row["delta_vs_327m"],
            row["delta_vs_fp16"],
            row["gamma"],
            row["decision_status"],
            row["recommendation"],
        ]
        for row in report["scenarios"]
    ]
    return "\n\n".join(
        [
            "# BitDistill Decision Scenarios",
            f"Generated: `{report['created_utc']}`",
            f"Quality claim: **{report['quality_claim']}**.",
            report["caveat"],
            "## Thresholds",
            md_table(
                ["threshold", "value"],
                [[key, value] for key, value in report["thresholds"].items()],
            ),
            "## Scenario Matrix",
            md_table(
                [
                    "scenario",
                    "hypothetical 655M accuracy",
                    "delta vs 327M",
                    "delta vs FP16",
                    "gamma status",
                    "decision",
                    "recommendation",
                ],
                scenario_rows,
            ),
            "## Interpretation",
            (
                "Use this matrix to audit the policy before the 655M result arrives. "
                "When the real downstream prediction trace exists, "
                "`bitdistill_next_decision_2026-05-23` is the authoritative report."
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
        default=Path("benchmarks/results/bitdistill_controlled_curve_2026-05-20.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/bitdistill_decision_scenarios_2026-05-23.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/bitdistill_decision_scenarios_2026-05-23.md"),
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
