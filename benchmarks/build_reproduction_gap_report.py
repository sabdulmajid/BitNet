#!/usr/bin/env python3
"""Build a manifest-style report for the current BitDistill reproduction gap."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PAPER_STAGE2_TOKENS = 10_000_000_000
SUCCESS_DELTA_FROM_FP = -0.01


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_path(path: str | Path, errors: list[str]) -> None:
    if not Path(path).exists():
        errors.append(f"missing artifact path: {path}")


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def pct(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        read_json(args.canonical_bundle),
        read_json(args.controlled_curve),
        read_json(args.bitnet_sft_budget),
        read_json(args.training_dynamics),
    )


def validate_inputs(
    canonical: dict[str, Any],
    controlled: dict[str, Any],
    bitnet_budget: dict[str, Any],
    dynamics: dict[str, Any],
    errors: list[str],
) -> None:
    if canonical.get("schema") != "bitnet-canonical-evidence-bundle-v1":
        errors.append(f"unexpected canonical schema: {canonical.get('schema')}")
    if controlled.get("schema") not in {"bitdistill-controlled-curve-v1", "bitdistill-controlled-curve-audit-v1"}:
        errors.append(f"unexpected controlled curve schema: {controlled.get('schema')}")
    if dynamics.get("schema") != "bitdistill-training-dynamics-audit-v1":
        errors.append(f"unexpected training-dynamics schema: {dynamics.get('schema')}")
    if not isinstance(bitnet_budget.get("best"), dict):
        errors.append("bitnet_sft_budget missing best row")
    if controlled.get("complete") != controlled.get("expected"):
        errors.append(f"controlled curve incomplete: {controlled.get('complete')}/{controlled.get('expected')}")

    claims = canonical.get("claims")
    if not isinstance(claims, dict):
        errors.append("canonical bundle missing claims")
        return
    bitdistill = claims.get("bitdistill_reproduction", {})
    if not isinstance(bitdistill, dict):
        errors.append("canonical bundle missing bitdistill_reproduction claim")
    elif bitdistill.get("status") == "reproduced":
        errors.append("canonical BitDistill claim says reproduced; this gap report is stale")

    artifacts = canonical.get("artifacts", {})
    if isinstance(artifacts, dict):
        for label, artifact in artifacts.items():
            if isinstance(artifact, dict) and isinstance(artifact.get("path"), str):
                require_path(artifact["path"], errors)

    best = bitnet_budget.get("best", {})
    if isinstance(best, dict):
        require_path(best.get("metrics_path", ""), errors)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    canonical, controlled, bitnet_budget, dynamics = load_inputs(args)
    errors: list[str] = []
    validate_inputs(canonical, controlled, bitnet_budget, dynamics, errors)
    if errors:
        raise RuntimeError("\n".join(errors))

    claims = canonical["claims"]
    bitdistill = claims["bitdistill_reproduction"]
    best_bitnet = bitnet_budget["best"]
    rows = controlled["rows"]
    controlled_by_tokens = {int(row["stage2_token_presentations"]): row for row in rows}
    row_40m = controlled_by_tokens[40_960_000]
    row_163m = controlled_by_tokens[163_840_000]
    row_327m = controlled_by_tokens[327_680_000]
    latest_tokens = max(controlled_by_tokens)
    latest_row = controlled_by_tokens[latest_tokens]

    fp16 = float(bitdistill["fp16_sft_mnli"])
    bitnet_best = float(best_bitnet["accuracy"])
    bitnet_default = float(bitnet_budget["default_baseline_accuracy"])
    paper_bitnet = float(bitnet_budget["paper_anchor"])
    bitdistill_327 = float(row_327m["metric_accuracy"])
    bitdistill_latest = float(latest_row["metric_accuracy"])
    latest_delta_vs_fp = bitdistill_latest - fp16
    status = "reproduced" if latest_delta_vs_fp >= SUCCESS_DELTA_FROM_FP else "not_reproduced"
    dynamics_traces = dynamics.get("traces", [])
    controlled_traces = [
        trace for trace in dynamics_traces if isinstance(trace, dict) and trace.get("kind") == "controlled"
    ]
    strongest_trace = max(
        controlled_traces,
        key=lambda trace: float(trace.get("final_grad_attention_to_ce") or 0.0),
        default={},
    )

    conclusions = [
        {
            "finding": "The short BitNet-SFT default was undertrained.",
            "evidence": (
                f"default {bitnet_default:.6f}; best budget row {bitnet_best:.6f}; "
                f"gain {bitnet_best - bitnet_default:+.6f}"
            ),
        },
        {
            "finding": "The local BitNet-SFT anchor is no longer the primary blocker.",
            "evidence": (
                f"best BitNet-SFT {bitnet_best:.6f}; paper BitNet-SFT anchor {paper_bitnet:.6f}; "
                f"delta {bitnet_best - paper_bitnet:+.6f}"
            ),
        },
        {
            "finding": "BitDistill recovery is evaluated against the latest completed Stage-2 row.",
            "evidence": (
                f"{latest_tokens / 1_000_000:.2f}M BitDistill MNLI {bitdistill_latest:.6f}; "
                f"FP16 {fp16:.6f}; delta {latest_delta_vs_fp:+.6f}; "
                f"gate {SUCCESS_DELTA_FROM_FP:+.6f}"
            ),
        },
        {
            "finding": "Stage-2 budget helps, but current budget is still small relative to the paper.",
            "evidence": (
                f"40.96M {float(row_40m['metric_accuracy']):.6f}; "
                f"163.84M {float(row_163m['metric_accuracy']):.6f}; "
                f"327.68M {bitdistill_327:.6f}; "
                f"latest {latest_tokens / 1_000_000:.2f}M {bitdistill_latest:.6f}; "
                f"latest paper fraction {pct(latest_tokens / PAPER_STAGE2_TOKENS)}"
            ),
        },
        {
            "finding": "Local paper-gamma training dynamics are still suspect.",
            "evidence": (
                f"final grad attention/CE {float(strongest_trace.get('final_grad_attention_to_ce') or 0.0):.6f}; "
                f"final loss attention/CE {float(strongest_trace.get('final_loss_attention_to_ce') or 0.0):.6f}"
            ),
        },
    ]

    if latest_tokens >= 655_360_000:
        budget_gate = {
            "gate": "Gamma-balanced downstream MNLI",
            "why": (
                "The 327.68M to 655.36M gain is modest while paper-gamma attention gradients "
                "dominate CE under the local reductions."
            ),
            "minimum_next_point": (
                "matched 10k-step MNLI from the 655.36M checkpoint with attention-KD gamma 60"
            ),
        }
    else:
        budget_gate = {
            "gate": "Stage-2 token-budget curve",
            "why": "Determine whether MNLI continues improving toward FP or saturates far below it.",
            "minimum_next_point": "655.36M cumulative token presentations with the same downstream recipe",
        }

    next_gates = [
        budget_gate,
        {
            "gate": "Loss-normalization/gradient-balance sweep",
            "why": "Paper gamma is only comparable if CE, logits KD, and attention KD reductions match.",
            "minimum_next_point": "full-quality comparison of gamma 60 versus paper gamma with all other axes fixed",
        },
        {
            "gate": "Same-artifact runtime quality",
            "why": "The strongest PyTorch classifier result and strongest packed causal runtime are still separate artifacts.",
            "minimum_next_point": "packed classifier head or primary causal prompt-scoring evaluation",
        },
        {
            "gate": "Backbone alignment",
            "why": "Paper-scale claims need exact/closest public Qwen3/Qwen2.5 recipe alignment.",
            "minimum_next_point": "one Qwen3-0.6B or exact Qwen2.5-0.5B MNLI run with matched logging",
        },
    ]

    artifacts = {
        "canonical_bundle": args.canonical_bundle,
        "controlled_curve": args.controlled_curve,
        "bitnet_sft_budget": args.bitnet_sft_budget,
        "training_dynamics": args.training_dynamics,
    }
    return {
        "schema": "bitnet-reproduction-gap-report-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "artifacts": {
            label: {"path": str(path), "sha256": sha256(path)} for label, path in artifacts.items()
        },
        "metrics": {
            "fp16_sft_mnli": fp16,
            "bitnet_sft_default_mnli": bitnet_default,
            "bitnet_sft_best_mnli": bitnet_best,
            "bitnet_sft_best_delta_vs_default": bitnet_best - bitnet_default,
            "bitnet_sft_best_delta_vs_paper_anchor": bitnet_best - paper_bitnet,
            "bitnet_sft_best_delta_vs_fp16": bitnet_best - fp16,
            "bitdistill_40_96m_mnli": float(row_40m["metric_accuracy"]),
            "bitdistill_163_84m_mnli": float(row_163m["metric_accuracy"]),
            "bitdistill_327_68m_mnli": bitdistill_327,
            "bitdistill_327_68m_delta_vs_bitnet_best": bitdistill_327 - bitnet_best,
            "bitdistill_327_68m_delta_vs_fp16": bitdistill_327 - fp16,
            "bitdistill_327_68m_ci95": row_327m["paired"]["paired_ci95"],
            "stage2_327_68m_fraction_of_paper": 327_680_000 / PAPER_STAGE2_TOKENS,
            "bitdistill_latest_stage2_tokens": latest_tokens,
            "bitdistill_latest_mnli": bitdistill_latest,
            "bitdistill_latest_delta_vs_bitnet_best": bitdistill_latest - bitnet_best,
            "bitdistill_latest_delta_vs_fp16": latest_delta_vs_fp,
            "bitdistill_latest_ci95": latest_row["paired"]["paired_ci95"],
            "bitdistill_latest_fraction_of_paper": latest_tokens / PAPER_STAGE2_TOKENS,
            "success_delta_from_fp16": SUCCESS_DELTA_FROM_FP,
            "final_grad_attention_to_ce": strongest_trace.get("final_grad_attention_to_ce"),
            "final_loss_attention_to_ce": strongest_trace.get("final_loss_attention_to_ce"),
            "controlled_trace_count": dynamics.get("controlled_trace_count"),
            "materialized_controlled_trace_count": dynamics.get("materialized_controlled_count"),
        },
        "conclusions": conclusions,
        "next_gates": next_gates,
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    conclusion_rows = [[row["finding"], row["evidence"]] for row in report["conclusions"]]
    gate_rows = [[row["gate"], row["why"], row["minimum_next_point"]] for row in report["next_gates"]]
    artifact_rows = [
        [label, artifact["path"], artifact["sha256"]]
        for label, artifact in sorted(report["artifacts"].items())
    ]
    return "\n\n".join(
        [
            "# BitDistill Reproduction Gap Report",
            (
                f"Status: **{report['status']}**. This report separates the now-improved "
                "BitNet-SFT baseline from the remaining BitDistill/FP recovery gap."
            ),
            md_table(
                ["metric", "value"],
                [
                    ["FP16-SFT MNLI", metrics["fp16_sft_mnli"]],
                    ["BitNet-SFT default MNLI", metrics["bitnet_sft_default_mnli"]],
                    ["BitNet-SFT best MNLI", metrics["bitnet_sft_best_mnli"]],
                    ["BitNet-SFT best vs default", metrics["bitnet_sft_best_delta_vs_default"]],
                    ["BitNet-SFT best vs paper anchor", metrics["bitnet_sft_best_delta_vs_paper_anchor"]],
                    ["BitNet-SFT best vs FP16", metrics["bitnet_sft_best_delta_vs_fp16"]],
                    ["BitDistill 40.96M MNLI", metrics["bitdistill_40_96m_mnli"]],
                    ["BitDistill 163.84M MNLI", metrics["bitdistill_163_84m_mnli"]],
                    ["BitDistill 327.68M MNLI", metrics["bitdistill_327_68m_mnli"]],
                    ["BitDistill 327.68M vs BitNet-SFT best", metrics["bitdistill_327_68m_delta_vs_bitnet_best"]],
                    ["BitDistill 327.68M vs FP16", metrics["bitdistill_327_68m_delta_vs_fp16"]],
                    ["BitDistill 327.68M CI95", metrics["bitdistill_327_68m_ci95"]],
                    ["327.68M as paper Stage-2 fraction", pct(metrics["stage2_327_68m_fraction_of_paper"])],
                    ["BitDistill latest Stage-2 tokens", metrics["bitdistill_latest_stage2_tokens"]],
                    ["BitDistill latest MNLI", metrics["bitdistill_latest_mnli"]],
                    ["BitDistill latest vs BitNet-SFT best", metrics["bitdistill_latest_delta_vs_bitnet_best"]],
                    ["BitDistill latest vs FP16", metrics["bitdistill_latest_delta_vs_fp16"]],
                    ["BitDistill latest CI95", metrics["bitdistill_latest_ci95"]],
                    ["latest as paper Stage-2 fraction", pct(metrics["bitdistill_latest_fraction_of_paper"])],
                    ["success delta from FP16", metrics["success_delta_from_fp16"]],
                    ["final grad attention/CE", metrics["final_grad_attention_to_ce"]],
                    ["final loss attention/CE", metrics["final_loss_attention_to_ce"]],
                    ["controlled telemetry traces", metrics["materialized_controlled_trace_count"]],
                ],
            ),
            "## Conclusions",
            md_table(["finding", "evidence"], conclusion_rows),
            "## Next Gates",
            md_table(["gate", "why", "minimum next point"], gate_rows),
            "## Artifact Inventory",
            md_table(["label", "path", "sha256"], artifact_rows),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-bundle",
        type=Path,
        default=Path("benchmarks/results/canonical_evidence_bundle_2026-05-20.json"),
    )
    parser.add_argument(
        "--controlled-curve",
        type=Path,
        default=Path("benchmarks/results/bitdistill_controlled_curve_2026-05-20.json"),
    )
    parser.add_argument(
        "--bitnet-sft-budget",
        type=Path,
        default=Path("benchmarks/results/bitnet_sft_budget_sweep_2026-05-23.json"),
    )
    parser.add_argument(
        "--training-dynamics",
        type=Path,
        default=Path("benchmarks/results/bitdistill_training_dynamics_2026-05-23.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/bitdistill_reproduction_gap_2026-05-23.md"),
    )
    args = parser.parse_args()

    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report).rstrip() + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
