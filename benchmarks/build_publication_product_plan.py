#!/usr/bin/env python3
"""Build a publication and product plan from current BitDistill evidence.

This report is designed for public/profile-facing planning and deep technical
review. It is intentionally explicit about claim boundaries, because the
project's useful direction is a bounded evaluator/runtime stack, not a universal
ternary converter.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def require_schema(path: Path, data: dict[str, Any], expected: str) -> None:
    if data.get("schema") != expected:
        raise RuntimeError(f"{path}: expected schema {expected}, got {data.get('schema')}")


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != 0.0 and (abs(value) >= 10000.0 or abs(value) < 0.0001):
            return f"{value:.3e}"
        return f"{value:.6f}"
    if isinstance(value, list):
        if not value:
            return "none"
        return ", ".join(fmt(item) for item in value)
    if isinstance(value, dict):
        if not value:
            return "none"
        return ", ".join(f"{key}={fmt(val)}" for key, val in value.items())
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def find_headline(scoreboard: dict[str, Any], area: str) -> dict[str, Any]:
    rows = scoreboard.get("headline_rows")
    if not isinstance(rows, list):
        raise RuntimeError("scoreboard missing headline_rows")
    for row in rows:
        if isinstance(row, dict) and row.get("area") == area:
            return row
    raise RuntimeError(f"scoreboard missing area {area}")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    bundle = read_json(args.canonical_bundle)
    scoreboard = read_json(args.scoreboard)
    traceability = read_json(args.traceability)
    product_scope = read_json(args.product_scope)
    next_decision = read_json(args.next_decision)
    require_schema(args.canonical_bundle, bundle, "bitnet-canonical-evidence-bundle-v1")
    require_schema(args.scoreboard, scoreboard, "bitdistill-benchmark-scoreboard-v1")
    require_schema(args.traceability, traceability, "bitdistill-goal-traceability-audit-v1")
    require_schema(args.product_scope, product_scope, "bitnet-product-scope-gate-v1")
    require_schema(args.next_decision, next_decision, "bitdistill-next-decision-v1")

    claims = bundle["claims"]
    blind = claims["blind_ptq"]
    qat = claims["qat_distill"]
    bitdistill = claims["bitdistill_reproduction"]
    row_contract = claims["row_scale_runtime_contract"]
    i2sr = claims["i2sr_cpu"]
    native = claims["native_classifier"]
    latest_controlled = next_decision["latest_controlled_row"]
    latest_tokens = int(latest_controlled["stage2_token_presentations"])
    latest_mnli = float(latest_controlled["accuracy"])
    latest_delta = float(latest_controlled["delta_vs_fp16"])

    publishable_units = [
        {
            "unit": "Negative PTQ result",
            "claim": "Blind FP/BF16-to-ternary projection is not viable for the tested dense-Qwen setup.",
            "evidence": find_headline(scoreboard, "Blind ternary PTQ")["evidence"],
            "publishable_now": True,
            "risk": "Scope must remain tested dense Qwen; do not universalize to every architecture.",
        },
        {
            "unit": "Row-scale runtime contract",
            "claim": "Row scales are learned model semantics and must be preserved by CPU formats.",
            "evidence": find_headline(scoreboard, "Row-scale runtime contract")["evidence"],
            "publishable_now": True,
            "risk": "This supports I2_SR/row-scale contracts, not TL2 row-scale support.",
        },
        {
            "unit": "I2_SR CPU runtime prototype",
            "claim": "A compatible row-scale ternary causal artifact can run through packed CPU I2_SR.",
            "evidence": find_headline(scoreboard, "Packed CPU I2_SR")["evidence"],
            "publishable_now": True,
            "risk": "Do not claim Q4-quality or Q4-storage competitiveness.",
        },
        {
            "unit": "BitDistill reproduction gap",
            "claim": "Local BitDistill-style training is improving but has not reproduced paper-level GLUE quality.",
            "evidence": (
                f"MNLI 40.96M {bitdistill['controlled_40_96m_mnli']:.6f}; "
                f"163.84M {bitdistill['controlled_163_84m_mnli']:.6f}; "
                f"327.68M {bitdistill['controlled_327_68m_mnli']:.6f}; "
                f"{latest_tokens / 1_000_000:.2f}M {latest_mnli:.6f}; "
                f"latest delta vs FP16 {latest_delta:+.6f}"
            ),
            "publishable_now": True,
            "risk": "Must frame as a reproduction-gap study until a within-1pt row is reproduced.",
        },
        {
            "unit": "Product-ready classifier",
            "claim": "Native packed sequence classification is not product-ready yet.",
            "evidence": find_headline(scoreboard, "Native sequence classification")["evidence"],
            "publishable_now": False,
            "risk": "Agreement and quality are below gate; useful only as a research demo.",
        },
        {
            "unit": "MoE/Kimi",
            "claim": "MoE/Kimi support is not proven beyond tiny Qwen2MoE plumbing.",
            "evidence": find_headline(scoreboard, "MoE / Kimi")["evidence"],
            "publishable_now": False,
            "risk": "No real Kimi artifact, quality benchmark, or routed expert locality proof exists.",
        },
    ]

    product_mvp = {
        "name": "CPU-first ternary retrofit evaluator",
        "target_user": "Engineers deciding whether a model-task pair is worth ternary distillation and CPU deployment.",
        "value": (
            "It prevents false converter claims by producing a fail-closed decision report: "
            "quality delta, paired confidence intervals, PPL, file size, RSS, prompt/decode speed, "
            "runtime compatibility, and claim label."
        ),
        "input": "A Hugging Face model, task/eval dataset, and target CPU/runtime profile.",
        "output": "Pass/fail evidence bundle plus suggested path: reject PTQ, try BitDistill/QAT, use row-scale I2_SR, or stop.",
        "current_readiness": product_scope["scope_status"],
        "why_useful_now": (
            "The negative PTQ and row-scale runtime findings are already actionable even when the answer is no."
        ),
    }

    gates = [
        {
            "gate": "Completed 655M Stage-2 downstream MNLI",
            "status": next_decision["status"],
            "decision_rule": next_decision["recommendation"],
            "success_condition": "Meaningful marginal gain or within-1pt FP recovery gate.",
            "failure_condition": "Saturation far below FP, requiring recipe/loss audit instead of broader sweeps.",
        },
        {
            "gate": "Gamma-60 component-gradient telemetry",
            "status": next_decision["gamma_balance"]["status"],
            "decision_rule": "If gamma-60 rebalances attention/CE updates, run matched downstream MNLI.",
            "success_condition": "Attention-KD gradient no longer dominates CE by the current local threshold.",
            "failure_condition": "Attention remains dominant, indicating deeper loss-normalization or recipe mismatch.",
        },
        {
            "gate": "Same-artifact task quality plus CPU runtime",
            "status": "not_ready",
            "decision_rule": "Choose packed classifier runtime or causal prompt-scoring product surface after MNLI recovery.",
            "success_condition": "One artifact provides task quality, agreement, RSS, file size, and throughput.",
            "failure_condition": "Quality proof remains PyTorch-only while runtime proof remains causal-only.",
        },
    ]

    paper_outline = [
        {
            "section": "Problem",
            "content": "Extreme ternary retrofit is a representation-learning and runtime-contract problem, not a file conversion problem.",
        },
        {
            "section": "Negative result",
            "content": (
                f"Blind PTQ collapses: FP PPL {blind['fp_wikitext_ppl']:.6f}, "
                f"PTQ PPL {blind['ptq_wikitext_ppl']:.6f}, FP mean {blind['fp_ten_task_mean']:.6f}, "
                f"PTQ mean {blind['ptq_ten_task_mean']:.6f}."
            ),
        },
        {
            "section": "Recovery path",
            "content": (
                f"Row-scale QAT recovers {qat['recovery_vs_ptq']:+.6f} over PTQ but remains "
                f"{qat['gap_vs_fp']:+.6f} below FP on the current ten-task mean."
            ),
        },
        {
            "section": "BitDistill reproduction",
            "content": (
                f"Current MNLI curve: 40.96M {bitdistill['controlled_40_96m_mnli']:.6f}, "
                f"163.84M {bitdistill['controlled_163_84m_mnli']:.6f}, "
                f"327.68M {bitdistill['controlled_327_68m_mnli']:.6f}, "
                f"{latest_tokens / 1_000_000:.2f}M {latest_mnli:.6f}; "
                f"latest delta vs FP16 {latest_delta:+.6f}."
            ),
        },
        {
            "section": "Runtime contract",
            "content": (
                f"One-scale TL2 relative RMS error {row_contract['one_scale_tl2_relative_rms_error']:.6f}; "
                f"exact row-scale error {row_contract['exact_fp16_row_scale_relative_rms_error']:.6f}."
            ),
        },
        {
            "section": "CPU results",
            "content": (
                f"I2_SR file {i2sr['row_i2sr']['file_mib']:.1f} MiB, "
                f"PPL {i2sr['row_i2sr']['ppl']:.4f}, decode {i2sr['row_i2sr']['decode_tok_s']:.2f} tok/s; "
                f"native classifier MNLI {native['mnli_accuracy']:.6f} is not product-ready."
            ),
        },
    ]

    return {
        "schema": "bitdistill-publication-product-plan-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quality_claim": "planning_from_existing_artifacts_not_new_benchmark",
        "status": "research_mvp_with_open_quality_gate",
        "executive_verdict": (
            "The work is publishable as a rigorous boundary study and systems-contract prototype, "
            "not as a universal BitNet converter and not yet as a complete BitDistill reproduction."
        ),
        "publishable_units": publishable_units,
        "product_mvp": product_mvp,
        "decision_gates": gates,
        "paper_outline": paper_outline,
        "claim_rules": {
            "safe_headline": (
                "Blind ternary PTQ fails for tested dense Qwen; row-scale ternary students require "
                "matching CPU runtime semantics; BitDistill-style recovery is still under gate."
            ),
            "avoid": [
                "universal converter",
                "lossless retrofit",
                "paper-level BitDistill reproduced",
                "I2_SR beats Q4 on quality/storage",
                "Kimi/MoE supported",
            ],
            "minimum_for_stronger_claim": [
                "matched 10k-step gamma-60 MNLI quality run from the 655M checkpoint",
                "replicated within-1pt MNLI recovery row",
                "QNLI/SST2 rows only after MNLI gate",
                "same-artifact task quality and CPU runtime proof",
            ],
        },
        "source_paths": {
            "canonical_bundle": str(args.canonical_bundle),
            "scoreboard": str(args.scoreboard),
            "traceability": str(args.traceability),
            "product_scope": str(args.product_scope),
            "next_decision": str(args.next_decision),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "# BitDistill Publication and Product Plan",
            f"Generated: `{report['created_utc']}`",
            f"Quality claim: **{report['quality_claim']}**.",
            f"Status: **{report['status']}**.",
            report["executive_verdict"],
            "## Publishable Units",
            md_table(
                ["unit", "claim", "evidence", "publishable now", "risk"],
                [
                    [row["unit"], row["claim"], row["evidence"], row["publishable_now"], row["risk"]]
                    for row in report["publishable_units"]
                ],
            ),
            "## Product MVP",
            md_table(["field", "value"], [[key, value] for key, value in report["product_mvp"].items()]),
            "## Decision Gates",
            md_table(
                ["gate", "status", "decision rule", "success condition", "failure condition"],
                [
                    [
                        row["gate"],
                        row["status"],
                        row["decision_rule"],
                        row["success_condition"],
                        row["failure_condition"],
                    ]
                    for row in report["decision_gates"]
                ],
            ),
            "## Paper Outline",
            md_table(["section", "content"], [[row["section"], row["content"]] for row in report["paper_outline"]]),
            "## Claim Rules",
            md_table(
                ["field", "value"],
                [
                    ["safe_headline", report["claim_rules"]["safe_headline"]],
                    ["avoid", report["claim_rules"]["avoid"]],
                    ["minimum_for_stronger_claim", report["claim_rules"]["minimum_for_stronger_claim"]],
                ],
            ),
            "## Source Artifacts",
            md_table(["artifact", "path"], [[key, value] for key, value in report["source_paths"].items()]),
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-bundle", type=Path, default=Path("benchmarks/results/canonical_evidence_bundle_2026-05-20.json"))
    parser.add_argument("--scoreboard", type=Path, default=Path("benchmarks/results/bitdistill_benchmark_scoreboard_2026-05-23.json"))
    parser.add_argument("--traceability", type=Path, default=Path("benchmarks/results/bitdistill_goal_traceability_2026-05-23.json"))
    parser.add_argument("--product-scope", type=Path, default=Path("benchmark_results/product_scope_gate_2026-05-15.json"))
    parser.add_argument("--next-decision", type=Path, default=Path("benchmarks/results/bitdistill_next_decision_2026-05-23.json"))
    parser.add_argument("--out-json", type=Path, default=Path("benchmarks/results/bitdistill_publication_product_plan_2026-05-23.json"))
    parser.add_argument("--out-md", type=Path, default=Path("benchmarks/results/bitdistill_publication_product_plan_2026-05-23.md"))
    args = parser.parse_args()

    report = build_report(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
