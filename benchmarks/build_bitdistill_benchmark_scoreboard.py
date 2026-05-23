#!/usr/bin/env python3
"""Build a consolidated scoreboard from existing BitNet/BitDistill artifacts.

This is a claim-status report, not a new benchmark. It exists to make the
current evidence easy to audit without mixing incompatible claims such as GLUE
quality, general-LM perplexity, packed causal runtime, and MoE plumbing.
"""

from __future__ import annotations

import argparse
import json
import math
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


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


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


def fmt_large(value: float) -> str:
    return f"{value:,.6f}" if abs(value) >= 10000.0 else f"{value:.6f}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def require_schema(path: Path, data: dict[str, Any], expected: str) -> None:
    actual = data.get("schema")
    if actual != expected:
        raise RuntimeError(f"{path}: expected schema {expected}, got {actual}")


def model_families(matrix: dict[str, Any]) -> list[str]:
    families: set[str] = set()
    for row in matrix.get("quality_benchmarks", []):
        if isinstance(row, dict) and isinstance(row.get("models"), dict):
            families.update(str(name) for name in row["models"])
    return sorted(families)


def quality_benchmark_names(matrix: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for row in matrix.get("quality_benchmarks", []):
        if isinstance(row, dict) and isinstance(row.get("benchmark"), str):
            names.append(row["benchmark"])
    return names


def product_gate_counts(moe: dict[str, Any]) -> dict[str, int]:
    gates = moe.get("productization_gates", [])
    passed = 0
    failed = 0
    if isinstance(gates, list):
        for gate in gates:
            if not isinstance(gate, dict):
                continue
            if gate.get("passed") is True:
                passed += 1
            else:
                failed += 1
    return {"passed": passed, "failed": failed, "total": passed + failed}


def headline_rows(
    claims: dict[str, Any],
    matrix: dict[str, Any],
    coverage: dict[str, Any],
    product: dict[str, Any],
    seqcls: dict[str, Any],
    moe: dict[str, Any],
    next_decision: dict[str, Any],
) -> list[dict[str, Any]]:
    blind = claims["blind_ptq"]
    qat = claims["qat_distill"]
    bitdistill = claims["bitdistill_reproduction"]
    gamma = claims["gamma_normalization"]
    row_contract = claims["row_scale_runtime_contract"]
    i2sr = claims["i2sr_cpu"]
    native = claims["native_classifier"]
    moe_claim = claims["moe_kimi"]
    runtime = matrix.get("cpu_runtime", {}) if isinstance(matrix.get("cpu_runtime"), dict) else {}
    q4_vs_i2sr = runtime.get("q4_vs_i2sr", {}) if isinstance(runtime.get("q4_vs_i2sr"), dict) else {}
    seq_native = (
        seqcls.get("seqcls_native_cpu_benchmark", {})
        if isinstance(seqcls.get("seqcls_native_cpu_benchmark"), dict)
        else {}
    )
    moe_gates = product_gate_counts(moe)

    return [
        {
            "area": "Blind ternary PTQ",
            "status": "rejected_for_tested_dense_qwen",
            "evidence": (
                f"FP WikiText PPL {fmt_large(float(blind['fp_wikitext_ppl']))}; "
                f"naive PTQ PPL {fmt_large(float(blind['ptq_wikitext_ppl']))}; "
                f"FP ten-task mean {blind['fp_ten_task_mean']:.6f}; "
                f"PTQ mean {blind['ptq_ten_task_mean']:.6f}"
            ),
            "impact": "The universal one-click arbitrary FP/BF16-to-ternary retrofit claim is not supported.",
            "limitation": blind["caveat"],
        },
        {
            "area": "QAT/distillation recovery",
            "status": qat["status"],
            "evidence": (
                f"best row-scale QAT mean {qat['best_row_scale_qat_ten_task_mean']:.6f}; "
                f"recovery vs PTQ {qat['recovery_vs_ptq']:+.6f}; "
                f"gap vs FP {qat['gap_vs_fp']:+.6f}"
            ),
            "impact": "Training under ternary constraints recovers real signal, but not FP quality.",
            "limitation": qat["caveat"],
        },
        {
            "area": "BitDistill reproduction",
            "status": bitdistill["status"],
            "evidence": (
                f"MNLI 40.96M {bitdistill['controlled_40_96m_mnli']:.6f}; "
                f"163.84M {bitdistill['controlled_163_84m_mnli']:.6f}; "
                f"327.68M {bitdistill['controlled_327_68m_mnli']:.6f}; "
                f"delta vs FP {bitdistill['controlled_327_68m_delta_vs_fp']:+.6f}"
            ),
            "impact": "Paper-level BitDistill quality is not reproduced yet; the curve is still improving.",
            "limitation": bitdistill["caveat"],
        },
        {
            "area": "Loss normalization / gamma",
            "status": gamma["status"],
            "evidence": (
                f"gamma-60 MNLI {gamma['gamma60_mnli']:.6f}; "
                f"delta vs FP {gamma['gamma60_delta_vs_fp']:+.6f}"
            ),
            "impact": "The attention-KD coefficient cannot be interpreted without matching loss reductions.",
            "limitation": gamma["caveat"],
        },
        {
            "area": "Row-scale runtime contract",
            "status": row_contract["status"],
            "evidence": (
                f"one-scale TL2 RMS error {row_contract['one_scale_tl2_relative_rms_error']:.6f}; "
                f"exact row-scale RMS error {row_contract['exact_fp16_row_scale_relative_rms_error']:.6f}"
            ),
            "impact": "Row scales are model semantics, not optional metadata.",
            "limitation": row_contract["caveat"],
        },
        {
            "area": "Packed CPU I2_SR",
            "status": i2sr["status"],
            "evidence": (
                f"I2_SR file {i2sr['row_i2sr']['file_mib']:.1f} MiB, "
                f"PPL {i2sr['row_i2sr']['ppl']:.4f}, "
                f"prompt {i2sr['row_i2sr']['prompt_tok_s']:.2f} tok/s, "
                f"decode {i2sr['row_i2sr']['decode_tok_s']:.2f} tok/s; "
                f"Q4_K_M file {i2sr['q4_k_m']['file_mib']:.1f} MiB, PPL {i2sr['q4_k_m']['ppl']:.4f}"
            ),
            "impact": (
                f"Dense row-scale ternary CPU execution works; audited Q4 comparison has "
                f"{q4_vs_i2sr.get('decode_speedup', 0.0):.3f}x decode and "
                f"{q4_vs_i2sr.get('prefill_speedup', 0.0):.3f}x prefill speedups for I2_SR."
            ),
            "limitation": i2sr["caveat"],
        },
        {
            "area": "Native sequence classification",
            "status": native["status"],
            "evidence": (
                f"MNLI {native['mnli_accuracy']:.6f}; PyTorch agreement {native['pytorch_agreement']:.6f}; "
                f"sequence-isolated {native['examples_per_second']:.6f} ex/s; "
                f"token-id runner {seq_native.get('examples_per_second', 0.0):.6f} ex/s"
            ),
            "impact": "Native packed classifier plumbing exists as a research demo.",
            "limitation": native["caveat"],
        },
        {
            "area": "MoE / Kimi",
            "status": moe_claim["status"],
            "evidence": (
                f"local Kimi artifacts {len(moe.get('local_kimi_artifacts', []))}; "
                f"MoE product gates passed {moe_gates['passed']}/{moe_gates['total']}"
            ),
            "impact": "Synthetic Qwen2MoE plumbing is useful, but Kimi is future work.",
            "limitation": moe_claim["caveat"],
        },
        {
            "area": "Benchmark coverage",
            "status": "passed" if coverage.get("passed") is True and not coverage.get("failed") else "failed",
            "evidence": (
                f"{matrix.get('quality_benchmark_count')} quality benchmarks; "
                f"{coverage.get('check_count')} coverage checks; failed checks {len(coverage.get('failed', []))}"
            ),
            "impact": "The current negative and partial-recovery claims are backed by broad audited coverage.",
            "limitation": "This coverage predates the active 655M Stage-2 extension.",
        },
        {
            "area": "Product scope",
            "status": product.get("scope_status"),
            "evidence": (
                f"supported claims {product.get('supported_claim_count')}; "
                f"unsupported claims {product.get('unsupported_claim_count')}"
            ),
            "impact": product.get("recommendation", {}).get("product", ""),
            "limitation": "This is a research MVP, not a universal converter product.",
        },
        {
            "area": "Active next decision",
            "status": next_decision.get("status"),
            "evidence": (
                "latest controlled row "
                f"{float(next_decision.get('latest_controlled_row', {}).get('accuracy')):.6f}; "
                "latest tokens "
                f"{int(next_decision.get('latest_controlled_row', {}).get('stage2_token_presentations')):,}; "
                f"gamma status {next_decision.get('gamma_balance', {}).get('status')}"
            ),
            "impact": next_decision.get("recommendation"),
            "limitation": "Decision is pending until the 655M downstream row and gamma-60 telemetry complete.",
        },
    ]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    bundle = read_json(args.canonical_bundle)
    matrix = read_json(args.benchmark_matrix)
    product = read_json(args.product_scope)
    coverage = read_json(args.coverage_gate)
    seqcls = read_json(args.seqcls_gap)
    moe = read_json(args.moe_support)
    next_decision = read_json(args.next_decision)

    require_schema(args.canonical_bundle, bundle, "bitnet-canonical-evidence-bundle-v1")
    require_schema(args.benchmark_matrix, matrix, "benchmark-matrix-audit-v1")
    require_schema(args.product_scope, product, "bitnet-product-scope-gate-v1")
    require_schema(args.coverage_gate, coverage, "benchmark_coverage_gate.v1")
    require_schema(args.seqcls_gap, seqcls, "seqcls_runtime_gap.v1")
    require_schema(args.moe_support, moe, "bitnet-moe-support-audit-v1")
    require_schema(args.next_decision, next_decision, "bitdistill-next-decision-v1")

    claims = bundle.get("claims")
    if not isinstance(claims, dict):
        raise RuntimeError("canonical bundle missing claims")
    required_claims = {
        "blind_ptq",
        "qat_distill",
        "bitdistill_reproduction",
        "gamma_normalization",
        "row_scale_runtime_contract",
        "i2sr_cpu",
        "native_classifier",
        "moe_kimi",
    }
    missing_claims = sorted(required_claims.difference(claims))
    if missing_claims:
        raise RuntimeError(f"canonical bundle missing claims: {missing_claims}")

    benchmarks = quality_benchmark_names(matrix)
    lm_eval_tasks = [name for name in benchmarks if "perplexity" not in name.lower()]
    sample_counts = matrix.get("sample_counts", {}) if isinstance(matrix.get("sample_counts"), dict) else {}
    runtime = matrix.get("cpu_runtime", {}) if isinstance(matrix.get("cpu_runtime"), dict) else {}
    report = {
        "schema": "bitdistill-benchmark-scoreboard-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quality_claim": "scoreboard_from_existing_artifacts_not_new_benchmark",
        "status": "mixed_supported_and_blocked",
        "publishability_assessment": {
            "publishable": True,
            "framing": (
                "Publishable as an independent negative/positive boundary study and systems-contract "
                "prototype, not as a universal BitNet converter and not yet as a paper-level "
                "BitDistill reproduction."
            ),
            "strongest_contribution": (
                "Blind PTQ failure plus row-scale runtime-contract evidence showing that trained "
                "ternary scale semantics must be preserved in CPU formats such as I2_SR."
            ),
            "main_blocker": (
                "The same artifact still does not jointly satisfy paper-level task quality, general-LM "
                "quality, mature Q4-level storage/quality tradeoffs, and product-ready packed runtime."
            ),
        },
        "coverage": {
            "quality_benchmark_count": matrix.get("quality_benchmark_count"),
            "quality_benchmarks": benchmarks,
            "lm_eval_task_count": len(lm_eval_tasks),
            "lm_eval_tasks": lm_eval_tasks,
            "model_families": model_families(matrix),
            "sample_counts": sample_counts,
            "coverage_gate_passed": coverage.get("passed"),
            "coverage_check_count": coverage.get("check_count"),
            "coverage_failed": coverage.get("failed"),
            "cpu_runtime": runtime,
        },
        "headline_rows": headline_rows(claims, matrix, coverage, product, seqcls, moe, next_decision),
        "novelty": [
            "A fail-closed evidence stack that separates blind PTQ, QAT/distillation, GLUE quality, general-LM perplexity, packed runtime, and MoE plumbing.",
            "Empirical evidence that row-scale ternary semantics materially affect output fidelity and require a matching CPU runtime contract.",
            "An I2_SR row-scale packed-runtime path for compatible dense causal artifacts.",
            "A bounded product direction: a CPU-first ternary retrofit evaluator, not a universal converter.",
        ],
        "nonclaims": [
            "No claim that arbitrary FP16/BF16 models can be converted losslessly to BitNet.",
            "No claim that paper-level BitDistill has been reproduced.",
            "No claim that I2_SR beats Q4_K_M on quality or file size.",
            "No claim that current causal exports are useful general-purpose LLMs.",
            "No claim that native packed sequence classification is product-ready.",
            "No claim that Kimi or real MoE CPU ternary quality is supported.",
        ],
        "next_steps": [
            "Wait for the active 655.36M Stage-2 extension, downstream MNLI row, and postprocess reports.",
            "Use gamma-60 component-gradient telemetry to determine whether the next run should extend Stage-2 or rebalance the attention-KD coefficient.",
            "If 655M shows meaningful marginal gain, continue the controlled Stage-2 token curve before broadening tasks.",
            "If 655M saturates, audit recipe alignment and loss normalization before spending more compute.",
            "Keep MoE/Kimi work outside the main claim path until dense quality/runtime evidence is resolved.",
        ],
        "source_paths": {
            "canonical_bundle": str(args.canonical_bundle),
            "benchmark_matrix": str(args.benchmark_matrix),
            "product_scope": str(args.product_scope),
            "coverage_gate": str(args.coverage_gate),
            "seqcls_gap": str(args.seqcls_gap),
            "moe_support": str(args.moe_support),
            "next_decision": str(args.next_decision),
        },
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    rows = report["headline_rows"]
    row_table = md_table(
        ["area", "status", "evidence", "impact", "limitation"],
        [[row["area"], row["status"], row["evidence"], row["impact"], row["limitation"]] for row in rows],
    )
    source_table = md_table(
        ["artifact", "path"],
        [[name, path] for name, path in report["source_paths"].items()],
    )
    coverage_table = md_table(
        ["field", "value"],
        [
            ["quality_benchmark_count", coverage["quality_benchmark_count"]],
            ["lm_eval_task_count", coverage["lm_eval_task_count"]],
            ["model_families", coverage["model_families"]],
            ["sample_counts", coverage["sample_counts"]],
            ["coverage_gate_passed", coverage["coverage_gate_passed"]],
            ["coverage_check_count", coverage["coverage_check_count"]],
            ["coverage_failed", coverage["coverage_failed"]],
        ],
    )
    return "\n\n".join(
        [
            "# BitDistill Benchmark Scoreboard",
            f"Generated: `{report['created_utc']}`",
            f"Quality claim: **{report['quality_claim']}**.",
            f"Status: **{report['status']}**.",
            "## Publishability Assessment",
            md_table(
                ["field", "value"],
                [[key, value] for key, value in report["publishability_assessment"].items()],
            ),
            "## Coverage",
            coverage_table,
            "Benchmarks covered: " + ", ".join(coverage["quality_benchmarks"]) + ".",
            "## Headline Scoreboard",
            row_table,
            "## Novelty",
            "\n".join(f"- {item}" for item in report["novelty"]),
            "## Nonclaims",
            "\n".join(f"- {item}" for item in report["nonclaims"]),
            "## Next Steps",
            "\n".join(f"- {item}" for item in report["next_steps"]),
            "## Source Artifacts",
            source_table,
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-bundle",
        type=Path,
        default=Path("benchmarks/results/canonical_evidence_bundle_2026-05-20.json"),
    )
    parser.add_argument(
        "--benchmark-matrix",
        type=Path,
        default=Path("benchmark_results/benchmark_matrix_audit_2026-05-15.json"),
    )
    parser.add_argument(
        "--product-scope",
        type=Path,
        default=Path("benchmark_results/product_scope_gate_2026-05-15.json"),
    )
    parser.add_argument(
        "--coverage-gate",
        type=Path,
        default=Path("benchmark_results/benchmark_coverage_gate_2026-05-15.json"),
    )
    parser.add_argument(
        "--seqcls-gap",
        type=Path,
        default=Path("benchmark_results/seqcls_runtime_gap_2026-05-15.json"),
    )
    parser.add_argument(
        "--moe-support",
        type=Path,
        default=Path("benchmark_results/moe_support_audit_2026-05-15.json"),
    )
    parser.add_argument(
        "--next-decision",
        type=Path,
        default=Path("benchmarks/results/bitdistill_next_decision_2026-05-23.json"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("benchmarks/results/bitdistill_benchmark_scoreboard_2026-05-23.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("benchmarks/results/bitdistill_benchmark_scoreboard_2026-05-23.md"),
    )
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
