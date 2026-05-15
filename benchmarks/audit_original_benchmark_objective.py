#!/usr/bin/env python3
"""Audit the original six-item benchmark objective against concrete artifacts.

This audit is narrower than objective_completion_audit.py.  It maps the user's
original benchmark plan item-by-item, without folding in later MoE/Kimi scope.
It intentionally treats the TL2 row-scale path as partial until the TL2 runtime
has explicit row/group-scale semantics and quality-valid CPU benchmark rows.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def row_by_prefix(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    for row in rows:
        if isinstance(row, dict) and str(row.get("requirement", "")).startswith(prefix):
            return row
    return {}


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != 0.0 and (abs(value) >= 1000 or abs(value) < 0.0001):
            return f"{value:.3e}"
        return f"{value:.6f}"
    return str(value).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(fmt(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def build_audit(root: Path, objective_path: Path, tl2_path: Path, coverage_path: Path) -> dict[str, Any]:
    objective = read_json(objective_path)
    tl2 = read_json(tl2_path)
    coverage = read_json(coverage_path)
    checklist = objective.get("checklist", []) if isinstance(objective.get("checklist"), list) else []
    metrics = objective.get("metrics", {}) if isinstance(objective.get("metrics"), dict) else {}

    exports = row_by_prefix(checklist, "Fix FSDP ternary export bug")
    prompts = row_by_prefix(checklist, "Run fixed prompt suites")
    ppl = row_by_prefix(checklist, "Add WikiText and FineWeb")
    lm_eval = row_by_prefix(checklist, "Add HellaSwag")
    baselines = row_by_prefix(checklist, "Add baselines")
    cpu = row_by_prefix(checklist, "Measure Xeon")
    productization = row_by_prefix(checklist, "Convert repaired checkpoints")

    tl2_checks = tl2.get("checks", []) if isinstance(tl2.get("checks"), list) else []
    tl2_failed = [check for check in tl2_checks if isinstance(check, dict) and not check.get("passed")]
    tl2_math = tl2.get("math", {}) if isinstance(tl2.get("math"), dict) else {}
    product = metrics.get("productization", {}) if isinstance(metrics.get("productization"), dict) else {}
    cpu_metrics = metrics.get("cpu", {}) if isinstance(metrics.get("cpu"), dict) else {}
    i2sr_cpu = cpu_metrics.get("row-scale I2_SR candidate", {}) if isinstance(cpu_metrics.get("row-scale I2_SR candidate"), dict) else {}
    q4_cpu = cpu_metrics.get("FP Q4_K_M", {}) if isinstance(cpu_metrics.get("FP Q4_K_M"), dict) else {}

    rows = [
        {
            "item": "1. Fix FSDP ternary export and re-export 1.5B step-5000",
            "status": exports.get("status", "unknown"),
            "evidence": exports.get("evidence", ""),
            "gap": exports.get("remaining_gap", ""),
        },
        {
            "item": "2. Run eval/prompt suites on repaired 1.5B and complete 0.5B",
            "status": prompts.get("status", "unknown"),
            "evidence": prompts.get("evidence", ""),
            "gap": prompts.get("remaining_gap", ""),
        },
        {
            "item": "3. Add WikiText, FineWeb, HellaSwag/PIQA/ARC lm-eval",
            "status": "complete" if ppl.get("status") == "complete" and lm_eval.get("status") == "complete" else "partial",
            "evidence": f"{ppl.get('evidence', '')}; {lm_eval.get('evidence', '')}",
            "gap": "; ".join(part for part in [ppl.get("remaining_gap", ""), lm_eval.get("remaining_gap", "")] if part),
        },
        {
            "item": "4. Add FP/PTQ/Q4/Q8/QAT/row-vs-tensor baselines",
            "status": baselines.get("status", "unknown"),
            "evidence": baselines.get("evidence", ""),
            "gap": baselines.get("remaining_gap", ""),
        },
        {
            "item": "5. Convert repaired checkpoints into GGUF/TL2/I2_S and run CPU inference",
            "status": "partial",
            "evidence": (
                f"{productization.get('evidence', '')}; TL2 ready={tl2.get('tl2_row_scale_runtime_ready')}; "
                f"TL2 current row-scale error={tl2_math.get('current_tl2_tensor_max_error')}; "
                f"TL2 failed checks={len(tl2_failed)}"
            ),
            "gap": (
                "Dense GGUF and row-scale I2_SR/I2_S CPU inference exist, but TL2 is not quality-preserving "
                "for learned row-scale checkpoints until row/group-scale metadata and kernels are implemented."
            ),
        },
        {
            "item": "6. Measure Xeon speed, prompt throughput, RSS, model size, and quality loss",
            "status": cpu.get("status", "unknown"),
            "evidence": (
                f"{cpu.get('evidence', '')}; I2_SR decode={i2sr_cpu.get('decode_tok_s')} tok/s; "
                f"Q4_K_M decode={q4_cpu.get('decode_tok_s')} tok/s"
            ),
            "gap": cpu.get("remaining_gap", ""),
        },
    ]

    complete_count = sum(1 for row in rows if row["status"] == "complete")
    return {
        "schema": "original-benchmark-objective-audit-v1",
        "date": DATE,
        "objective_json": str(objective_path.relative_to(root)),
        "tl2_contract_json": str(tl2_path.relative_to(root)),
        "coverage_json": str(coverage_path.relative_to(root)),
        "benchmark_coverage_passed": coverage.get("passed"),
        "benchmark_coverage_checks": coverage.get("check_count"),
        "complete_count": complete_count,
        "check_count": len(rows),
        "objective_achieved": complete_count == len(rows),
        "completion_status": "complete" if complete_count == len(rows) else "partial",
        "rows": rows,
        "tl2_failed_blockers": [check.get("blocker") for check in tl2_failed],
        "interpretation": (
            "The original benchmark objective is substantially satisfied for dense Qwen through the stable I2_SR path. "
            "It is not fully complete because the requested TL2 path is not quality-preserving for learned row-scale "
            "checkpoints; current TL2 collapses row scales to one tensor scale."
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            f"# Original Benchmark Objective Audit, {result['date']}",
            "This audit maps the original six requested benchmark deliverables to concrete artifacts. It does not include later MoE/Kimi scope.",
            md_table(
                ["field", "value"],
                [
                    ["objective achieved", result["objective_achieved"]],
                    ["completion", f"{result['complete_count']}/{result['check_count']}"],
                    ["coverage passed", result["benchmark_coverage_passed"]],
                    ["coverage checks", result["benchmark_coverage_checks"]],
                ],
            ),
            "## Checklist",
            md_table(
                ["item", "status", "evidence", "remaining gap"],
                [[row["item"], row["status"], row["evidence"], row["gap"]] for row in result["rows"]],
            ),
            "## TL2 Blockers",
            md_table(["blocker"], [[blocker] for blocker in result["tl2_failed_blockers"]]),
            "## Interpretation",
            result["interpretation"],
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--objective-json",
        type=Path,
        default=Path(f"benchmark_results/objective_completion_audit_{DATE}.json"),
    )
    parser.add_argument(
        "--tl2-json",
        type=Path,
        default=Path(f"benchmark_results/tl2_row_scale_runtime_contract_{DATE}.json"),
    )
    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=Path(f"benchmark_results/benchmark_coverage_gate_{DATE}.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"benchmark_results/original_benchmark_objective_audit_{DATE}.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"benchmarks/results/original_benchmark_objective_audit_{DATE}.md"),
    )
    args = parser.parse_args()

    root = args.repo_root.resolve()
    objective_path = args.objective_json if args.objective_json.is_absolute() else root / args.objective_json
    tl2_path = args.tl2_json if args.tl2_json.is_absolute() else root / args.tl2_json
    coverage_path = args.coverage_json if args.coverage_json.is_absolute() else root / args.coverage_json
    result = build_audit(root, objective_path, tl2_path, coverage_path)

    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
