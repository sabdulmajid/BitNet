#!/usr/bin/env python3
"""Summarize what the current TL2 evidence proves and does not prove.

This is a negative-result audit.  The original benchmark objective asked for
GGUF/TL2/I2_S CPU inference, but the later row-scale work found that the strong
QAT checkpoint stores learned row scales.  Current TL2 can execute for audited
Qwen0.5B scalar/tensor-style probes, but it does not preserve learned row-scale
semantics.  That distinction should be visible in public reports.
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if value != 0 and (abs(value) >= 1000 or abs(value) < 0.0001):
            return f"{value:.3e}"
        return f"{value:.6f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell).replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def strategy_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = result.get("strategies", [])
    if not isinstance(rows, list):
        return {}
    return {str(row.get("name")): row for row in rows if isinstance(row, dict)}


def row_by_label(results: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for row in results:
        if isinstance(row, dict) and row.get("label") == label:
            return row
    return {}


def parse_tl2_run(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    rows = data.get("rows", [])
    parsed: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return parsed
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", ""))
        if "tl2" not in name.lower():
            continue
        bench = row.get("bench", {}) if isinstance(row.get("bench"), dict) else {}
        prefill = bench.get("prefill", {}) if isinstance(bench.get("prefill"), dict) else {}
        decode = bench.get("decode", {}) if isinstance(bench.get("decode"), dict) else {}
        perplexity = row.get("perplexity", {}) if isinstance(row.get("perplexity"), dict) else {}
        parsed.append(
            {
                "path": str(path),
                "name": name,
                "file_mib": row.get("file_mib"),
                "ppl": perplexity.get("ppl"),
                "prefill_tok_s": prefill.get("tok_s"),
                "decode_tok_s": decode.get("tok_s"),
                "cpu_executed": finite(prefill.get("tok_s")) and finite(decode.get("tok_s")),
                "quality_finite": finite(perplexity.get("ppl")),
            }
        )
    return parsed


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    design = read_json(args.design_json)
    runtime = read_json(args.runtime_contract_json)
    design_results = design.get("results", []) if isinstance(design.get("results"), list) else []
    tensor = row_by_label(design_results, "qwen15b_tensor_scale")
    row = row_by_label(design_results, "qwen15b_row_scale")
    tensor_strategies = strategy_map(tensor)
    row_strategies = strategy_map(row)
    current_row = row_strategies.get("current_tl2_tensor_max_fp32", {})
    best_one_scale = row_strategies.get("tensor_l2_optimal_fp32", {})
    group32 = row_strategies.get("group32_l2_optimal_fp16", {})
    row_exact = row_strategies.get("row_exact_fp16", {})
    tensor_current = tensor_strategies.get("current_tl2_tensor_max_fp32", {})

    tl2_runs: list[dict[str, Any]] = []
    for path in args.tl2_summary:
        tl2_runs.extend(parse_tl2_run(path))

    failed_checks = [
        check for check in runtime.get("checks", []) if isinstance(check, dict) and not check.get("passed")
    ]
    cpu_executed = any(row.get("cpu_executed") for row in tl2_runs)
    finite_quality = any(row.get("quality_finite") for row in tl2_runs)
    current_row_error = current_row.get("expected_relative_output_rms_error")
    row_exact_error = row_exact.get("expected_relative_output_rms_error")
    row_scale_ready = bool(runtime.get("tl2_row_scale_runtime_ready"))
    negative_result_supported = (
        cpu_executed
        and not row_scale_ready
        and finite(current_row_error)
        and float(current_row_error) > args.max_acceptable_row_error
        and finite(row_exact_error)
        and float(row_exact_error) < args.max_acceptable_row_error
        and len(failed_checks) > 0
    )
    return {
        "schema": "tl2-negative-result-audit-v1",
        "date": DATE,
        "tl2_cpu_probe_rows": tl2_runs,
        "tl2_cpu_executed": cpu_executed,
        "tl2_probe_has_finite_quality": finite_quality,
        "qwen15b_tensor_scale_error": tensor_current.get("expected_relative_output_rms_error"),
        "qwen15b_row_scale_current_tl2_error": current_row_error,
        "qwen15b_row_scale_best_one_scale_error": best_one_scale.get("expected_relative_output_rms_error"),
        "qwen15b_row_scale_group32_fp16_error": group32.get("expected_relative_output_rms_error"),
        "qwen15b_row_scale_exact_fp16_error": row_exact_error,
        "qwen15b_row_scale_exact_fp16_scale_mib": row_exact.get("scale_mib_fp16"),
        "runtime_ready": row_scale_ready,
        "runtime_failed_checks": len(failed_checks),
        "runtime_blockers": runtime.get("blockers", []),
        "negative_result_supported": negative_result_supported,
        "verdict": (
            "TL2 has CPU execution evidence, but current one-scale TL2 is a negative "
            "result for learned row-scale checkpoints. I2_SR remains the supported "
            "row-scale packed path until TL2 gains row/group-scale metadata and kernels."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    run_rows = [
        [
            row["name"],
            row["file_mib"],
            row["ppl"],
            row["prefill_tok_s"],
            row["decode_tok_s"],
            row["cpu_executed"],
            row["quality_finite"],
            row["path"],
        ]
        for row in summary["tl2_cpu_probe_rows"]
    ]
    scale_rows = [
        ["tensor-scale checkpoint through current TL2", summary["qwen15b_tensor_scale_error"]],
        ["row-scale checkpoint through current TL2", summary["qwen15b_row_scale_current_tl2_error"]],
        ["row-scale best possible one tensor scale", summary["qwen15b_row_scale_best_one_scale_error"]],
        ["row-scale group32 fp16 scales", summary["qwen15b_row_scale_group32_fp16_error"]],
        ["row-scale exact fp16 row scales", summary["qwen15b_row_scale_exact_fp16_error"]],
        ["exact fp16 row-scale overhead MiB", summary["qwen15b_row_scale_exact_fp16_scale_mib"]],
    ]
    blockers = summary.get("runtime_blockers", [])
    blocker_text = "\n".join(f"- {item}" for item in blockers) if blockers else "- none"
    return "\n\n".join(
        [
            f"# TL2 Negative Result Audit, {summary['date']}",
            summary["verdict"],
            "## Status",
            md_table(
                ["field", "value"],
                [
                    ["TL2 CPU probes executed", summary["tl2_cpu_executed"]],
                    ["TL2 probe has finite quality", summary["tl2_probe_has_finite_quality"]],
                    ["row-scale TL2 runtime ready", summary["runtime_ready"]],
                    ["runtime failed checks", summary["runtime_failed_checks"]],
                    ["negative result supported", summary["negative_result_supported"]],
                ],
            ),
            "## TL2 CPU Probe Rows",
            md_table(
                [
                    "row",
                    "file MiB",
                    "PPL",
                    "prefill tok/s",
                    "decode tok/s",
                    "CPU executed",
                    "finite quality",
                    "path",
                ],
                run_rows,
            ),
            "## Scale Semantics",
            md_table(["case", "relative output RMS error / value"], scale_rows),
            "## Blockers",
            blocker_text,
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tl2-summary",
        type=Path,
        nargs="+",
        default=[
            Path("benchmark_results/gguf-qwen05b-tl2-probe-2026-05-05/summary.json"),
            Path("benchmark_results/gguf-qwen05b-tl2-avx512-2026-05-05/summary.json"),
        ],
    )
    parser.add_argument("--design-json", type=Path, default=Path("benchmark_results/tl2_row_scale_design_2026-05-13.json"))
    parser.add_argument(
        "--runtime-contract-json",
        type=Path,
        default=Path(f"benchmark_results/tl2_row_scale_runtime_contract_{DATE}.json"),
    )
    parser.add_argument("--max-acceptable-row-error", type=float, default=0.01)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/tl2_negative_result_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/tl2_negative_result_{DATE}.md"))
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
