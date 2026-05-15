#!/usr/bin/env python3
"""Compute the CPU quality/speed/memory tradeoff table for headline GGUF rows.

This is an analysis pass over already-run Xeon benchmarks. It is deliberately
separate from the benchmark runner so public claims can distinguish raw
measurements from normalized comparisons and domination checks.
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

HEADLINE_ROWS = {
    "FP F16": ("benchmark_results/gguf-qwen15b-row-i2s-prototype-suite/summary.json", "qwen15b_fp_f16"),
    "FP Q8_0": ("benchmark_results/gguf-qwen15b-row-i2s-prototype-suite/summary.json", "qwen15b_fp_q8_0"),
    "FP Q4_K_M": ("benchmark_results/gguf-qwen15b-row-i2s-prototype-suite/summary.json", "qwen15b_fp_q4_k_m"),
    "row TQ2_0": (
        "benchmark_results/gguf-qwen15b-row-i2s-prototype-suite/summary.json",
        "qwen15b_klonly_row_notie_static_ternary_tq2_0",
    ),
    "row I2_S": (
        "benchmark_results/gguf-qwen15b-row-i2s-heapfix-confirm/summary.json",
        "qwen15b_klonly_row_notie_static_ternary_i2_s_rowscale",
    ),
    "row I2_SR": (
        "benchmark_results/i2sr-row-scale-qwen15b-x86act-suite-2026-05-13/summary.json",
        "qwen15b_klonly_row_notie_static_ternary_i2_sr_x86act",
    ),
}

RSS_SUMMARIES = [
    "benchmark_results/gguf-rss-qwen15b-context-scaling-2026-05-05/summary.json",
    "benchmark_results/gguf-rss-qwen15b-i2sr-x86act-context-2026-05-13/summary.json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if abs(value) >= 1000 or (0.0 < abs(value) < 0.0001):
            return f"{value:.6e}"
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


def find_row(summary: dict[str, Any], name: str) -> dict[str, Any] | None:
    for row in summary.get("rows", []):
        if isinstance(row, dict) and row.get("name") == name:
            return row
    return None


def load_rss(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for rel_path in RSS_SUMMARIES:
        path = root / rel_path
        if not path.exists():
            continue
        for row in read_json(path).get("rows", []):
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            ctx = row.get("ctx_size")
            if isinstance(name, str) and isinstance(ctx, int) and row.get("returncode") == 0:
                rows[(name, ctx)] = row
    return rows


def row_metrics(root: Path, label: str, rel_path: str, name: str, rss_rows: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    path = root / rel_path
    summary = read_json(path) if path.exists() else {}
    row = find_row(summary, name) if path.exists() else None
    if not row:
        return {"label": label, "name": name, "exists": False, "source": rel_path}
    rss_512 = rss_rows.get((name, 512), {})
    rss_32768 = rss_rows.get((name, 32768), {})
    return {
        "label": label,
        "name": name,
        "exists": True,
        "source": rel_path,
        "kind": row.get("kind"),
        "cpu": row.get("bench", {}).get("decode", {}).get("cpu"),
        "file_mib": row.get("file_mib"),
        "ppl": row.get("perplexity", {}).get("ppl"),
        "ppl_stderr": row.get("perplexity", {}).get("stderr"),
        "prefill_tok_s": row.get("bench", {}).get("prefill", {}).get("tok_s"),
        "decode_tok_s": row.get("bench", {}).get("decode", {}).get("tok_s"),
        "prefill_stddev_tok_s": row.get("bench", {}).get("prefill", {}).get("stddev_tok_s"),
        "decode_stddev_tok_s": row.get("bench", {}).get("decode", {}).get("stddev_tok_s"),
        "rss_512_gib": rss_512.get("max_rss_gib"),
        "rss_32768_gib": rss_32768.get("max_rss_gib"),
    }


def ratio(value: Any, reference: Any) -> float | None:
    if not finite(value) or not finite(reference) or float(reference) == 0.0:
        return None
    return float(value) / float(reference)


def add_reference_ratios(row: dict[str, Any], reference: dict[str, Any], prefix: str) -> None:
    row[f"{prefix}_ppl_ratio"] = ratio(row.get("ppl"), reference.get("ppl"))
    row[f"{prefix}_file_ratio"] = ratio(row.get("file_mib"), reference.get("file_mib"))
    row[f"{prefix}_rss512_ratio"] = ratio(row.get("rss_512_gib"), reference.get("rss_512_gib"))
    row[f"{prefix}_prefill_speedup"] = ratio(row.get("prefill_tok_s"), reference.get("prefill_tok_s"))
    row[f"{prefix}_decode_speedup"] = ratio(row.get("decode_tok_s"), reference.get("decode_tok_s"))


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    lower_is_better = ("ppl", "file_mib", "rss_512_gib")
    higher_is_better = ("prefill_tok_s", "decode_tok_s")
    comparisons: list[tuple[float, float, str]] = []
    for key in lower_is_better:
        if not finite(a.get(key)) or not finite(b.get(key)):
            return False
        comparisons.append((float(a[key]), float(b[key]), "lower"))
    for key in higher_is_better:
        if not finite(a.get(key)) or not finite(b.get(key)):
            return False
        comparisons.append((float(a[key]), float(b[key]), "higher"))
    no_worse = all(x <= y if sense == "lower" else x >= y for x, y, sense in comparisons)
    strictly_better = any(x < y if sense == "lower" else x > y for x, y, sense in comparisons)
    return no_worse and strictly_better


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    rss_rows = load_rss(root)
    rows = [
        row_metrics(root, label, rel_path, name, rss_rows)
        for label, (rel_path, name) in HEADLINE_ROWS.items()
    ]
    by_label = {row["label"]: row for row in rows}
    fp = by_label.get("FP F16", {})
    q4 = by_label.get("FP Q4_K_M", {})
    for row in rows:
        if row.get("exists"):
            add_reference_ratios(row, fp, "vs_fp_f16")
            add_reference_ratios(row, q4, "vs_q4_k_m")
    dominated_by: dict[str, list[str]] = {}
    for row in rows:
        if not row.get("exists"):
            continue
        dominators = [
            other["label"]
            for other in rows
            if other is not row and other.get("exists") and dominates(other, row)
        ]
        dominated_by[row["label"]] = dominators
        row["dominated_by"] = dominators
        row["pareto_frontier"] = not dominators

    i2sr = by_label.get("row I2_SR", {})
    q4_ratio = {
        "file_ratio": i2sr.get("vs_q4_k_m_file_ratio"),
        "rss512_ratio": i2sr.get("vs_q4_k_m_rss512_ratio"),
        "prefill_speedup": i2sr.get("vs_q4_k_m_prefill_speedup"),
        "decode_speedup": i2sr.get("vs_q4_k_m_decode_speedup"),
        "ppl_ratio": i2sr.get("vs_q4_k_m_ppl_ratio"),
    }
    return {
        "schema": "cpu-tradeoff-frontier-v1",
        "date": DATE,
        "rows": rows,
        "q4_vs_i2sr": q4_ratio,
        "frontier": [row["label"] for row in rows if row.get("pareto_frontier")],
        "dominated": dominated_by,
        "interpretation": (
            "I2_SR is a speed-oriented proof of row-scale ternary runtime semantics. "
            "It improves decode speed versus FP16 and is faster than Q4_K_M in the audited run, "
            "but it is larger than Q4_K_M and has much worse PPL. It should not be claimed as a "
            "quality/storage win over mature Q4 quantization."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary["rows"]:
        rows.append(
            [
                row["label"],
                row.get("file_mib"),
                row.get("rss_512_gib"),
                row.get("ppl"),
                row.get("prefill_tok_s"),
                row.get("decode_tok_s"),
                row.get("vs_fp_f16_file_ratio"),
                row.get("vs_fp_f16_ppl_ratio"),
                row.get("vs_fp_f16_decode_speedup"),
                row.get("vs_q4_k_m_file_ratio"),
                row.get("vs_q4_k_m_ppl_ratio"),
                row.get("vs_q4_k_m_decode_speedup"),
                ", ".join(row.get("dominated_by", [])) or "-",
            ]
        )
    q4 = summary["q4_vs_i2sr"]
    return "\n\n".join(
        [
            f"# CPU Tradeoff Frontier Audit, {summary['date']}",
            summary["interpretation"],
            "",
            "## Headline Ratios",
            (
                "Compared with FP Q4_K_M, row-scale `I2_SR` has "
                f"`{fmt(q4.get('file_ratio'))}x` file size, "
                f"`{fmt(q4.get('rss512_ratio'))}x` RSS at ctx 512, "
                f"`{fmt(q4.get('prefill_speedup'))}x` prefill throughput, "
                f"`{fmt(q4.get('decode_speedup'))}x` decode throughput, and "
                f"`{fmt(q4.get('ppl_ratio'))}x` PPL."
            ),
            "",
            f"Pareto frontier over PPL, file size, RSS, prefill, and decode: `{', '.join(summary['frontier'])}`.",
            "",
            "## Rows",
            md_table(
                [
                    "artifact",
                    "file MiB",
                    "RSS512 GiB",
                    "PPL",
                    "prefill tok/s",
                    "decode tok/s",
                    "file/FP",
                    "PPL/FP",
                    "decode/FP",
                    "file/Q4",
                    "PPL/Q4",
                    "decode/Q4",
                    "dominated by",
                ],
                rows,
            ),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/cpu_tradeoff_frontier_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/cpu_tradeoff_frontier_{DATE}.md"))
    args = parser.parse_args()

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    args.output_md.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
