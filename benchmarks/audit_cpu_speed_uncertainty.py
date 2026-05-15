#!/usr/bin/env python3
"""Add confidence intervals to headline Xeon llama-bench throughput rows.

The raw benchmark summaries contain mean tok/s and standard deviation across
repetitions. This audit turns those into conservative 95% intervals for the
means and for selected speedup ratios, so CPU speed claims are not presented as
single unqualified point estimates.
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

T_CRIT_95 = {
    1: 12.706204736432095,
    2: 4.302652729749464,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.570581835636314,
    6: 2.4469118511449692,
    7: 2.3646242510102993,
    8: 2.306004135204166,
    9: 2.2621571628540993,
    10: 2.2281388519649385,
}


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
        lines.append("| " + " | ".join(fmt(cell).replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def tcrit95(n: int) -> float:
    df = max(int(n) - 1, 1)
    return T_CRIT_95.get(df, 1.959963984540054)


def mean_ci(mean: float, stddev: float, n: int) -> list[float]:
    half = tcrit95(n) * stddev / math.sqrt(float(n))
    return [mean - half, mean + half]


def ratio_ci(
    numerator_mean: float,
    numerator_stddev: float,
    denominator_mean: float,
    denominator_stddev: float,
    n: int,
) -> list[float]:
    ratio = numerator_mean / denominator_mean
    # Delta method on log ratio. This assumes independent benchmark samples and
    # is meant as a reviewer-facing uncertainty check, not a perfect timing model.
    se_log = math.sqrt(
        (numerator_stddev / (numerator_mean * math.sqrt(float(n)))) ** 2
        + (denominator_stddev / (denominator_mean * math.sqrt(float(n)))) ** 2
    )
    half = tcrit95(n) * se_log
    return [ratio * math.exp(-half), ratio * math.exp(half)]


def find_row(summary: dict[str, Any], name: str) -> dict[str, Any] | None:
    for row in summary.get("rows", []):
        if isinstance(row, dict) and row.get("name") == name:
            return row
    return None


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for label, (rel_path, name) in HEADLINE_ROWS.items():
        path = root / rel_path
        if not path.exists():
            rows.append({"label": label, "exists": False, "source": rel_path})
            continue
        summary = read_json(path)
        row = find_row(summary, name)
        if row is None:
            rows.append({"label": label, "exists": False, "source": rel_path})
            continue
        n = int(summary.get("repetitions") or 1)
        prefill = row.get("bench", {}).get("prefill", {})
        decode = row.get("bench", {}).get("decode", {})
        item = {
            "label": label,
            "name": name,
            "exists": True,
            "source": rel_path,
            "repetitions": n,
            "prefill_mean": prefill.get("tok_s"),
            "prefill_stddev": prefill.get("stddev_tok_s"),
            "decode_mean": decode.get("tok_s"),
            "decode_stddev": decode.get("stddev_tok_s"),
        }
        if finite(item["prefill_mean"]) and finite(item["prefill_stddev"]):
            item["prefill_ci95"] = mean_ci(float(item["prefill_mean"]), float(item["prefill_stddev"]), n)
        if finite(item["decode_mean"]) and finite(item["decode_stddev"]):
            item["decode_ci95"] = mean_ci(float(item["decode_mean"]), float(item["decode_stddev"]), n)
        rows.append(item)
    return rows


def add_ratio(row: dict[str, Any], reference: dict[str, Any], key: str, prefix: str) -> None:
    mean_key = f"{key}_mean"
    sd_key = f"{key}_stddev"
    if not (finite(row.get(mean_key)) and finite(row.get(sd_key)) and finite(reference.get(mean_key)) and finite(reference.get(sd_key))):
        return
    n = min(int(row.get("repetitions") or 1), int(reference.get("repetitions") or 1))
    if n <= 1:
        return
    ratio = float(row[mean_key]) / float(reference[mean_key])
    row[f"{prefix}_{key}_speedup"] = ratio
    row[f"{prefix}_{key}_speedup_ci95"] = ratio_ci(
        float(row[mean_key]),
        float(row[sd_key]),
        float(reference[mean_key]),
        float(reference[sd_key]),
        n,
    )


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(args.repo_root.resolve())
    by_label = {row["label"]: row for row in rows}
    fp = by_label.get("FP F16", {})
    q4 = by_label.get("FP Q4_K_M", {})
    for row in rows:
        if row.get("exists"):
            add_ratio(row, fp, "prefill", "vs_fp_f16")
            add_ratio(row, fp, "decode", "vs_fp_f16")
            add_ratio(row, q4, "prefill", "vs_q4_k_m")
            add_ratio(row, q4, "decode", "vs_q4_k_m")
    i2sr = by_label.get("row I2_SR", {})
    return {
        "schema": "cpu-speed-uncertainty-v1",
        "date": DATE,
        "rows": rows,
        "i2sr_vs_q4": {
            "prefill_speedup": i2sr.get("vs_q4_k_m_prefill_speedup"),
            "prefill_speedup_ci95": i2sr.get("vs_q4_k_m_prefill_speedup_ci95"),
            "decode_speedup": i2sr.get("vs_q4_k_m_decode_speedup"),
            "decode_speedup_ci95": i2sr.get("vs_q4_k_m_decode_speedup_ci95"),
        },
        "interpretation": (
            "The intervals use benchmark-reported standard deviations across repetitions and a conservative "
            "Student-t multiplier. They quantify run-to-run timing uncertainty only; they do not cover "
            "machine-to-machine variation or quality uncertainty."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    rows = [
        [
            row["label"],
            row.get("repetitions"),
            row.get("prefill_mean"),
            fmt_ci(row.get("prefill_ci95")),
            row.get("decode_mean"),
            fmt_ci(row.get("decode_ci95")),
            row.get("vs_q4_k_m_prefill_speedup"),
            fmt_ci(row.get("vs_q4_k_m_prefill_speedup_ci95")),
            row.get("vs_q4_k_m_decode_speedup"),
            fmt_ci(row.get("vs_q4_k_m_decode_speedup_ci95")),
        ]
        for row in summary["rows"]
    ]
    i2sr = summary["i2sr_vs_q4"]
    return "\n\n".join(
        [
            f"# CPU Speed Uncertainty Audit, {summary['date']}",
            summary["interpretation"],
            "",
            "## I2_SR Versus Q4_K_M",
            (
                "row-scale `I2_SR` versus FP `Q4_K_M`: prefill speedup "
                f"`{fmt(i2sr.get('prefill_speedup'))}` with 95% CI "
                f"`{fmt_ci(i2sr.get('prefill_speedup_ci95'))}`; decode speedup "
                f"`{fmt(i2sr.get('decode_speedup'))}` with 95% CI "
                f"`{fmt_ci(i2sr.get('decode_speedup_ci95'))}`."
            ),
            "",
            "## Rows",
            md_table(
                [
                    "artifact",
                    "n",
                    "prefill tok/s",
                    "prefill 95% CI",
                    "decode tok/s",
                    "decode 95% CI",
                    "prefill/Q4",
                    "prefill/Q4 95% CI",
                    "decode/Q4",
                    "decode/Q4 95% CI",
                ],
                rows,
            ),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/cpu_speed_uncertainty_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/cpu_speed_uncertainty_{DATE}.md"))
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
