#!/usr/bin/env python3
"""Mechanically audit the Qwen0.5B TL2 probe negative result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_name(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = summary.get("rows", [])
    if not isinstance(rows, list):
        raise TypeError(f"summary rows must be a list, got {type(rows).__name__}")
    return {
        str(row.get("name")): row
        for row in rows
        if isinstance(row, dict)
    }


def ppl_value(row: dict[str, Any]) -> Any:
    perplexity = row.get("perplexity", {})
    return perplexity.get("ppl") if isinstance(perplexity, dict) else None


def tok_s(row: dict[str, Any], mode: str) -> float | None:
    value = row.get("bench", {}).get(mode, {}).get("tok_s")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def is_nan_ppl(row: dict[str, Any]) -> bool:
    value = ppl_value(row)
    if isinstance(value, str):
        return value.lower() == "nan"
    return isinstance(value, float) and math.isnan(value)


def finite_ppl(row: dict[str, Any], max_value: float) -> bool:
    value = ppl_value(row)
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) <= max_value


def add_check(rows: list[list[str]], label: str, ok: bool, observed: str, expected: str) -> bool:
    rows.append([label, "PASS" if ok else "FAIL", observed, expected])
    return ok


def audit(generic_path: Path, avx512_path: Path) -> tuple[bool, str]:
    generic = rows_by_name(load_summary(generic_path))
    avx512 = rows_by_name(load_summary(avx512_path))
    checks: list[list[str]] = []

    generic_tl2 = generic.get("qwen05b_qat_tl2", {})
    avx_tl2 = avx512.get("qwen05b_qat_tl2", {})
    avx_i2s = avx512.get("qwen05b_qat_i2_s", {})
    avx_f16 = avx512.get("qwen05b_fp_f16", {})
    avx_q4 = avx512.get("qwen05b_fp_q4_k_m", {})

    ok_values = [
        add_check(
            checks,
            "generic TL2 smoke segfault",
            int(generic_tl2.get("smoke_returncode", 0)) == -11,
            str(generic_tl2.get("smoke_returncode")),
            "-11",
        ),
        add_check(
            checks,
            "generic TL2 PPL segfault",
            int(generic_tl2.get("ppl_returncode", 0)) == -11,
            str(generic_tl2.get("ppl_returncode")),
            "-11",
        ),
        add_check(
            checks,
            "generic TL2 bench ran",
            int(generic_tl2.get("bench_returncode", 1)) == 0
            and (tok_s(generic_tl2, "prefill") or 0.0) > 0.0
            and (tok_s(generic_tl2, "decode") or 0.0) > 0.0,
            f"rc={generic_tl2.get('bench_returncode')} prefill={tok_s(generic_tl2, 'prefill')} decode={tok_s(generic_tl2, 'decode')}",
            "rc=0 and positive tok/s",
        ),
        add_check(
            checks,
            "AVX512 TL2 all commands ran",
            all(int(avx_tl2.get(key, 1)) == 0 for key in ("smoke_returncode", "bench_returncode", "ppl_returncode")),
            f"smoke={avx_tl2.get('smoke_returncode')} bench={avx_tl2.get('bench_returncode')} ppl={avx_tl2.get('ppl_returncode')}",
            "all 0",
        ),
        add_check(
            checks,
            "AVX512 FP16 control finite",
            finite_ppl(avx_f16, 30.0),
            str(ppl_value(avx_f16)),
            "finite PPL <= 30",
        ),
        add_check(
            checks,
            "AVX512 Q4_K_M control finite",
            finite_ppl(avx_q4, 30.0),
            str(ppl_value(avx_q4)),
            "finite PPL <= 30",
        ),
        add_check(
            checks,
            "AVX512 I2_S quality failed",
            is_nan_ppl(avx_i2s),
            str(ppl_value(avx_i2s)),
            "NaN",
        ),
        add_check(
            checks,
            "AVX512 TL2 quality failed",
            is_nan_ppl(avx_tl2),
            str(ppl_value(avx_tl2)),
            "NaN",
        ),
        add_check(
            checks,
            "AVX512 TL2 throughput measured",
            (tok_s(avx_tl2, "prefill") or 0.0) > 0.0 and (tok_s(avx_tl2, "decode") or 0.0) > 0.0,
            f"prefill={tok_s(avx_tl2, 'prefill')} decode={tok_s(avx_tl2, 'decode')}",
            "positive tok/s",
        ),
    ]

    report = "\n\n".join(
        [
            "# Qwen0.5B TL2 Probe Evidence Audit",
            md_table(["check", "status", "observed", "expected"], checks),
            "## Verdict",
            (
                "PASS: the artifacts support the scoped negative TL2 claim."
                if all(ok_values)
                else "FAIL: one or more TL2 probe expectations did not match the artifacts."
            ),
        ]
    )
    return all(ok_values), report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generic-summary", type=Path, required=True)
    parser.add_argument("--avx512-summary", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    ok, report = audit(args.generic_summary, args.avx512_summary)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report + "\n", encoding="utf-8")
    print(report)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
