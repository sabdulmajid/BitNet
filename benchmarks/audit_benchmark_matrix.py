#!/usr/bin/env python3
"""Audit the side-by-side benchmark matrix against concrete artifacts.

This is a reviewer-facing coverage audit for the original benchmark objective:
it verifies that the public comparison is not based on one-off samples or proxy
signals. The quality matrix must contain at least ten distinct non-runtime
benchmarks across the same FP/PTQ/QAT model families, and the runtime matrix
must contain finite Xeon speed, size, RSS, and PPL evidence.
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

SELECTED_METRICS = {
    "arc_challenge": "acc_norm",
    "arc_easy": "acc_norm",
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
    "boolq": "acc",
    "copa": "acc",
    "openbookqa": "acc_norm",
    "sciq": "acc_norm",
    "truthfulqa_mc1": "acc",
}

PPL_RUNS = {
    "FP": (
        "benchmark_results/quality-9735/qwen15b_fp_wikitext.json",
        "benchmark_results/quality-9735/qwen15b_fp_fineweb_heldout.json",
    ),
    "naive PTQ": (
        "benchmark_results/quality-ptq-qwen15b/qwen15b_naive_ptq_wikitext.json",
        "benchmark_results/quality-ptq-qwen15b/qwen15b_naive_ptq_fineweb_heldout.json",
    ),
    "QAT hidden-MSE": (
        "benchmark_results/quality-9735/qwen15b_ternary_wikitext.json",
        "benchmark_results/quality-9735/qwen15b_ternary_fineweb_heldout.json",
    ),
    "QAT KL-only": (
        "benchmark_results/quality-qwen15b-klonly-5000/qwen15b_ternary_wikitext.json",
        "benchmark_results/quality-qwen15b-klonly-5000/qwen15b_ternary_fineweb_heldout.json",
    ),
    "QAT KL-only dense lm_head": (
        "benchmark_results/quality-qwen15b-klonly-notiehead-5000/qwen15b_ternary_wikitext.json",
        "benchmark_results/quality-qwen15b-klonly-notiehead-5000/qwen15b_ternary_fineweb_heldout.json",
    ),
    "QAT KL-only row dense lm_head": (
        "benchmark_results/quality-qwen15b-klonly-row-notiehead-5000/qwen15b_ternary_wikitext.json",
        "benchmark_results/quality-qwen15b-klonly-row-notiehead-5000/qwen15b_ternary_fineweb_heldout.json",
    ),
}

LM_EVAL_RUNS = {
    "FP": "benchmark_results/lm-eval-qwen15b-full10/qwen15b_fp.json",
    "naive PTQ": "benchmark_results/lm-eval-qwen15b-full10/qwen15b_naive_ptq.json",
    "QAT hidden-MSE": "benchmark_results/lm-eval-qwen15b-full10/qwen15b_qat_ternary.json",
    "QAT KL-only": "benchmark_results/lm-eval-qwen15b-klonly-full10/qwen15b_qat_ternary.json",
    "QAT KL-only dense lm_head": "benchmark_results/lm-eval-qwen15b-klonly-notiehead-full10/qwen15b_qat_ternary.json",
    "QAT KL-only row dense lm_head": "benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/qwen15b_qat_ternary.json",
}

PAIRED_REPORTS = [
    "benchmarks/results/paired_row_densehead_minus_fp_2026-05-13.md",
    "benchmarks/results/paired_row_densehead_minus_ptq_2026-05-13.md",
    "benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/paired_row_densehead_minus_tensor_densehead.md",
    "benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/paired_row_densehead_minus_klonly.md",
]

EXPECTED_SAMPLES = 22382
EXPECTED_RSS_CONTEXTS = [512, 2048, 8192, 32768]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def metric_value(task_results: dict[str, Any], metric: str) -> float | None:
    for key in (metric, f"{metric},none"):
        value = task_results.get(key)
        if finite(value):
            return float(value)
    return None


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if finite(value):
        return f"{float(value):.6f}"
    return str(value)


def make_check(name: str, passed: bool, evidence: str, blocker: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "evidence": evidence, "blocker": "" if passed else blocker}


def collect_ppl_benchmarks(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark, index in [("WikiText perplexity", 0), ("FineWeb heldout perplexity", 1)]:
        values: dict[str, float | None] = {}
        paths: dict[str, str] = {}
        for model, rel_paths in PPL_RUNS.items():
            path = root / rel_paths[index]
            data = read_json(path)
            values[model] = float(data["perplexity"]) if data and finite(data.get("perplexity")) else None
            paths[model] = rel_paths[index]
        rows.append(
            {
                "benchmark": benchmark,
                "metric": "perplexity",
                "models": values,
                "complete": all(value is not None for value in values.values()),
                "artifact_paths": paths,
            }
        )
    return rows


def collect_lm_eval_benchmarks(root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    loaded = {model: read_json(root / rel_path) for model, rel_path in LM_EVAL_RUNS.items()}
    sample_counts: dict[str, int] = {}
    for model, data in loaded.items():
        total = 0
        if data is not None:
            samples = data.get("samples", {})
            for task in SELECTED_METRICS:
                task_samples = samples.get(task, [])
                if isinstance(task_samples, list):
                    total += len(task_samples)
        sample_counts[model] = total

    rows: list[dict[str, Any]] = []
    for task, metric in SELECTED_METRICS.items():
        values: dict[str, float | None] = {}
        for model, data in loaded.items():
            task_results = data.get("results", {}).get(task) if data else None
            values[model] = metric_value(task_results, metric) if isinstance(task_results, dict) else None
        rows.append(
            {
                "benchmark": task,
                "metric": metric,
                "models": values,
                "complete": all(value is not None for value in values.values()),
                "artifact_paths": LM_EVAL_RUNS,
            }
        )
    return rows, sample_counts


def collect_cpu_runtime(root: Path) -> dict[str, Any]:
    frontier = read_json(root / "benchmark_results/cpu_tradeoff_frontier_2026-05-15.json") or {}
    rows = frontier.get("rows", []) if isinstance(frontier.get("rows"), list) else []
    headline_labels = {"FP F16", "FP Q8_0", "FP Q4_K_M", "row I2_SR", "row TQ2_0"}
    finite_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("label") in headline_labels
        and all(finite(row.get(key)) for key in ["file_mib", "ppl", "prefill_tok_s", "decode_tok_s", "rss_512_gib"])
        and "Xeon" in str(row.get("cpu", ""))
    ]
    q4 = frontier.get("q4_vs_i2sr", {}) if isinstance(frontier.get("q4_vs_i2sr"), dict) else {}
    rss = read_json(root / "benchmark_results/gguf-rss-qwen15b-i2sr-x86act-context-2026-05-13/summary.json") or {}
    rss_rows = rss.get("rows", []) if isinstance(rss.get("rows"), list) else []
    contexts = sorted({int(row.get("ctx_size")) for row in rss_rows if isinstance(row, dict) and finite(row.get("ctx_size")) and finite(row.get("max_rss_gib"))})
    return {
        "finite_headline_rows": len(finite_rows),
        "headline_labels": sorted(row.get("label") for row in finite_rows),
        "q4_vs_i2sr": q4,
        "rss_contexts": contexts,
        "frontier_path": "benchmark_results/cpu_tradeoff_frontier_2026-05-15.json",
        "rss_path": "benchmark_results/gguf-rss-qwen15b-i2sr-x86act-context-2026-05-13/summary.json",
    }


def collect_tl2_status(root: Path) -> dict[str, Any]:
    tl2 = read_json(root / f"benchmark_results/tl2_row_scale_runtime_contract_{DATE}.json") or {}
    return {
        "ready": tl2.get("tl2_row_scale_runtime_ready"),
        "checks": len(tl2.get("checks", [])) if isinstance(tl2.get("checks"), list) else 0,
        "failed": [check.get("name") for check in tl2.get("checks", []) if isinstance(check, dict) and not check.get("passed")]
        if isinstance(tl2.get("checks"), list)
        else [],
        "path": f"benchmark_results/tl2_row_scale_runtime_contract_{DATE}.json",
    }


def build_audit(root: Path) -> dict[str, Any]:
    ppl_rows = collect_ppl_benchmarks(root)
    lm_rows, sample_counts = collect_lm_eval_benchmarks(root)
    quality_rows = ppl_rows + lm_rows
    complete_quality = [row for row in quality_rows if row["complete"]]
    paired_existing = [path for path in PAIRED_REPORTS if (root / path).exists()]
    cpu = collect_cpu_runtime(root)
    tl2 = collect_tl2_status(root)

    checks = [
        make_check(
            "At least ten distinct side-by-side quality benchmarks are complete",
            len(complete_quality) >= 10,
            f"complete={len(complete_quality)}/12; benchmarks={[row['benchmark'] for row in complete_quality]}",
            "The quality matrix has fewer than ten complete benchmark rows.",
        ),
        make_check(
            "Both perplexity benchmarks cover all six model families",
            all(row["complete"] for row in ppl_rows) and len(ppl_rows) == 2,
            f"complete={[row['benchmark'] for row in ppl_rows if row['complete']]}",
            "WikiText or FineWeb perplexity is missing for one or more model families.",
        ),
        make_check(
            "All ten lm-eval tasks cover all six model families",
            all(row["complete"] for row in lm_rows) and len(lm_rows) == len(SELECTED_METRICS),
            f"complete_tasks={sum(1 for row in lm_rows if row['complete'])}/{len(SELECTED_METRICS)}",
            "One or more selected lm-eval tasks is missing for a model family.",
        ),
        make_check(
            "Each lm-eval model family has full logged sample count",
            all(count == EXPECTED_SAMPLES for count in sample_counts.values()) and len(sample_counts) == len(LM_EVAL_RUNS),
            f"sample_counts={sample_counts}",
            f"Expected {EXPECTED_SAMPLES} logged samples for each lm-eval model family.",
        ),
        make_check(
            "Paired statistical reports exist for the key quality comparisons",
            len(paired_existing) == len(PAIRED_REPORTS),
            f"present={len(paired_existing)}/{len(PAIRED_REPORTS)}",
            "A key paired comparison report is missing.",
        ),
        make_check(
            "Xeon CPU matrix has finite PPL, size, RSS, prefill, and decode rows",
            cpu["finite_headline_rows"] >= 5,
            f"rows={cpu['finite_headline_rows']}; labels={cpu['headline_labels']}",
            "CPU runtime matrix lacks finite headline rows.",
        ),
        make_check(
            "I2_SR RSS is measured at four context lengths",
            cpu["rss_contexts"] == EXPECTED_RSS_CONTEXTS,
            f"contexts={cpu['rss_contexts']}",
            "I2_SR RSS/context scaling is incomplete.",
        ),
        make_check(
            "TL2 row-scale is explicitly excluded from success claims",
            tl2["ready"] is False and len(tl2["failed"]) > 0,
            f"ready={tl2['ready']}; failed={len(tl2['failed'])}; path={tl2['path']}",
            "TL2 row-scale status is not explicitly blocked.",
        ),
    ]

    return {
        "schema": "benchmark-matrix-audit-v1",
        "date": DATE,
        "passed": all(check["passed"] for check in checks),
        "quality_benchmark_count": len(complete_quality),
        "quality_benchmarks": quality_rows,
        "sample_counts": sample_counts,
        "paired_reports": paired_existing,
        "cpu_runtime": cpu,
        "tl2_status": tl2,
        "checks": checks,
        "failed": [check["name"] for check in checks if not check["passed"]],
        "verdict": (
            "The side-by-side dense-Qwen comparison has at least ten complete quality benchmarks "
            "plus finite Xeon runtime/RSS evidence. The full original objective remains partial "
            "only because TL2 row-scale support is explicitly blocked."
        ),
    }


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    quality_rows = [
        [
            row["benchmark"],
            row["metric"],
            "pass" if row["complete"] else "fail",
            str(sum(1 for value in row["models"].values() if value is not None)),
        ]
        for row in result["quality_benchmarks"]
    ]
    check_rows = [
        [check["name"], "pass" if check["passed"] else "fail", check["evidence"], check["blocker"]]
        for check in result["checks"]
    ]
    cpu = result["cpu_runtime"]
    return "\n\n".join(
        [
            f"# Benchmark Matrix Audit, {result['date']}",
            (
                "This audit verifies that the public side-by-side comparison is backed by "
                "complete benchmark artifacts rather than one-off samples."
            ),
            f"Overall status: {'PASS' if result['passed'] else 'FAIL'}.",
            f"Complete quality benchmarks: `{result['quality_benchmark_count']}`.",
            "## Quality Matrix",
            md_table(["benchmark", "metric", "status", "model families covered"], quality_rows),
            "## Runtime Matrix",
            md_table(
                ["field", "value"],
                [
                    ["finite Xeon rows", str(cpu["finite_headline_rows"])],
                    ["headline labels", ", ".join(cpu["headline_labels"])],
                    ["RSS contexts", ", ".join(str(ctx) for ctx in cpu["rss_contexts"])],
                    ["Q4 vs I2_SR ratios", json.dumps(cpu["q4_vs_i2sr"], sort_keys=True)],
                ],
            ),
            "## Checks",
            md_table(["check", "status", "evidence", "blocker"], check_rows),
            "## Verdict",
            result["verdict"],
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/benchmark_matrix_audit_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/benchmark_matrix_audit_{DATE}.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    result = build_audit(root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(result)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
