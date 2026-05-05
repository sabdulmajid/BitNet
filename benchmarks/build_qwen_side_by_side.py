#!/usr/bin/env python3
"""Build a side-by-side Qwen2.5-1.5B retrofit summary from artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SELECTED_LM_EVAL_METRICS = {
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


PPL_RUNS = [
    ("FP", "benchmark_results/quality-9735/qwen15b_fp_wikitext.json", "benchmark_results/quality-9735/qwen15b_fp_fineweb_heldout.json"),
    ("naive PTQ", "benchmark_results/quality-ptq-qwen15b/qwen15b_naive_ptq_wikitext.json", "benchmark_results/quality-ptq-qwen15b/qwen15b_naive_ptq_fineweb_heldout.json"),
    ("QAT hidden-MSE", "benchmark_results/quality-9735/qwen15b_ternary_wikitext.json", "benchmark_results/quality-9735/qwen15b_ternary_fineweb_heldout.json"),
    ("QAT KL-only", "benchmark_results/quality-qwen15b-klonly-5000/qwen15b_ternary_wikitext.json", "benchmark_results/quality-qwen15b-klonly-5000/qwen15b_ternary_fineweb_heldout.json"),
    ("QAT KL-only dense lm_head", "benchmark_results/quality-qwen15b-klonly-notiehead-5000/qwen15b_ternary_wikitext.json", "benchmark_results/quality-qwen15b-klonly-notiehead-5000/qwen15b_ternary_fineweb_heldout.json"),
    ("QAT KL-only row dense lm_head", "benchmark_results/quality-qwen15b-klonly-row-notiehead-5000/qwen15b_ternary_wikitext.json", "benchmark_results/quality-qwen15b-klonly-row-notiehead-5000/qwen15b_ternary_fineweb_heldout.json"),
]

LM_EVAL_RUNS = [
    ("FP", "benchmark_results/lm-eval-qwen15b-full10/qwen15b_fp.json"),
    ("naive PTQ", "benchmark_results/lm-eval-qwen15b-full10/qwen15b_naive_ptq.json"),
    ("QAT hidden-MSE", "benchmark_results/lm-eval-qwen15b-full10/qwen15b_qat_ternary.json"),
    ("QAT KL-only", "benchmark_results/lm-eval-qwen15b-klonly-full10/qwen15b_qat_ternary.json"),
    ("QAT KL-only dense lm_head", "benchmark_results/lm-eval-qwen15b-klonly-notiehead-full10/qwen15b_qat_ternary.json"),
    ("QAT KL-only row dense lm_head", "benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/qwen15b_qat_ternary.json"),
]

GGUF_SUMMARIES = [
    ("KL-only all-linear static ternary suite", "benchmark_results/gguf-qwen15b-klonly-suite/summary.json"),
    ("KL-only dense lm_head static ternary suite", "benchmark_results/gguf-qwen15b-klonly-notiehead-suite/summary.json"),
    ("KL-only row dense lm_head static ternary suite", "benchmark_results/gguf-qwen15b-klonly-row-notiehead-suite/summary.json"),
    ("KL-only row dense lm_head I2_S row-scale prototype suite", "benchmark_results/gguf-qwen15b-row-i2s-prototype-suite/summary.json"),
    ("KL-only row dense lm_head I2_S row-scale prototype native suite", "benchmark_results/gguf-qwen15b-row-i2s-prototype-native-suite/summary.json"),
]

GGUF_MEMORY_SUMMARIES = [
    ("Qwen2.5-1.5B row-scale I2_S RSS probe", "benchmark_results/gguf-rss-qwen15b-row-i2s-fixed-2026-05-05/summary.json"),
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.{digits}f}"
    return "-" if value is None else str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def metric_value(task_results: dict[str, Any], metric: str) -> float:
    for key in (metric, f"{metric},none"):
        value = task_results.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    raise KeyError(metric)


def selected_lm_eval_mean(path: Path) -> tuple[float | None, int, int]:
    data = read_json(path)
    if data is None:
        return None, 0, 0
    results = data.get("results", {})
    samples = data.get("samples", {})
    values: list[float] = []
    sample_count = 0
    for task, metric in SELECTED_LM_EVAL_METRICS.items():
        task_results = results.get(task)
        if not isinstance(task_results, dict):
            continue
        values.append(metric_value(task_results, metric))
        task_samples = samples.get(task, [])
        if isinstance(task_samples, list):
            sample_count += len(task_samples)
    return (sum(values) / len(values) if values else None), len(values), sample_count


def build_ppl_table() -> str:
    rows: list[list[str]] = []
    for label, wiki_path, fineweb_path in PPL_RUNS:
        wiki = read_json(Path(wiki_path))
        fineweb = read_json(Path(fineweb_path))
        rows.append([
            label,
            fmt(wiki.get("perplexity") if wiki else None),
            fmt(fineweb.get("perplexity") if fineweb else None),
            str(int(wiki.get("eval_tokens", 0))) if wiki else "-",
            str(int(fineweb.get("eval_tokens", 0))) if fineweb else "-",
            "present" if wiki and fineweb else "missing",
        ])
    return md_table(["run", "WikiText PPL", "FineWeb PPL", "Wiki tokens", "FineWeb tokens", "status"], rows)


def build_lm_eval_table() -> str:
    rows: list[list[str]] = []
    for label, path in LM_EVAL_RUNS:
        mean, tasks, samples = selected_lm_eval_mean(Path(path))
        rows.append([label, fmt(mean, 6), str(tasks), str(samples), "present" if mean is not None else "missing"])
    return md_table(["run", "selected mean", "tasks", "samples", "status"], rows)


def build_lm_eval_detail_table() -> str:
    loaded = [(label, read_json(Path(path))) for label, path in LM_EVAL_RUNS]
    rows: list[list[str]] = []
    for task, metric in SELECTED_LM_EVAL_METRICS.items():
        row = [task, metric]
        for _, data in loaded:
            value = None
            if data is not None:
                task_results = data.get("results", {}).get(task)
                if isinstance(task_results, dict):
                    try:
                        value = metric_value(task_results, metric)
                    except KeyError:
                        value = None
            row.append(fmt(value, 3))
        rows.append(row)
    return md_table(["task", "metric", *[label for label, _ in loaded]], rows)


def build_gguf_table() -> str:
    rows: list[list[str]] = []
    for suite_label, path in GGUF_SUMMARIES:
        summary = read_json(Path(path))
        if summary is None:
            rows.append([suite_label, "-", "-", "-", "-", "-", "missing"])
            continue
        for row in summary.get("rows", []):
            name = str(row.get("name", ""))
            if not name.endswith(("f16", "q8_0", "q4_k_m", "tq2_0", "i2_s", "i2_s_rowscale")):
                continue
            ppl = row.get("perplexity", {}).get("ppl")
            bench = row.get("bench", {})
            cpu = bench.get("prefill", {}).get("cpu") or bench.get("decode", {}).get("cpu") or "-"
            rows.append([
                suite_label,
                str(cpu),
                name,
                str(row.get("kind", "")),
                fmt(row.get("file_mib"), 1),
                fmt(bench.get("prefill", {}).get("tok_s"), 2),
                fmt(bench.get("decode", {}).get("tok_s"), 2),
                fmt(ppl, 4),
            ])
    return md_table(["suite", "CPU", "artifact", "kind", "file MiB", "prefill tok/s", "decode tok/s", "PPL"], rows)


def build_gguf_memory_table() -> str:
    rows: list[list[str]] = []
    for suite_label, path in GGUF_MEMORY_SUMMARIES:
        summary = read_json(Path(path))
        if summary is None:
            rows.append([suite_label, "-", "-", "-", "-", "-", "missing"])
            continue
        for row in summary.get("rows", []):
            rows.append([
                suite_label,
                str(row.get("name", "")),
                str(row.get("kind", "")),
                fmt(row.get("file_mib"), 1),
                fmt(row.get("max_rss_gib"), 3),
                str(row.get("returncode", "")),
            ])
    return md_table(["suite", "artifact", "kind", "file MiB", "max RSS GiB", "return code"], rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    report = "\n\n".join([
        "# Qwen2.5-1.5B Side-by-Side Artifact Summary",
        "Generated from benchmark JSON artifacts. Missing rows are intentionally shown as missing.",
        "## Perplexity",
        build_ppl_table(),
        "## Full Ten-Task lm-eval",
        build_lm_eval_table(),
        "## Full Ten-Task Detail",
        build_lm_eval_detail_table(),
        "## Packed GGUF CPU",
        build_gguf_table(),
        "## Packed GGUF RSS",
        build_gguf_memory_table(),
    ])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
