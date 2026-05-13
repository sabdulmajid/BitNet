#!/usr/bin/env python3
"""Build a side-by-side Qwen2.5-1.5B retrofit summary from artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
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


CATASTROPHIC_PPL_THRESHOLD = 1.0e4


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
    ("KL-only row dense lm_head I2_S heap-fix confirmation", "benchmark_results/gguf-qwen15b-row-i2s-heapfix-confirm/summary.json"),
    ("KL-only row dense lm_head I2_SR fixed x86 ACT candidate", "benchmark_results/i2sr-row-scale-qwen15b-x86act-suite-2026-05-13/summary.json"),
]

GGUF_MEMORY_SUMMARIES = [
    ("Qwen2.5-1.5B row-scale I2_S RSS probe", "benchmark_results/gguf-rss-qwen15b-row-i2s-fixed-2026-05-05/summary.json"),
    ("Qwen2.5-1.5B row-scale I2_S RSS context scaling", "benchmark_results/gguf-rss-qwen15b-context-scaling-2026-05-05/summary.json"),
    ("Qwen2.5-1.5B row-scale I2_SR fixed x86 ACT RSS context scaling", "benchmark_results/gguf-rss-qwen15b-i2sr-x86act-context-2026-05-13/summary.json"),
]

HEADLINE_CPU_ROWS = [
    ("FP F16", "benchmark_results/gguf-qwen15b-row-i2s-prototype-suite/summary.json", "qwen15b_fp_f16"),
    ("FP Q8_0", "benchmark_results/gguf-qwen15b-row-i2s-prototype-suite/summary.json", "qwen15b_fp_q8_0"),
    ("FP Q4_K_M", "benchmark_results/gguf-qwen15b-row-i2s-prototype-suite/summary.json", "qwen15b_fp_q4_k_m"),
    ("row-scale ternary TQ2_0", "benchmark_results/gguf-qwen15b-row-i2s-prototype-suite/summary.json", "qwen15b_klonly_row_notie_static_ternary_tq2_0"),
    ("row-scale ternary I2_S prototype", "benchmark_results/gguf-qwen15b-row-i2s-heapfix-confirm/summary.json", "qwen15b_klonly_row_notie_static_ternary_i2_s_rowscale"),
    ("row-scale ternary I2_SR fixed candidate", "benchmark_results/i2sr-row-scale-qwen15b-x86act-suite-2026-05-13/summary.json", "qwen15b_klonly_row_notie_static_ternary_i2_sr_x86act"),
]

PRODUCTIZATION_GATE = "benchmark_results/row_scale_qtype_productization_gate_2026-05-13.json"
OBJECTIVE_AUDIT = "benchmark_results/objective_completion_audit_2026-05-13.json"
PRODUCT_SCOPE_GATE = "benchmark_results/product_scope_gate_2026-05-13.json"
I2SR_PROMOTION_AUDIT = "benchmark_results/i2sr_submodule_promotion_audit_2026-05-13.json"
MOE_PACKING_CONTRACT = "benchmark_results/moe_packing_contract_2026-05-13.json"

PAIRED_DELTA_REPORTS = [
    ("QAT row-scale minus FP", "benchmarks/results/paired_row_densehead_minus_fp_2026-05-13.md"),
    ("QAT row-scale minus naive PTQ", "benchmarks/results/paired_row_densehead_minus_ptq_2026-05-13.md"),
    (
        "QAT row-scale minus tensor-scale dense lm_head",
        "benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/paired_row_densehead_minus_tensor_densehead.md",
    ),
    (
        "QAT row-scale minus KL-only tensor-scale",
        "benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/paired_row_densehead_minus_klonly.md",
    ),
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.{digits}f}"
    return "-" if value is None else str(value)


def quality_status(ppl: Any) -> str:
    if ppl is None:
        return "missing"
    if not isinstance(ppl, (int, float)) or not math.isfinite(float(ppl)):
        return "nan-fail"
    if float(ppl) >= CATASTROPHIC_PPL_THRESHOLD:
        return "catastrophic"
    return "ok"


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


def find_summary_row(path: Path, name: str) -> dict[str, Any] | None:
    summary = read_json(path)
    if summary is None:
        return None
    for row in summary.get("rows", []):
        if row.get("name") == name:
            return row
    return None


def build_headline_table() -> str:
    rows: list[list[str]] = []
    fp_lm_mean, _, _ = selected_lm_eval_mean(Path("benchmark_results/lm-eval-qwen15b-full10/qwen15b_fp.json"))
    best_lm_mean, _, _ = selected_lm_eval_mean(Path("benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/qwen15b_qat_ternary.json"))
    fp_wiki = read_json(Path("benchmark_results/quality-9735/qwen15b_fp_wikitext.json"))
    best_wiki = read_json(Path("benchmark_results/quality-qwen15b-klonly-row-notiehead-5000/qwen15b_ternary_wikitext.json"))
    fp_fineweb = read_json(Path("benchmark_results/quality-9735/qwen15b_fp_fineweb_heldout.json"))
    best_fineweb = read_json(Path("benchmark_results/quality-qwen15b-klonly-row-notiehead-5000/qwen15b_ternary_fineweb_heldout.json"))

    rows.append([
        "best HF quality recovery",
        "QAT KL-only row-scale dense lm_head",
        fmt(best_wiki.get("perplexity") if best_wiki else None, 3),
        fmt(fp_wiki.get("perplexity") if fp_wiki else None, 3),
        fmt(best_fineweb.get("perplexity") if best_fineweb else None, 3),
        fmt(fp_fineweb.get("perplexity") if fp_fineweb else None, 3),
        fmt(best_lm_mean, 6),
        fmt(fp_lm_mean, 6),
        "not FP-quality",
    ])

    gate = read_json(Path(PRODUCTIZATION_GATE))
    gate_status = "missing"
    if gate is not None:
        failed = [item.get("name", "") for item in gate.get("gates", []) if not item.get("passed")]
        gate_status = "production gate pass" if gate.get("passed") else f"not production-ready; {len(failed)} qtype/runtime gates fail"

    rows.append([
        "packed CPU candidate",
        "direct I2_SR fixed x86 ACT",
        "38.848",
        "12.281",
        "-",
        "-",
        "-",
        "-",
        gate_status,
    ])

    return md_table(
        [
            "claim area",
            "best current artifact",
            "artifact Wiki/CPU PPL",
            "reference FP PPL",
            "artifact FineWeb/PPL",
            "reference FP FineWeb/PPL",
            "artifact ten-task mean",
            "FP ten-task mean",
            "status",
        ],
        rows,
    )


def build_reviewer_gate_table() -> str:
    objective = read_json(Path(OBJECTIVE_AUDIT)) or {}
    scope = read_json(Path(PRODUCT_SCOPE_GATE)) or {}
    i2sr = read_json(Path(I2SR_PROMOTION_AUDIT)) or {}
    moe = read_json(Path(MOE_PACKING_CONTRACT)) or {}
    moe_verdict = moe.get("verdict", {}) if isinstance(moe.get("verdict"), dict) else {}
    rows = [
        [
            "benchmark coverage",
            "pass",
            "full ten-task, paired deltas, CPU quality/speed/RSS, manifest",
            "This confirms artifact coverage, not product completion.",
        ],
        [
            "objective completion",
            str(objective.get("completion_status", "missing")),
            f"{objective.get('complete_count', '-')}/{objective.get('check_count', '-')} complete",
            "Open items are default row-scale runtime promotion and MoE/Kimi evidence.",
        ],
        [
            "product scope",
            str(scope.get("scope_status", "missing")),
            str(scope.get("publishable_angle", "-")),
            "Do not claim arbitrary lossless retrofit or MoE/Kimi support.",
        ],
        [
            "I2_SR active submodule",
            "ready" if i2sr.get("promotion_ready") else "blocked",
            f"active={i2sr.get('active_runtime_support')}; patch_applies={i2sr.get('patch_applies_cleanly')}; blockers={len(i2sr.get('blockers', []))}",
            "Quality-valid CPU path exists only with the downstream patch until a writable llama.cpp fork/branch is provided.",
        ],
        [
            "MoE/Kimi packing",
            "ready" if moe_verdict.get("moe_packing_ready") else "blocked",
            (
                f"tl2_3d={moe_verdict.get('merged_3d_tl2_supported')}; "
                f"i2sr_3d={moe_verdict.get('merged_3d_i2s_i2sr_supported')}; "
                f"2d_control={moe_verdict.get('dense_2d_i2s_control_supported')}"
            ),
            "Synthetic contract now separates direct I2_S/I2_SR packing from TL2; no Kimi artifact exists.",
        ],
    ]
    return md_table(["gate", "status", "evidence", "reviewer implication"], rows)


def build_cpu_headline_table() -> str:
    rows: list[list[str]] = []
    for label, path, name in HEADLINE_CPU_ROWS:
        row = find_summary_row(Path(path), name)
        if row is None:
            rows.append([label, "-", "-", "-", "-", "-", "missing"])
            continue
        bench = row.get("bench", {})
        ppl = row.get("perplexity", {}).get("ppl")
        cpu = bench.get("prefill", {}).get("cpu") or bench.get("decode", {}).get("cpu") or "-"
        rows.append([
            label,
            str(cpu),
            fmt(row.get("file_mib"), 1),
            fmt(ppl, 4),
            fmt(bench.get("prefill", {}).get("tok_s"), 2),
            fmt(bench.get("decode", {}).get("tok_s"), 2),
            quality_status(ppl),
        ])
    return md_table(["artifact", "CPU", "file MiB", "PPL", "prefill tok/s", "decode tok/s", "quality status"], rows)


def parse_paired_summary(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    macro = re.search(r"\| macro mean delta \| ([^|]+) \|", text)
    weighted = re.search(r"\| example-weighted delta \| ([^|]+) \|", text)
    tasks = re.findall(r"^\| [a-z0-9_]+ \| [a-z_]+ \| ([0-9]+) \|", text, flags=re.MULTILINE)
    return {
        "macro": macro.group(1).strip() if macro else "-",
        "weighted": weighted.group(1).strip() if weighted else "-",
        "tasks": str(len(tasks)),
        "matched": str(sum(int(value) for value in tasks)) if tasks else "-",
    }


def build_paired_delta_table() -> str:
    rows: list[list[str]] = []
    for label, path in PAIRED_DELTA_REPORTS:
        summary = parse_paired_summary(Path(path))
        if summary is None:
            rows.append([label, "-", "-", "-", "missing"])
            continue
        rows.append([label, summary["macro"], summary["weighted"], summary["matched"], "present"])
    return md_table(["comparison", "macro mean delta with 95% CI", "example-weighted delta", "matched examples", "status"], rows)


def build_gguf_table() -> str:
    rows: list[list[str]] = []
    for suite_label, path in GGUF_SUMMARIES:
        summary = read_json(Path(path))
        if summary is None:
            rows.append([suite_label, "-", "-", "-", "-", "-", "missing"])
            continue
        for row in summary.get("rows", []):
            name = str(row.get("name", ""))
            if not name.endswith(("f16", "q8_0", "q4_k_m", "tq2_0", "i2_s", "i2_s_rowscale", "i2_sr_x86act")):
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
                quality_status(ppl),
            ])
    return md_table(["suite", "CPU", "artifact", "kind", "file MiB", "prefill tok/s", "decode tok/s", "PPL", "quality status"], rows)


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
                str(row.get("ctx_size", "-")),
                fmt(row.get("file_mib"), 1),
                fmt(row.get("max_rss_gib"), 3),
                str(row.get("returncode", "")),
            ])
    return md_table(["suite", "artifact", "kind", "ctx", "file MiB", "max RSS GiB", "return code"], rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    report = "\n\n".join([
        "# Qwen2.5-1.5B Side-by-Side Artifact Summary",
        (
            "Generated from benchmark JSON artifacts. Missing rows are intentionally "
            "shown as missing. The Xeon headline isolates the Intel Xeon Silver 4116 "
            "runs; the longer GGUF table also preserves older Threadripper control "
            "runs and should not be used for cross-machine speed ratios."
        ),
        "## Headline Verdict",
        build_headline_table(),
        "## Reviewer Gate Summary",
        build_reviewer_gate_table(),
        "## Perplexity",
        build_ppl_table(),
        "## Full Ten-Task lm-eval",
        build_lm_eval_table(),
        "## Full Ten-Task Detail",
        build_lm_eval_detail_table(),
        "## Paired Ten-Task Delta Checks",
        build_paired_delta_table(),
        "## Xeon Packed Runtime Headline",
        build_cpu_headline_table(),
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
