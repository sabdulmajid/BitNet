#!/usr/bin/env python3
"""Audit whether the public benchmark matrix is backed by concrete artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
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

LM_EVAL_RUNS = {
    "FP": "benchmark_results/lm-eval-qwen15b-full10/qwen15b_fp.json",
    "naive PTQ": "benchmark_results/lm-eval-qwen15b-full10/qwen15b_naive_ptq.json",
    "QAT hidden-MSE": "benchmark_results/lm-eval-qwen15b-full10/qwen15b_qat_ternary.json",
    "QAT KL-only": "benchmark_results/lm-eval-qwen15b-klonly-full10/qwen15b_qat_ternary.json",
    "QAT KL-only dense lm_head": "benchmark_results/lm-eval-qwen15b-klonly-notiehead-full10/qwen15b_qat_ternary.json",
    "QAT KL-only row dense lm_head": "benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/qwen15b_qat_ternary.json",
}

PAIRED_REPORTS = {
    "row minus FP": "benchmarks/results/paired_row_densehead_minus_fp_2026-05-13.md",
    "row minus naive PTQ": "benchmarks/results/paired_row_densehead_minus_ptq_2026-05-13.md",
    "row minus tensor dense-head": "benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/paired_row_densehead_minus_tensor_densehead.md",
    "row minus KL-only": "benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/paired_row_densehead_minus_klonly.md",
}

CPU_ROWS = {
    "FP F16": ("benchmark_results/gguf-qwen15b-row-i2s-prototype-suite/summary.json", "qwen15b_fp_f16"),
    "FP Q8_0": ("benchmark_results/gguf-qwen15b-row-i2s-prototype-suite/summary.json", "qwen15b_fp_q8_0"),
    "FP Q4_K_M": ("benchmark_results/gguf-qwen15b-row-i2s-prototype-suite/summary.json", "qwen15b_fp_q4_k_m"),
    "row-scale TQ2_0": ("benchmark_results/gguf-qwen15b-row-i2s-prototype-suite/summary.json", "qwen15b_klonly_row_notie_static_ternary_tq2_0"),
    "row-scale I2_S": ("benchmark_results/gguf-qwen15b-row-i2s-heapfix-confirm/summary.json", "qwen15b_klonly_row_notie_static_ternary_i2_s_rowscale"),
    "row-scale I2_SR": ("benchmark_results/i2sr-row-scale-qwen15b-x86act-suite-2026-05-13/summary.json", "qwen15b_klonly_row_notie_static_ternary_i2_sr_x86act"),
}

EXPECTED_TASKS = len(SELECTED_METRICS)
EXPECTED_SAMPLES = 22382
BITDISTILL_GLUE_EXPECTED = {"mnli": 9815, "qnli": 5463, "sst2": 872}
EXPECTED_RSS_CONTEXTS = [512, 2048, 8192, 32768]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_artifact(root: Path, pattern: str, fallback: str) -> Path:
    matches = sorted(root.glob(pattern))
    return matches[-1] if matches else root / fallback


def metric_value(task_results: dict[str, Any], metric: str) -> float | None:
    for key in (metric, f"{metric},none"):
        value = task_results.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, evidence: str, blocker: str = "") -> None:
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence, "blocker": "" if passed else blocker})


def audit_lm_eval(root: Path, checks: list[dict[str, Any]]) -> None:
    for label, rel_path in LM_EVAL_RUNS.items():
        path = root / rel_path
        if not path.exists():
            add_check(checks, f"{label} lm-eval file exists", False, rel_path, "missing file")
            continue
        data = read_json(path)
        results = data.get("results", {})
        samples = data.get("samples", {})
        present_tasks = 0
        sample_count = 0
        missing: list[str] = []
        for task, metric in SELECTED_METRICS.items():
            task_results = results.get(task)
            if not isinstance(task_results, dict) or metric_value(task_results, metric) is None:
                missing.append(task)
                continue
            present_tasks += 1
            task_samples = samples.get(task, [])
            if isinstance(task_samples, list):
                sample_count += len(task_samples)
        add_check(
            checks,
            f"{label} has ten selected lm-eval tasks",
            present_tasks == EXPECTED_TASKS and not missing,
            f"tasks={present_tasks}, missing={missing}",
            "selected metric missing from one or more tasks",
        )
        add_check(
            checks,
            f"{label} has expected logged samples",
            sample_count == EXPECTED_SAMPLES,
            f"samples={sample_count}",
            f"expected {EXPECTED_SAMPLES} logged samples",
        )


def audit_paired_reports(root: Path, checks: list[dict[str, Any]]) -> None:
    row_re = re.compile(r"^\| [a-z0-9_]+ \| [a-z_]+ \| ([0-9]+) \|", re.MULTILINE)
    macro_re = re.compile(r"\| macro mean delta \| ([^|]+) \|")
    for label, rel_path in PAIRED_REPORTS.items():
        path = root / rel_path
        if not path.exists():
            add_check(checks, f"{label} paired report exists", False, rel_path, "missing report")
            continue
        text = path.read_text(encoding="utf-8")
        rows = [int(value) for value in row_re.findall(text)]
        macro = macro_re.search(text)
        add_check(
            checks,
            f"{label} has ten paired task rows",
            len(rows) == EXPECTED_TASKS,
            f"rows={len(rows)}",
            "paired task row count mismatch",
        )
        add_check(
            checks,
            f"{label} has expected paired examples",
            sum(rows) == EXPECTED_SAMPLES,
            f"matched={sum(rows)}",
            f"expected {EXPECTED_SAMPLES} matched examples",
        )
        add_check(
            checks,
            f"{label} has macro CI",
            macro is not None and "[" in macro.group(1) and "]" in macro.group(1),
            macro.group(1).strip() if macro else "missing",
            "macro delta CI not found",
        )


def audit_bitdistill_paired_baselines(root: Path, checks: list[dict[str, Any]]) -> None:
    path = latest_artifact(
        root,
        "benchmark_results/bitdistill_paired_predictions_*.json",
        f"benchmark_results/bitdistill_paired_predictions_{DATE}.json",
    )
    if not path.exists():
        add_check(checks, "BitDistill paired prediction audit exists", False, str(path), "missing paired audit")
        return
    data = read_json(path)
    rows = data.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    complete_rows = [row for row in rows if isinstance(row, dict) and row.get("status") == "pass"]
    stats_complete_rows = [
        row
        for row in complete_rows
        if isinstance(row.get("paired_ci95"), list)
        and len(row.get("paired_ci95")) == 2
        and isinstance(row.get("mcnemar_exact_p"), (int, float))
    ]
    baseline_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("family") == "baseline_vs_fp" and row.get("candidate_label") == "BitNet-SFT"
    ]
    full_rows = [
        row
        for row in baseline_rows
        if row.get("status") == "pass"
        and row.get("matched") == BITDISTILL_GLUE_EXPECTED.get(str(row.get("task")))
        and row.get("expected_examples") == BITDISTILL_GLUE_EXPECTED.get(str(row.get("task")))
    ]
    stats_rows = [
        row
        for row in full_rows
        if isinstance(row.get("paired_ci95"), list)
        and len(row.get("paired_ci95")) == 2
        and isinstance(row.get("mcnemar_exact_p"), (int, float))
    ]
    add_check(
        checks,
        "BitDistill paired audit is complete",
        data.get("status") == "pass"
        and data.get("complete") == data.get("total") == len(rows)
        and data.get("pending") == 0
        and data.get("failed") == 0,
        f"complete={data.get('complete')}/{data.get('total')}, pending={data.get('pending')}, failed={data.get('failed')}",
        "paired prediction audit still has pending or failed rows",
    )
    add_check(
        checks,
        "BitDistill paired audit has paired statistics for every row",
        len(stats_complete_rows) == len(rows) and len(rows) > 0,
        f"stats_rows={len(stats_complete_rows)}/{len(rows)}",
        "at least one completed paired row lacks CI or McNemar statistics",
    )
    add_check(
        checks,
        "BitDistill paired audit has BitNet baseline rows",
        len(baseline_rows) == len(BITDISTILL_GLUE_EXPECTED),
        f"rows={len(baseline_rows)}, path={path.relative_to(root)}",
        "expected one BitNet-SFT-vs-FP16 row for each GLUE task",
    )
    add_check(
        checks,
        "BitNet baseline paired rows cover full GLUE validation",
        len(full_rows) == len(BITDISTILL_GLUE_EXPECTED)
        and sum(int(row.get("matched", 0)) for row in full_rows) == sum(BITDISTILL_GLUE_EXPECTED.values()),
        f"full_rows={len(full_rows)}, matched={sum(int(row.get('matched', 0)) for row in full_rows)}",
        "paired baseline rows are missing or partial",
    )
    add_check(
        checks,
        "BitNet baseline paired rows have paired statistics",
        len(stats_rows) == len(BITDISTILL_GLUE_EXPECTED),
        f"stats_rows={len(stats_rows)}",
        "paired CI or McNemar p-value missing",
    )


def audit_bitnet_sft_budget_paired(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/bitnet_sft_budget_paired_{DATE}.json"
    if not path.exists():
        add_check(checks, "BitNet-SFT budget paired audit exists", False, str(path.relative_to(root)), "missing paired audit")
        return
    data = read_json(path)
    best = data.get("best", {}) if isinstance(data.get("best"), dict) else {}
    complete = data.get("complete")
    total = data.get("total")
    ci = best.get("paired_ci95")
    pvalue = best.get("mcnemar_exact_p")
    add_check(
        checks,
        "BitNet-SFT budget paired audit has completed full-MNLI rows",
        isinstance(complete, int)
        and isinstance(total, int)
        and complete >= 1
        and total >= complete
        and best.get("matched") == BITDISTILL_GLUE_EXPECTED["mnli"],
        f"complete={complete}/{total}, best_matched={best.get('matched')}, path={path.relative_to(root)}",
        "no full-validation paired BitNet-SFT budget row is available",
    )
    add_check(
        checks,
        "BitNet-SFT best budget row has paired CI and McNemar test",
        isinstance(ci, list)
        and len(ci) == 2
        and isinstance(pvalue, (int, float))
        and isinstance(best.get("delta_vs_reference"), (int, float)),
        f"delta={best.get('delta_vs_reference')}, ci={ci}, mcnemar={pvalue}",
        "best budget row lacks paired statistical evidence",
    )


def find_row(summary: dict[str, Any], name: str) -> dict[str, Any] | None:
    for row in summary.get("rows", []):
        if row.get("name") == name:
            return row
    return None


def audit_cpu_rows(root: Path, checks: list[dict[str, Any]]) -> None:
    for label, (rel_path, name) in CPU_ROWS.items():
        path = root / rel_path
        row = find_row(read_json(path), name) if path.exists() else None
        ppl = row.get("perplexity", {}).get("ppl") if row else None
        prefill = row.get("bench", {}).get("prefill", {}).get("tok_s") if row else None
        decode = row.get("bench", {}).get("decode", {}).get("tok_s") if row else None
        add_check(
            checks,
            f"{label} CPU row is finite",
            isinstance(ppl, (int, float)) and isinstance(prefill, (int, float)) and isinstance(decode, (int, float)),
            f"ppl={ppl}, prefill={prefill}, decode={decode}",
            "missing or non-finite PPL/throughput",
        )


def manifest_missing_is_only_self_coverage(manifest: dict[str, Any]) -> bool:
    missing = manifest.get("missing", [])
    if not isinstance(missing, list):
        return False
    optional_preflight = {
        "benchmark_coverage_gate_report",
        "benchmark_coverage_gate_json",
        "bitdistill_postprocess_dependency_report",
        "bitdistill_postprocess_dependency_json",
        "bitdistill_producer_script_audit_report",
        "bitdistill_producer_script_audit_json",
    }
    return set(missing) <= optional_preflight


def audit_rss_and_gates(root: Path, checks: list[dict[str, Any]], manifest_path_arg: Path | None) -> None:
    rss_path = root / "benchmark_results/gguf-rss-qwen15b-i2sr-x86act-context-2026-05-13/summary.json"
    rss = read_json(rss_path)
    contexts = sorted({int(row.get("ctx_size")) for row in rss.get("rows", []) if row.get("returncode") == 0})
    add_check(
        checks,
        "fixed I2_SR RSS has four context rows",
        contexts == EXPECTED_RSS_CONTEXTS,
        f"contexts={contexts}",
        f"expected {EXPECTED_RSS_CONTEXTS}",
    )

    manifest_path = (
        manifest_path_arg
        if manifest_path_arg is not None
        else latest_artifact(root, "benchmarks/results/evidence_manifest_*.json", "benchmarks/results/evidence_manifest_2026-05-13.json")
    )
    manifest = read_json(manifest_path)
    missing_ok = manifest.get("missing_count") == 0 or manifest_missing_is_only_self_coverage(manifest)
    add_check(
        checks,
        "evidence manifest has no missing artifacts",
        missing_ok and manifest.get("artifact_count", 0) >= 78,
        f"path={manifest_path.relative_to(root)}, artifacts={manifest.get('artifact_count')}, missing={manifest.get('missing_count')}, missing_labels={manifest.get('missing', [])}",
        "manifest is missing one or more cited artifacts",
    )

    gate = read_json(root / "benchmark_results/row_scale_qtype_productization_gate_2026-05-13.json")
    failed = [item for item in gate.get("gates", []) if not item.get("passed")]
    add_check(
        checks,
        "productization gate passes for stable I2_SR",
        gate.get("passed") is True
        and len(failed) == 0
        and gate.get("observations", {}).get("stable_benchmark_quality_ok") is True
        and gate.get("observations", {}).get("packing_verification_passed") is True,
        f"passed={gate.get('passed')}, failed={len(failed)}, stable_quality={gate.get('observations', {}).get('stable_benchmark_quality_ok')}, layout={gate.get('observations', {}).get('packing_verification_passed')}",
        "stable I2_SR productization gate did not pass",
    )


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    rows = [
        [
            check["name"],
            "pass" if check["passed"] else "fail",
            str(check["evidence"]),
            str(check.get("blocker", "")),
        ]
        for check in result["checks"]
    ]
    status = "PASS" if result["passed"] else "FAIL"
    return "\n\n".join(
        [
            f"# Benchmark Coverage Gate, {result['date']}",
            f"Overall status: **{status}**.",
            md_table(["check", "status", "evidence", "blocker"], rows),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/benchmark_coverage_gate_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/benchmark_coverage_gate_{DATE}.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    checks: list[dict[str, Any]] = []
    audit_lm_eval(root, checks)
    audit_paired_reports(root, checks)
    audit_bitdistill_paired_baselines(root, checks)
    audit_bitnet_sft_budget_paired(root, checks)
    audit_cpu_rows(root, checks)
    manifest_path = args.manifest_path.resolve() if args.manifest_path is not None else None
    audit_rss_and_gates(root, checks, manifest_path)

    result = {
        "schema": "benchmark_coverage_gate.v1",
        "date": DATE,
        "passed": all(check["passed"] for check in checks),
        "check_count": len(checks),
        "failed": [check["name"] for check in checks if not check["passed"]],
        "checks": checks,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(result) + "\n", encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
