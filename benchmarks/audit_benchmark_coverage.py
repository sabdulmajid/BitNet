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


def audit_bitnet_sft_mechanics(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/bitnet_sft_mechanics_audit_{DATE}.json"
    if not path.exists():
        add_check(checks, "BitNet-SFT mechanics audit exists", False, str(path.relative_to(root)), "missing mechanics audit")
        return
    data = read_json(path)
    ternary = data.get("ternary_state", {}) if isinstance(data.get("ternary_state"), dict) else {}
    family_counts = ternary.get("family_tensor_counts", {}) if isinstance(ternary.get("family_tensor_counts"), dict) else {}
    code_fractions = ternary.get("code_fractions", {}) if isinstance(ternary.get("code_fractions"), dict) else {}
    expected_families = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    add_check(
        checks,
        "BitNet-SFT mechanics audit passes",
        data.get("passed") is True and data.get("verdict") == "basic_mechanics_pass_bitdistill_recovery_pending",
        f"passed={data.get('passed')}, verdict={data.get('verdict')}, path={path.relative_to(root)}",
        "mechanics audit did not pass",
    )
    add_check(
        checks,
        "BitNet-SFT mechanics audit has exact projection replacement counts",
        ternary.get("ternary_weight_count") == 168
        and set(family_counts) == expected_families
        and all(family_counts.get(name) == 24 for name in expected_families),
        f"ternary={ternary.get('ternary_weight_count')}, families={family_counts}",
        "missing or extra ternary projection families",
    )
    add_check(
        checks,
        "BitNet-SFT mechanics audit confirms dense non-projection tensors",
        ternary.get("score_weight_dense") is True
        and ternary.get("score_ternary_present") is False
        and ternary.get("forbidden_ternary_keys") == [],
        f"score_dense={ternary.get('score_weight_dense')}, score_ternary={ternary.get('score_ternary_present')}, forbidden={ternary.get('forbidden_ternary_keys')}",
        "sequence head, embedding, or norm was unexpectedly ternarized",
    )
    add_check(
        checks,
        "BitNet-SFT mechanics audit confirms three-symbol ternary distribution",
        set(code_fractions) == {"-1", "0", "1"}
        and all(isinstance(code_fractions.get(key), (int, float)) for key in ["-1", "0", "1"]),
        f"fractions={code_fractions}, entropy={ternary.get('code_entropy_bits')}",
        "ternary code distribution is incomplete or malformed",
    )


def audit_subln_activation_variance(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/subln_activation_variance_{DATE}.json"
    if not path.exists():
        add_check(checks, "SubLN activation-variance audit exists", False, str(path.relative_to(root)), "missing SubLN audit")
        return
    data = read_json(path)
    rel = data.get("logit_relative_rms")
    cosine = data.get("logit_cosine")
    inserted = data.get("subln_inserted")
    families = data.get("families", {}) if isinstance(data.get("families"), dict) else {}
    output_rms = []
    for family in (".self_attn.o_proj", ".mlp.down_proj"):
        row = families.get(family, {})
        subln_output = row.get("subln_output", {}) if isinstance(row, dict) else {}
        value = subln_output.get("token_rms_mean") if isinstance(subln_output, dict) else None
        if isinstance(value, (int, float)):
            output_rms.append(float(value))
    add_check(
        checks,
        "SubLN activation-variance audit has finite logit drift",
        isinstance(inserted, int)
        and inserted > 0
        and isinstance(rel, (int, float))
        and isinstance(cosine, (int, float)),
        f"inserted={inserted}, rel_rms={rel}, cosine={cosine}, path={path.relative_to(root)}",
        "SubLN audit did not quantify finite logit drift",
    )
    add_check(
        checks,
        "SubLN audit confirms projection-input normalization",
        len(output_rms) == 2 and all(0.9 <= value <= 1.1 for value in output_rms),
        f"subln_output_rms={output_rms}",
        "SubLN output RMS is not near unit scale for both audited projection families",
    )


def audit_bitdistill_root_cause(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/bitdistill_root_cause_audit_{DATE}.json"
    if not path.exists():
        add_check(checks, "BitDistill root-cause audit exists", False, str(path.relative_to(root)), "missing root-cause audit")
        return
    data = read_json(path)
    claims = data.get("claims", []) if isinstance(data.get("claims"), list) else []
    claim_status = {
        str(claim.get("claim", "")): claim.get("status")
        for claim in claims
        if isinstance(claim, dict)
    }
    required_claims = {
        "Blind ternary PTQ is not a viable universal retrofit for tested Qwen.": "supported_for_tested_setup",
        "BitDistill paper-level recovery has not been locally reproduced.": "not_proven",
        "Row-scale I2_SR is a runtime-semantics contribution, not a Q4 quality/storage win.": "supported",
        "TL2 row-scale and real Kimi/MoE product claims remain open.": "not_proven",
    }
    missing_or_wrong = {
        claim: {"expected": status, "actual": claim_status.get(claim)}
        for claim, status in required_claims.items()
        if claim_status.get(claim) != status
    }
    controlled = data.get("bitdistill", {}) if isinstance(data.get("bitdistill"), dict) else {}
    runtime = data.get("runtime", {}) if isinstance(data.get("runtime"), dict) else {}
    q4 = runtime.get("q4_vs_i2sr", {}) if isinstance(runtime.get("q4_vs_i2sr"), dict) else {}
    add_check(
        checks,
        "BitDistill root-cause audit has required claim statuses",
        not missing_or_wrong,
        f"claims={len(claims)}, mismatches={missing_or_wrong}",
        "root-cause claim ledger is missing or has unsafe statuses",
    )
    add_check(
        checks,
        "BitDistill root-cause audit marks controlled recovery incomplete",
        controlled.get("controlled_all_complete") is False
        and isinstance(controlled.get("controlled_complete"), int)
        and 0 <= controlled.get("controlled_complete") < controlled.get("controlled_expected", 0)
        and controlled.get("controlled_expected") == 3,
        f"controlled={controlled.get('controlled_complete')}/{controlled.get('controlled_expected')}, all={controlled.get('controlled_all_complete')}",
        "root-cause audit should not claim controlled BitDistill recovery before all queued rows finish and pass",
    )
    add_check(
        checks,
        "BitDistill root-cause audit carries Q4-vs-I2_SR boundary ratios",
        all(isinstance(q4.get(key), (int, float)) for key in ["file_ratio", "rss512_ratio", "prefill_speedup", "decode_speedup", "ppl_ratio"]),
        f"q4_vs_i2sr={q4}",
        "root-cause audit is missing runtime boundary ratios",
    )


def audit_bitdistill_telemetry_coverage(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/bitdistill_telemetry_coverage_{DATE}.json"
    if not path.exists():
        add_check(checks, "BitDistill telemetry coverage audit exists", False, str(path.relative_to(root)), "missing telemetry audit")
        return
    data = read_json(path)
    measured = data.get("measured", []) if isinstance(data.get("measured"), list) else []
    missing = data.get("missing", []) if isinstance(data.get("missing"), list) else []
    missing_names = {
        str(row.get("telemetry", ""))
        for row in missing
        if isinstance(row, dict)
    }
    required_missing = {"materialized training-dynamics telemetry rows"}
    measured_names = {
        str(row.get("telemetry", ""))
        for row in measured
        if isinstance(row, dict)
    }
    materialized_controlled_count = int(data.get("materialized_controlled_count") or 0)
    add_check(
        checks,
        "BitDistill telemetry coverage audit measures current loss diagnostics",
        data.get("status") in {"partial_observability", "controlled_observability"}
        and data.get("measured_count") == data.get("measured_expected")
        and data.get("measured_count", 0) >= 5,
        f"status={data.get('status')}, measured={data.get('measured_count')}/{data.get('measured_expected')}",
        "telemetry audit should pass current loss/static diagnostics",
    )
    add_check(
        checks,
        "BitDistill telemetry coverage audit handles advanced telemetry materialization",
        (
            required_missing.issubset(missing_names)
            if materialized_controlled_count == 0
            else not missing_names
        )
        and "BitLinear activation int8 saturation" in measured_names
        and "ternary flip-rate and scale trajectory" in measured_names,
        (
            f"missing={sorted(missing_names)}, "
            f"controlled={materialized_controlled_count}, "
            f"measured_activation={'BitLinear activation int8 saturation' in measured_names}, "
            f"measured_dynamics={'ternary flip-rate and scale trajectory' in measured_names}"
        ),
        "telemetry audit must either block missing controlled traces or record that controlled telemetry has materialized",
    )


def audit_bitdistill_training_dynamics(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/bitdistill_training_dynamics_{DATE}.json"
    if not path.exists():
        add_check(checks, "BitDistill training-dynamics audit exists", False, str(path.relative_to(root)), "missing training-dynamics audit")
        return
    data = read_json(path)
    status = data.get("status")
    add_check(
        checks,
        "BitDistill training-dynamics audit parses materialized telemetry",
        status in {"smoke_only", "controlled_materialized"}
        and data.get("trace_count", 0) > 0
        and data.get("smoke_materialized_count", 0) > 0,
        (
            f"status={status}, traces={data.get('trace_count')}, "
            f"smoke={data.get('smoke_materialized_count')}, "
            f"controlled={data.get('materialized_controlled_count')}"
        ),
        "training-dynamics audit must parse at least the smoke telemetry hooks",
    )
    add_check(
        checks,
        "BitDistill training-dynamics audit blocks controlled claims until real traces exist",
        status == "controlled_materialized" or data.get("materialized_controlled_count") == 0,
        f"status={status}, controlled={data.get('materialized_controlled_count')}",
        "training-dynamics audit must distinguish smoke parser validation from controlled-run evidence",
    )


def audit_bitdistill_loss_contract(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/bitdistill_loss_contract_{DATE}.json"
    if not path.exists():
        add_check(checks, "BitDistill loss-contract audit exists", False, str(path.relative_to(root)), "missing loss-contract audit")
        return
    data = read_json(path)
    static_checks = data.get("checks", []) if isinstance(data.get("checks"), list) else []
    live = data.get("live", {}) if isinstance(data.get("live"), dict) else {}
    max_ratio = live.get("max_observed_weighted_attention_to_ce")
    add_check(
        checks,
        "BitDistill loss-contract static checks pass",
        data.get("passed") is True and len(static_checks) >= 6,
        f"passed={data.get('passed')}, checks={len(static_checks)}, status={data.get('status')}",
        "loss-contract source checks failed",
    )
    add_check(
        checks,
        "BitDistill loss-contract records paper-gamma dominance risk",
        data.get("status") == "loss_normalization_risk"
        and isinstance(max_ratio, (int, float))
        and max_ratio >= 100.0,
        f"status={data.get('status')}, max_attn_ce={max_ratio}",
        "loss-contract audit should flag current paper-gamma loss-balance risk",
    )


def audit_original_benchmark_objective(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/original_benchmark_objective_audit_{DATE}.json"
    if not path.exists():
        add_check(
            checks,
            "Original benchmark objective audit exists",
            False,
            str(path.relative_to(root)),
            "missing original six-item objective audit",
        )
        return
    data = read_json(path)
    rows = data.get("rows", []) if isinstance(data.get("rows"), list) else []
    partial_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("status") != "complete"
    ]
    partial_text = " ".join(str(row.get("item", "")) + " " + str(row.get("gap", "")) for row in partial_rows)
    add_check(
        checks,
        "Original benchmark objective audit maps all six requested deliverables",
        data.get("check_count") == 6 and len(rows) == 6,
        f"completion={data.get('complete_count')}/{data.get('check_count')}, status={data.get('completion_status')}",
        "audit should map each explicit item in the original six-item benchmark objective",
    )
    add_check(
        checks,
        "Original benchmark objective audit keeps TL2 row-scale blocker explicit",
        data.get("complete_count") == 5
        and len(partial_rows) == 1
        and "TL2" in partial_text
        and "row-scale" in partial_text,
        f"partial_rows={len(partial_rows)}, partial={partial_text[:240]}",
        "the only incomplete original objective item should be quality-preserving TL2 row-scale support",
    )


def audit_tl2_negative_result(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/tl2_negative_result_{DATE}.json"
    if not path.exists():
        add_check(checks, "TL2 negative-result audit exists", False, str(path.relative_to(root)), "missing TL2 negative-result audit")
        return
    data = read_json(path)
    add_check(
        checks,
        "TL2 negative-result audit has CPU probe evidence",
        data.get("tl2_cpu_executed") is True
        and data.get("tl2_probe_has_finite_quality") is False,
        f"cpu_executed={data.get('tl2_cpu_executed')}, finite_quality={data.get('tl2_probe_has_finite_quality')}",
        "TL2 negative result should include actual CPU probe evidence and quality failure status",
    )
    add_check(
        checks,
        "TL2 negative-result audit proves row-scale mismatch",
        data.get("negative_result_supported") is True
        and isinstance(data.get("qwen15b_row_scale_current_tl2_error"), (int, float))
        and data.get("qwen15b_row_scale_current_tl2_error") > 1.0
        and isinstance(data.get("qwen15b_row_scale_exact_fp16_error"), (int, float))
        and data.get("qwen15b_row_scale_exact_fp16_error") < 0.01
        and int(data.get("runtime_failed_checks") or 0) > 0,
        (
            f"supported={data.get('negative_result_supported')}, "
            f"current={data.get('qwen15b_row_scale_current_tl2_error')}, "
            f"row_fp16={data.get('qwen15b_row_scale_exact_fp16_error')}, "
            f"failed_checks={data.get('runtime_failed_checks')}"
        ),
        "TL2 negative result should connect CPU probes, row-scale math, and runtime blockers",
    )


def audit_tl2_implementation_plan(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/tl2_row_scale_implementation_plan_{DATE}.json"
    if not path.exists():
        add_check(
            checks,
            "TL2 row-scale implementation plan exists",
            False,
            str(path.relative_to(root)),
            "missing TL2 row-scale implementation plan",
        )
        return
    data = read_json(path)
    steps = data.get("implementation_steps", []) if isinstance(data.get("implementation_steps"), list) else []
    blockers = data.get("blockers", []) if isinstance(data.get("blockers"), list) else []
    add_check(
        checks,
        "TL2 row-scale implementation plan keeps runtime blocked",
        data.get("current_ready") is False
        and int(data.get("failed_check_count") or 0) >= 1
        and len(blockers) >= 1,
        (
            f"ready={data.get('current_ready')}, "
            f"failed_checks={data.get('failed_check_count')}, blockers={len(blockers)}"
        ),
        "TL2 implementation plan should preserve the blocked status until the row-scale runtime contract passes",
    )
    add_check(
        checks,
        "TL2 row-scale implementation plan has source-mapped patch sequence",
        len(steps) >= 6
        and all(isinstance(step.get("files"), list) and step.get("files") for step in steps if isinstance(step, dict))
        and all(step.get("exit_gate") for step in steps if isinstance(step, dict)),
        f"step_count={len(steps)}",
        "TL2 implementation plan should list file ownership and exit gates for every required patch phase",
    )
    add_check(
        checks,
        "TL2 row-scale implementation plan carries scale-error proof",
        isinstance(data.get("current_one_scale_error"), (int, float))
        and data.get("current_one_scale_error") > 1.0
        and isinstance(data.get("exact_row_fp16_error"), (int, float))
        and data.get("exact_row_fp16_error") < 0.01
        and isinstance(data.get("row_scale_storage_mib"), (int, float)),
        (
            f"one_scale={data.get('current_one_scale_error')}, "
            f"row_fp16={data.get('exact_row_fp16_error')}, "
            f"scale_mib={data.get('row_scale_storage_mib')}"
        ),
        "TL2 implementation plan should cite the row-scale mismatch math, not just a prose TODO",
    )


def audit_tl2_group_scale_viability(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/tl2_group_scale_viability_{DATE}.json"
    if not path.exists():
        add_check(
            checks,
            "TL2 group-scale viability audit exists",
            False,
            str(path.relative_to(root)),
            "missing TL2 group-scale viability audit",
        )
        return
    data = read_json(path)
    best_group = data.get("best_group_fp16", {}) if isinstance(data.get("best_group_fp16"), dict) else {}
    exact_row = data.get("exact_row_fp16", {}) if isinstance(data.get("exact_row_fp16"), dict) else {}
    group_rows = data.get("group_fp16_rows", []) if isinstance(data.get("group_fp16_rows"), list) else []
    best_error = best_group.get("expected_relative_output_rms_error")
    exact_error = exact_row.get("expected_relative_output_rms_error")
    ratio = data.get("best_group_to_exact_row_error_ratio")
    add_check(
        checks,
        "TL2 group-scale viability audit has group sweep",
        len(group_rows) >= 10
        and isinstance(best_error, (int, float))
        and isinstance(exact_error, (int, float)),
        f"groups={len(group_rows)}, best_group={best_error}, exact_row={exact_error}",
        "group-scale viability should quantify the row-group error frontier",
    )
    add_check(
        checks,
        "TL2 group-scale viability keeps strict row-scale blocker",
        data.get("strict_group_scale_viable") is False
        and data.get("exact_row_required_for_strict_fidelity") is True
        and isinstance(best_error, (int, float))
        and best_error > 0.05
        and isinstance(exact_error, (int, float))
        and exact_error < 0.01
        and isinstance(ratio, (int, float))
        and ratio > 100,
        (
            f"strict_viable={data.get('strict_group_scale_viable')}, "
            f"best_group={best_error}, exact_row={exact_error}, ratio={ratio}"
        ),
        "group-scale TL2 should not be allowed to close the row-scale objective blocker unless it meets strict fidelity",
    )


def audit_ternary_flip_dynamics(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/ternary_flip_dynamics_{DATE}.json"
    if not path.exists():
        add_check(checks, "Ternary flip-dynamics audit exists", False, str(path.relative_to(root)), "missing flip-dynamics audit")
        return
    data = read_json(path)
    pairs = data.get("pairs", []) if isinstance(data.get("pairs"), list) else []
    max_flip = data.get("max_flip_rate")
    min_flip = data.get("min_flip_rate")
    add_check(
        checks,
        "Ternary flip-dynamics audit has nonzero saved-snapshot flips",
        data.get("status") == "pass"
        and len(pairs) >= 2
        and isinstance(max_flip, (int, float))
        and isinstance(min_flip, (int, float))
        and max_flip > 0.0
        and min_flip > 0.0,
        f"status={data.get('status')}, pairs={len(pairs)}, min_flip={min_flip}, max_flip={max_flip}",
        "saved Stage-2 snapshots did not show measurable ternary code movement",
    )


def audit_ternary_threshold_dynamics(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/ternary_threshold_dynamics_{DATE}.json"
    if not path.exists():
        add_check(
            checks,
            "Ternary threshold-dynamics audit exists",
            False,
            str(path.relative_to(root)),
            "missing threshold-dynamics audit",
        )
        return
    data = read_json(path)
    snapshots = data.get("snapshots", []) if isinstance(data.get("snapshots"), list) else []
    delta = data.get("threshold_band_delta_first_to_last")
    final_band = None
    if snapshots:
        final = snapshots[-1]
        total = final.get("total", {}) if isinstance(final.get("total"), dict) else {}
        bands = total.get("threshold_band_fractions", {}) if isinstance(total.get("threshold_band_fractions"), dict) else {}
        final_band = bands.get(str(data.get("primary_band")))
    add_check(
        checks,
        "Ternary threshold-dynamics audit has measured boundary movement",
        data.get("status") == "measured_increase"
        and len(snapshots) >= 2
        and data.get("monotonic_non_decreasing") is True
        and isinstance(delta, (int, float))
        and delta > 0.0
        and isinstance(final_band, (int, float)),
        (
            f"status={data.get('status')}, snapshots={len(snapshots)}, "
            f"delta={delta}, final_band={final_band}, monotonic={data.get('monotonic_non_decreasing')}"
        ),
        "saved Stage-2 snapshots did not show increasing mass near the ternary transition boundary",
    )


def audit_second_order_ternary_init(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/second_order_ternary_init_{DATE}.json"
    if not path.exists():
        add_check(
            checks,
            "Second-order ternary init audit exists",
            False,
            str(path.relative_to(root)),
            "missing second-order ternary init audit",
        )
        return
    data = read_json(path)
    profiles = data.get("profiles", {}) if isinstance(data.get("profiles"), dict) else {}
    deltas = [
        profile.get("row_diag_hessian_ls_minus_row_absmean", {})
        for profile in profiles.values()
        if isinstance(profile, dict)
    ]
    add_check(
        checks,
        "Second-order ternary init audit improves synthetic reconstruction",
        data.get("status")
        in {
            "synthetic_promising_ls_integrated_diag_calibration_pending",
            "synthetic_promising_diag_calibration_integrated_quality_pending",
            "synthetic_promising_task_quality_rejected",
            "synthetic_promising_task_quality_supported",
        }
        and len(deltas) >= 2
        and all(delta.get("mean", 0.0) < -0.02 for delta in deltas)
        and all(delta.get("wins") == delta.get("trials") for delta in deltas),
        f"status={data.get('status')}, deltas={[delta.get('mean') for delta in deltas]}",
        "diagonal-Hessian LS should beat row absmean in the synthetic reconstruction audit before promotion",
    )
    task_quality_audits = data.get("task_quality_audits", {})
    diag_audit = task_quality_audits.get("diag_ls", {}) if isinstance(task_quality_audits, dict) else {}
    if diag_audit.get("status") == "complete":
        add_check(
            checks,
            "Second-order ternary init audit ingests diag-LS task-quality result",
            data.get("status") in {"synthetic_promising_task_quality_rejected", "synthetic_promising_task_quality_supported"}
            and diag_audit.get("candidate_improves_absmean_baseline") is False,
            (
                f"status={data.get('status')}, diag_status={diag_audit.get('status')}, "
                f"diag_delta={diag_audit.get('delta_vs_absmean_baseline')}"
            ),
            "completed diag-LS quality audit should update the second-order initializer status instead of remaining pending",
        )
    add_check(
        checks,
        "Second-order ternary init audit blocks quality overclaim",
        data.get("quality_proven") is False
        and data.get("training_integrated") is True
        and isinstance(data.get("diag_hessian_training_integrated"), bool),
        (
            f"quality_proven={data.get('quality_proven')}, "
            f"ls_integrated={data.get('training_integrated')}, "
            f"diag_integrated={data.get('diag_hessian_training_integrated')}"
        ),
        "audit should distinguish synthetic reconstruction and initializer integration from real model quality",
    )


def audit_bitnet_sft_ls_init_submission(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/bitnet_sft_ls_init_submission_{DATE}.json"
    if not path.exists():
        add_check(
            checks,
            "BitNet-SFT LS-init submission exists",
            False,
            str(path.relative_to(root)),
            "missing BitNet-SFT LS-init submission record",
        )
        return
    data = read_json(path)
    add_check(
        checks,
        "BitNet-SFT LS-init submission changes only initializer",
        data.get("status") == "submitted"
        and data.get("changed_axis") == "ternary_init_mode_only"
        and data.get("ternary_init_mode") == "ls"
        and data.get("method") == "bitnet_sft"
        and data.get("task") == "mnli",
        (
            f"status={data.get('status')}, changed_axis={data.get('changed_axis')}, "
            f"init={data.get('ternary_init_mode')}, method={data.get('method')}, task={data.get('task')}"
        ),
        "LS-init submission should be a controlled MNLI BitNet-SFT initializer-only comparison",
    )
    add_check(
        checks,
        "BitNet-SFT LS-init submission has matched absmean baseline",
        bool(data.get("baseline_output_dir"))
        and (root / str(data.get("baseline_output_dir")) / "metrics.json").exists()
        and int(data.get("steps") or 0) == 10000
        and float(data.get("learning_rate") or 0.0) == 2e-5,
        (
            f"baseline={data.get('baseline_output_dir')}, "
            f"steps={data.get('steps')}, lr={data.get('learning_rate')}"
        ),
        "LS-init submission should compare against the existing matched 10000-step absmean baseline",
    )


def audit_bitnet_sft_ls_init_result(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/bitnet_sft_ls_init_audit_{DATE}.json"
    if not path.exists():
        add_check(
            checks,
            "BitNet-SFT LS-init result audit exists",
            False,
            str(path.relative_to(root)),
            "missing BitNet-SFT LS-init result audit",
        )
        return
    data = read_json(path)
    status = data.get("status")
    add_check(
        checks,
        "BitNet-SFT LS-init result audit is pending or complete",
        status in {
            "pending",
            "complete",
            "complete_incomplete_eval",
            "complete_init_contract_failed",
            "complete_prediction_contract_failed",
        }
        and data.get("baseline_accuracy") is not None
        and data.get("quality_proven") in {False, True},
        (
            f"status={status}, baseline={data.get('baseline_accuracy')}, "
            f"candidate={data.get('candidate_accuracy')}, quality_proven={data.get('quality_proven')}"
        ),
        "LS-init result audit should either explicitly wait for Slurm output or report a complete comparison",
    )
    if status == "complete":
        paired = data.get("paired", {}) if isinstance(data.get("paired"), dict) else {}
        add_check(
            checks,
            "BitNet-SFT LS-init complete result has paired MNLI statistics",
            paired.get("status") == "pass"
            and paired.get("matched") == 9815
            and isinstance(paired.get("paired_ci95"), list)
            and isinstance(data.get("delta_vs_absmean_baseline"), (int, float)),
            (
                f"matched={paired.get('matched')}, ci={paired.get('paired_ci95')}, "
                f"delta={data.get('delta_vs_absmean_baseline')}"
            ),
            "complete LS-init audit should include full paired MNLI statistics",
        )


def audit_bitnet_sft_diag_ls_init_submission(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/bitnet_sft_diag_ls_init_submission_{DATE}.json"
    if not path.exists():
        add_check(
            checks,
            "BitNet-SFT diag-LS init submission exists",
            False,
            str(path.relative_to(root)),
            "missing BitNet-SFT diag-LS init submission record",
        )
        return
    data = read_json(path)
    add_check(
        checks,
        "BitNet-SFT diag-LS init submission changes only initializer",
        data.get("status") == "submitted"
        and data.get("changed_axis") == "ternary_init_mode_only"
        and data.get("ternary_init_mode") == "diag_ls"
        and data.get("ternary_init_calibration_batches") == 8
        and data.get("method") == "bitnet_sft"
        and data.get("task") == "mnli",
        (
            f"status={data.get('status')}, changed_axis={data.get('changed_axis')}, "
            f"init={data.get('ternary_init_mode')}, cal={data.get('ternary_init_calibration_batches')}, "
            f"method={data.get('method')}, task={data.get('task')}"
        ),
        "diag-LS submission should be a controlled MNLI BitNet-SFT initializer-only comparison",
    )
    add_check(
        checks,
        "BitNet-SFT diag-LS init submission has matched absmean baseline",
        bool(data.get("baseline_output_dir"))
        and (root / str(data.get("baseline_output_dir")) / "metrics.json").exists()
        and int(data.get("steps") or 0) == 10000
        and float(data.get("learning_rate") or 0.0) == 2e-5,
        (
            f"baseline={data.get('baseline_output_dir')}, "
            f"steps={data.get('steps')}, lr={data.get('learning_rate')}"
        ),
        "diag-LS submission should compare against the existing matched 10000-step absmean baseline",
    )


def audit_bitnet_sft_diag_ls_init_result(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/bitnet_sft_diag_ls_init_audit_{DATE}.json"
    if not path.exists():
        add_check(
            checks,
            "BitNet-SFT diag-LS init result audit exists",
            False,
            str(path.relative_to(root)),
            "missing BitNet-SFT diag-LS init result audit",
        )
        return
    data = read_json(path)
    status = data.get("status")
    add_check(
        checks,
        "BitNet-SFT diag-LS init result audit is pending or complete",
        status in {
            "pending",
            "complete",
            "complete_incomplete_eval",
            "complete_init_contract_failed",
            "complete_prediction_contract_failed",
        }
        and data.get("baseline_accuracy") is not None
        and data.get("quality_proven") in {False, True},
        (
            f"status={status}, baseline={data.get('baseline_accuracy')}, "
            f"candidate={data.get('candidate_accuracy')}, quality_proven={data.get('quality_proven')}"
        ),
        "diag-LS result audit should either explicitly wait for Slurm output or report a complete comparison",
    )
    if status == "complete":
        paired = data.get("paired", {}) if isinstance(data.get("paired"), dict) else {}
        add_check(
            checks,
            "BitNet-SFT diag-LS complete result has paired MNLI statistics",
            paired.get("status") == "pass"
            and paired.get("matched") == 9815
            and isinstance(paired.get("paired_ci95"), list)
            and isinstance(data.get("delta_vs_absmean_baseline"), (int, float)),
            (
                f"matched={paired.get('matched')}, ci={paired.get('paired_ci95')}, "
                f"delta={data.get('delta_vs_absmean_baseline')}"
            ),
            "complete diag-LS audit should include full paired MNLI statistics",
        )


def audit_seqcls_runtime_gap(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/seqcls_runtime_gap_{DATE}.json"
    if not path.exists():
        add_check(checks, "Sequence-classification runtime gap audit exists", False, str(path.relative_to(root)), "missing seqcls runtime gap audit")
        return
    data = read_json(path)
    seqcls = data.get("sequence_classification", {}) if isinstance(data.get("sequence_classification"), dict) else {}
    causal = data.get("causal_runtime", {}) if isinstance(data.get("causal_runtime"), dict) else {}
    export = data.get("causal_export_summary", {}) if isinstance(data.get("causal_export_summary"), dict) else {}
    smoke = data.get("seqcls_sidecar_smoke", {}) if isinstance(data.get("seqcls_sidecar_smoke"), dict) else {}
    sidecar_cpu = (
        data.get("seqcls_sidecar_cpu_benchmark", {})
        if isinstance(data.get("seqcls_sidecar_cpu_benchmark"), dict)
        else {}
    )
    hidden_contract = (
        data.get("seqcls_hidden_contract", {})
        if isinstance(data.get("seqcls_hidden_contract"), dict)
        else {}
    )
    native_cpu = (
        data.get("seqcls_native_cpu_benchmark", {})
        if isinstance(data.get("seqcls_native_cpu_benchmark"), dict)
        else {}
    )
    gap_status = data.get("status")
    accepted_gap_statuses = {
        "blocked_by_classifier_runtime",
        "sidecar_prototype_available_native_runtime_blocked",
        "sidecar_qwen_contract_available_native_head_blocked",
        "native_classifier_smoke_available_full_validation_blocked",
        "native_classifier_sample_quality_mismatch_full_validation_blocked",
        "native_classifier_sample_available_full_validation_blocked",
        "native_classifier_full_validation_batching_blocked",
    }
    full_validation_status = gap_status == "native_classifier_full_validation_batching_blocked"
    add_check(
        checks,
        "Sequence-classification runtime gap is narrowed but not closed",
        gap_status in accepted_gap_statuses
        and data.get("same_artifact_quality_cpu_ready") is False
        and seqcls.get("sequence_classification", 0) > 0
        and seqcls.get("causal_export_compatible") == 0
        and causal.get("causal_export_compatible", 0) > 0
        and export.get("exported", 0) > 0
        and (
            not full_validation_status
            or (
                native_cpu.get("full_validation_complete") is True
                and native_cpu.get("batching_parity_ready") is False
                and native_cpu.get("ready_to_productize") is False
                and native_cpu.get("examples") == 9815
            )
        ),
        (
            f"status={gap_status}, seqcls={seqcls.get('sequence_classification')}, "
            f"seqcls_exportable={seqcls.get('causal_export_compatible')}, "
            f"causal_exportable={causal.get('causal_export_compatible')}, exports={export.get('exported')}, "
            f"native_full={native_cpu.get('full_validation_complete')}, "
            f"batching={native_cpu.get('batching_parity_ready')}, ready={native_cpu.get('ready_to_productize')}"
        ),
        "sequence-classification runtime status must remain blocked unless full validation and batching parity are both product-ready",
    )
    add_check(
        checks,
        "Sequence-classification I2_SR sidecar smoke passes",
        smoke.get("passed") is True
        and smoke.get("runtime_returncode") == 0
        and smoke.get("finite_logits") is True
        and smoke.get("head_shape") == [3, 896],
        (
            f"status={smoke.get('status')}, returncode={smoke.get('runtime_returncode')}, "
            f"head_shape={smoke.get('head_shape')}, finite_logits={smoke.get('finite_logits')}"
        ),
        "the sidecar prototype did not load the packed backbone and produce finite classifier logits",
    )
    add_check(
        checks,
        "Sequence-classification sidecar CPU quality mismatch is recorded",
        sidecar_cpu.get("status") == "quality_mismatch"
        and sidecar_cpu.get("examples", 0) >= 64
        and isinstance(sidecar_cpu.get("agreement_with_saved_pytorch_predictions"), (int, float))
        and sidecar_cpu.get("agreement_with_saved_pytorch_predictions") < 0.95,
        (
            f"status={sidecar_cpu.get('status')}, examples={sidecar_cpu.get('examples')}, "
            f"agreement={sidecar_cpu.get('agreement_with_saved_pytorch_predictions')}, "
            f"accuracy={sidecar_cpu.get('accuracy')}"
        ),
        "sampled sidecar CPU benchmark should make the runtime-contract mismatch explicit",
    )
    add_check(
        checks,
        "Sequence-classification hidden contract is near but not exact",
        hidden_contract.get("status") == "hidden_contract_mismatch"
        and hidden_contract.get("token_id_match") is True
        and isinstance(hidden_contract.get("hidden_relative_rms"), (int, float))
        and 0.05 < hidden_contract.get("hidden_relative_rms") < 0.2
        and isinstance(hidden_contract.get("hidden_cosine"), (int, float))
        and hidden_contract.get("hidden_cosine") > 0.99,
        (
            f"status={hidden_contract.get('status')}, token_match={hidden_contract.get('token_id_match')}, "
            f"hidden_rel_rms={hidden_contract.get('hidden_relative_rms')}, "
            f"hidden_cosine={hidden_contract.get('hidden_cosine')}, "
            f"logit_rel_rms={hidden_contract.get('logit_relative_rms')}"
        ),
        "the sidecar hidden-contract audit did not record the expected near-pass-but-not-bit-exact state",
    )
    arch_path = root / f"benchmark_results/seqcls_i2sr_arch_contract_{DATE}.json"
    if not arch_path.exists():
        add_check(
            checks,
            "Sequence-classification architecture contract mismatch is identified",
            False,
            str(arch_path.relative_to(root)),
            "missing seqcls architecture-contract audit",
        )
        return
    arch = read_json(arch_path)
    arch_checks = arch.get("checks", {}) if isinstance(arch.get("checks"), dict) else {}
    runtime = arch.get("runtime_source", {}) if isinstance(arch.get("runtime_source"), dict) else {}
    qwen = arch.get("bitnet_qwen_contract", {}) if isinstance(arch.get("bitnet_qwen_contract"), dict) else {}
    biases = arch.get("checkpoint_biases", {}) if isinstance(arch.get("checkpoint_biases"), dict) else {}
    add_check(
        checks,
        "Sequence-classification architecture contract is identified and repaired",
        arch.get("status") == "bitnet_qwen_contract_available"
        and arch_checks.get("activation_mismatch") is True
        and arch_checks.get("plain_bitnet_has_silu_graph") is True
        and arch_checks.get("plain_bitnet_bias_contract_gap") is True
        and arch_checks.get("bitnet_qwen_contract_available") is True
        and runtime.get("bitnet25_ffn_activation") == "relu_sqr"
        and qwen.get("available") is True
        and qwen.get("ffn_activation") == "silu"
        and qwen.get("loader_has_qkv_bias") is True
        and biases.get("projection_bias_count") == 72,
        (
            f"status={arch.get('status')}, hidden_act={arch.get('checkpoint_config', {}).get('hidden_act')}, "
            f"bitnet25_activation={runtime.get('bitnet25_ffn_activation')}, "
            f"bitnet_qwen={qwen}, projection_biases={biases.get('projection_bias_count')}, checks={arch_checks}"
        ),
        "the seqcls runtime gap lacks the expected bitnet-qwen architecture-contract repair",
    )


def audit_seqcls_runtime_implementation_plan(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/seqcls_runtime_implementation_plan_{DATE}.json"
    if not path.exists():
        add_check(
            checks,
            "Sequence-classification runtime implementation plan exists",
            False,
            str(path.relative_to(root)),
            "missing seqcls runtime implementation plan",
        )
        return
    data = read_json(path)
    steps = data.get("implementation_steps", []) if isinstance(data.get("implementation_steps"), list) else []
    add_check(
        checks,
        "Sequence-classification runtime implementation plan keeps product blocked",
        data.get("same_artifact_quality_cpu_ready") is False
        and data.get("ready_to_productize") is False
        and data.get("seqcls_causal_export_compatible") == 0,
        (
            f"same_artifact={data.get('same_artifact_quality_cpu_ready')}, "
            f"ready={data.get('ready_to_productize')}, "
            f"seqcls_exportable={data.get('seqcls_causal_export_compatible')}"
        ),
        "implementation plan should not promote sequence-classification product status before native runtime evidence",
    )
    add_check(
        checks,
        "Sequence-classification runtime implementation plan has source-owned steps",
        len(steps) >= 5
        and all(isinstance(step.get("files"), list) and step.get("files") for step in steps if isinstance(step, dict))
        and all(step.get("exit_gate") for step in steps if isinstance(step, dict)),
        f"step_count={len(steps)}",
        "seqcls runtime plan should list source ownership and exit gates",
    )


def audit_seqcls_native_i2sr_smoke(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/seqcls_native_i2sr_smoke_{DATE}.json"
    if not path.exists():
        add_check(
            checks,
            "Sequence-classification native I2_SR smoke exists",
            False,
            str(path.relative_to(root)),
            "missing native seqcls I2_SR smoke audit",
        )
        return
    data = read_json(path)
    runtime = data.get("runtime", {}) if isinstance(data.get("runtime"), dict) else {}
    add_check(
        checks,
        "Sequence-classification native I2_SR smoke passes",
        data.get("status") == "pass"
        and data.get("single_artifact") is True
        and data.get("returncode") == 0
        and data.get("logit_count") == 3
        and data.get("prediction") == data.get("sidecar_prediction")
        and isinstance(data.get("relative_rms_logit_delta"), (int, float))
        and data.get("relative_rms_logit_delta") < 1e-5,
        (
            f"status={data.get('status')}, single_artifact={data.get('single_artifact')}, "
            f"returncode={data.get('returncode')}, logits={data.get('logit_count')}, "
            f"prediction={data.get('prediction')}, sidecar_prediction={data.get('sidecar_prediction')}, "
            f"rel_rms={data.get('relative_rms_logit_delta')}"
        ),
        "native GGUF classifier-head smoke did not match the sidecar reference logits",
    )
    add_check(
        checks,
        "Sequence-classification native I2_SR smoke keeps product blocked",
        data.get("full_validation_complete") is False
        and data.get("ready_to_productize") is False
        and isinstance(runtime.get("prompt_eval_tokens_per_second"), (int, float))
        and runtime.get("prompt_eval_tokens_per_second") > 0,
        (
            f"full_validation={data.get('full_validation_complete')}, "
            f"ready={data.get('ready_to_productize')}, "
            f"prompt_tok_s={runtime.get('prompt_eval_tokens_per_second')}"
        ),
        "single-prompt smoke must not be promoted to a full quality/runtime product claim",
    )


def audit_seqcls_native_i2sr_cpu_sample(root: Path, checks: list[dict[str, Any]]) -> None:
    full_path = root / f"benchmark_results/seqcls_native_i2sr_cpu_mnli_full_token_ids_{DATE}.json"
    sample_path = root / f"benchmark_results/seqcls_native_i2sr_cpu_mnli_64_token_ids_{DATE}.json"
    path = full_path if full_path.exists() else sample_path
    expected_examples = 9815 if path == full_path else 64
    if not path.exists():
        add_check(
            checks,
            "Sequence-classification native I2_SR CPU result exists",
            False,
            str(path.relative_to(root)),
            "missing native seqcls I2_SR CPU sample",
        )
        return
    data = read_json(path)
    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
    runtime = data.get("runtime", {}) if isinstance(data.get("runtime"), dict) else {}
    add_check(
        checks,
        "Sequence-classification native I2_SR CPU token-ID result records agreement",
        data.get("status") in {"pass", "sample_only", "quality_mismatch"}
        and data.get("prompt_input") == "token_ids"
        and summary.get("examples") == expected_examples
        and isinstance(summary.get("agreement_with_saved_pytorch_predictions"), (int, float))
        and summary.get("agreement_with_saved_pytorch_predictions") >= 0.95
        and isinstance(summary.get("accuracy"), (int, float)),
        (
            f"status={data.get('status')}, prompt_input={data.get('prompt_input')}, examples={summary.get('examples')}, "
            f"accuracy={summary.get('accuracy')}, agreement={summary.get('agreement_with_saved_pytorch_predictions')}"
        ),
        "native sampled classifier result should use the repaired token-ID path and report agreement",
    )
    add_check(
        checks,
        "Sequence-classification native I2_SR CPU result keeps product blocked until batching parity passes",
        data.get("ready_to_productize") is False
        and data.get("batching_parity_ready") is not True
        and isinstance(runtime.get("examples_per_second"), (int, float))
        and runtime.get("examples_per_second") > 0
        and isinstance(runtime.get("child_peak_rss_mib"), (int, float))
        and runtime.get("child_peak_rss_mib") > 0,
        (
            f"full_validation={data.get('full_validation_complete')}, ready={data.get('ready_to_productize')}, "
            f"batching={data.get('batching_parity_ready')}, examples_per_second={runtime.get('examples_per_second')}, "
            f"rss_mib={runtime.get('child_peak_rss_mib')}"
        ),
        "native classifier result should include runtime/RSS evidence and remain blocked until batching parity passes",
    )
    mismatch_path = root / f"benchmark_results/seqcls_native_mismatch_audit_{DATE}.json"
    mismatch = read_json(mismatch_path)
    mismatch_summary = mismatch.get("summary", {}) if isinstance(mismatch.get("summary"), dict) else {}
    add_check(
        checks,
        "Sequence-classification native mismatch audit isolates runtime drift",
        mismatch.get("status") == "runtime_hidden_drift"
        and mismatch_summary.get("all_token_ids_match") is True
        and mismatch_summary.get("text_roundtrip_all_token_ids_match") is False
        and mismatch_summary.get("native_vs_sidecar_logits_match") is True,
        (
            f"status={mismatch.get('status')}, token_ids={mismatch_summary.get('all_token_ids_match')}, "
            f"text_roundtrip={mismatch_summary.get('text_roundtrip_all_token_ids_match')}, "
            f"native_sidecar={mismatch_summary.get('native_vs_sidecar_logits_match')}"
        ),
        "mismatch audit should prove token-ID repair and isolate residual packed-runtime drift",
    )
    batching_path = root / f"benchmark_results/seqcls_native_batching_audit_{DATE}.json"
    batching = read_json(batching_path)
    batching_summary = batching.get("summary", {}) if isinstance(batching.get("summary"), dict) else {}
    add_check(
        checks,
        "Sequence-classification native batching audit blocks batched throughput",
        batching.get("status") == "batching_parity_mismatch"
        and batching_summary.get("all_predictions_invariant") is False
        and isinstance(batching_summary.get("changed_case_count"), int)
        and batching_summary.get("changed_case_count") > 0
        and batching.get("ready_for_batched_product_benchmark") is False,
        (
            f"status={batching.get('status')}, invariant={batching_summary.get('all_predictions_invariant')}, "
            f"changed={batching_summary.get('changed_case_count')}, ready={batching.get('ready_for_batched_product_benchmark')}"
        ),
        "batched native seqcls timing should remain blocked until logits are batch-invariant",
    )
    add_check(
        checks,
        "Sequence-classification native batching audit diagnoses non-swap drift",
        batching.get("status") == "batching_parity_mismatch"
        and batching_summary.get("mapping_diagnosis") == "position_dependent_drift_not_row_swap"
        and batching_summary.get("drifted_rows_nearest_self") is True
        and isinstance(batching_summary.get("drifted_case_count"), int)
        and batching_summary.get("drifted_case_count") > 0,
        (
            f"diagnosis={batching_summary.get('mapping_diagnosis')}, "
            f"nearest_self={batching_summary.get('drifted_rows_nearest_self')}, "
            f"drifted={batching_summary.get('drifted_case_count')}"
        ),
        "batching audit should show whether the failure is a row swap or true batched-runtime drift",
    )
    duplicate_path = root / f"benchmark_results/seqcls_native_duplicate_batching_audit_{DATE}.json"
    duplicate = read_json(duplicate_path)
    duplicate_summary = duplicate.get("summary", {}) if isinstance(duplicate.get("summary"), dict) else {}
    duplicate_controls = {
        control.get("label"): control
        for control in duplicate.get("control_models", [])
        if isinstance(control, dict)
    }
    fp_control = duplicate_controls.get("fp_qwen05b_f16", {})
    fp_control_summary = fp_control.get("summary", {}) if isinstance(fp_control, dict) else {}
    bitnet_backbone_control = duplicate_controls.get("bitnet_qwen_i2sr_backbone", {})
    bitnet_backbone_summary = (
        bitnet_backbone_control.get("summary", {}) if isinstance(bitnet_backbone_control, dict) else {}
    )
    add_check(
        checks,
        "Sequence-classification duplicate-prompt batching audit rules out prompt formatting",
        duplicate.get("status") == "duplicate_batching_parity_mismatch"
        and duplicate_summary.get("same_prompt_repeated") is True
        and duplicate_summary.get("formatting_and_tokenization_ruled_out") is True
        and duplicate_summary.get("all_logits_invariant") is False
        and isinstance(duplicate_summary.get("max_relative_rms_vs_alone"), (int, float))
        and duplicate_summary.get("max_relative_rms_vs_alone") > 0.05
        and duplicate.get("ready_for_batched_product_benchmark") is False,
        (
            f"status={duplicate.get('status')}, same_prompt={duplicate_summary.get('same_prompt_repeated')}, "
            f"logits_invariant={duplicate_summary.get('all_logits_invariant')}, "
            f"predictions_invariant={duplicate_summary.get('all_predictions_invariant')}, "
            f"changed_predictions={duplicate_summary.get('changed_prediction_count')}, "
            f"max_rel={duplicate_summary.get('max_relative_rms_vs_alone')}, "
            f"ready={duplicate.get('ready_for_batched_product_benchmark')}"
        ),
        "duplicate token-ID prompts should prove whether batching drift is independent of prompt formatting/tokenization",
    )
    add_check(
        checks,
        "Sequence-classification duplicate-prompt controls isolate BitNet/I2_SR drift",
        isinstance(fp_control_summary.get("max_relative_rms_vs_alone"), (int, float))
        and fp_control_summary.get("max_relative_rms_vs_alone") < 0.001
        and isinstance(bitnet_backbone_summary.get("max_relative_rms_vs_alone"), (int, float))
        and bitnet_backbone_summary.get("max_relative_rms_vs_alone") > 0.05,
        (
            f"fp_max_rel={fp_control_summary.get('max_relative_rms_vs_alone')}, "
            f"fp_logits_invariant={fp_control_summary.get('all_logits_invariant')}, "
            f"bitnet_backbone_max_rel={bitnet_backbone_summary.get('max_relative_rms_vs_alone')}, "
            f"bitnet_backbone_logits_invariant={bitnet_backbone_summary.get('all_logits_invariant')}"
        ),
        "duplicate batching controls should distinguish generic FP16 pooled-embedding drift from BitNet/I2_SR drift",
    )
    full_progress_path = root / f"benchmark_results/seqcls_native_full_progress_{DATE}.json"
    full_progress = read_json(full_progress_path)
    add_check(
        checks,
        "Sequence-classification native full CPU progress is resumable",
        full_progress.get("status") in {"partial", "complete"}
        and isinstance(full_progress.get("completed_examples"), int)
        and full_progress.get("completed_examples") > 0
        and full_progress.get("contiguous_prefix") is True
        and full_progress.get("is_product_evidence") is False,
        (
            f"status={full_progress.get('status')}, completed={full_progress.get('completed_examples')}, "
            f"contiguous={full_progress.get('contiguous_prefix')}, product={full_progress.get('is_product_evidence')}"
        ),
        "full native CPU run should expose a resumable progress trace without treating partial progress as product evidence",
    )


def audit_qwen3_paper_alignment(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/qwen3_paper_alignment_{DATE}.json"
    if not path.exists():
        add_check(checks, "Qwen3 paper-alignment audit exists", False, str(path.relative_to(root)), "missing Qwen3 audit")
        return
    data = read_json(path)
    rows = data.get("rows", []) if isinstance(data.get("rows"), list) else []
    methods = {(row.get("task"), row.get("method"), row.get("scale"), row.get("phase")) for row in rows if isinstance(row, dict)}
    required_rows = {
        ("mnli", "fp16_sft", "tensor", "paper_baseline"),
        ("mnli", "bitnet_sft", "tensor", "paper_baseline"),
        ("mnli", "bitdistill", "tensor", "paper_baseline"),
        ("mnli", "bitdistill", "row", "novelty_row_scale"),
        ("qnli", "fp16_sft", "tensor", "paper_baseline"),
        ("qnli", "bitnet_sft", "tensor", "paper_baseline"),
        ("qnli", "bitdistill", "tensor", "paper_baseline"),
        ("sst2", "fp16_sft", "tensor", "paper_baseline"),
        ("sst2", "bitnet_sft", "tensor", "paper_baseline"),
        ("sst2", "bitdistill", "tensor", "paper_baseline"),
    }
    add_check(
        checks,
        "Qwen3 paper-alignment audit tracks required GLUE rows",
        data.get("job_count") == 16 and required_rows.issubset(methods),
        f"jobs={data.get('job_count')}, complete={data.get('complete_count')}, ready={data.get('paper_reproduction_ready')}",
        "Qwen3 audit is missing one or more paper-alignment rows",
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


def audit_cpu_tradeoff_frontier(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/cpu_tradeoff_frontier_{DATE}.json"
    if not path.exists():
        add_check(checks, "CPU tradeoff frontier audit exists", False, str(path.relative_to(root)), "missing CPU tradeoff audit")
        return
    data = read_json(path)
    q4 = data.get("q4_vs_i2sr", {}) if isinstance(data.get("q4_vs_i2sr"), dict) else {}
    rows = data.get("rows", []) if isinstance(data.get("rows"), list) else []
    required = ["FP F16", "FP Q8_0", "FP Q4_K_M", "row TQ2_0", "row I2_S", "row I2_SR"]
    labels = {row.get("label") for row in rows if isinstance(row, dict)}
    add_check(
        checks,
        "CPU tradeoff frontier has headline rows",
        set(required).issubset(labels),
        f"labels={sorted(label for label in labels if isinstance(label, str))}",
        "tradeoff frontier is missing a headline CPU artifact",
    )
    add_check(
        checks,
        "CPU tradeoff frontier reports Q4-vs-I2_SR ratios",
        all(isinstance(q4.get(key), (int, float)) for key in ["file_ratio", "rss512_ratio", "prefill_speedup", "decode_speedup", "ppl_ratio"]),
        f"q4_vs_i2sr={q4}",
        "Q4-vs-I2_SR normalized ratios are missing",
    )


def audit_cpu_speed_uncertainty(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/cpu_speed_uncertainty_{DATE}.json"
    if not path.exists():
        add_check(checks, "CPU speed uncertainty audit exists", False, str(path.relative_to(root)), "missing CPU speed uncertainty audit")
        return
    data = read_json(path)
    i2sr = data.get("i2sr_vs_q4", {}) if isinstance(data.get("i2sr_vs_q4"), dict) else {}
    decode_ci = i2sr.get("decode_speedup_ci95")
    prefill_ci = i2sr.get("prefill_speedup_ci95")
    add_check(
        checks,
        "CPU speed uncertainty audit has I2_SR-vs-Q4 intervals",
        isinstance(decode_ci, list)
        and len(decode_ci) == 2
        and isinstance(prefill_ci, list)
        and len(prefill_ci) == 2
        and all(isinstance(value, (int, float)) for value in decode_ci + prefill_ci),
        f"prefill_ci={prefill_ci}, decode_ci={decode_ci}",
        "missing uncertainty intervals for I2_SR-vs-Q4 speedup",
    )
    add_check(
        checks,
        "I2_SR-vs-Q4 speedup intervals stay above 1",
        isinstance(decode_ci, list)
        and len(decode_ci) == 2
        and isinstance(prefill_ci, list)
        and len(prefill_ci) == 2
        and min(decode_ci) > 1.0
        and min(prefill_ci) > 1.0,
        f"prefill_ci={prefill_ci}, decode_ci={decode_ci}",
        "I2_SR speedup over Q4 is not robust under recorded benchmark uncertainty",
    )


def audit_benchmark_matrix(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/benchmark_matrix_audit_{DATE}.json"
    if not path.exists():
        add_check(checks, "Benchmark matrix audit exists", False, str(path.relative_to(root)), "missing benchmark matrix audit")
        return
    data = read_json(path)
    cpu = data.get("cpu_runtime", {}) if isinstance(data.get("cpu_runtime"), dict) else {}
    tl2 = data.get("tl2_status", {}) if isinstance(data.get("tl2_status"), dict) else {}
    add_check(
        checks,
        "Benchmark matrix audit has at least ten complete quality benchmarks",
        data.get("passed") is True and int(data.get("quality_benchmark_count") or 0) >= 10,
        f"passed={data.get('passed')}, quality_benchmark_count={data.get('quality_benchmark_count')}",
        "benchmark matrix has fewer than ten complete quality benchmarks",
    )
    add_check(
        checks,
        "Benchmark matrix audit has Xeon runtime and RSS evidence",
        int(cpu.get("finite_headline_rows") or 0) >= 5 and cpu.get("rss_contexts") == EXPECTED_RSS_CONTEXTS,
        f"finite_rows={cpu.get('finite_headline_rows')}, rss_contexts={cpu.get('rss_contexts')}",
        "benchmark matrix lacks finite Xeon runtime or RSS context coverage",
    )
    add_check(
        checks,
        "Benchmark matrix audit keeps TL2 row-scale excluded",
        tl2.get("ready") is False and len(tl2.get("failed", [])) > 0,
        f"tl2_ready={tl2.get('ready')}, failed={len(tl2.get('failed', []))}",
        "TL2 row-scale is not explicitly blocked in benchmark matrix",
    )


def audit_research_redirect_claims(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / f"benchmark_results/research_redirect_claims_{DATE}.json"
    if not path.exists():
        add_check(checks, "Research redirect claim gate exists", False, str(path.relative_to(root)), "missing redirect claim gate")
        return
    data = read_json(path)
    claims = data.get("claims", []) if isinstance(data.get("claims"), list) else []
    by_name = {claim.get("name"): claim for claim in claims if isinstance(claim, dict)}
    required = {
        "Blind arbitrary FP/BF16-to-ternary PTQ retrofit",
        "QAT/distillation recovery over blind PTQ",
        "Paper-level BitDistill reproduction",
        "Row-scale I2_SR runtime semantics",
        "TL2 row-scale runtime readiness",
        "Native packed sequence-classification deployment",
        "Kimi/MoE product support",
    }
    supported = {name for name, claim in by_name.items() if claim.get("supported") is True}
    add_check(
        checks,
        "Research redirect claim gate passes",
        data.get("passed") is True and data.get("status") == "claim_guardrail_passed",
        f"status={data.get('status')}, supported={data.get('supported_guardrail_count')}/{data.get('claim_count')}",
        "redirect claim gate failed",
    )
    add_check(
        checks,
        "Research redirect gate covers required claims",
        required <= supported,
        f"missing={sorted(required - supported)}",
        "one or more required redirect claims is missing or unsupported",
    )
    blocked_statuses = {
        by_name.get("Paper-level BitDistill reproduction", {}).get("status"),
        by_name.get("TL2 row-scale runtime readiness", {}).get("status"),
        by_name.get("Native packed sequence-classification deployment", {}).get("status"),
        by_name.get("Kimi/MoE product support", {}).get("status"),
    }
    add_check(
        checks,
        "Research redirect gate blocks overclaims",
        {"not_proven", "blocked"} <= blocked_statuses
        and bool({"prototype_only", "full_validation_batching_blocked"} & blocked_statuses),
        f"blocked_statuses={sorted(str(item) for item in blocked_statuses)}",
        "paper reproduction, TL2, native classifier, or Kimi/MoE overclaim is not blocked",
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
    audit_bitnet_sft_mechanics(root, checks)
    audit_subln_activation_variance(root, checks)
    audit_bitdistill_root_cause(root, checks)
    audit_bitdistill_telemetry_coverage(root, checks)
    audit_bitdistill_training_dynamics(root, checks)
    audit_bitdistill_loss_contract(root, checks)
    audit_original_benchmark_objective(root, checks)
    audit_tl2_negative_result(root, checks)
    audit_tl2_implementation_plan(root, checks)
    audit_tl2_group_scale_viability(root, checks)
    audit_ternary_flip_dynamics(root, checks)
    audit_ternary_threshold_dynamics(root, checks)
    audit_second_order_ternary_init(root, checks)
    audit_bitnet_sft_ls_init_submission(root, checks)
    audit_bitnet_sft_ls_init_result(root, checks)
    audit_bitnet_sft_diag_ls_init_submission(root, checks)
    audit_bitnet_sft_diag_ls_init_result(root, checks)
    audit_seqcls_runtime_gap(root, checks)
    audit_seqcls_runtime_implementation_plan(root, checks)
    audit_seqcls_native_i2sr_smoke(root, checks)
    audit_seqcls_native_i2sr_cpu_sample(root, checks)
    audit_qwen3_paper_alignment(root, checks)
    audit_cpu_rows(root, checks)
    audit_cpu_tradeoff_frontier(root, checks)
    audit_cpu_speed_uncertainty(root, checks)
    audit_benchmark_matrix(root, checks)
    audit_research_redirect_claims(root, checks)
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
