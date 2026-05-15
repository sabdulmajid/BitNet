#!/usr/bin/env python3
"""Audit the active BitDistill reproduction goal against concrete evidence.

This gate is intentionally separate from the dense-Qwen objective audit.  It
tracks the current BitDistill workstream: paper-style reproduction, missing
training components, row-scale novelty, CPU export/runtime evidence, and the
publication/product boundary.
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
TASKS = ("mnli", "qnli", "sst2")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def status_rank(status: str) -> int:
    return {"complete": 0, "pending": 1, "partial": 2, "not_complete": 3}.get(status, 3)


def add_row(
    rows: list[dict[str, Any]],
    requirement: str,
    status: str,
    evidence: str,
    remaining_gap: str = "",
) -> None:
    rows.append(
        {
            "requirement": requirement,
            "status": status,
            "evidence": evidence,
            "remaining_gap": "" if status == "complete" else remaining_gap,
        }
    )


def latest_step(monitor: dict[str, Any]) -> tuple[int | None, int | None, float | None]:
    warmup = monitor.get("warmup", {}) if isinstance(monitor.get("warmup"), dict) else {}
    latest = warmup.get("latest_step", {}) if isinstance(warmup.get("latest_step"), dict) else {}
    step = latest.get("step")
    max_steps = warmup.get("max_steps")
    progress = warmup.get("progress")
    return (
        int(step) if isinstance(step, int) else None,
        int(max_steps) if isinstance(max_steps, int) else None,
        float(progress) if finite_number(progress) else None,
    )


def materialized_rows(gate: dict[str, Any], family: str) -> list[dict[str, Any]]:
    rows = gate.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("family") == family
        and row.get("exists") is True
        and finite_number(row.get("accuracy"))
    ]


def audit_reproduction(
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    reproduction: dict[str, Any],
    matrix: dict[str, Any],
    monitor: dict[str, Any],
) -> None:
    fp = materialized_rows(reproduction, "baseline")
    gamma100 = materialized_rows(reproduction, "longwarmup_gamma100")
    paper = materialized_rows(reproduction, "paper_hparam_candidate")
    paper_row = materialized_rows(reproduction, "paper_hparam_row_candidate")
    row_scale = materialized_rows(reproduction, "row_scale_candidate")
    baseline_tasks = sorted({row.get("task") for row in fp if row.get("run") == "FP16-SFT"})
    bitnet_tasks = sorted({row.get("task") for row in fp if row.get("run") == "BitNet-SFT"})
    step, max_steps, progress = latest_step(monitor)
    matrix_passed = matrix.get("passed") is True
    configured = matrix.get("configured_rows")
    expected = matrix.get("expected_rows")
    inferred_rows = len(matrix.get("inferred_field_rows", [])) if isinstance(matrix.get("inferred_field_rows"), list) else None
    pending = matrix_passed and configured == expected and inferred_rows == 0
    strict_complete = reproduction.get("paper_style_tensor_complete") is True
    strict_passed = reproduction.get("paper_style_tensor_passed") is True
    search_complete = reproduction.get("paper_search_tensor_complete") is True
    search_passed = reproduction.get("paper_search_tensor_passed") is True
    complete = strict_complete and strict_passed
    status = "complete" if complete else ("partial" if strict_complete else ("pending" if pending else "partial"))
    if search_complete:
        search_state = "LR/head-init search is complete"
        search_result = "passed" if search_passed else "did not pass"
    else:
        search_state = "LR/head-init search remains incomplete"
        search_result = "pending"
    metrics["paper_reproduction"] = {
        "fp16_tasks": baseline_tasks,
        "bitnet_tasks": bitnet_tasks,
        "gamma100_rows": len(gamma100),
        "paper_rows": len(paper),
        "paper_row_rows": len(paper_row),
        "row_scale_rows": len(row_scale),
        "paper_style_tensor_complete": reproduction.get("paper_style_tensor_complete"),
        "paper_style_tensor_passed": reproduction.get("paper_style_tensor_passed"),
        "paper_search_tensor_complete": reproduction.get("paper_search_tensor_complete"),
        "paper_search_tensor_passed": reproduction.get("paper_search_tensor_passed"),
        "job_matrix_passed": matrix_passed,
        "configured_rows": configured,
        "expected_rows": expected,
        "inferred_rows": inferred_rows,
        "warmup_step": step,
        "warmup_max_steps": max_steps,
        "warmup_progress": progress,
    }
    add_row(
        rows,
        "Reproduce BitDistill GLUE3 baseline on Qwen2.5-0.5B with FP16-SFT, BitNet-SFT, and BitDistill",
        status,
        (
            f"FP16 tasks={baseline_tasks}; BitNet tasks={bitnet_tasks}; "
            f"gamma100 rows={len(gamma100)}/3; strict paper rows={len(paper)}/3; "
            f"paper row rows={len(paper_row)}/3; "
            f"matrix={configured}/{expected}, inferred={inferred_rows}; "
            f"warm-up={step}/{max_steps}; {search_state}, {search_result}"
        ),
        (
            "Gamma=100, strict paper-gamma tensor, strict paper-gamma row, and LR/head-init "
            "BitDistill searches are complete and below the FP16-gap target; clean row-warmup "
            "and full-budget candidates remain pending."
            if search_complete
            else "Gamma=100, strict paper-gamma tensor, and strict paper-gamma row BitDistill are complete and below the FP16-gap target; LR-search, head-init, and full-budget candidates remain pending."
        ),
    )


def audit_components(rows: list[dict[str, Any]], metrics: dict[str, Any], smoke: dict[str, Any], alignment: dict[str, Any]) -> None:
    feature_checks = alignment.get("code_features", {})
    if not isinstance(feature_checks, dict):
        feature_checks = {}
    failed_features = sorted(name for name, value in feature_checks.items() if value is not True)
    smoke_passed = smoke.get("passed") is True
    checks = smoke.get("check_count")
    failed = smoke.get("failed", [])
    complete = smoke_passed and not failed_features
    metrics["components"] = {
        "smoke_passed": smoke_passed,
        "smoke_checks": checks,
        "smoke_failed": failed,
        "feature_failed": failed_features,
    }
    add_row(
        rows,
        "Implement SubLN, Stage-2 CE, Stage-3 CE+logits KL+attention-relation KD, and layer selection",
        "complete" if complete else "partial",
        f"smoke={smoke_passed}, smoke checks={checks}, failed features={failed_features}",
        "Implementation smoke contract or paper-alignment feature checks failed.",
    )


def audit_novelty_and_runtime(
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    reproduction: dict[str, Any],
    rowwarmup: dict[str, Any],
    i2sr: dict[str, Any],
    i2sr_local: dict[str, Any],
    cpu: dict[str, Any],
    cpu_xeon: dict[str, Any],
    cpu_fast: dict[str, Any],
) -> None:
    row_complete = reproduction.get("row_scale_complete") is True
    row_passed = reproduction.get("row_scale_passed") is True
    rowwarmup_families = rowwarmup.get("family_status", {}) if isinstance(rowwarmup.get("family_status"), dict) else {}
    rowwarmup_complete = any(
        isinstance(status, dict) and status.get("complete") is True
        for status in rowwarmup_families.values()
    )
    rowwarmup_passed = any(
        isinstance(status, dict) and status.get("passed") is True
        for status in rowwarmup_families.values()
    )
    i2sr_passed = i2sr.get("passed") is True
    i2sr_local_passed = i2sr_local.get("passed") is True
    cpu_passed = cpu.get("passed") is True
    cpu_xeon_passed = cpu_xeon.get("passed") is True
    cpu_fast_passed = cpu_fast.get("passed") is True
    i2sr_rows = i2sr.get("rows", []) if isinstance(i2sr.get("rows"), list) else []
    i2sr_local_rows = i2sr_local.get("rows", []) if isinstance(i2sr_local.get("rows"), list) else []
    i2sr_complete = sum(1 for row in i2sr_rows if isinstance(row, dict) and row.get("complete"))
    i2sr_local_complete = sum(1 for row in i2sr_local_rows if isinstance(row, dict) and row.get("complete"))
    cpu_critical = cpu.get("critical", []) if isinstance(cpu.get("critical"), list) else []
    cpu_xeon_critical = cpu_xeon.get("critical", []) if isinstance(cpu_xeon.get("critical"), list) else []
    cpu_fast_critical = cpu_fast.get("critical", []) if isinstance(cpu_fast.get("critical"), list) else []
    cpu_complete = sum(1 for row in cpu_critical if isinstance(row, dict) and row.get("complete"))
    cpu_xeon_complete = sum(1 for row in cpu_xeon_critical if isinstance(row, dict) and row.get("complete"))
    cpu_fast_complete = sum(1 for row in cpu_fast_critical if isinstance(row, dict) and row.get("complete"))
    cpu_xeon_expected = len(cpu_xeon_critical) or len(cpu_critical)
    cpu_xeon_status: Any = cpu_xeon_passed if cpu_xeon else "pending"
    cpu_hardware = cpu.get("hardware", {}) if isinstance(cpu.get("hardware"), dict) else {}
    cpu_xeon_hardware = cpu_xeon.get("hardware", {}) if isinstance(cpu_xeon.get("hardware"), dict) else {}
    cpu_fast_hardware = cpu_fast.get("hardware", {}) if isinstance(cpu_fast.get("hardware"), dict) else {}
    metrics["row_scale_runtime"] = {
        "row_scale_complete": row_complete,
        "row_scale_passed": row_passed,
        "row_warmup_complete": rowwarmup_complete,
        "row_warmup_passed": rowwarmup_passed,
        "row_warmup_families": rowwarmup_families,
        "i2sr_passed": i2sr_passed,
        "i2sr_complete": i2sr_complete,
        "i2sr_expected": len(i2sr_rows),
        "i2sr_local_passed": i2sr_local_passed,
        "i2sr_local_complete": i2sr_local_complete,
        "i2sr_local_expected": len(i2sr_local_rows),
        "cpu_passed": cpu_passed,
        "cpu_complete": cpu_complete,
        "cpu_expected": len(cpu_critical),
        "cpu_model": cpu_hardware.get("cpu_model"),
        "cpu_xeon_passed": cpu_xeon_passed,
        "cpu_xeon_present": bool(cpu_xeon),
        "cpu_xeon_complete": cpu_xeon_complete,
        "cpu_xeon_expected": cpu_xeon_expected,
        "cpu_xeon_model": cpu_xeon_hardware.get("cpu_model"),
        "cpu_fast_passed": cpu_fast_passed,
        "cpu_fast_complete": cpu_fast_complete,
        "cpu_fast_expected": len(cpu_fast_critical),
        "cpu_fast_model": cpu_fast_hardware.get("cpu_model"),
        "cpu_blockers": cpu.get("blockers", []),
        "cpu_xeon_blockers": cpu_xeon.get("blockers", []),
    }
    add_row(
        rows,
        "Compare paper-style per-tensor BitDistill against row-scale BitDistill",
        "complete" if (row_complete and row_passed) or rowwarmup_passed else ("partial" if row_complete else "pending"),
        (
            f"tensor-warmup row gate complete={row_complete}, passed={row_passed}; "
            f"row-warmup gate complete={rowwarmup_complete}, passed={rowwarmup_passed}"
        ),
        "Gamma=100 and paper-gamma tensor-warmup row comparisons are complete but do not pass the FP16-gap gate; row-warmup comparisons remain pending.",
    )
    add_row(
        rows,
        "Export row-scale checkpoints through I2_SR and benchmark CPU speed, memory/RSS, and task quality on Xeon",
        "complete" if i2sr_passed and cpu_xeon_passed else ("partial" if i2sr_local_passed or cpu_fast_passed or cpu_passed else "pending"),
        (
            f"I2_SR gate={i2sr_passed} ({i2sr_complete}/{len(i2sr_rows)} rows); "
            f"local isolated I2_SR={i2sr_local_passed} ({i2sr_local_complete}/{len(i2sr_local_rows)} rows); "
            f"full CPU gate={cpu_passed} on {cpu_hardware.get('cpu_model', 'unknown')} "
            f"({cpu_complete}/{len(cpu_critical)} critical rows); "
            f"Xeon full CPU gate={cpu_xeon_status} on {cpu_xeon_hardware.get('cpu_model', 'hardware pending')} "
            f"({cpu_xeon_complete}/{cpu_xeon_expected} critical rows); "
            f"scoped CPU slice={cpu_fast_passed} on {cpu_fast_hardware.get('cpu_model', 'unknown')} "
            f"({cpu_fast_complete}/{len(cpu_fast_critical)} critical rows)"
        ),
        "Causal export/runtime and non-Xeon CPU rows have passed; the Xeon-local full CPU gate must pass before this row is complete.",
    )


def audit_publishability(
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    product: dict[str, Any],
    paper_alignment: dict[str, Any],
) -> None:
    supported = product.get("supported_claim_count")
    unsupported = product.get("unsupported_claim_count")
    scope = product.get("scope_status")
    alignment_rows = paper_alignment.get("alignment", []) if isinstance(paper_alignment.get("alignment"), list) else []
    partial = [row.get("dimension") for row in alignment_rows if isinstance(row, dict) and row.get("status") in {"partial", "pending"}]
    lr_headinit_complete = metrics.get("paper_reproduction", {}).get("paper_search_tensor_complete") is True
    quality_blocker = (
        "Strict tensor LR/head-init searches are complete and negative; remaining quality claims need clean row-warmup/full-budget evidence and full CPU-quality gates."
        if lr_headinit_complete
        else "Publishable quality claims must wait for strict paper-hyperparameter BitDistill results and full CPU-quality gates; current support is implementation/provenance plus dense-Qwen I2_SR evidence."
    )
    metrics["publication_scope"] = {
        "scope_status": scope,
        "supported_claim_count": supported,
        "unsupported_claim_count": unsupported,
        "paper_alignment_partial_or_pending": partial,
        "lr_headinit_search_complete": lr_headinit_complete,
    }
    add_row(
        rows,
        "Define publishable scope: independent reproduction, open training implementation, row-scale I2_SR extension, boundary study, and MoE/Kimi limits",
        "partial",
        f"product scope={scope}; supported={supported}; unsupported={unsupported}; paper gaps={partial}",
        quality_blocker,
    )


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    reproduction = read_json(root / args.reproduction_json)
    matrix = read_json(root / args.matrix_json)
    monitor = read_json(root / args.monitor_json)
    row_monitor = read_json(root / args.row_monitor_json)
    rowwarmup = read_json(root / args.rowwarmup_json)
    smoke = read_json(root / args.smoke_json)
    paper_alignment = read_json(root / args.paper_alignment_json)
    i2sr = read_json(root / args.i2sr_json)
    i2sr_local = read_json(root / args.i2sr_local_json)
    cpu = read_json(root / args.cpu_json)
    cpu_xeon = read_json(root / args.cpu_xeon_json)
    cpu_fast = read_json(root / args.cpu_fast_json)
    product = read_json(root / args.product_json)

    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    audit_reproduction(rows, metrics, reproduction, matrix, monitor)
    row_step, row_max_steps, row_progress = latest_step(row_monitor)
    metrics["row_warmup"] = {
        "warmup_step": row_step,
        "warmup_max_steps": row_max_steps,
        "warmup_progress": row_progress,
    }
    audit_components(rows, metrics, smoke, paper_alignment)
    audit_novelty_and_runtime(rows, metrics, reproduction, rowwarmup, i2sr, i2sr_local, cpu, cpu_xeon, cpu_fast)
    audit_publishability(rows, metrics, product, paper_alignment)
    complete_count = sum(1 for row in rows if row["status"] == "complete")
    pending_count = sum(1 for row in rows if row["status"] == "pending")
    status = max((row["status"] for row in rows), key=status_rank)
    achieved = all(row["status"] == "complete" for row in rows)
    return {
        "schema": "bitdistill-active-goal-audit-v1",
        "date": DATE,
        "objective_achieved": achieved,
        "completion_status": status,
        "check_count": len(rows),
        "complete_count": complete_count,
        "pending_count": pending_count,
        "open_requirements": [row["requirement"] for row in rows if row["status"] != "complete"],
        "inputs": {
            "reproduction_json": rel(root / args.reproduction_json, root),
            "matrix_json": rel(root / args.matrix_json, root),
            "monitor_json": rel(root / args.monitor_json, root),
            "row_monitor_json": rel(root / args.row_monitor_json, root),
            "rowwarmup_json": rel(root / args.rowwarmup_json, root),
            "smoke_json": rel(root / args.smoke_json, root),
            "paper_alignment_json": rel(root / args.paper_alignment_json, root),
            "i2sr_json": rel(root / args.i2sr_json, root),
            "i2sr_local_json": rel(root / args.i2sr_local_json, root),
            "cpu_json": rel(root / args.cpu_json, root),
            "cpu_xeon_json": rel(root / args.cpu_xeon_json, root),
            "cpu_fast_json": rel(root / args.cpu_fast_json, root),
            "product_json": rel(root / args.product_json, root),
        },
        "checklist": rows,
        "metrics": metrics,
    }


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    checklist_rows = [
        [row["requirement"], row["status"], row["evidence"], row["remaining_gap"]]
        for row in result["checklist"]
    ]
    input_rows = [[name, path] for name, path in result["inputs"].items()]
    open_rows = [[requirement] for requirement in result["open_requirements"]] or [["none"]]
    metrics = result["metrics"]
    warmup = metrics.get("paper_reproduction", {})
    row_warmup = metrics.get("row_warmup", {})
    runtime = metrics.get("row_scale_runtime", {})
    return "\n\n".join(
        [
            f"# BitDistill Active Goal Audit, {result['date']}",
            "This audit maps the current BitDistill reproduction/productization goal to concrete artifacts. It is not a success declaration.",
            "## Verdict",
            f"Objective achieved: `{result['objective_achieved']}`.",
            f"Completion status: `{result['completion_status']}`.",
            f"Complete rows: `{result['complete_count']}` / `{result['check_count']}`.",
            f"Pending rows: `{result['pending_count']}`.",
            (
                f"Tensor warm-up progress: `{fmt(warmup.get('warmup_step'))}` / `{fmt(warmup.get('warmup_max_steps'))}` "
                f"(`{fmt(warmup.get('warmup_progress'))}`)."
            ),
            (
                f"Row warm-up progress: `{fmt(row_warmup.get('warmup_step'))}` / `{fmt(row_warmup.get('warmup_max_steps'))}` "
                f"(`{fmt(row_warmup.get('warmup_progress'))}`)."
            ),
            (
                f"Runtime gates: row-scale complete=`{runtime.get('row_scale_complete')}`, "
                f"row-warmup complete=`{runtime.get('row_warmup_complete')}`, "
                f"I2_SR=`{runtime.get('i2sr_passed')}`, local I2_SR=`{runtime.get('i2sr_local_passed')}`, "
                f"CPU=`{runtime.get('cpu_passed')}`, Xeon CPU=`{runtime.get('cpu_xeon_passed')}`, "
                f"scoped CPU=`{runtime.get('cpu_fast_passed')}`."
            ),
            "## Prompt-To-Artifact Checklist",
            md_table(["requirement", "status", "evidence", "remaining gap"], checklist_rows),
            "## Open Requirements",
            md_table(["requirement"], open_rows),
            "## Inputs",
            md_table(["input", "path"], input_rows),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--reproduction-json", type=Path, default=Path(f"benchmark_results/bitdistill_reproduction_gate_{DATE}.json"))
    parser.add_argument("--matrix-json", type=Path, default=Path(f"benchmark_results/bitdistill_job_matrix_audit_{DATE}.json"))
    parser.add_argument("--monitor-json", type=Path, default=Path(f"benchmark_results/bitdistill_job_monitor_{DATE}.json"))
    parser.add_argument("--row-monitor-json", type=Path, default=Path(f"benchmark_results/bitdistill_row_warmup_monitor_{DATE}.json"))
    parser.add_argument("--rowwarmup-json", type=Path, default=Path(f"benchmark_results/bitdistill_rowwarmup_gate_{DATE}.json"))
    parser.add_argument("--smoke-json", type=Path, default=Path(f"benchmark_results/bitdistill_smoke_contract_{DATE}.json"))
    parser.add_argument("--paper-alignment-json", type=Path, default=Path(f"benchmark_results/bitdistill_paper_alignment_{DATE}.json"))
    parser.add_argument("--i2sr-json", type=Path, default=Path(f"benchmark_results/bitdistill_i2sr_export_gate_{DATE}.json"))
    parser.add_argument("--i2sr-local-json", type=Path, default=Path(f"benchmark_results/bitdistill_i2sr_export_gate_local_{DATE}.json"))
    parser.add_argument("--cpu-json", type=Path, default=Path(f"benchmark_results/bitdistill_glue_cpu_gate_{DATE}.json"))
    parser.add_argument("--cpu-xeon-json", type=Path, default=Path(f"benchmark_results/bitdistill_glue_cpu_xeon_gate_{DATE}.json"))
    parser.add_argument("--cpu-fast-json", type=Path, default=Path(f"benchmark_results/bitdistill_glue_cpu_fast_gate_{DATE}.json"))
    parser.add_argument("--product-json", type=Path, default=Path(f"benchmark_results/product_scope_gate_{DATE}.json"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_active_goal_audit_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_active_goal_audit_{DATE}.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    result = build_audit(args)
    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(result)
    output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
