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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = datetime.now(timezone.utc).date().isoformat()
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
    paper = materialized_rows(reproduction, "paper_hparam_candidate")
    row_scale = materialized_rows(reproduction, "row_scale_candidate")
    baseline_tasks = sorted({row.get("task") for row in fp if row.get("run") == "FP16-SFT"})
    bitnet_tasks = sorted({row.get("task") for row in fp if row.get("run") == "BitNet-SFT"})
    step, max_steps, progress = latest_step(monitor)
    matrix_passed = matrix.get("passed") is True
    configured = matrix.get("configured_rows")
    expected = matrix.get("expected_rows")
    inferred_rows = len(matrix.get("inferred_field_rows", [])) if isinstance(matrix.get("inferred_field_rows"), list) else None
    pending = matrix_passed and configured == expected and inferred_rows == 0
    complete = reproduction.get("paper_style_tensor_complete") is True and reproduction.get("paper_style_tensor_passed") is True
    status = "complete" if complete else ("pending" if pending else "partial")
    metrics["paper_reproduction"] = {
        "fp16_tasks": baseline_tasks,
        "bitnet_tasks": bitnet_tasks,
        "paper_rows": len(paper),
        "row_scale_rows": len(row_scale),
        "paper_style_tensor_complete": reproduction.get("paper_style_tensor_complete"),
        "paper_style_tensor_passed": reproduction.get("paper_style_tensor_passed"),
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
            f"paper rows={len(paper)}/3; matrix={configured}/{expected}, inferred={inferred_rows}; "
            f"warm-up={step}/{max_steps}"
        ),
        "Long-warmup BitDistill metrics are still pending; short-budget diagnostics are not a paper reproduction.",
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
    i2sr: dict[str, Any],
    cpu: dict[str, Any],
) -> None:
    row_complete = reproduction.get("row_scale_complete") is True
    row_passed = reproduction.get("row_scale_passed") is True
    i2sr_passed = i2sr.get("passed") is True
    cpu_passed = cpu.get("passed") is True
    i2sr_rows = i2sr.get("rows", []) if isinstance(i2sr.get("rows"), list) else []
    i2sr_complete = sum(1 for row in i2sr_rows if isinstance(row, dict) and row.get("complete"))
    cpu_critical = cpu.get("critical", []) if isinstance(cpu.get("critical"), list) else []
    cpu_complete = sum(1 for row in cpu_critical if isinstance(row, dict) and row.get("complete"))
    metrics["row_scale_runtime"] = {
        "row_scale_complete": row_complete,
        "row_scale_passed": row_passed,
        "i2sr_passed": i2sr_passed,
        "i2sr_complete": i2sr_complete,
        "i2sr_expected": len(i2sr_rows),
        "cpu_passed": cpu_passed,
        "cpu_complete": cpu_complete,
        "cpu_expected": len(cpu_critical),
        "cpu_blockers": cpu.get("blockers", []),
    }
    add_row(
        rows,
        "Compare paper-style per-tensor BitDistill against row-scale BitDistill",
        "complete" if row_complete and row_passed else "pending",
        f"row-scale gate complete={row_complete}, passed={row_passed}",
        "Row-scale and tensor long-warmup metrics must finish before the novelty comparison is known.",
    )
    add_row(
        rows,
        "Export row-scale checkpoints through I2_SR and benchmark CPU speed, memory/RSS, and task quality on Xeon",
        "complete" if i2sr_passed and cpu_passed else "pending",
        f"I2_SR gate={i2sr_passed} ({i2sr_complete}/{len(i2sr_rows)} rows); CPU gate={cpu_passed} ({cpu_complete}/{len(cpu_critical)} critical rows)",
        "Causal export/CPU benchmark jobs are dependency-blocked on unfinished BitDistill checkpoints.",
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
    metrics["publication_scope"] = {
        "scope_status": scope,
        "supported_claim_count": supported,
        "unsupported_claim_count": unsupported,
        "paper_alignment_partial_or_pending": partial,
    }
    add_row(
        rows,
        "Define publishable scope: independent reproduction, open training implementation, row-scale I2_SR extension, boundary study, and MoE/Kimi limits",
        "partial",
        f"product scope={scope}; supported={supported}; unsupported={unsupported}; paper gaps={partial}",
        "Publishable claims must wait for the long-warmup BitDistill results; current support is implementation/provenance plus dense-Qwen I2_SR evidence.",
    )


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    reproduction = read_json(root / args.reproduction_json)
    matrix = read_json(root / args.matrix_json)
    monitor = read_json(root / args.monitor_json)
    smoke = read_json(root / args.smoke_json)
    paper_alignment = read_json(root / args.paper_alignment_json)
    i2sr = read_json(root / args.i2sr_json)
    cpu = read_json(root / args.cpu_json)
    product = read_json(root / args.product_json)

    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    audit_reproduction(rows, metrics, reproduction, matrix, monitor)
    audit_components(rows, metrics, smoke, paper_alignment)
    audit_novelty_and_runtime(rows, metrics, reproduction, i2sr, cpu)
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
            "smoke_json": rel(root / args.smoke_json, root),
            "paper_alignment_json": rel(root / args.paper_alignment_json, root),
            "i2sr_json": rel(root / args.i2sr_json, root),
            "cpu_json": rel(root / args.cpu_json, root),
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
                f"Warm-up progress: `{fmt(warmup.get('warmup_step'))}` / `{fmt(warmup.get('warmup_max_steps'))}` "
                f"(`{fmt(warmup.get('warmup_progress'))}`)."
            ),
            (
                f"Runtime gates: row-scale complete=`{runtime.get('row_scale_complete')}`, "
                f"I2_SR=`{runtime.get('i2sr_passed')}`, CPU=`{runtime.get('cpu_passed')}`."
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
    parser.add_argument("--smoke-json", type=Path, default=Path(f"benchmark_results/bitdistill_smoke_contract_{DATE}.json"))
    parser.add_argument("--paper-alignment-json", type=Path, default=Path(f"benchmark_results/bitdistill_paper_alignment_{DATE}.json"))
    parser.add_argument("--i2sr-json", type=Path, default=Path(f"benchmark_results/bitdistill_i2sr_export_gate_{DATE}.json"))
    parser.add_argument("--cpu-json", type=Path, default=Path(f"benchmark_results/bitdistill_glue_cpu_gate_{DATE}.json"))
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
