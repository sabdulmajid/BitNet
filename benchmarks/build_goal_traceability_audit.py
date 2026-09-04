#!/usr/bin/env python3
"""Build a requirement-level traceability audit for the active BitDistill goal.

The report answers: what was requested, what current evidence proves, what is
still pending, and what the next evidence-producing action should be. It is a
snapshot for technical review and does not create new benchmark evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STEP_RE = re.compile(
    r"step=(?P<step>\d+)\s+ce=(?P<ce>[0-9.eE+-]+)\s+lr=(?P<lr>[0-9.eE+-]+)\s+elapsed=(?P<elapsed>[0-9.eE+-]+)s"
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def require_schema(path: Path, data: dict[str, Any], expected: str) -> None:
    actual = data.get("schema")
    if actual != expected:
        raise RuntimeError(f"{path}: expected schema {expected}, got {actual}")


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != 0.0 and (abs(value) >= 10000.0 or abs(value) < 0.0001):
            return f"{value:.3e}"
        return f"{value:.6f}"
    if isinstance(value, list):
        if not value:
            return "none"
        return ", ".join(fmt(item) for item in value)
    if isinstance(value, dict):
        if not value:
            return "none"
        return ", ".join(f"{key}={fmt(val)}" for key, val in value.items())
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def squeue_rows(job_ids: list[str]) -> dict[str, dict[str, str]]:
    if not job_ids:
        return {}
    result = subprocess.run(
        ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i\t%j\t%T\t%M\t%D\t%R"],
        check=False,
        capture_output=True,
        text=True,
    )
    rows: dict[str, dict[str, str]] = {}
    if result.returncode != 0:
        return rows
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 6:
            continue
        job_id, name, state, time_used, nodes, reason = parts
        rows[job_id] = {
            "job_id": job_id,
            "name": name,
            "state": state,
            "time": time_used,
            "nodes": nodes,
            "reason": reason,
        }
    return rows


def parse_latest_step(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"log_exists": False, "path": str(log_path)}
    latest: dict[str, Any] = {"log_exists": True, "path": str(log_path)}
    rows: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        row = {
            "step": int(match.group("step")),
            "ce": float(match.group("ce")),
            "lr": float(match.group("lr")),
            "elapsed_seconds": float(match.group("elapsed")),
        }
        rows.append(row)
        latest.update(row)
    if rows:
        recent = rows[-20:]
        recent_ce = [float(row["ce"]) for row in recent]
        latest.update(
            {
                "parsed_log_rows": len(rows),
                "recent_window_rows": len(recent),
                "recent_ce_mean": sum(recent_ce) / len(recent_ce),
                "recent_ce_min": min(recent_ce),
                "recent_ce_max": max(recent_ce),
            }
        )
    return latest


def estimate_stage2(latest: dict[str, Any], max_steps: int) -> dict[str, Any]:
    step = latest.get("step")
    elapsed = latest.get("elapsed_seconds")
    if not isinstance(step, int) or step <= 0 or not isinstance(elapsed, (int, float)) or elapsed <= 0:
        return {"progress": None, "eta_hours": None}
    seconds_per_step = float(elapsed) / float(step)
    eta_seconds = max(max_steps - step, 0) * seconds_per_step
    return {
        "progress": float(step) / float(max_steps),
        "seconds_per_step": seconds_per_step,
        "eta_hours": eta_seconds / 3600.0,
    }


def source_checks(root: Path) -> list[dict[str, Any]]:
    checks = [
        {
            "label": "SubLN wrapper implemented",
            "path": "train_bitdistill.py",
            "patterns": ["class SubLNLinear", "Insert SubLN before attention output and FFN down projections"],
        },
        {
            "label": "Stage-3 loss combines CE, logits KD, and attention KD",
            "path": "train_bitdistill.py",
            "patterns": ["loss = ce + weighted_logit_kd + weighted_attention_kd", "logits_kd_loss", "attention_relation_distillation_components"],
        },
        {
            "label": "component-gradient telemetry exists",
            "path": "train_bitdistill.py",
            "patterns": ["telemetry_component_grad_norms", "component_grad_norms_microbatch"],
        },
        {
            "label": "math viability test exists",
            "path": "experiments/math_viability_test.py",
            "patterns": ["theoretical_mean_abs_relative_fro_error", "relative_output_fro_error"],
        },
        {
            "label": "row-scale scoreboard exists",
            "path": "benchmarks/results/bitdistill_benchmark_scoreboard_2026-05-23.json",
            "patterns": ["bitdistill-benchmark-scoreboard-v1", "scoreboard_from_existing_artifacts_not_new_benchmark"],
        },
    ]
    results: list[dict[str, Any]] = []
    for check in checks:
        path = root / str(check["path"])
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        missing = [pattern for pattern in check["patterns"] if pattern not in text]
        results.append(
            {
                "label": check["label"],
                "path": check["path"],
                "exists": path.exists(),
                "passed": path.exists() and not missing,
                "missing_patterns": missing,
            }
        )
    return results


def latest_controlled_accuracy(controlled: dict[str, Any]) -> dict[str, Any]:
    rows = controlled.get("rows", [])
    complete = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("metrics_exists") is True and isinstance(row.get("metric_accuracy"), (int, float)):
                complete.append(row)
    complete.sort(key=lambda row: int(row.get("stage2_token_presentations", 0)))
    if not complete:
        return {}
    return complete[-1]


def requirement_rows(
    *,
    bundle: dict[str, Any],
    scoreboard: dict[str, Any],
    reproduction_gap: dict[str, Any],
    controlled_curve: dict[str, Any],
    product_scope: dict[str, Any],
    seqcls_gap: dict[str, Any],
    moe_support: dict[str, Any],
    next_decision: dict[str, Any],
    live: dict[str, Any],
    checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    claims = bundle["claims"]
    blind = claims["blind_ptq"]
    qat = claims["qat_distill"]
    bitdistill = claims["bitdistill_reproduction"]
    row_contract = claims["row_scale_runtime_contract"]
    i2sr = claims["i2sr_cpu"]
    native = claims["native_classifier"]
    latest = latest_controlled_accuracy(controlled_curve)
    latest_tokens = int(latest.get("stage2_token_presentations", 0))
    latest_mnli = float(latest.get("metric_accuracy", 0.0))
    latest_delta = float(
        reproduction_gap["metrics"].get(
            "bitdistill_latest_delta_vs_fp16",
            bitdistill["controlled_327_68m_delta_vs_fp"],
        )
    )
    stage2_gate_complete = latest_tokens >= 655_360_000
    source_passed = {check["label"]: check["passed"] for check in checks}
    seq_native = seqcls_gap.get("seqcls_native_cpu_benchmark", {})
    live_stage2 = live.get("stage2", {})
    live_jobs = live.get("jobs", {})
    moe_gates = moe_support.get("productization_gates", [])
    moe_passed = sum(1 for gate in moe_gates if isinstance(gate, dict) and gate.get("passed") is True)
    moe_total = sum(1 for gate in moe_gates if isinstance(gate, dict))

    return [
        {
            "requirement": "Post-training ternary math audit",
            "requested_scope": "Determine whether naive FP/BF16 to ternary conversion is lossless or destructive.",
            "status": "proven_negative_for_tested_dense_qwen",
            "proof_strength": "strong empirical plus analytic probe",
            "evidence": (
                f"FP PPL {blind['fp_wikitext_ppl']:.6f}; PTQ PPL {blind['ptq_wikitext_ppl']:.6f}; "
                f"FP ten-task mean {blind['fp_ten_task_mean']:.6f}; PTQ mean {blind['ptq_ten_task_mean']:.6f}; "
                f"math test present {source_passed.get('math viability test exists')}"
            ),
            "remaining_gap": "The claim is scoped to tested dense Qwen checkpoints, not every possible architecture.",
            "next_action": "Keep this as the headline negative result; do not market universal conversion.",
        },
        {
            "requirement": "BitLinear/SubLN implementation",
            "requested_scope": "Add SubLN and BitDistill-style model surgery for Qwen-family models.",
            "status": "implemented_alignment_still_under_quality_audit",
            "proof_strength": "source evidence plus training artifacts",
            "evidence": (
                f"SubLN source check {source_passed.get('SubLN wrapper implemented')}; "
                f"BitNet-SFT best budget row {reproduction_gap['metrics']['bitnet_sft_best_mnli']:.6f}; "
                f"default row {reproduction_gap['metrics']['bitnet_sft_default_mnli']:.6f}"
            ),
            "remaining_gap": "Implementation exists, but paper-level BitDistill recovery is not proven.",
            "next_action": str(next_decision.get("recommendation")),
        },
        {
            "requirement": "Stage-2 continued pretraining",
            "requested_scope": "Run controlled continued-pretraining budgets before downstream distillation.",
            "status": "completed_655m_curve" if stage2_gate_complete else "active_extension_running",
            "proof_strength": (
                "completed 655.36M row with paired predictions"
                if stage2_gate_complete
                else "completed 327.68M row plus live 655.36M job"
            ),
            "evidence": (
                f"completed latest tokens {latest_tokens:,}; "
                f"latest MNLI {latest_mnli:.6f}; "
                f"live job {live_jobs.get('10250', {}).get('state')}; "
                f"live step {live_stage2.get('latest_step')}; ETA hours {float(live_stage2.get('eta_hours')):.2f}"
            ),
            "remaining_gap": (
                f"The completed row remains {abs(latest_delta) * 100:.3f} accuracy points below FP16."
                if stage2_gate_complete
                else "655.36M downstream MNLI and paired prediction trace are pending."
            ),
            "next_action": str(next_decision.get("recommendation")),
        },
        {
            "requirement": "Stage-3 downstream CE + logits KL + attention-relation KD",
            "requested_scope": "Implement and evaluate downstream BitDistill training.",
            "status": "implemented_but_not_reproduced",
            "proof_strength": "source evidence plus MNLI curve",
            "evidence": (
                f"loss source check {source_passed.get('Stage-3 loss combines CE, logits KD, and attention KD')}; "
                f"FP16-SFT MNLI {bitdistill['fp16_sft_mnli']:.6f}; "
                f"{latest_tokens / 1_000_000:.2f}M BitDistill MNLI {latest_mnli:.6f}; "
                f"delta {latest_delta:+.6f}"
            ),
            "remaining_gap": "Not within the 0.5-1.0 point FP recovery target.",
            "next_action": str(next_decision.get("recommendation")),
        },
        {
            "requirement": "MNLI/QNLI/SST2 paper-style baseline reproduction",
            "requested_scope": "Compare FP16-SFT, BitNet-SFT, and BitDistill on GLUE tasks.",
            "status": "partial_mnli_focused_not_complete",
            "proof_strength": "MNLI controlled rows and earlier GLUE audits",
            "evidence": (
                f"FP16-SFT MNLI {bitdistill['fp16_sft_mnli']:.6f}; "
                f"BitNet-SFT best MNLI {reproduction_gap['metrics']['bitnet_sft_best_mnli']:.6f}; "
                f"BitDistill latest MNLI {latest_mnli:.6f}; "
                f"scoreboard status {scoreboard['status']}"
            ),
            "remaining_gap": "QNLI/SST2 should be run only after a credible MNLI recovery row or recipe fix.",
            "next_action": "Gate QNLI/SST2 on the MNLI recovery decision to avoid wasting compute.",
        },
        {
            "requirement": "Row-scale novelty vs paper-style tensor scale",
            "requested_scope": "Compare tensor-scale BitDistill with row-scale retrofit variants.",
            "status": "supported_as_retrofit_variant",
            "proof_strength": "paired quality evidence plus runtime contract",
            "evidence": (
                f"row-scale QAT mean {qat['best_row_scale_qat_ten_task_mean']:.6f}; "
                f"recovery vs PTQ {qat['recovery_vs_ptq']:+.6f}; "
                f"row-scale RMS {row_contract['exact_fp16_row_scale_relative_rms_error']:.6f}; "
                f"one-scale RMS {row_contract['one_scale_tl2_relative_rms_error']:.6f}"
            ),
            "remaining_gap": "Row scale is not standard BitDistill and does not close FP gap yet.",
            "next_action": "Keep row-scale results labeled as retrofit-variant systems work.",
        },
        {
            "requirement": "I2_SR export and CPU benchmarking on Xeon",
            "requested_scope": "Export row-scale checkpoints and measure speed, memory, RSS, and quality.",
            "status": "working_not_q4_quality_competitive",
            "proof_strength": "CPU benchmark artifact",
            "evidence": (
                f"I2_SR file {i2sr['row_i2sr']['file_mib']:.1f} MiB; "
                f"PPL {i2sr['row_i2sr']['ppl']:.4f}; prompt {i2sr['row_i2sr']['prompt_tok_s']:.2f}; "
                f"decode {i2sr['row_i2sr']['decode_tok_s']:.2f}; Q4 PPL {i2sr['q4_k_m']['ppl']:.4f}"
            ),
            "remaining_gap": "Same-artifact task quality plus product-ready packed runtime remains unsolved.",
            "next_action": "Decide whether product target is packed classifier or causal prompt scorer.",
        },
        {
            "requirement": "At least ten benchmark comparisons",
            "requested_scope": "Provide enough benchmark evidence to prove or disprove the hypothesis.",
            "status": "complete_for_existing_qwen15b_boundary_study",
            "proof_strength": "coverage gate",
            "evidence": (
                f"quality benchmarks {scoreboard['coverage']['quality_benchmark_count']}; "
                f"lm-eval tasks {scoreboard['coverage']['lm_eval_task_count']}; "
                f"coverage checks {scoreboard['coverage']['coverage_check_count']}; "
                f"failed checks {len(scoreboard['coverage']['coverage_failed'])}"
            ),
            "remaining_gap": (
                "The 655M paired MNLI row is now included; broader task coverage remains gated on recovery."
                if stage2_gate_complete
                else "These benchmarks do not include the active 655M BitDistill row."
            ),
            "next_action": "Keep the 655M result in the controlled curve and preserve task-specific claim boundaries.",
        },
        {
            "requirement": "MoE/Kimi feasibility",
            "requested_scope": "Assess expert routing and Kimi/MoE viability.",
            "status": "not_supported_beyond_tiny_plumbing",
            "proof_strength": "negative scope audit",
            "evidence": (
                f"local Kimi artifacts {len(moe_support.get('local_kimi_artifacts', []))}; "
                f"MoE product gates passed {moe_passed}/{moe_total}"
            ),
            "remaining_gap": "No real Kimi mapping, trained MoE quality, routed expert-locality benchmark, or product CPU runtime.",
            "next_action": "Keep MoE/Kimi in future work until dense path is resolved.",
        },
        {
            "requirement": "Product-ready packed sequence classification",
            "requested_scope": "Create a useful deployable CPU artifact for task quality.",
            "status": "research_demo_not_product_ready",
            "proof_strength": "native classifier audit",
            "evidence": (
                f"MNLI {native['mnli_accuracy']:.6f}; agreement {native['pytorch_agreement']:.6f}; "
                f"sequence-isolated {native['examples_per_second']:.6f} ex/s; "
                f"token-id runner {seq_native.get('examples_per_second', 0.0):.6f} ex/s"
            ),
            "remaining_gap": "Agreement below 0.99 product gate and quality weak.",
            "next_action": "Choose product surface after MNLI recovery result: classifier runtime or causal prompt-scoring evaluator.",
        },
        {
            "requirement": "Publishable framing",
            "requested_scope": "Determine whether the work is publishable and under what claim boundaries.",
            "status": "publishable_as_boundary_study_not_converter",
            "proof_strength": "claim ledger plus scoreboard",
            "evidence": (
                f"product scope {product_scope['scope_status']}; "
                f"supported claims {product_scope['supported_claim_count']}; "
                f"unsupported claims {product_scope['unsupported_claim_count']}; "
                f"scoreboard publishable {scoreboard['publishability_assessment']['publishable']}"
            ),
            "remaining_gap": "Paper-level BitDistill reproduction and product artifact remain incomplete.",
            "next_action": "Frame as negative PTQ result plus row-scale runtime contract; keep stronger claims gated.",
        },
    ]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd()
    bundle = read_json(args.canonical_bundle)
    scoreboard = read_json(args.scoreboard)
    reproduction_gap = read_json(args.reproduction_gap)
    controlled_curve = read_json(args.controlled_curve)
    product_scope = read_json(args.product_scope)
    seqcls_gap = read_json(args.seqcls_gap)
    moe_support = read_json(args.moe_support)
    next_decision = read_json(args.next_decision)
    handoff_submission = read_json(args.handoff_submission)
    handoff_job_id = str(handoff_submission.get("handoff_job_id") or args.handoff_job_id)
    require_schema(args.canonical_bundle, bundle, "bitnet-canonical-evidence-bundle-v1")
    require_schema(args.scoreboard, scoreboard, "bitdistill-benchmark-scoreboard-v1")
    require_schema(args.reproduction_gap, reproduction_gap, "bitnet-reproduction-gap-report-v1")
    require_schema(args.controlled_curve, controlled_curve, "bitdistill-controlled-curve-audit-v1")
    require_schema(args.product_scope, product_scope, "bitnet-product-scope-gate-v1")
    require_schema(args.seqcls_gap, seqcls_gap, "seqcls_runtime_gap.v1")
    require_schema(args.moe_support, moe_support, "bitnet-moe-support-audit-v1")
    require_schema(args.next_decision, next_decision, "bitdistill-next-decision-v1")

    live_jobs = squeue_rows([args.stage2_job_id, handoff_job_id, args.telemetry_job_id])
    latest_step = parse_latest_step(args.stage2_log)
    estimate = estimate_stage2(latest_step, args.stage2_max_steps)
    live = {
        "jobs": live_jobs,
        "stage2": {
            "job_id": args.stage2_job_id,
            "latest_step": latest_step.get("step"),
            "latest_ce": latest_step.get("ce"),
            "latest_lr": latest_step.get("lr"),
            "progress": estimate.get("progress"),
            "eta_hours": estimate.get("eta_hours"),
            "recent_ce_mean": latest_step.get("recent_ce_mean"),
            "max_steps": args.stage2_max_steps,
            "log_path": str(args.stage2_log),
        },
    }
    checks = source_checks(root)
    requirements = requirement_rows(
        bundle=bundle,
        scoreboard=scoreboard,
        reproduction_gap=reproduction_gap,
        controlled_curve=controlled_curve,
        product_scope=product_scope,
        seqcls_gap=seqcls_gap,
        moe_support=moe_support,
        next_decision=next_decision,
        live=live,
        checks=checks,
    )
    status_counts: dict[str, int] = {}
    for row in requirements:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1

    return {
        "schema": "bitdistill-goal-traceability-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quality_claim": "traceability_from_existing_artifacts_not_new_benchmark",
        "objective_achieved": False,
        "completion_status": "in_progress",
        "verdict": (
            "The original universal retrofit thesis is disproven for the tested dense-Qwen setup. "
            "The active goal is now a bounded BitDistill/row-scale runtime study. The completed 655M "
            "gate remains below FP16, so the next test is a matched gamma-balanced downstream run."
        ),
        "live_state": live,
        "active_job_ids": {
            "stage2": args.stage2_job_id,
            "handoff": handoff_job_id,
            "telemetry": args.telemetry_job_id,
        },
        "status_counts": status_counts,
        "requirements": requirements,
        "source_checks": checks,
        "research_agent_summary": {
            "what_is_solved": [
                "Blind ternary PTQ is rejected for tested dense Qwen.",
                "BitDistill-style components and telemetry exist in source.",
                "Row-scale runtime semantics are proven material for current row-scale checkpoints.",
                "I2_SR packed CPU runtime works for compatible dense causal artifacts.",
                "MoE/Kimi is correctly scoped as not supported beyond tiny fixtures.",
            ],
            "what_is_being_tested_now": [
                "Whether gamma-60's measured gradient rebalance improves full 10k-step MNLI quality from the 655M checkpoint.",
                "Whether loss normalization, rather than more Stage-2 tokens, explains the remaining FP16 gap.",
            ],
            "next_decision_rule": next_decision.get("recommendation"),
            "publishability": scoreboard["publishability_assessment"],
        },
        "source_paths": {
            "canonical_bundle": str(args.canonical_bundle),
            "scoreboard": str(args.scoreboard),
            "reproduction_gap": str(args.reproduction_gap),
            "controlled_curve": str(args.controlled_curve),
            "product_scope": str(args.product_scope),
            "seqcls_gap": str(args.seqcls_gap),
            "moe_support": str(args.moe_support),
            "next_decision": str(args.next_decision),
            "handoff_submission": str(args.handoff_submission),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    live = report["live_state"]
    active_job_ids = report.get("active_job_ids", {})
    live_rows = []
    for job_id in (
        active_job_ids.get("stage2", "10250"),
        active_job_ids.get("handoff", "10255"),
        active_job_ids.get("telemetry", "10257"),
    ):
        job = live["jobs"].get(job_id, {"job_id": job_id, "state": "not_in_squeue"})
        live_rows.append([job_id, job.get("name", ""), job.get("state"), job.get("time", ""), job.get("reason", "")])
    stage2 = live["stage2"]
    return "\n\n".join(
        [
            "# BitDistill Goal Traceability Audit",
            f"Generated: `{report['created_utc']}`",
            f"Quality claim: **{report['quality_claim']}**.",
            f"Objective achieved: **{report['objective_achieved']}**.",
            f"Completion status: **{report['completion_status']}**.",
            report["verdict"],
            "## Live State",
            md_table(["job id", "name", "state", "time", "reason"], live_rows),
            md_table(
                ["stage2 field", "value"],
                [
                    ["latest_step", stage2.get("latest_step")],
                    ["max_steps", stage2.get("max_steps")],
                    ["progress", stage2.get("progress")],
                    ["latest_ce", stage2.get("latest_ce")],
                    ["recent_ce_mean", stage2.get("recent_ce_mean")],
                    ["eta_hours", stage2.get("eta_hours")],
                    ["log_path", stage2.get("log_path")],
                ],
            ),
            "## Requirement Traceability",
            md_table(
                ["requirement", "status", "proof strength", "evidence", "remaining gap", "next action"],
                [
                    [
                        row["requirement"],
                        row["status"],
                        row["proof_strength"],
                        row["evidence"],
                        row["remaining_gap"],
                        row["next_action"],
                    ]
                    for row in report["requirements"]
                ],
            ),
            "## Source Checks",
            md_table(
                ["check", "path", "passed", "missing patterns"],
                [[row["label"], row["path"], row["passed"], row["missing_patterns"]] for row in report["source_checks"]],
            ),
            "## What Is Solved",
            "\n".join(f"- {item}" for item in report["research_agent_summary"]["what_is_solved"]),
            "## What Is Being Tested Now",
            "\n".join(f"- {item}" for item in report["research_agent_summary"]["what_is_being_tested_now"]),
            "## Publishability",
            md_table(
                ["field", "value"],
                [[key, value] for key, value in report["research_agent_summary"]["publishability"].items()],
            ),
            "## Source Artifacts",
            md_table(["artifact", "path"], [[key, value] for key, value in report["source_paths"].items()]),
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-bundle", type=Path, default=Path("benchmarks/results/canonical_evidence_bundle_2026-05-20.json"))
    parser.add_argument("--scoreboard", type=Path, default=Path("benchmarks/results/bitdistill_benchmark_scoreboard_2026-05-23.json"))
    parser.add_argument("--reproduction-gap", type=Path, default=Path("benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json"))
    parser.add_argument("--controlled-curve", type=Path, default=Path("benchmarks/results/bitdistill_controlled_curve_2026-05-23.json"))
    parser.add_argument("--product-scope", type=Path, default=Path("benchmark_results/product_scope_gate_2026-05-15.json"))
    parser.add_argument("--seqcls-gap", type=Path, default=Path("benchmark_results/seqcls_runtime_gap_2026-05-15.json"))
    parser.add_argument("--moe-support", type=Path, default=Path("benchmark_results/moe_support_audit_2026-05-15.json"))
    parser.add_argument("--next-decision", type=Path, default=Path("benchmarks/results/bitdistill_next_decision_2026-05-23.json"))
    parser.add_argument("--handoff-submission", type=Path, default=Path("benchmarks/results/stage2_655m_handoff_submission_2026-05-23.json"))
    parser.add_argument("--stage2-log", type=Path, default=Path("logs/bd-s2-655m-10250.out"))
    parser.add_argument("--stage2-job-id", default="10250")
    parser.add_argument("--handoff-job-id", default="10255")
    parser.add_argument("--telemetry-job-id", default="10257")
    parser.add_argument("--stage2-max-steps", type=int, default=40000)
    parser.add_argument("--out-json", type=Path, default=Path("benchmarks/results/bitdistill_goal_traceability_2026-05-23.json"))
    parser.add_argument("--out-md", type=Path, default=Path("benchmarks/results/bitdistill_goal_traceability_2026-05-23.md"))
    args = parser.parse_args()

    report = build_report(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
