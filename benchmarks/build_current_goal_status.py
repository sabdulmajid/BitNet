#!/usr/bin/env python3
"""Build a current objective-status report from canonical evidence artifacts.

This report is intentionally a snapshot, not a success declaration. It gathers
the current claim ledger, BitDistill reproduction gap, and live 655M monitor so
reviewers can see what is proven, what is pending, and what remains blocked by
missing evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(item).replace("|", "\\|") for item in row) + " |")
    return "\n".join(lines)


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def validate_inputs(canonical: dict[str, Any], gap: dict[str, Any], monitor: dict[str, Any]) -> None:
    errors: list[str] = []
    if canonical.get("schema") != "bitnet-canonical-evidence-bundle-v1":
        errors.append(f"unexpected canonical schema: {canonical.get('schema')}")
    if gap.get("schema") != "bitnet-reproduction-gap-report-v1":
        errors.append(f"unexpected reproduction-gap schema: {gap.get('schema')}")
    if gap.get("status") != "not_reproduced":
        errors.append(f"unexpected reproduction-gap status: {gap.get('status')}")
    if monitor.get("schema") != "bitnet-active-stage2-extension-monitor-v1":
        errors.append(f"unexpected monitor schema: {monitor.get('schema')}")
    if monitor.get("quality_claim") != "none":
        errors.append(f"monitor quality_claim must be none, got {monitor.get('quality_claim')}")
    downstream = monitor.get("downstream")
    if not isinstance(downstream, dict):
        errors.append("monitor missing downstream object")
    elif "does not compute or claim MNLI accuracy" not in str(downstream.get("caveat", "")):
        errors.append("monitor downstream caveat is missing non-quality language")
    if errors:
        raise RuntimeError("\n".join(errors))


def build_status(args: argparse.Namespace) -> dict[str, Any]:
    canonical = read_json(args.canonical_bundle)
    gap = read_json(args.reproduction_gap)
    monitor = read_json(args.active_monitor)
    validate_inputs(canonical, gap, monitor)

    claims = canonical["claims"]
    blind = claims["blind_ptq"]
    qat = claims["qat_distill"]
    bitdistill = claims["bitdistill_reproduction"]
    runtime = claims["row_scale_runtime_contract"]
    i2sr = claims["i2sr_cpu"]
    native = claims["native_classifier"]
    moe = claims["moe_kimi"]
    gap_metrics = gap["metrics"]
    latest_tokens = int(gap_metrics.get("bitdistill_latest_stage2_tokens") or 327_680_000)
    latest_mnli = float(gap_metrics.get("bitdistill_latest_mnli", gap_metrics["bitdistill_327_68m_mnli"]))
    latest_delta_vs_fp = float(
        gap_metrics.get("bitdistill_latest_delta_vs_fp16", gap_metrics["bitdistill_327_68m_delta_vs_fp16"])
    )
    stage2 = monitor["stage2"]
    downstream = monitor["downstream"]
    telemetry = monitor["telemetry"]

    requirements = [
        {
            "requirement": "Arbitrary FP/BF16 to ternary retrofit",
            "status": "rejected_for_tested_dense_qwen_setup",
            "evidence": (
                f"FP WikiText PPL {blind['fp_wikitext_ppl']:.3f}; naive PTQ PPL "
                f"{blind['ptq_wikitext_ppl']:.3f}; FP ten-task mean "
                f"{blind['fp_ten_task_mean']:.6f}; PTQ mean {blind['ptq_ten_task_mean']:.6f}"
            ),
            "remaining_gap": "Do not market as a universal converter.",
        },
        {
            "requirement": "BitDistill paper-level MNLI recovery",
            "status": str(gap.get("status", "not_reproduced")),
            "evidence": (
                f"FP16-SFT {gap_metrics['fp16_sft_mnli']:.6f}; "
                f"latest {latest_tokens / 1_000_000:.2f}M BitDistill {latest_mnli:.6f}; "
                f"delta {latest_delta_vs_fp:+.6f}"
            ),
            "remaining_gap": "655.36M downstream MNLI is pending behind the active Stage-2 producer.",
        },
        {
            "requirement": "BitNet-SFT baseline sanity",
            "status": "locally_sanity_checked",
            "evidence": (
                f"default {gap_metrics['bitnet_sft_default_mnli']:.6f}; "
                f"best budget row {gap_metrics['bitnet_sft_best_mnli']:.6f}; "
                f"delta vs paper anchor {gap_metrics['bitnet_sft_best_delta_vs_paper_anchor']:+.6f}"
            ),
            "remaining_gap": "This does not reproduce BitDistill recovery.",
        },
        {
            "requirement": "Row-scale runtime contract",
            "status": "supported",
            "evidence": (
                f"one-scale TL2 RMS error {runtime['one_scale_tl2_relative_rms_error']:.6f}; "
                f"exact row-scale RMS error {runtime['exact_fp16_row_scale_relative_rms_error']:.6f}"
            ),
            "remaining_gap": "TL2 row-scale kernels are not implemented; I2_SR is the supported row-scale path.",
        },
        {
            "requirement": "Packed CPU I2_SR path",
            "status": i2sr["status"],
            "evidence": (
                f"I2_SR file {i2sr['row_i2sr']['file_mib']:.1f} MiB; PPL "
                f"{i2sr['row_i2sr']['ppl']:.4f}; prompt {i2sr['row_i2sr']['prompt_tok_s']:.2f} tok/s; "
                f"decode {i2sr['row_i2sr']['decode_tok_s']:.2f} tok/s"
            ),
            "remaining_gap": "Not quality/storage competitive with Q4_K_M.",
        },
        {
            "requirement": "Native packed classifier product",
            "status": native["status"],
            "evidence": (
                f"MNLI accuracy {native['mnli_accuracy']:.6f}; PyTorch agreement "
                f"{native['pytorch_agreement']:.6f}; RSS {native['rss_mib']:.2f} MiB"
            ),
            "remaining_gap": "Agreement and task quality remain below product gates.",
        },
        {
            "requirement": "MoE/Kimi support",
            "status": moe["status"],
            "evidence": moe["caveat"],
            "remaining_gap": "Needs real routed model mapping, quality, and CPU runtime evidence.",
        },
    ]

    active_gate = {
        "stage2_job_id": stage2["job_id"],
        "stage2_status": monitor["status"],
        "stage2_slurm_state": stage2["slurm"].get("state"),
        "latest_step": stage2["latest_step"].get("step"),
        "max_steps": stage2["max_steps"],
        "progress": stage2["progress"],
        "latest_ce": stage2["latest_step"].get("ce"),
        "eta_hours": stage2["progress_estimate"].get("eta_hours"),
        "latest_complete_snapshot_step": stage2["latest_complete_snapshot_step"],
        "downstream_status": downstream["status"],
        "downstream_complete": downstream["complete"],
        "telemetry_job_id": telemetry["job_id"],
        "telemetry_slurm_state": telemetry["slurm"].get("state"),
    }

    publishable_scope = {
        "not_publishable_as": [
            "universal BitNet converter",
            "paper-level BitDistill reproduction",
            "Q4-quality I2_SR replacement",
            "Kimi/MoE runtime support",
        ],
        "potentially_publishable_as": [
            "negative blind-ternary-PTQ result for tested dense Qwen models",
            "independent BitDistill reproduction-gap study",
            "row-scale ternary runtime-contract evidence",
            "I2_SR packed CPU row-scale extension for compatible causal artifacts",
            "boundary study separating task quality, LM perplexity, RSS, file size, and throughput",
        ],
    }

    return {
        "schema": "bitnet-current-goal-status-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "objective_achieved": False,
        "completion_status": "in_progress",
        "quality_claim": "mixed_completed_evidence_plus_active_pending_gate",
        "current_verdict": (
            "Blind ternary PTQ is rejected for the tested dense-Qwen setup. "
            f"BitDistill-style recovery status is {gap.get('status')}; the active 655.36M "
            "Stage-2 gate is testing whether recovery continues with more tokens."
        ),
        "artifacts": {
            "canonical_bundle": artifact(args.canonical_bundle),
            "reproduction_gap": artifact(args.reproduction_gap),
            "active_monitor": artifact(args.active_monitor),
        },
        "headline_metrics": {
            "blind_ptq_fp_ppl": blind["fp_wikitext_ppl"],
            "blind_ptq_naive_ppl": blind["ptq_wikitext_ppl"],
            "qat_row_scale_ten_task_mean": qat["best_row_scale_qat_ten_task_mean"],
            "qat_recovery_vs_ptq": qat["recovery_vs_ptq"],
            "qat_gap_vs_fp": qat["gap_vs_fp"],
            "fp16_sft_mnli": bitdistill["fp16_sft_mnli"],
            "bitdistill_327_68m_mnli": bitdistill["controlled_327_68m_mnli"],
            "bitdistill_327_68m_delta_vs_fp": bitdistill["controlled_327_68m_delta_vs_fp"],
            "bitdistill_latest_stage2_tokens": latest_tokens,
            "bitdistill_latest_mnli": latest_mnli,
            "bitdistill_latest_delta_vs_fp": latest_delta_vs_fp,
            "bitdistill_655_36m_status": downstream["status"],
        },
        "requirements": requirements,
        "active_gate": active_gate,
        "next_gates": gap["next_gates"],
        "publishable_scope": publishable_scope,
    }


def render_markdown(report: dict[str, Any]) -> str:
    requirements = report["requirements"]
    active = report["active_gate"]
    headline = report["headline_metrics"]
    scope = report["publishable_scope"]
    return "\n\n".join(
        [
            "# Current Goal Status",
            f"Generated: `{report['created_utc']}`",
            f"Git HEAD: `{report['git_head']}`",
            f"Objective achieved: **{report['objective_achieved']}**.",
            f"Completion status: **{report['completion_status']}**.",
            "## Verdict",
            report["current_verdict"],
            "## Headline Metrics",
            md_table(
                ["metric", "value"],
                [[key, value] for key, value in headline.items()],
            ),
            "## Requirement Audit",
            md_table(
                ["requirement", "status", "evidence", "remaining gap"],
                [
                    [row["requirement"], row["status"], row["evidence"], row["remaining_gap"]]
                    for row in requirements
                ],
            ),
            "## Active 655M Gate",
            md_table(["field", "value"], [[key, value] for key, value in active.items()]),
            "## Next Gates",
            md_table(
                ["gate", "minimum next point", "why"],
                [[row["gate"], row["minimum_next_point"], row["why"]] for row in report["next_gates"]],
            ),
            "## Publishable Scope",
            "Not publishable as:\n"
            + "\n".join(f"- {item}" for item in scope["not_publishable_as"])
            + "\n\nPotentially publishable as:\n"
            + "\n".join(f"- {item}" for item in scope["potentially_publishable_as"]),
            "## Inputs",
            md_table(
                ["artifact", "path", "sha256"],
                [[label, item["path"], item["sha256"]] for label, item in report["artifacts"].items()],
            ),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-bundle",
        type=Path,
        default=Path("benchmarks/results/canonical_evidence_bundle_2026-05-20.json"),
    )
    parser.add_argument(
        "--reproduction-gap",
        type=Path,
        default=Path("benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json"),
    )
    parser.add_argument(
        "--active-monitor",
        type=Path,
        default=Path("benchmarks/results/active_stage2_extension_monitor_2026-05-23.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/current_goal_status_2026-05-23.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/current_goal_status_2026-05-23.md"),
    )
    args = parser.parse_args()

    report = build_status(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report).rstrip() + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
