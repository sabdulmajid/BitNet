#!/usr/bin/env python3
"""Audit how the local BitDistill work aligns with the paper recipe.

This is not a benchmark. It is an interpretation guardrail: it identifies which
paper conditions are matched, approximated, pending, or missing so local results
are not overread as an exact reproduction.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PAPER_STAGE2_TOKENS = 10_000_000_000


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def require_schema(path: Path, data: dict[str, Any], expected: str) -> None:
    if data.get("schema") != expected:
        raise RuntimeError(f"{path}: expected {expected}, got {data.get('schema')}")


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
        return ", ".join(fmt(item) for item in value) if value else "none"
    if isinstance(value, dict):
        return ", ".join(f"{key}={fmt(val)}" for key, val in value.items()) if value else "none"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def source_check(traceability: dict[str, Any], label: str) -> bool:
    checks = traceability.get("source_checks", [])
    if not isinstance(checks, list):
        return False
    for check in checks:
        if isinstance(check, dict) and check.get("label") == label:
            return check.get("passed") is True
    return False


def rows(
    *,
    reproduction_gap: dict[str, Any],
    stage2_submission: dict[str, Any],
    traceability: dict[str, Any],
    next_decision: dict[str, Any],
) -> list[dict[str, Any]]:
    metrics = reproduction_gap["metrics"]
    config = stage2_submission["run_config"]
    completed_tokens = int(metrics.get("bitdistill_latest_stage2_tokens", 327_680_000))
    latest_mnli = float(metrics.get("bitdistill_latest_mnli", metrics["bitdistill_327_68m_mnli"]))
    latest_delta_vs_fp16 = float(
        metrics.get("bitdistill_latest_delta_vs_fp16", metrics["bitdistill_327_68m_delta_vs_fp16"])
    )
    success_delta = float(metrics.get("success_delta_from_fp16", -0.01))
    active_tokens = int(config["cumulative_token_presentations"])
    paper_fraction_completed = completed_tokens / PAPER_STAGE2_TOKENS
    paper_fraction_active = active_tokens / PAPER_STAGE2_TOKENS
    effective_batch = int(config["per_device_batch_size"]) * int(config["grad_accum_steps"])
    return [
        {
            "axis": "Goal",
            "paper_recipe": "Task-specific finetuning of FP LLMs into 1.58-bit BitNet models.",
            "local_state": "The repo is now framed as task-specific ternary distillation plus runtime-contract testing.",
            "status": "aligned",
            "risk": "Do not revert to a universal arbitrary converter claim.",
        },
        {
            "axis": "Backbone",
            "paper_recipe": "Qwen3 0.6B/1.7B/4B primary; Qwen2.5-0.5B and Gemma ablations.",
            "local_state": f"Primary controlled reproduction is {stage2_submission['model']}; dense Qwen2.5-1.5B used for PTQ/runtime boundary study.",
            "status": "paper_ablation_backbone_not_primary_backbone",
            "risk": "Good for Qwen2.5 ablation alignment, not exact Qwen3 main-table reproduction.",
        },
        {
            "axis": "Tasks",
            "paper_recipe": "MNLI, QNLI, SST2, and CNNDM.",
            "local_state": "Current controlled BitDistill gate is MNLI first; QNLI/SST2 are intentionally gated; CNNDM not run.",
            "status": "partial",
            "risk": "Do not claim full GLUE/CNNDM reproduction.",
        },
        {
            "axis": "Baselines",
            "paper_recipe": "FP16-SFT, BitNet-SFT, and BitDistill.",
            "local_state": (
                f"MNLI FP16-SFT {metrics['fp16_sft_mnli']:.6f}; "
                f"BitNet-SFT best {metrics['bitnet_sft_best_mnli']:.6f}; "
                f"BitDistill latest {latest_mnli:.6f}."
            ),
            "status": "mnli_present",
            "risk": "BitDistill is still below FP; QNLI/SST2 baselines should wait for MNLI recovery gate.",
        },
        {
            "axis": "Stage-1 SubLN",
            "paper_recipe": "Insert SubLN before attention output projection and FFN down projection.",
            "local_state": f"Source check passed: {source_check(traceability, 'SubLN wrapper implemented')}; active Stage-2 USE_SUBLN={config['use_subln']}.",
            "status": "implemented",
            "risk": "Source presence does not itself prove paper-level optimization behavior.",
        },
        {
            "axis": "Ternary weight quantization",
            "paper_recipe": "Per-tensor absmean W1.58 in the paper equation.",
            "local_state": f"Active controlled Stage-2 scale_mode={config['scale_mode']}; row-scale work is separately labeled retrofit variant.",
            "status": "active_tensor_matches_paper_equation",
            "risk": "Row-scale I2_SR results must not be labeled as standard BitDistill.",
        },
        {
            "axis": "Activation quantization",
            "paper_recipe": "8-bit activation quantization.",
            "local_state": f"Active Stage-2 activation_quantization={config['activation_quantization']}.",
            "status": "matched_in_active_gate",
            "risk": "Kernel/runtime parity still needs same-artifact proof for product claims.",
        },
        {
            "axis": "Stage-2 corpus",
            "paper_recipe": "10B tokens sampled from FALCON corpus.",
            "local_state": "Local Slurm defaults use HuggingFaceFW/fineweb-edu sample-10BT unless overridden.",
            "status": "mismatch",
            "risk": "Corpus mismatch is a plausible reproduction gap and must be named.",
        },
        {
            "axis": "Stage-2 token budget",
            "paper_recipe": "10B continued-pretraining tokens.",
            "local_state": (
                f"Completed controlled row {completed_tokens:,} tokens ({100.0 * paper_fraction_completed:.4f}% of paper); "
                f"active gate targets {active_tokens:,} tokens ({100.0 * paper_fraction_active:.4f}% of paper)."
            ),
            "status": "under_budget_active_extension_running",
            "risk": "Current non-reproduction cannot disprove paper-scale BitDistill.",
        },
        {
            "axis": "Stage-3 loss terms",
            "paper_recipe": "CE + logits KL + attention-relation KD over Q/K/V.",
            "local_state": f"Source check passed: {source_check(traceability, 'Stage-3 loss combines CE, logits KD, and attention KD')}.",
            "status": "implemented",
            "risk": "Loss terms exist, but normalization and gradient balance remain suspect.",
        },
        {
            "axis": "Logits distillation temperature",
            "paper_recipe": "Temperature 5.0.",
            "local_state": "Active handoff downstream recipe sets LOGIT_TEMPERATURE=5.0.",
            "status": "matched_for_active_downstream",
            "risk": "Only applies after handoff/downstream job runs.",
        },
        {
            "axis": "Attention-relation coefficient",
            "paper_recipe": "Classification uses large attention-KD coefficient; exact meaning depends on reductions.",
            "local_state": (
                f"Paper-gamma local telemetry showed attention/CE imbalance; gamma status is "
                f"{next_decision['gamma_balance']['status']}."
            ),
            "status": "normalization_not_proven_equivalent",
            "risk": "Copying gamma numerically is not enough unless reductions match.",
        },
        {
            "axis": "Attention layer selection",
            "paper_recipe": "Distill a single selected layer, often later layers.",
            "local_state": "Active downstream handoff uses DISTILL_LAYER=-1.",
            "status": "matched_strategy",
            "risk": "Layer choice still requires sweep/replication evidence.",
        },
        {
            "axis": "Sequence length",
            "paper_recipe": "Max sequence length 512 for GLUE setup.",
            "local_state": f"Active Stage-2/downstream configuration uses max_seq_len={config['max_seq_len']}.",
            "status": "matched",
            "risk": "Padding/tokenization details still need exact paper parity if claiming reproduction.",
        },
        {
            "axis": "Batch size",
            "paper_recipe": "Batch size 32.",
            "local_state": (
                f"Active Stage-2 effective local batch is {effective_batch} "
                f"({config['per_device_batch_size']} per device x grad_accum {config['grad_accum_steps']})."
            ),
            "status": "mismatch_or_unproven",
            "risk": "Optimizer dynamics may differ from paper setup.",
        },
        {
            "axis": "Hardware",
            "paper_recipe": "8x AMD MI300X servers for paper experiments.",
            "local_state": "Local active gate is single midcard GPU; CPU runtime measured on Xeon Silver 4116.",
            "status": "mismatch",
            "risk": "Throughput and feasible token budget are not paper-comparable.",
        },
        {
            "axis": "Success criterion",
            "paper_recipe": "BitDistill comparable to FP16-SFT on downstream tasks.",
            "local_state": (
                f"Latest completed MNLI delta vs FP16 is {latest_delta_vs_fp16:+.6f}; "
                f"configured recovery gate is {success_delta:+.6f}."
            ),
            "status": "not_met",
            "risk": "No public claim of paper-level reproduction.",
        },
    ]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    reproduction_gap = read_json(args.reproduction_gap)
    stage2_submission = read_json(args.stage2_submission)
    traceability = read_json(args.traceability)
    next_decision = read_json(args.next_decision)
    require_schema(args.reproduction_gap, reproduction_gap, "bitnet-reproduction-gap-report-v1")
    require_schema(args.stage2_submission, stage2_submission, "bitnet-stage2-extension-submission-v1")
    require_schema(args.traceability, traceability, "bitdistill-goal-traceability-audit-v1")
    require_schema(args.next_decision, next_decision, "bitdistill-next-decision-v1")
    alignment_rows = rows(
        reproduction_gap=reproduction_gap,
        stage2_submission=stage2_submission,
        traceability=traceability,
        next_decision=next_decision,
    )
    counts: dict[str, int] = {}
    for row in alignment_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    highest_risks = [
        "Stage-2 token budget is far below 10B and the 655M gate is still running.",
        "Stage-2 corpus differs from the paper's FALCON corpus unless explicitly overridden.",
        "Attention-KD coefficient equivalence is not proven because loss reductions may differ.",
        "Batch size/hardware differ materially from paper conditions.",
        "QNLI/SST2/CNNDM are not yet paper-level reproduction rows.",
    ]
    return {
        "schema": "bitdistill-paper-alignment-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quality_claim": "paper_alignment_not_new_benchmark",
        "status": "not_exact_reproduction",
        "verdict": (
            "The local work is a paper-inspired Qwen2.5 MNLI reproduction-gap study with "
            "several implemented BitDistill components. It is not an exact paper reproduction."
        ),
        "alignment_counts": counts,
        "rows": alignment_rows,
        "highest_risks": highest_risks,
        "source_paths": {
            "reproduction_gap": str(args.reproduction_gap),
            "stage2_submission": str(args.stage2_submission),
            "traceability": str(args.traceability),
            "next_decision": str(args.next_decision),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "# BitDistill Paper Alignment Audit",
            f"Generated: `{report['created_utc']}`",
            f"Status: **{report['status']}**.",
            f"Quality claim: **{report['quality_claim']}**.",
            report["verdict"],
            "## Alignment Matrix",
            md_table(
                ["axis", "paper recipe", "local state", "status", "risk"],
                [
                    [row["axis"], row["paper_recipe"], row["local_state"], row["status"], row["risk"]]
                    for row in report["rows"]
                ],
            ),
            "## Highest-Risk Mismatches",
            "\n".join(f"- {risk}" for risk in report["highest_risks"]),
            "## Source Artifacts",
            md_table(["artifact", "path"], [[key, value] for key, value in report["source_paths"].items()]),
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduction-gap", type=Path, default=Path("benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json"))
    parser.add_argument("--stage2-submission", type=Path, default=Path("benchmarks/results/stage2_655m_submission_2026-05-23.json"))
    parser.add_argument("--traceability", type=Path, default=Path("benchmarks/results/bitdistill_goal_traceability_2026-05-23.json"))
    parser.add_argument("--next-decision", type=Path, default=Path("benchmarks/results/bitdistill_next_decision_2026-05-23.json"))
    parser.add_argument("--out-json", type=Path, default=Path("benchmarks/results/bitdistill_paper_alignment_2026-05-23.json"))
    parser.add_argument("--out-md", type=Path, default=Path("benchmarks/results/bitdistill_paper_alignment_2026-05-23.md"))
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
