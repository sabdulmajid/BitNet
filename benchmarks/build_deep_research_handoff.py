#!/usr/bin/env python3
"""Build a deep-research handoff brief from current BitNet evidence.

The brief is designed for an external research agent: it summarizes the tested
hypothesis, the mathematical/runtime interpretation, what is original in this
fork, what is not original, and the next questions that need evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def validate_inputs(status: dict[str, Any], canonical: dict[str, Any], gap: dict[str, Any]) -> None:
    errors: list[str] = []
    if status.get("schema") != "bitnet-current-goal-status-v1":
        errors.append(f"unexpected status schema {status.get('schema')}")
    if status.get("objective_achieved") is not False:
        errors.append("current status should not declare objective complete")
    if canonical.get("schema") != "bitnet-canonical-evidence-bundle-v1":
        errors.append(f"unexpected canonical schema {canonical.get('schema')}")
    if gap.get("schema") != "bitnet-reproduction-gap-report-v1":
        errors.append(f"unexpected gap schema {gap.get('schema')}")
    if errors:
        raise RuntimeError("\n".join(errors))


def validate_decision_inputs(next_decision: dict[str, Any], blueprint: dict[str, Any]) -> None:
    errors: list[str] = []
    if next_decision.get("schema") != "bitdistill-next-decision-v1":
        errors.append(f"unexpected next-decision schema {next_decision.get('schema')}")
    if blueprint.get("schema") != "bitdistill-next-experiment-blueprint-v1":
        errors.append(f"unexpected blueprint schema {blueprint.get('schema')}")
    if next_decision.get("status") != blueprint.get("status"):
        errors.append(
            "next-decision and blueprint status mismatch: "
            f"{next_decision.get('status')} != {blueprint.get('status')}"
        )
    if errors:
        raise RuntimeError("\n".join(errors))


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def build_handoff(args: argparse.Namespace) -> dict[str, Any]:
    status = read_json(args.current_status)
    canonical = read_json(args.canonical_bundle)
    gap = read_json(args.reproduction_gap)
    next_decision = read_json(args.next_decision)
    blueprint = read_json(args.next_experiment_blueprint)
    validate_inputs(status, canonical, gap)
    validate_decision_inputs(next_decision, blueprint)

    claims = canonical["claims"]
    blind = claims["blind_ptq"]
    qat = claims["qat_distill"]
    bitdistill = claims["bitdistill_reproduction"]
    runtime = claims["row_scale_runtime_contract"]
    i2sr = claims["i2sr_cpu"]
    native = claims["native_classifier"]
    gap_metrics = gap["metrics"]
    latest_tokens = int(gap_metrics.get("bitdistill_latest_stage2_tokens") or 327_680_000)
    latest_mnli = float(gap_metrics.get("bitdistill_latest_mnli", bitdistill["controlled_327_68m_mnli"]))
    latest_delta_vs_fp = float(
        gap_metrics.get("bitdistill_latest_delta_vs_fp16", bitdistill["controlled_327_68m_delta_vs_fp"])
    )
    latest_gain_vs_327m = latest_mnli - float(bitdistill["controlled_327_68m_mnli"])
    gamma_balance = (
        next_decision.get("gamma_balance", {})
        if isinstance(next_decision.get("gamma_balance"), dict)
        else {}
    )

    thesis = {
        "original_question": "Can arbitrary pretrained FP16/BF16 models be post-hoc converted to BitNet-style W1.58A8 CPU inference?",
        "current_answer": "No for the tested dense-Qwen setup; blind ternary PTQ collapses.",
        "redirected_question": (
            "Can task-specific ternary students be trained from pretrained teachers, "
            "and can CPU formats preserve the scale semantics those students learn?"
        ),
        "core_interpretation": (
            "Extreme ternary quantization is representation learning plus a runtime contract, "
            "not only compression or file conversion."
        ),
    }

    completed_findings = [
        {
            "finding": "Blind ternary PTQ is a strong negative result in the tested dense-Qwen setup.",
            "evidence": (
                f"FP PPL {blind['fp_wikitext_ppl']:.3f} vs naive PTQ PPL "
                f"{blind['ptq_wikitext_ppl']:.3f}; FP ten-task mean "
                f"{blind['fp_ten_task_mean']:.6f} vs PTQ {blind['ptq_ten_task_mean']:.6f}."
            ),
            "interpretation": "The FP weight geometry is not preserved by a blind ternary projection.",
        },
        {
            "finding": "QAT/distillation recovers signal but not FP quality.",
            "evidence": (
                f"Best row-scale QAT mean {qat['best_row_scale_qat_ten_task_mean']:.6f}; "
                f"recovery over PTQ {qat['recovery_vs_ptq']:+.6f}; "
                f"gap to FP {qat['gap_vs_fp']:+.6f}."
            ),
            "interpretation": "Training can move some function into the ternary family, but current runs do not close the gap.",
        },
        {
            "finding": "BitDistill paper-level recovery remains governed by the latest completed Stage-2 row.",
            "evidence": (
                f"FP16-SFT MNLI {bitdistill['fp16_sft_mnli']:.6f}; "
                f"latest {latest_tokens / 1_000_000:.2f}M BitDistill {latest_mnli:.6f}; "
                f"delta {latest_delta_vs_fp:+.6f}; status {gap.get('status')}."
            ),
            "interpretation": (
                f"The 655.36M row is complete; its marginal gain over 327.68M is "
                f"{latest_gain_vs_327m:+.6f}, so loss balance is the next controlled variable."
            ),
        },
        {
            "finding": "The earlier weak BitNet-SFT baseline was mostly undertraining, not the main blocker.",
            "evidence": (
                f"default BitNet-SFT {gap_metrics['bitnet_sft_default_mnli']:.6f}; "
                f"best budget row {gap_metrics['bitnet_sft_best_mnli']:.6f}; "
                f"delta vs paper anchor {gap_metrics['bitnet_sft_best_delta_vs_paper_anchor']:+.6f}."
            ),
            "interpretation": "The remaining problem is BitDistill recovery/loss dynamics, not merely BitLinear replacement.",
        },
        {
            "finding": "Row-scale semantics are material to the learned function.",
            "evidence": (
                f"TL2 one-scale output RMS error {runtime['one_scale_tl2_relative_rms_error']:.6f}; "
                f"exact row-scale RMS error {runtime['exact_fp16_row_scale_relative_rms_error']:.6f}."
            ),
            "interpretation": "A row-scale ternary student represents W approximately as s_row times T, so scales are model semantics.",
        },
        {
            "finding": "I2_SR is a working row-scale packed CPU path but not a Q4 replacement.",
            "evidence": (
                f"I2_SR file {i2sr['row_i2sr']['file_mib']:.1f} MiB, PPL "
                f"{i2sr['row_i2sr']['ppl']:.4f}, prompt {i2sr['row_i2sr']['prompt_tok_s']:.2f} tok/s, "
                f"decode {i2sr['row_i2sr']['decode_tok_s']:.2f} tok/s; Q4_K_M PPL "
                f"{i2sr['q4_k_m']['ppl']:.4f}, file {i2sr['q4_k_m']['file_mib']:.1f} MiB."
            ),
            "interpretation": "The systems path is real, but quality/storage tradeoffs remain unfavorable versus mature Q4.",
        },
    ]

    novelty = [
        {
            "item": "Not novel",
            "description": "BitDistill as a concept: SubLN, continued pretraining, logits KD, and attention-relation KD are Microsoft paper contributions.",
        },
        {
            "item": "Potentially novel",
            "description": "Independent reproduction-gap study with fail-closed artifacts and paired evidence for where local BitDistill diverges.",
        },
        {
            "item": "Potentially novel",
            "description": "Row-scale ternary retrofit variant and the measured requirement that runtime formats preserve row-scale semantics.",
        },
        {
            "item": "Potentially novel",
            "description": "I2_SR packed CPU runtime extension for compatible row-scale causal artifacts.",
        },
        {
            "item": "Potentially novel",
            "description": "Boundary study separating task quality, LM perplexity, file size, RSS, prompt speed, and decode speed.",
        },
    ]

    active_gate = status["active_gate"]
    current_action = blueprint.get("current_action", {}) if isinstance(blueprint.get("current_action"), dict) else {}
    next_action = {
        "decision_status": next_decision.get("status"),
        "recommendation": next_decision.get("recommendation"),
        "blueprint_action": current_action.get("action"),
        "runnable_now": current_action.get("runnable_now"),
        "claim_boundary": current_action.get("claim_boundary"),
        "required_evidence": current_action.get("evidence_required", []),
        "commands": current_action.get("commands", []),
    }
    open_questions = [
        {
            "question": "Did doubling Stage-2 from 327.68M to 655.36M close the MNLI gap?",
            "evidence_needed": "Completed 655M paired prediction trace against the fixed FP16 reference.",
            "current_state": (
                f"Answered: gain {latest_gain_vs_327m:+.6f}; latest MNLI {latest_mnli:.6f}; "
                f"delta vs FP16 {latest_delta_vs_fp:+.6f}."
            ),
        },
        {
            "question": "Is the remaining BitDistill gap mostly compute budget or loss-normalization mismatch?",
            "evidence_needed": "Matched 10k-step gamma-60 and paper-gamma MNLI runs from the same 655M checkpoint.",
            "current_state": (
                f"paper-gamma grad attention/CE {gap_metrics['final_grad_attention_to_ce']:.6f}; "
                f"gamma-60 {float(gamma_balance.get('gamma60_final_grad_attention_to_ce', 0.0)):.6f}; "
                "quality ablation not yet run."
            ),
        },
        {
            "question": "Can the same artifact provide both quality and CPU runtime evidence?",
            "evidence_needed": "Packed classifier or causal prompt-scoring artifact with task quality, RSS, file size, and throughput.",
            "current_state": (
                f"native classifier MNLI {native['mnli_accuracy']:.6f}, agreement "
                f"{native['pytorch_agreement']:.6f}; not product-ready."
            ),
        },
        {
            "question": "Do row-scale variants help generally or only in specific retrofit regimes?",
            "evidence_needed": "Controlled tensor/row/group-scale comparisons across tasks/backbones with paired confidence intervals.",
            "current_state": "Row-scale runtime contract is strong; row-scale accuracy is not a universal guarantee.",
        },
        {
            "question": "Is MoE/Kimi feasible in this runtime path?",
            "evidence_needed": "Real routed model mapping, expert layout, trained quality, and CPU expert-selection benchmarks.",
            "current_state": claims["moe_kimi"]["caveat"],
        },
    ]

    return {
        "schema": "bitnet-deep-research-handoff-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "handoff_not_completion",
        "thesis": thesis,
        "completed_findings": completed_findings,
        "novelty": novelty,
        "active_gate": active_gate,
        "next_action": next_action,
        "open_questions": open_questions,
        "nonclaims": status["publishable_scope"]["not_publishable_as"],
        "publishable_angles": status["publishable_scope"]["potentially_publishable_as"],
        "source_artifacts": {
            "current_status": artifact(args.current_status),
            "canonical_bundle": artifact(args.canonical_bundle),
            "reproduction_gap": artifact(args.reproduction_gap),
            "next_decision": artifact(args.next_decision),
            "next_experiment_blueprint": artifact(args.next_experiment_blueprint),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    thesis = report["thesis"]
    return "\n\n".join(
        [
            "# Deep Research Handoff",
            f"Generated: `{report['created_utc']}`",
            f"Status: **{report['status']}**.",
            "## Thesis",
            "\n".join(
                [
                    f"- Original question: {thesis['original_question']}",
                    f"- Current answer: {thesis['current_answer']}",
                    f"- Redirected question: {thesis['redirected_question']}",
                    f"- Core interpretation: {thesis['core_interpretation']}",
                ]
            ),
            "## Completed Findings",
            md_table(
                ["finding", "evidence", "interpretation"],
                [[row["finding"], row["evidence"], row["interpretation"]] for row in report["completed_findings"]],
            ),
            "## Novelty Boundary",
            md_table(
                ["classification", "description"],
                [[row["item"], row["description"]] for row in report["novelty"]],
            ),
            "## Completed 655M Gate",
            md_table(["field", "value"], [[key, value] for key, value in report["active_gate"].items()]),
            "## Next Action Policy",
            md_table(
                ["field", "value"],
                [
                    ["decision_status", report["next_action"]["decision_status"]],
                    ["recommendation", report["next_action"]["recommendation"]],
                    ["blueprint_action", report["next_action"]["blueprint_action"]],
                    ["runnable_now", report["next_action"]["runnable_now"]],
                    ["claim_boundary", report["next_action"]["claim_boundary"]],
                    ["required_evidence", ", ".join(report["next_action"]["required_evidence"])],
                    ["commands", " ; ".join(report["next_action"]["commands"])],
                ],
            ),
            "## Open Research Questions",
            md_table(
                ["question", "evidence needed", "current state"],
                [[row["question"], row["evidence_needed"], row["current_state"]] for row in report["open_questions"]],
            ),
            "## Nonclaims",
            "\n".join(f"- {item}" for item in report["nonclaims"]),
            "## Publishable Angles",
            "\n".join(f"- {item}" for item in report["publishable_angles"]),
            "## Source Artifacts",
            md_table(
                ["artifact", "path", "sha256"],
                [[label, item["path"], item["sha256"]] for label, item in report["source_artifacts"].items()],
            ),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current-status",
        type=Path,
        default=Path("benchmarks/results/current_goal_status_2026-05-23.json"),
    )
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
        "--next-decision",
        type=Path,
        default=Path("benchmarks/results/bitdistill_next_decision_2026-05-23.json"),
    )
    parser.add_argument(
        "--next-experiment-blueprint",
        type=Path,
        default=Path("benchmarks/results/bitdistill_next_experiment_blueprint_2026-05-23.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/deep_research_handoff_2026-05-23.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/deep_research_handoff_2026-05-23.md"),
    )
    args = parser.parse_args()

    report = build_handoff(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report).rstrip() + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
