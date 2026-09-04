#!/usr/bin/env python3
"""Build concrete next-experiment blueprints for the BitDistill gate.

This report intentionally separates a decision from a launch.  It consumes the
next-decision report and records the exact command templates that are allowed
under each possible decision state.  Pending states only permit status refreshes
and audits, not new quality claims.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_bitdistill_next_decision import fmt, md_table, read_json


DATE = os.environ.get("BITNET_REPORT_DATE", "2026-05-23")


def command(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def build_catalog() -> dict[str, dict[str, Any]]:
    gamma60_downstream = command(
        r"""
        MODEL=Qwen/Qwen2.5-0.5B \
        STAGE=task_sft \
        METHOD=bitdistill \
        TASK_NAME=mnli \
        TASK_FORMAT=sequence_classification \
        LABEL_SCHEME=letters \
        CANDIDATE_SCORE=mean \
        TEACHER_MODEL=checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1 \
        INIT_STATE_MANIFEST=benchmarks/results/stage2_manifest_655m_2026-05-23.json \
        SCALE_MODE=tensor \
        EXCLUDE_LINEAR_REGEX='score|classifier' \
        DISTILL_LAYER=-1 \
        ATTENTION_SPLIT_HEADS=8 \
        ACTIVATION_QUANTIZATION=1 \
        USE_SUBLN=1 \
        LOGIT_KD_WEIGHT=10 \
        ATTENTION_KD_WEIGHT=60 \
        LOGIT_TEMPERATURE=5.0 \
        LOGIT_KD_TEMPERATURE_SCALE=none \
        ATTENTION_TEMPERATURE=1.0 \
        INIT_OUTPUT_HEAD_FROM_TEACHER=1 \
        MAX_SEQ_LEN=512 \
        MAX_STEPS=10000 \
        PER_DEVICE_BATCH_SIZE=4 \
        GRAD_ACCUM_STEPS=4 \
        LR=2e-5 \
        LR_SCHEDULER=cosine \
        SAVE_EVERY_STEPS=0 \
        SAVE_MODEL_ARTIFACTS=0 \
        OUTPUT_DIR=checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-gamma60-headinit \
        sbatch --partition=midcard --job-name=bd-mnli-655m-g60 slurm_bitdistill_glue.sh
        """
    )
    replicate_recovery = command(
        r"""
        MODEL=Qwen/Qwen2.5-0.5B \
        STAGE=task_sft \
        METHOD=bitdistill \
        TASK_NAME=mnli \
        TASK_FORMAT=sequence_classification \
        LABEL_SCHEME=letters \
        CANDIDATE_SCORE=mean \
        TEACHER_MODEL=checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1 \
        INIT_STATE_MANIFEST=benchmarks/results/stage2_manifest_655m_2026-05-23.json \
        SCALE_MODE=tensor \
        EXCLUDE_LINEAR_REGEX='score|classifier' \
        DISTILL_LAYER=-1 \
        ATTENTION_SPLIT_HEADS=8 \
        ACTIVATION_QUANTIZATION=1 \
        USE_SUBLN=1 \
        LOGIT_KD_WEIGHT=10 \
        ATTENTION_KD_WEIGHT=100000 \
        LOGIT_TEMPERATURE=5.0 \
        LOGIT_KD_TEMPERATURE_SCALE=none \
        ATTENTION_TEMPERATURE=1.0 \
        INIT_OUTPUT_HEAD_FROM_TEACHER=1 \
        MAX_SEQ_LEN=512 \
        MAX_STEPS=10000 \
        PER_DEVICE_BATCH_SIZE=4 \
        GRAD_ACCUM_STEPS=4 \
        LR=2e-5 \
        LR_SCHEDULER=cosine \
        SAVE_EVERY_STEPS=0 \
        SAVE_MODEL_ARTIFACTS=0 \
        OUTPUT_DIR=checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit-replicate1 \
        sbatch --partition=midcard --job-name=bd-mnli-655m-rep1 slurm_bitdistill_glue.sh
        """
    )
    qnli_sst2_note = command(
        r"""
        # Only after the replicated MNLI row passes the within-1pt FP gate:
        # 1. Replace TASK_NAME=mnli with TASK_NAME=qnli and the matching QNLI FP16 teacher path.
        # 2. Repeat with TASK_NAME=sst2 and the matching SST2 FP16 teacher path.
        # 3. Keep INIT_STATE_MANIFEST, scale_mode, SubLN, distillation layer, batch, LR,
        #    and loss coefficients identical to the passing MNLI recipe.
        """
    )
    return {
        "pending_no_controlled_rows": {
            "action": "materialize_controlled_row",
            "runnable_now": False,
            "why": "A controlled downstream row is missing, so no broader run is justified.",
            "evidence_required": [
                "controlled-curve JSON with at least one completed row",
                "paired FP16 prediction trace",
            ],
            "commands": [
                "python benchmarks/build_reproduction_gap_report.py",
                "python benchmarks/run_active_gate_watchdog.py",
            ],
            "claim_boundary": "status repair only; no quality claim",
        },
        "pending_655m_downstream": {
            "action": "wait_and_watch_655m_gate",
            "runnable_now": True,
            "why": "The active 655.36M producer/downstream chain is already queued; launching another broad run would confound the token-budget curve.",
            "evidence_required": [
                "655M Stage-2 manifest",
                "655M downstream metrics.json",
                "655M downstream eval_predictions.jsonl",
                "rebuilt controlled curve and next-decision report",
            ],
            "commands": [
                "python benchmarks/run_active_gate_watchdog.py",
                "python benchmarks/audit_stage2_655m_ingestion.py",
            ],
            "claim_boundary": "status only; quality_claim remains none until ingestion is ingested_reports_rebuilt",
        },
        "hold_for_gamma_balance": {
            "action": "wait_for_gamma60_telemetry",
            "runnable_now": True,
            "why": "The quality row alone is insufficient; component-gradient telemetry is needed before choosing more Stage-2 tokens or a rebalance run.",
            "evidence_required": [
                "gamma60 telemetry status report",
                "gamma60_gradient_balance report",
                "rebuilt next-decision report",
            ],
            "commands": [
                "python benchmarks/run_active_gate_watchdog.py",
                "python benchmarks/audit_bitdistill_gamma_balance.py",
            ],
            "claim_boundary": "diagnostic only; gamma60 telemetry is not a quality benchmark",
        },
        "run_gamma_balanced_downstream": {
            "action": "run_matched_gamma60_mnli_downstream",
            "runnable_now": True,
            "why": (
                "The completed 655M row has weak marginal gain and gamma60 telemetry shows "
                "attention-KD updates are rebalanced, so the matched one-axis MNLI ablation is ready."
            ),
            "evidence_required": [
                "next-decision status run_gamma_balanced_downstream",
                "stage2_manifest_655m_2026-05-23.json exists and validates",
                "gamma60_gradient_balance status indicates rebalanced updates",
            ],
            "commands": [gamma60_downstream],
            "claim_boundary": "single MNLI ablation; do not broaden to QNLI/SST2 until paired MNLI result is ingested",
        },
        "extend_stage2_curve": {
            "action": "prepare_next_controlled_stage2_point",
            "runnable_now": False,
            "why": "If 655M still has meaningful marginal gain, the next experiment is another controlled Stage-2 point with the same downstream recipe.",
            "evidence_required": [
                "655M marginal gain >= configured meaningful-gain threshold",
                "compute budget and target token count selected explicitly",
                "new submission report created before launch",
            ],
            "commands": [
                "python benchmarks/build_bitdistill_next_decision.py",
                "# Create a new Stage-2 submission report before sbatch; do not launch from an undocumented one-off command.",
            ],
            "claim_boundary": "budget-curve extension only; keep recipe fixed and do not add new task axes",
        },
        "replicate_recovery_gate": {
            "action": "replicate_passing_mnli_then_expand_glue",
            "runnable_now": False,
            "why": "A within-1pt MNLI row is not enough; it must replicate before QNLI/SST2 are credible.",
            "evidence_required": [
                "next-decision status replicate_recovery_gate",
                "655M row reaches the configured FP recovery gate",
                "replicate prediction trace and paired CI",
            ],
            "commands": [replicate_recovery, qnli_sst2_note],
            "claim_boundary": "reproducibility gate; QNLI/SST2 remain gated behind replicated MNLI",
        },
        "pause_broad_stage2_audit_recipe": {
            "action": "stop_broad_scaling_and_audit_recipe",
            "runnable_now": True,
            "why": "Saturation plus unresolved update imbalance means more broad compute is not justified.",
            "evidence_required": [
                "655M marginal gain <= saturation threshold",
                "gamma60 does not rebalance updates",
            ],
            "commands": [
                "python benchmarks/build_bitdistill_paper_alignment_audit.py",
                "python benchmarks/audit_bitdistill_loss_contract.py",
                "python benchmarks/audit_bitdistill_training_dynamics.py",
            ],
            "claim_boundary": "root-cause audit only; do not submit larger Stage-2 runs before resolving recipe mismatch",
        },
        "ambiguous_recovery_continue_with_controls": {
            "action": "choose_one_narrow_ablation",
            "runnable_now": False,
            "why": "Evidence is mixed; a broad sweep would be hard to interpret.",
            "evidence_required": [
                "single selected ablation objective",
                "pre-registered output directory",
                "postprocess report that rebuilds the decision artifact",
            ],
            "commands": [
                "python benchmarks/build_bitdistill_next_decision.py",
                "# Choose exactly one: next Stage-2 point OR matched gamma60 downstream, not both.",
            ],
            "claim_boundary": "one-axis ablation only",
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    decision = read_json(args.next_decision)
    if decision.get("schema") != "bitdistill-next-decision-v1":
        raise RuntimeError(f"unexpected next-decision schema: {decision.get('schema')}")
    catalog = build_catalog()
    status = str(decision.get("status"))
    current = catalog.get(
        status,
        {
            "action": "unknown_decision_status",
            "runnable_now": False,
            "why": f"No blueprint is registered for decision status {status!r}.",
            "evidence_required": ["update the blueprint catalog"],
            "commands": ["python benchmarks/build_bitdistill_next_decision.py"],
            "claim_boundary": "no launch",
        },
    )
    return {
        "schema": "bitdistill-next-experiment-blueprint-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "quality_claim": "experiment_blueprint_not_benchmark",
        "recommendation": decision.get("recommendation"),
        "current_action": current,
        "action_catalog": catalog,
        "nonclaims": [
            "This report does not add benchmark evidence.",
            "A runnable command is not permission to update quality claims.",
            "Broad sweeps remain disallowed until the matched gamma-60 MNLI result is ingested.",
        ],
        "source_paths": {
            "next_decision": str(args.next_decision),
            "stage2_ingestion": str(args.stage2_ingestion),
            "gamma_balance": str(args.gamma_balance),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    current = report["current_action"]
    catalog_rows = [
        [
            status,
            data["action"],
            data["runnable_now"],
            data["claim_boundary"],
        ]
        for status, data in report["action_catalog"].items()
    ]
    current_commands = "\n\n".join(f"```bash\n{cmd}\n```" for cmd in current["commands"])
    return "\n\n".join(
        [
            "# BitDistill Next Experiment Blueprint",
            f"Generated: `{report['created_utc']}`",
            f"Status: **{report['status']}**.",
            f"Quality claim: **{report['quality_claim']}**.",
            "## Current Recommendation",
            str(report["recommendation"]),
            "## Current Action",
            md_table(
                ["field", "value"],
                [
                    ["action", current["action"]],
                    ["runnable now", current["runnable_now"]],
                    ["why", current["why"]],
                    ["claim boundary", current["claim_boundary"]],
                ],
            ),
            "## Evidence Required",
            md_table(["required evidence"], [[item] for item in current["evidence_required"]]),
            "## Commands",
            current_commands,
            "## Action Catalog",
            md_table(["decision status", "action", "runnable now", "claim boundary"], catalog_rows),
            "## Nonclaims",
            md_table(["nonclaim"], [[item] for item in report["nonclaims"]]),
            "## Source Paths",
            md_table(["artifact", "path"], [[key, value] for key, value in report["source_paths"].items()]),
            (
                "This blueprint is decision support. Regenerate it whenever the controlled "
                "curve, gamma telemetry, or next-decision report changes."
            ),
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--next-decision",
        type=Path,
        default=Path(f"benchmarks/results/bitdistill_next_decision_{DATE}.json"),
    )
    parser.add_argument(
        "--stage2-ingestion",
        type=Path,
        default=Path(f"benchmarks/results/stage2_655m_ingestion_{DATE}.json"),
    )
    parser.add_argument(
        "--gamma-balance",
        type=Path,
        default=Path(f"benchmarks/results/gamma60_gradient_balance_{DATE}.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"benchmarks/results/bitdistill_next_experiment_blueprint_{DATE}.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"benchmarks/results/bitdistill_next_experiment_blueprint_{DATE}.md"),
    )
    args = parser.parse_args()

    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
