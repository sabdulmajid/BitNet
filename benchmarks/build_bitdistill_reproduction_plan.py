#!/usr/bin/env python3
"""Generate the narrow BitDistill reproduction and novelty experiment matrix."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
TASKS = ["mnli", "qnli", "sst2"]
BASELINES = ["fp16_sft", "bitnet_sft", "bitdistill"]


def make_command(
    *,
    model: str,
    task: str,
    task_format: str,
    label_scheme: str,
    candidate_score: str,
    method: str,
    scale_mode: str,
    distill_layer: int,
    teacher_root: str,
    warmup_state: str,
    max_steps: int,
) -> str:
    teacher = ""
    init = ""
    if method == "bitdistill":
        teacher_path = f"{teacher_root}/{model.replace('/', '-')}/{task}/fp16_sft-tensor-layer-1"
        teacher = f" TEACHER_MODEL={teacher_path}"
        init = f" INIT_STATE_DICT={warmup_state}"
        init += " LOGIT_KD_TEMPERATURE_SCALE=none"
    return (
        f"MODEL={model} TASK_FORMAT={task_format} LABEL_SCHEME={label_scheme} CANDIDATE_SCORE={candidate_score} "
        f"TASK_NAME={task} METHOD={method} SCALE_MODE={scale_mode} DISTILL_LAYER={distill_layer} "
        f"MAX_STEPS={max_steps}{teacher}{init} "
        "sbatch slurm_bitdistill_glue.sh"
    )


def build_matrix(args: argparse.Namespace) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    tasks = [args.primary_task]
    if args.include_secondary_tasks:
        tasks = TASKS
    warmup_dir = f"{args.warmup_root}/{args.model.replace('/', '-')}/continued_pretrain/bitdistill-tensor"
    warmup_state = f"{warmup_dir}/custom_state_dict.pt"
    runs.append(
        {
            "phase": "stage2_continued_pretraining",
            "task": "-",
            "method": "bitdistill",
            "scale_mode": "tensor",
            "distill_layer": "-",
            "command": (
                f"MODEL={args.model} STAGE=continued_pretrain METHOD=bitdistill SCALE_MODE=tensor "
                f"MAX_STEPS={args.continued_pretrain_steps} SAVE_EVERY_STEPS={args.continued_pretrain_save_every_steps} "
                f"OUTPUT_DIR={warmup_dir} "
                "sbatch slurm_bitdistill_glue.sh"
            ),
        }
    )
    for task in tasks:
        for method in BASELINES:
            runs.append(
                {
                    "phase": "paper_baseline",
                    "task": task,
                    "method": method,
                    "scale_mode": "tensor",
                    "distill_layer": -1,
                    "command": make_command(
                        model=args.model,
                        task=task,
                        task_format=args.task_format,
                        label_scheme=args.label_scheme,
                        candidate_score=args.candidate_score,
                        method=method,
                        scale_mode="tensor",
                        distill_layer=-1,
                        teacher_root=args.teacher_root,
                        warmup_state=warmup_state,
                        max_steps=args.max_steps,
                    ),
                }
            )
        if args.include_row_scale:
            runs.append(
                {
                    "phase": "novelty_row_scale",
                    "task": task,
                    "method": "bitdistill",
                    "scale_mode": "row",
                    "distill_layer": -1,
                    "command": make_command(
                        model=args.model,
                        task=task,
                        task_format=args.task_format,
                        label_scheme=args.label_scheme,
                        candidate_score=args.candidate_score,
                        method="bitdistill",
                        scale_mode="row",
                        distill_layer=-1,
                        teacher_root=args.teacher_root,
                        warmup_state=warmup_state,
                        max_steps=args.max_steps,
                    ),
                }
            )
    if args.include_layer_sweep:
        for layer in args.layer_sweep:
            runs.append(
                {
                    "phase": "attention_layer_sweep",
                    "task": args.sweep_task,
                    "method": "bitdistill",
                    "scale_mode": "tensor",
                    "distill_layer": layer,
                    "command": make_command(
                        model=args.model,
                        task=args.sweep_task,
                        task_format=args.task_format,
                        label_scheme=args.label_scheme,
                        candidate_score=args.candidate_score,
                        method="bitdistill",
                        scale_mode="tensor",
                        distill_layer=layer,
                        teacher_root=args.teacher_root,
                        warmup_state=warmup_state,
                        max_steps=args.max_steps,
                    ),
                }
            )
    return {
        "schema": "bitdistill-reproduction-plan-v1",
        "date": DATE,
        "plan_mode": "canonical_mnli_first",
        "model": args.model,
        "primary_task": args.primary_task,
        "included_tasks": tasks,
        "deferred_axes": {
            "secondary_tasks": not args.include_secondary_tasks,
            "row_scale": not args.include_row_scale,
            "attention_layer_sweep": not args.include_layer_sweep,
        },
        "task_format": args.task_format,
        "success_criterion": "BitDistill within 0.5-1.0 accuracy point (0.005-0.010 absolute accuracy) of FP16-SFT on the primary task before expanding axes.",
        "required_first": "Run FP16-SFT for each included task; those checkpoints become task teachers for BitDistill.",
        "required_warmup": f"Run continued pretraining first and pass `{warmup_state}` to every BitDistill task run.",
        "logits_kd": "Use paper-style logits KL with `LOGIT_KD_TEMPERATURE_SCALE=none`; the tau-squared convention is available only as an explicit diagnostic.",
        "expansion_rule": "Do not add QNLI/SST2, row-scale novelty, or attention-layer sweeps until the MNLI tensor-scale BitDistill gate is interpretable.",
        "blocking_question": (
            "Does tensor-scale BitDistill move monotonically toward the local FP16-SFT task model "
            "as Stage-2 tokens increase, or does it saturate far below FP16 despite a viable CE-only "
            "BitNet-SFT baseline?"
        ),
        "baseline_mismatch_audit": [
            "Hold the sequence-classification formulation fixed and compare FP16-SFT, BitNet-SFT, and BitDistill on full MNLI validation.",
            "Verify q/k/v/o/gate/up/down replacement counts and keep embeddings, norms, and classifier heads dense.",
            "Run W1.58-only and W1.58A8 controls to isolate activation quantization damage.",
            "Record ternary code fractions, threshold-band occupancy, scale distributions, A8 clipping, and int8 edge occupancy.",
            "Materialize component-gradient telemetry so CE, logits KD, and attention KD update magnitudes are comparable.",
            "Treat row-scale, QNLI/SST2, and attention-layer sweeps as blocked until the tensor-scale MNLI gate is interpretable.",
        ],
        "stage2_budget_curve": ["0", "40.96M", "163.84M", "327.68M", "640M+", "2.5B+", "10B target"],
        "runs": runs,
    }


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(plan: dict[str, Any]) -> str:
    rows = [
        [
            str(index),
            run["phase"],
            run["task"],
            run["method"],
            run["scale_mode"],
            str(run["distill_layer"]),
            f"`{run['command']}`",
        ]
        for index, run in enumerate(plan["runs"], start=1)
    ]
    return "\n\n".join(
        [
            f"# BitDistill Reproduction Plan, {plan['date']}",
            f"Plan mode: `{plan['plan_mode']}`.",
            f"Model: `{plan['model']}`.",
            f"Primary task: `{plan['primary_task']}`.",
            f"Included tasks: `{', '.join(plan['included_tasks'])}`.",
            f"Task format: `{plan['task_format']}`.",
            f"Success criterion: {plan['success_criterion']}",
            f"Ordering constraint: {plan['required_first']}",
            f"Warmup constraint: {plan['required_warmup']}",
            f"Logits-KD constraint: {plan['logits_kd']}",
            f"Expansion rule: {plan['expansion_rule']}",
            f"Blocking question: {plan['blocking_question']}",
            "This matrix keeps paper reproduction separate from this fork's novelty claim. The default plan uses tensor-scale BitDistill on MNLI first. Secondary tasks, row-scale novelty, and attention-layer sweeps require explicit opt-in flags.",
            "Baseline mismatch audit:\n\n"
            + "\n".join(f"- {item}" for item in plan["baseline_mismatch_audit"]),
            "Stage-2 budget curve:\n\n"
            + ", ".join(f"`{item}`" for item in plan["stage2_budget_curve"]),
            md_table(["#", "phase", "task", "method", "scale", "layer", "command"], rows),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--teacher-root", default="checkpoints/bitdistill-glue-seqcls")
    parser.add_argument("--warmup-root", default="checkpoints/bitdistill-glue")
    parser.add_argument("--task-format", choices=["sequence_classification", "causal_lm"], default="sequence_classification")
    parser.add_argument("--label-scheme", choices=["letters", "words"], default="letters")
    parser.add_argument("--candidate-score", choices=["mean", "sum"], default="mean")
    parser.add_argument("--primary-task", default="mnli", choices=TASKS)
    parser.add_argument("--include-secondary-tasks", action="store_true")
    parser.add_argument("--include-row-scale", action="store_true")
    parser.add_argument("--include-layer-sweep", action="store_true")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--continued-pretrain-steps", type=int, default=5000)
    parser.add_argument("--continued-pretrain-save-every-steps", type=int, default=1000)
    parser.add_argument("--sweep-task", default="mnli", choices=TASKS)
    parser.add_argument("--layer-sweep", type=int, nargs="+", default=[-1, -2, -4, -8])
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_reproduction_plan_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_reproduction_plan_{DATE}.md"))
    args = parser.parse_args()

    plan = build_matrix(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(plan), encoding="utf-8")
    print(render_markdown(plan))


if __name__ == "__main__":
    main()
