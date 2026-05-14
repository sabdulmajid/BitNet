#!/usr/bin/env python3
"""Generate the BitDistill reproduction and novelty experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
    return (
        f"MODEL={model} TASK_FORMAT={task_format} LABEL_SCHEME={label_scheme} CANDIDATE_SCORE={candidate_score} "
        f"TASK_NAME={task} METHOD={method} SCALE_MODE={scale_mode} DISTILL_LAYER={distill_layer} "
        f"MAX_STEPS={max_steps}{teacher}{init} "
        "sbatch slurm_bitdistill_glue.sh"
    )


def build_matrix(args: argparse.Namespace) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
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
                f"MAX_STEPS={args.continued_pretrain_steps} OUTPUT_DIR={warmup_dir} "
                "sbatch slurm_bitdistill_glue.sh"
            ),
        }
    )
    for task in TASKS:
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
        "model": args.model,
        "task_format": args.task_format,
        "success_criterion": "BitDistill within 0.5-1.0 accuracy point of FP16-SFT on MNLI/QNLI/SST2.",
        "required_first": "Run FP16-SFT for each task; those checkpoints become task teachers for BitDistill.",
        "required_warmup": f"Run continued pretraining first and pass `{warmup_state}` to every BitDistill task run.",
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
            "# BitDistill Reproduction Plan, 2026-05-14",
            f"Model: `{plan['model']}`.",
            f"Task format: `{plan['task_format']}`.",
            f"Success criterion: {plan['success_criterion']}",
            f"Ordering constraint: {plan['required_first']}",
            f"Warmup constraint: {plan['required_warmup']}",
            "This matrix separates paper reproduction from this fork's novelty claim. The paper reproduction uses tensor-scale BitDistill. The novelty run changes only the scale mode to row and then exports through the stable `I2_SR` path after quality is proven.",
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
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--continued-pretrain-steps", type=int, default=5000)
    parser.add_argument("--sweep-task", default="mnli", choices=TASKS)
    parser.add_argument("--layer-sweep", type=int, nargs="+", default=[-1, -2, -4, -8])
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_results/bitdistill_reproduction_plan_2026-05-14.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/results/bitdistill_reproduction_plan_2026-05-14.md"))
    args = parser.parse_args()

    plan = build_matrix(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(plan), encoding="utf-8")
    print(render_markdown(plan))


if __name__ == "__main__":
    main()
