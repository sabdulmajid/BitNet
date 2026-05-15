#!/usr/bin/env python3
"""Audit whether local GLUE measurements match the BitDistill claim surface.

This report is deliberately conservative.  It separates:

* sequence-classification GLUE runs, which are the closest local match to the
  Qwen2.5-0.5B MNLI numbers quoted in the BitDistill excerpt, and
* causal-LM prompt/classification runs, which are useful diagnostics but should
  not be described as the same experimental formulation without paper/code
  confirmation.
"""

from __future__ import annotations

import os
import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
TASKS = ("mnli", "qnli", "sst2")
EXPECTED_EVAL_EXAMPLES = {
    "mnli": 9815,
    "qnli": 5463,
    "sst2": 872,
}

# User-provided BitDistill paper excerpt, Table 3 robustness row for
# Qwen2.5-0.5B on MNLI.  The excerpt did not provide Qwen2.5 QNLI/SST2 anchors.
PAPER_QWEN25_MNLI = {
    "fp16_sft": 0.7991,
    "bitnet_sft": 0.6080,
    "bitdistill": 0.7998,
}


@dataclass(frozen=True)
class RunRef:
    label: str
    root: str
    template: str
    formulation: str
    paper_role: str


RUNS = [
    RunRef("FP16-SFT", "checkpoints/bitdistill-glue-seqcls", "{task}/fp16_sft-tensor-layer-1", "sequence_classification", "baseline"),
    RunRef("BitNet-SFT", "checkpoints/bitdistill-glue-seqcls", "{task}/bitnet_sft-tensor-layer-1", "sequence_classification", "baseline"),
    RunRef("BitDistill short tensor", "checkpoints/bitdistill-glue-seqcls", "{task}/bitdistill-tensor-layer-1", "sequence_classification", "diagnostic"),
    RunRef("BitDistill short row", "checkpoints/bitdistill-glue-seqcls", "{task}/bitdistill-row-layer-1", "sequence_classification", "diagnostic"),
    RunRef("BitDistill longwarmup tensor gamma100", "checkpoints/bitdistill-glue-seqcls-longwarmup", "{task}/bitdistill-longwarmup-tensor-layer-8", "sequence_classification", "pending_candidate"),
    RunRef("BitDistill longwarmup tensor paper gamma", "checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma", "{task}/bitdistill-longwarmup-tensor-layer-8", "sequence_classification", "pending_paper_candidate"),
    RunRef("BitDistill longwarmup row paper gamma", "checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row", "{task}/bitdistill-longwarmup-row-layer-8", "sequence_classification", "pending_row_candidate"),
    RunRef("Causal FP16-SFT words", "checkpoints/bitdistill-glue", "{task}/fp16_sft-tensor-layer-1", "causal_lm_words", "diagnostic"),
    RunRef("Causal BitNet-SFT words", "checkpoints/bitdistill-glue", "{task}/bitnet_sft-tensor-layer-1", "causal_lm_words", "diagnostic"),
    RunRef("Causal BitDistill short tensor words", "checkpoints/bitdistill-glue", "{task}/bitdistill-tensor-layer-1", "causal_lm_words", "diagnostic"),
    RunRef("Causal FP16-SFT letters", "checkpoints/bitdistill-glue-letters", "{task}/fp16_sft-tensor-layer-1", "causal_lm_letters", "diagnostic"),
    RunRef("Causal BitNet-SFT letters", "checkpoints/bitdistill-glue-letters", "{task}/bitnet_sft-tensor-layer-1", "causal_lm_letters", "diagnostic"),
    RunRef("Causal BitDistill short tensor letters", "checkpoints/bitdistill-glue-letters", "{task}/bitdistill-tensor-layer-1", "causal_lm_letters", "diagnostic"),
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_path(root: Path, model: str, ref: RunRef, task: str) -> Path:
    return root / ref.root / model.replace("/", "-") / ref.template.format(task=task) / "metrics.json"


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def metric_row(root: Path, model: str, ref: RunRef, task: str) -> dict[str, Any]:
    path = run_path(root, model, ref, task)
    metrics = read_json(path)
    eval_metrics = metrics.get("eval", {}) if isinstance(metrics.get("eval"), dict) else {}
    accuracy = eval_metrics.get("accuracy")
    examples = eval_metrics.get("eval_examples")
    accuracy_value = float(accuracy) if isinstance(accuracy, (int, float)) else None
    example_count = int(examples) if isinstance(examples, (int, float)) else None
    recorded_format = metrics.get("task_format")
    recorded_label_scheme = metrics.get("label_scheme")
    return {
        "task": task,
        "run": ref.label,
        "formulation": ref.formulation,
        "paper_role": ref.paper_role,
        "path": rel(path, root),
        "exists": path.exists(),
        "accuracy": accuracy_value,
        "eval_examples": example_count,
        "expected_examples": EXPECTED_EVAL_EXAMPLES[task],
        "full_eval_examples": example_count == EXPECTED_EVAL_EXAMPLES[task],
        "recorded_task_format": recorded_format,
        "recorded_label_scheme": recorded_label_scheme,
    }


def add_paper_anchor(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchored = []
    for row in rows:
        anchor = None
        if row["task"] == "mnli" and row["formulation"] == "sequence_classification":
            if row["run"] == "FP16-SFT":
                anchor = PAPER_QWEN25_MNLI["fp16_sft"]
            elif row["run"] == "BitNet-SFT":
                anchor = PAPER_QWEN25_MNLI["bitnet_sft"]
            elif row["run"].startswith("BitDistill"):
                anchor = PAPER_QWEN25_MNLI["bitdistill"]
        row = dict(row)
        row["paper_qwen25_mnli_anchor"] = anchor
        row["local_minus_anchor"] = (
            row["accuracy"] - anchor if isinstance(row["accuracy"], float) and isinstance(anchor, float) else None
        )
        anchored.append(row)
    return anchored


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    rows = add_paper_anchor([metric_row(root, args.model, ref, task) for ref in RUNS for task in args.tasks])
    seq_rows = [row for row in rows if row["formulation"] == "sequence_classification"]
    causal_rows = [row for row in rows if row["formulation"].startswith("causal_lm")]
    pending_paper_candidates = [
        row
        for row in seq_rows
        if row["paper_role"].startswith("pending") and not (row["exists"] and row["full_eval_examples"])
    ]
    full_seq_baselines = [
        row
        for row in seq_rows
        if row["paper_role"] == "baseline" and row["exists"] and row["full_eval_examples"]
    ]
    return {
        "schema": "bitdistill-task-formulation-audit-v1",
        "date": DATE,
        "model": args.model,
        "tasks": list(args.tasks),
        "paper_anchor_source": "BitDistill paper excerpt, Table 3 Qwen2.5-0.5B MNLI row.",
        "paper_anchor_qwen25_mnli": PAPER_QWEN25_MNLI,
        "claim_controls": [
            "Current strict reproduction claim should be limited to the sequence-classification branch until paper training code confirms a different task head/prompt formulation.",
            "Causal-LM GLUE rows are diagnostics for deployment-style prompting and should not be mixed with sequence-classification rows in one headline accuracy table.",
            "The provided excerpt only gives Qwen2.5-0.5B anchors for MNLI; QNLI/SST2 local Qwen2.5 rows are reproduction targets by task, not direct table-value reproductions.",
            "BitDistill success remains pending until long-warmup tensor/row candidates finish full validation.",
        ],
        "sequence_baselines_full": len(full_seq_baselines),
        "causal_rows_materialized": sum(1 for row in causal_rows if row["exists"]),
        "pending_paper_candidates": len(pending_paper_candidates),
        "rows": rows,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    control_rows = [[item] for item in summary["claim_controls"]]
    rows = [
        [
            row["task"],
            row["run"],
            row["formulation"],
            row["paper_role"],
            fmt(row["exists"]),
            fmt(row["accuracy"]),
            fmt(row["eval_examples"]),
            fmt(row["full_eval_examples"]),
            fmt(row["paper_qwen25_mnli_anchor"]),
            fmt(row["local_minus_anchor"]),
            row["path"],
        ]
        for row in summary["rows"]
    ]
    return "\n\n".join(
        [
            f"# BitDistill Task Formulation Audit, {summary['date']}",
            f"Model: `{summary['model']}`.",
            f"Paper anchor source: {summary['paper_anchor_source']}",
            "This audit prevents sequence-classification, causal prompt scoring, and paper table anchors from being mixed into one overbroad claim.",
            f"Sequence-classification full baselines: `{summary['sequence_baselines_full']}`. "
            f"Causal diagnostic rows materialized: `{summary['causal_rows_materialized']}`. "
            f"Pending paper candidates: `{summary['pending_paper_candidates']}`.",
            "## Claim Controls",
            md_table(["control"], control_rows),
            "## Rows",
            md_table(
                [
                    "task",
                    "run",
                    "formulation",
                    "paper role",
                    "exists",
                    "accuracy",
                    "eval n",
                    "full eval",
                    "paper Qwen2.5 MNLI anchor",
                    "local-anchor",
                    "metrics path",
                ],
                rows,
            ),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_task_formulation_audit_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_task_formulation_audit_{DATE}.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    summary = build_summary(args)
    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
