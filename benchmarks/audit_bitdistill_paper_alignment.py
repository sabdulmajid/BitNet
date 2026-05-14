#!/usr/bin/env python3
"""Audit local BitDistill experiments against the paper recipe.

This is a claim-control report.  It does not decide quality by itself; it
records whether the local runs are actually comparable to the BitDistill paper
setup before a result is described as a reproduction.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = datetime.now(timezone.utc).date().isoformat()
TASKS = ["mnli", "qnli", "sst2"]
PAPER_WARMUP_TOKENS = 10_000_000_000


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def accuracy(path: Path) -> float | None:
    data = read_json(path)
    eval_metrics = data.get("eval", {}) if isinstance(data.get("eval"), dict) else {}
    value = eval_metrics.get("accuracy")
    return float(value) if isinstance(value, (int, float)) else None


def metric_path(root: Path, model: str, task: str, run: str) -> Path:
    return root / model.replace("/", "-") / task / run / "metrics.json"


def code_features(root: Path) -> dict[str, bool]:
    train = read_text(root / "train_bitdistill.py")
    submit = read_text(root / "benchmarks" / "submit_bitdistill_longwarmup_downstream.sh")
    gate = read_text(root / "benchmarks" / "gate_bitdistill_reproduction.py")
    return {
        "subln_wrapper": "class SubLNLinear" in train,
        "subln_o_proj": 'hasattr(self_attn, "o_proj")' in train,
        "subln_down_proj": 'hasattr(mlp, "down_proj")' in train,
        "continued_pretrain_stage": '"continued_pretrain"' in train and "train_continued_pretrain" in train,
        "sequence_classification_stage": '"sequence_classification"' in train,
        "logits_kd": "def logits_kd_loss" in train,
        "paper_logit_temperature_scale_default": 'choices=["none", "square"], default="none"' in train,
        "attention_relation_kd": "def attention_relation_distillation_loss" in train,
        "attention_relation_l2_normalization": "F.normalize(states, dim=-1)" in train and "relation = torch.matmul(states" in train,
        "single_layer_selection": "--distill-layer" in train,
        "row_scale_mode": "--scale-mode" in train and "row" in train,
        "longwarmup_submitter": "WARMUP_STATE" in submit and "INIT_STATE_DICT" in submit,
        "strict_paper_gamma_gate": "paper_hparam_candidate" in gate and "paper_hparam_root" in gate,
    }


def run_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_specs = [
        ("FP16-SFT", args.baseline_root, "fp16_sft-tensor-layer-1", "baseline", None),
        ("BitNet-SFT", args.baseline_root, "bitnet_sft-tensor-layer-1", "baseline", None),
        ("BitDistill short tensor gamma100", args.baseline_root, "bitdistill-tensor-layer-1", "diagnostic", None),
        ("BitDistill short row gamma100", args.baseline_root, "bitdistill-row-layer-1", "diagnostic", None),
        ("BitDistill short tensor layer -8", args.baseline_root, "bitdistill-tensor-layer-8", "diagnostic", None),
        ("BitDistill longwarmup tensor gamma100", args.longwarmup_root, "bitdistill-longwarmup-tensor-layer-8", "diagnostic_pending", None),
        ("BitDistill longwarmup row gamma100", args.longwarmup_root, "bitdistill-longwarmup-row-layer-8", "novelty_pending", None),
        ("BitDistill longwarmup tensor paper gamma", args.paper_hparam_root, "bitdistill-longwarmup-tensor-layer-8", "paper_candidate", None),
        ("BitDistill longwarmup tensor gamma1k", args.gamma1k_root, "bitdistill-longwarmup-tensor-layer-8", "mnli_gamma_sweep_pending", {"mnli"}),
        ("BitDistill longwarmup tensor gamma10k", args.gamma10k_root, "bitdistill-longwarmup-tensor-layer-8", "mnli_gamma_sweep_pending", {"mnli"}),
    ]
    fp_by_task = {
        task: accuracy(metric_path(args.baseline_root, args.model, task, "fp16_sft-tensor-layer-1"))
        for task in args.tasks
    }
    for task in args.tasks:
        for label, root, run, family, task_filter in run_specs:
            if task_filter is not None and task not in task_filter:
                continue
            path = metric_path(root, args.model, task, run)
            acc = accuracy(path)
            fp = fp_by_task.get(task)
            rows.append(
                {
                    "task": task,
                    "run": label,
                    "family": family,
                    "exists": path.exists(),
                    "accuracy": acc,
                    "fp16_accuracy": fp,
                    "fp_minus_run": (fp - acc) if fp is not None and acc is not None else None,
                    "metrics_path": str(path),
                }
            )
    return rows


def warmup_status(args: argparse.Namespace) -> dict[str, Any]:
    monitor = read_json(args.monitor_json)
    warmup = monitor.get("warmup", {}) if isinstance(monitor.get("warmup"), dict) else {}
    target = warmup.get("target_token_presentations")
    effective = warmup.get("effective_token_presentations")
    return {
        "paper_tokens": PAPER_WARMUP_TOKENS,
        "active_target_tokens": target,
        "active_effective_tokens": effective,
        "active_target_vs_paper": (float(target) / PAPER_WARMUP_TOKENS) if isinstance(target, (int, float)) else None,
        "active_effective_vs_paper": (float(effective) / PAPER_WARMUP_TOKENS) if isinstance(effective, (int, float)) else None,
        "latest_step": (warmup.get("latest_step") or {}).get("step") if isinstance(warmup.get("latest_step"), dict) else None,
        "max_steps": warmup.get("max_steps"),
        "eta_seconds": warmup.get("eta_seconds"),
    }


def alignment_rows(features: dict[str, bool], warmup: dict[str, Any]) -> list[dict[str, str]]:
    rows = [
        {
            "dimension": "Backbone",
            "paper": "Qwen3 0.6B/1.7B/4B primary, plus Qwen2.5/Gemma robustness",
            "local": "Qwen2.5-0.5B",
            "status": "partial",
            "note": "Useful robustness-style target, but not the paper's primary Qwen3 scale ladder.",
        },
        {
            "dimension": "Tasks",
            "paper": "MNLI, QNLI, SST2 first for classification",
            "local": "MNLI, QNLI, SST2",
            "status": "matched",
            "note": "Task set is aligned.",
        },
        {
            "dimension": "Baselines",
            "paper": "FP16-SFT, BitNet-SFT, BitDistill",
            "local": "All three exist for short-budget GLUE3; long-warmup BitDistill is pending",
            "status": "partial",
            "note": "The final paper candidate is not complete until long-warmup downstream metrics exist.",
        },
        {
            "dimension": "Stage-1 SubLN",
            "paper": "SubLN before attention output projection and FFN down projection",
            "local": "Implemented" if features["subln_wrapper"] and features["subln_o_proj"] and features["subln_down_proj"] else "Missing",
            "status": "matched" if features["subln_wrapper"] and features["subln_o_proj"] and features["subln_down_proj"] else "missing",
            "note": "Implemented as RMSNorm wrappers around Qwen `o_proj` and `down_proj`.",
        },
        {
            "dimension": "Stage-2 warm-up",
            "paper": "10B-token continued pretraining",
            "local": f"active target {warmup.get('active_target_tokens')} token presentations",
            "status": "partial",
            "note": f"Current target is {warmup.get('active_target_vs_paper')} of the paper token budget.",
        },
        {
            "dimension": "Stage-3 logits KD",
            "paper": "temperature 5, lambda 10",
            "local": "temperature 5, weight 10, no tau^2 scaling by default",
            "status": "matched" if features["logits_kd"] and features["paper_logit_temperature_scale_default"] else "partial",
            "note": "First completed wave used tau^2; current code and pending runs use paper-style scaling.",
        },
        {
            "dimension": "Stage-3 attention KD",
            "paper": "single-layer Q/K/V relation KD, gamma 1e5 for classification",
            "local": "single-layer L2-normalized Q/K/V relation KD implemented; completed runs gamma 100; long-warmup gamma 1e3/1e4/1e5 branches pending",
            "status": "pending",
            "note": "The gamma sweep is intentional because local loss-scale probes show the paper gamma can dominate CE.",
        },
        {
            "dimension": "Hyperparameter search",
            "paper": "greedy search over learning rate and epochs",
            "local": "fixed 1000-step downstream schedule plus selected diagnostics",
            "status": "partial",
            "note": "A strict reproduction needs at least a small LR/epoch search after the long warm-up.",
        },
        {
            "dimension": "Hardware/resources",
            "paper": "8x AMD MI300X training, CPU throughput with 16 threads",
            "local": "single-GPU Slurm jobs; Xeon CPU runtime for local inference",
            "status": "partial",
            "note": "Resource gap affects training budget and wall-clock, not the mathematical objective.",
        },
    ]
    return rows


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
    alignment = [
        [row["dimension"], row["paper"], row["local"], row["status"], row["note"]]
        for row in summary["alignment"]
    ]
    metrics = [
        [
            row["task"],
            row["run"],
            row["family"],
            fmt(row["exists"]),
            fmt(row["accuracy"]),
            fmt(row["fp16_accuracy"]),
            fmt(row["fp_minus_run"]),
        ]
        for row in summary["runs"]
    ]
    features = [[key, "pass" if value else "fail"] for key, value in summary["code_features"].items()]
    warmup = summary["warmup"]
    warmup_rows = [
        ["paper warm-up tokens", fmt(warmup["paper_tokens"])],
        ["active target tokens", fmt(warmup["active_target_tokens"])],
        ["active target / paper", fmt(warmup["active_target_vs_paper"])],
        ["active effective tokens", fmt(warmup["active_effective_tokens"])],
        ["active effective / paper", fmt(warmup["active_effective_vs_paper"])],
        ["latest step", fmt(warmup["latest_step"])],
        ["max steps", fmt(warmup["max_steps"])],
    ]
    return "\n\n".join(
        [
            f"# BitDistill Paper Alignment Audit, {summary['date']}",
            "Verdict: local code contains the major BitDistill mechanisms, but the completed results are not a strict paper reproduction. The strict paper-hyperparameter branch is queued/pending.",
            "## Alignment",
            md_table(["dimension", "paper", "local", "status", "note"], alignment),
            "## Warm-Up Budget",
            md_table(["field", "value"], warmup_rows),
            "## Current Accuracy Matrix",
            md_table(["task", "run", "family", "exists", "accuracy", "FP16", "FP-run"], metrics),
            "## Code Feature Checks",
            md_table(["feature", "status"], features),
            "## Interpretation",
            "\n".join(
                [
                    "- The existing negative GLUE result is a valid short-budget boundary result.",
                    "- It is not a disproof of BitDistill because warm-up budget, attention-KD gamma, backbone scale, and search are not paper-matched yet.",
                    "- The publishable angle remains independent reproduction plus a row-scale CPU-runtime extension if the strict branch closes the quality gap; otherwise the publishable angle becomes a resource-sensitivity and boundary study.",
                ]
            ),
        ]
    ) + "\n"


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    features = code_features(root)
    warmup = warmup_status(args)
    return {
        "schema": "bitdistill-paper-alignment-v1",
        "date": DATE,
        "model": args.model,
        "tasks": args.tasks,
        "code_features": features,
        "warmup": warmup,
        "alignment": alignment_rows(features, warmup),
        "runs": run_matrix(args),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--baseline-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls"))
    parser.add_argument("--longwarmup-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup"))
    parser.add_argument("--paper-hparam-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma"))
    parser.add_argument("--gamma1k-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-gamma1k"))
    parser.add_argument("--gamma10k-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-gamma10k"))
    parser.add_argument("--monitor-json", type=Path, default=Path(f"benchmark_results/bitdistill_job_monitor_{DATE}.json"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_paper_alignment_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_paper_alignment_{DATE}.md"))
    args = parser.parse_args()

    if not args.monitor_json.is_absolute():
        args.monitor_json = args.repo_root / args.monitor_json
    summary = build_summary(args)
    output_json = args.output_json if args.output_json.is_absolute() else args.repo_root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else args.repo_root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
