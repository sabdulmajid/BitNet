#!/usr/bin/env python3
"""Audit the completed BitDistill Stage-2 warm-up budget evidence.

This is a diagnostic, not a paper-reproduction claim. The completed local
Qwen2.5-0.5B rows do not form a perfectly controlled token-budget curve because
some downstream recipe details changed between runs. The point of this report
is to prevent a false conclusion: the current local evidence is too small and
too confounded to disprove BitDistill, but it is enough to show that the local
short/medium warm-up budget has not recovered FP16-level MNLI quality.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
PAPER_STAGE2_TOKENS = 10_000_000_000
DEFAULT_TOKENS_PER_STEP = 8192


@dataclass(frozen=True)
class RunSpec:
    label: str
    family: str
    metrics_path: str
    note: str = ""
    warmup_token_presentations: int | None = None


RUNS = [
    RunSpec(
        "FP16-SFT reference",
        "reference",
        "checkpoints/bitdistill-glue-seqcls-predtrace/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1/metrics.json",
        "Dense FP16 task model; not a ternary student.",
    ),
    RunSpec(
        "BitNet-SFT default",
        "ce_only",
        "checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-layer-1/metrics.json",
        "Short CE-only baseline; undertrained relative to later budget rows.",
    ),
    RunSpec(
        "BitNet-SFT best completed budget",
        "ce_only",
        "checkpoints/bitdistill-glue-seqcls-bitnet-sft-budget/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-steps10000-lr2em5/metrics.json",
        "CE-only tensor-scale row that clears the paper BitNet-SFT anchor but remains far below FP16.",
    ),
    RunSpec(
        "BitDistill tensor short warm-up",
        "bitdistill_tensor",
        "checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-layer-8/metrics.json",
        "Uses the older 5k-step Stage-2 checkpoint; attention Q/K/V reduction predates the later explicit sum setting.",
    ),
    RunSpec(
        "BitDistill tensor 20k warm-up",
        "bitdistill_tensor",
        "checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json",
        "Uses 20k-step tensor Stage-2 warm-up and explicit Q/K/V-sum attention relation loss.",
    ),
    RunSpec(
        "BitDistill tensor 20k paper-gamma",
        "bitdistill_tensor",
        "checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json",
        "Same tensor warm-up family with gamma=100000 under the local normalization.",
    ),
    RunSpec(
        "BitDistill row downstream, tensor warm-up",
        "retrofit_variant",
        "checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json",
        "Row-scale downstream retrofit variant loaded from the tensor Stage-2 checkpoint.",
    ),
    RunSpec(
        "BitDistill row downstream, row warm-up",
        "retrofit_variant",
        "checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json",
        "Row-scale downstream variant loaded from the row-scale Stage-2 checkpoint.",
    ),
    RunSpec(
        "Controlled recovery, 5k warm-up",
        "controlled_curve",
        "checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-5kwarmup-steps10000-lr2em5-papergamma-headinit/metrics.json",
        "Queued fixed-recipe control: tensor scale, layer -1, gamma 100000, head init, 10000 downstream steps.",
        40_960_000,
    ),
    RunSpec(
        "Controlled recovery, 20k warm-up",
        "controlled_curve",
        "checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-steps10000-lr2em5-papergamma-headinit/metrics.json",
        "Queued fixed-recipe control: tensor scale, layer -1, gamma 100000, head init, 10000 downstream steps.",
        163_840_000,
    ),
    RunSpec(
        "Controlled recovery, 40k warm-up",
        "controlled_curve",
        "checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-40kwarmup-steps10000-lr2em5-papergamma-headinit/metrics.json",
        "Queued fixed-recipe control: tensor scale, layer -1, gamma 100000, head init, 10000 downstream steps.",
        327_680_000,
    ),
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if 0.0 < abs(value) < 0.0001:
            return f"{value:.6e}"
        return f"{value:.6f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(fmt(cell).replace("|", "\\|") for cell in row) + " |")
    return "\n".join(out)


def warmup_metrics_path(metrics: dict[str, Any]) -> Path | None:
    state_path = nested(metrics, "state_load", "path")
    if not isinstance(state_path, str) or not state_path:
        return None
    path = Path(state_path)
    if path.name != "custom_state_dict.pt":
        return path.parent / "metrics.json"
    return path.with_name("metrics.json")


def infer_warmup_tokens(warmup_metrics: dict[str, Any]) -> tuple[float | None, str]:
    explicit = finite(warmup_metrics.get("effective_train_token_presentations"))
    if explicit is not None:
        return explicit, "recorded"
    steps = finite(warmup_metrics.get("steps"))
    if steps is not None and steps > 0:
        return steps * DEFAULT_TOKENS_PER_STEP, f"inferred_steps_x_{DEFAULT_TOKENS_PER_STEP}"
    return None, "none"


def summarize_run(root: Path, spec: RunSpec) -> dict[str, Any]:
    path = root / spec.metrics_path
    metrics = read_json(path)
    warmup_path = None if spec.family in {"reference", "ce_only"} else warmup_metrics_path(metrics)
    warmup = read_json(root / warmup_path) if warmup_path is not None and not warmup_path.is_absolute() else read_json(warmup_path) if warmup_path is not None else {}
    warmup_tokens, token_source = infer_warmup_tokens(warmup)
    if warmup_tokens is None and spec.warmup_token_presentations is not None:
        warmup_tokens = float(spec.warmup_token_presentations)
        token_source = "expected_from_submission"
    has_kd = spec.family in {"bitdistill_tensor", "retrofit_variant"}
    accuracy = finite(nested(metrics, "eval", "accuracy"))
    ce = finite(nested(metrics, "last", "ce"))
    weighted_logit = finite(nested(metrics, "last", "weighted_logit_kd")) if has_kd else None
    weighted_attention = finite(nested(metrics, "last", "weighted_attention_kd")) if has_kd else None
    return {
        "label": spec.label,
        "family": spec.family,
        "note": spec.note,
        "metrics_path": spec.metrics_path,
        "exists": path.exists(),
        "accuracy": accuracy,
        "eval_examples": finite(nested(metrics, "eval", "eval_examples")),
        "method": metrics.get("method"),
        "stage": metrics.get("stage"),
        "task_format": metrics.get("task_format"),
        "scale_mode": metrics.get("scale_mode"),
        "steps": metrics.get("steps"),
        "distill_layer": metrics.get("distill_layer"),
        "attention_kd_weight": finite(nested(metrics, "loss_weights", "attention_kd_weight")) if has_kd else None,
        "attention_qkv_reduction": nested(metrics, "loss_weights", "attention_qkv_reduction", default="legacy_or_absent") if has_kd else "",
        "output_head_copied": nested(metrics, "output_head_init", "copied"),
        "warmup_metrics_path": str(warmup_path) if warmup_path is not None else "",
        "warmup_steps": warmup.get("steps"),
        "warmup_scale_mode": warmup.get("scale_mode"),
        "warmup_tokens": warmup_tokens,
        "warmup_token_source": token_source,
        "paper_stage2_fraction": warmup_tokens / PAPER_STAGE2_TOKENS if warmup_tokens is not None else None,
        "last_ce": ce,
        "weighted_logit_kd": weighted_logit,
        "weighted_attention_kd": weighted_attention,
        "weighted_logit_to_ce": weighted_logit / ce if weighted_logit is not None and ce not in (None, 0.0) else None,
        "weighted_attention_to_ce": weighted_attention / ce if weighted_attention is not None and ce not in (None, 0.0) else None,
    }


def best_by_family(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        acc = row.get("accuracy")
        if not isinstance(acc, float):
            continue
        family = str(row["family"])
        previous = out.get(family)
        if previous is None or acc > previous["accuracy"]:
            out[family] = row
    return out


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    rows = [summarize_run(root, spec) for spec in RUNS]
    fp = next((row for row in rows if row["family"] == "reference" and isinstance(row.get("accuracy"), float)), None)
    fp_acc = fp["accuracy"] if fp else None
    for row in rows:
        acc = row.get("accuracy")
        row["delta_vs_fp16"] = acc - fp_acc if isinstance(acc, float) and isinstance(fp_acc, float) else None
    best = best_by_family(rows)
    tensor_rows = [row for row in rows if row["family"] == "bitdistill_tensor" and isinstance(row.get("accuracy"), float)]
    tensor_rows.sort(key=lambda row: (row.get("warmup_tokens") or -1, row["accuracy"]))
    tensor_delta = None
    if len(tensor_rows) >= 2:
        tensor_delta = tensor_rows[-1]["accuracy"] - tensor_rows[0]["accuracy"]
    return {
        "schema": "bitdistill-stage2-curve-audit-v1",
        "date": DATE,
        "paper_stage2_tokens": PAPER_STAGE2_TOKENS,
        "rows": rows,
        "best_by_family": best,
        "fp16_accuracy": fp_acc,
        "best_tensor_accuracy": best.get("bitdistill_tensor", {}).get("accuracy"),
        "best_retrofit_variant_accuracy": best.get("retrofit_variant", {}).get("accuracy"),
        "best_tensor_delta_vs_fp16": best.get("bitdistill_tensor", {}).get("delta_vs_fp16"),
        "best_retrofit_variant_delta_vs_fp16": best.get("retrofit_variant", {}).get("delta_vs_fp16"),
        "diagnostic_tensor_short_to_long_delta": tensor_delta,
        "controlled_curve": False,
        "confounders": [
            "The short and 20k tensor warm-up rows changed attention Q/K/V reduction reporting/semantics.",
            "The row-scale rows are retrofit variants and are not paper-style tensor-scale BitDistill reproduction rows.",
            "The largest completed local Stage-2 budget is 163.84M token presentations, only 1.6384% of the paper's 10B-token warm-up.",
            "Downstream Stage-3 budgets also differ from the paper's epoch/LR search.",
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    rows = summary["rows"]
    table_rows = [
        [
            row["label"],
            row["family"],
            row["accuracy"],
            row["delta_vs_fp16"],
            row["warmup_tokens"],
            row["paper_stage2_fraction"],
            row["warmup_token_source"],
            row["scale_mode"],
            row["attention_kd_weight"],
            row["attention_qkv_reduction"],
            row["weighted_attention_to_ce"],
            row["note"],
        ]
        for row in rows
    ]
    verdict = (
        "This is not a controlled proof that Stage-2 tokens alone explain the gap. "
        "It does show that moving from the older short warm-up row to the completed "
        "20k tensor warm-up family coincides with a large MNLI gain, while the best "
        "completed tensor BitDistill row remains far below FP16."
    )
    best_tensor = summary.get("best_tensor_accuracy")
    best_tensor_delta = summary.get("best_tensor_delta_vs_fp16")
    best_row = summary.get("best_retrofit_variant_accuracy")
    best_row_delta = summary.get("best_retrofit_variant_delta_vs_fp16")
    lines = [
        f"# BitDistill Stage-2 Budget Curve Audit, {summary['date']}",
        verdict,
        "## Summary",
        md_table(
            ["metric", "value"],
            [
                ["paper Stage-2 warm-up tokens", summary["paper_stage2_tokens"]],
                ["best tensor BitDistill MNLI", best_tensor],
                ["best tensor delta vs FP16", best_tensor_delta],
                ["best row-scale retrofit MNLI", best_row],
                ["best row-scale delta vs FP16", best_row_delta],
                ["diagnostic tensor short-to-long delta", summary["diagnostic_tensor_short_to_long_delta"]],
                ["controlled token curve", summary["controlled_curve"]],
            ],
        ),
        "## Rows",
        md_table(
            [
                "run",
                "family",
                "MNLI acc",
                "delta vs FP16",
                "Stage-2 tokens",
                "paper fraction",
                "token source",
                "scale",
                "gamma",
                "QKV reduction",
                "weighted AD / CE",
                "note",
            ],
            table_rows,
        ),
        "## Confounders",
        "\n".join(f"- {item}" for item in summary["confounders"]),
        "## Interpretation",
        (
            "The correct next experiment is a fixed-recipe Stage-2 budget curve "
            "rather than more broad ablations: keep Qwen2.5-0.5B, MNLI, tensor-scale "
            "sequence classification, SubLN policy, dense head policy, attention layer, "
            "loss normalization, LR, and downstream steps fixed while varying only the "
            "continued-pretraining token budget."
        ),
        "",
    ]
    return "\n\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_stage2_curve_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_stage2_curve_{DATE}.md"))
    args = parser.parse_args()

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
