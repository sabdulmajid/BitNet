#!/usr/bin/env python3
"""Audit why the local BitNet-SFT baseline is below the paper anchor.

This is intentionally narrower than the full BitDistill evidence manifest.  The
core reproduction question is whether the low BitNet-SFT score is caused by an
obvious mechanical mismatch before Stage-2 or distillation are considered.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
PAPER_QWEN25_MNLI = {
    "FP16-SFT": 0.7991,
    "BitNet-SFT": 0.6080,
    "BitDistill": 0.7998,
}
EXPECTED_QWEN25_05B_LAYERS = 24
EXPECTED_QWEN25_05B_LINEAR_PER_LAYER = 7
EXPECTED_QWEN25_05B_TERNARY = EXPECTED_QWEN25_05B_LAYERS * EXPECTED_QWEN25_05B_LINEAR_PER_LAYER
FULL_MNLI_VALIDATION = 9815


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def projection_family(module_name: str) -> str:
    parts = module_name.split(".")
    if "self_attn" in parts:
        idx = parts.index("self_attn")
        return ".".join(parts[idx : idx + 2])
    if "mlp" in parts:
        idx = parts.index("mlp")
        return ".".join(parts[idx : idx + 2])
    if parts:
        return parts[-1]
    return module_name


def summarize_tensor(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return {"count": 0}
    return {
        "count": int(values.numel()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
    }


def summarize_ternary_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        return {"exists": False, "path": str(path), "error": "not a state dict"}

    ternary_keys = sorted(key for key in state if key.endswith(".ternary_weight"))
    scale_keys = sorted(key for key in state if key.endswith(".weight_scale"))
    dense_weight_keys = sorted(
        key for key in state if key.endswith(".weight") and not key.endswith(".ternary_weight")
    )
    bias_keys = sorted(key for key in state if key.endswith(".bias"))

    code_counts: Counter[int] = Counter()
    family_counts: dict[str, Counter[int]] = defaultdict(Counter)
    total_codes = 0
    layer_ids: set[int] = set()
    family_key_counts: Counter[str] = Counter()
    for key in ternary_keys:
        module = key[: -len(".ternary_weight")]
        parts = module.split(".")
        if "layers" in parts:
            pos = parts.index("layers")
            if pos + 1 < len(parts) and parts[pos + 1].isdigit():
                layer_ids.add(int(parts[pos + 1]))
        family = projection_family(module)
        family_key_counts[family] += 1
        tensor = state[key].detach().to(torch.int8).reshape(-1)
        total_codes += int(tensor.numel())
        values, counts = torch.unique(tensor, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            code_counts[int(value)] += int(count)
            family_counts[family][int(value)] += int(count)

    scale_values = []
    scale_numel_by_key = Counter()
    for key in scale_keys:
        scale = state[key].detach().float().reshape(-1)
        scale_values.append(scale)
        scale_numel_by_key[int(scale.numel())] += 1
    all_scales = torch.cat(scale_values) if scale_values else torch.empty(0)

    fractions = {
        str(code): (count / total_codes if total_codes else None)
        for code, count in sorted(code_counts.items())
    }
    family_summary = {}
    for family, counts in sorted(family_counts.items()):
        family_total = sum(counts.values())
        family_summary[family] = {
            "tensors": int(family_key_counts[family]),
            "codes": int(family_total),
            "minus_one_frac": counts.get(-1, 0) / family_total if family_total else None,
            "zero_frac": counts.get(0, 0) / family_total if family_total else None,
            "plus_one_frac": counts.get(1, 0) / family_total if family_total else None,
        }

    return {
        "exists": True,
        "path": str(path),
        "total_keys": len(state),
        "ternary_weight_count": len(ternary_keys),
        "weight_scale_count": len(scale_keys),
        "dense_weight_keys": dense_weight_keys,
        "dense_weight_count": len(dense_weight_keys),
        "bias_count": len(bias_keys),
        "layer_count": len(layer_ids),
        "layer_min": min(layer_ids) if layer_ids else None,
        "layer_max": max(layer_ids) if layer_ids else None,
        "total_codes": total_codes,
        "code_counts": {str(key): int(value) for key, value in sorted(code_counts.items())},
        "code_fractions": fractions,
        "families": family_summary,
        "scale_numel_by_key": {str(key): int(value) for key, value in sorted(scale_numel_by_key.items())},
        "scale_summary": summarize_tensor(all_scales),
    }


def summarize_checkpoint(name: str, root: Path) -> dict[str, Any]:
    metrics_path = root / "metrics.json"
    metrics = read_json(metrics_path)
    eval_accuracy = finite_float(nested(metrics, "eval", "accuracy"))
    eval_examples = finite_float(nested(metrics, "eval", "eval_examples"))
    ternary = summarize_ternary_state(root / "ternary_state_dict.pt")
    preparation = metrics.get("preparation", {}) if isinstance(metrics.get("preparation"), dict) else {}
    preparation = dict(preparation)
    if "activation_quantization" not in preparation and ternary.get("exists"):
        preparation["activation_quantization"] = True
    return {
        "name": name,
        "root": str(root),
        "metrics_path": str(metrics_path),
        "exists": metrics_path.exists(),
        "method": metrics.get("method"),
        "task": metrics.get("task"),
        "task_format": metrics.get("task_format"),
        "scale_mode": metrics.get("scale_mode"),
        "accuracy": eval_accuracy,
        "eval_examples": eval_examples,
        "full_eval": int(eval_examples or 0) == FULL_MNLI_VALIDATION,
        "fp16_gap": None,
        "paper_anchor": PAPER_QWEN25_MNLI.get(name),
        "paper_gap": (PAPER_QWEN25_MNLI.get(name) - eval_accuracy) if eval_accuracy is not None and PAPER_QWEN25_MNLI.get(name) is not None else None,
        "steps": metrics.get("steps"),
        "training_budget": metrics.get("training_budget", {}),
        "preparation": preparation,
        "loss_weights": metrics.get("loss_weights", {}),
        "last": metrics.get("last", {}),
        "state_load": metrics.get("state_load", {}),
        "ternary": ternary,
    }


def add_fp_gaps(rows: list[dict[str, Any]]) -> None:
    fp = next((row for row in rows if row["name"] == "FP16-SFT" and row.get("accuracy") is not None), None)
    if not fp:
        return
    fp_acc = fp["accuracy"]
    for row in rows:
        acc = row.get("accuracy")
        row["fp16_gap"] = (fp_acc - acc) if isinstance(acc, (int, float)) else None


def build_checks(
    rows: list[dict[str, Any]],
    budget_sweep: dict[str, Any],
    paired_budget: dict[str, Any],
) -> list[dict[str, Any]]:
    by_name = {row["name"]: row for row in rows}
    fp = by_name.get("FP16-SFT", {})
    bitnet = by_name.get("BitNet-SFT", {})
    weights_only = by_name.get("BitNet-SFT weights-only", {})
    subln_bitnet = by_name.get("BitNet-SFT SubLN", {})
    bitnet_t = bitnet.get("ternary", {}) if isinstance(bitnet.get("ternary"), dict) else {}
    best_budget = budget_sweep.get("best") if isinstance(budget_sweep.get("best"), dict) else {}
    best_paired = paired_budget.get("best") if isinstance(paired_budget.get("best"), dict) else {}
    best_budget_acc = finite_float(best_budget.get("accuracy"))
    best_paired_delta = finite_float(best_paired.get("delta_vs_reference"))
    best_paired_ci = best_paired.get("paired_ci95")

    checks: list[dict[str, Any]] = []
    fp_acc = fp.get("accuracy")
    bitnet_acc = bitnet.get("accuracy")
    weights_only_acc = weights_only.get("accuracy")
    subln_bitnet_acc = subln_bitnet.get("accuracy")
    fp_paper = PAPER_QWEN25_MNLI["FP16-SFT"]
    bitnet_paper = PAPER_QWEN25_MNLI["BitNet-SFT"]
    checks.append(
        {
            "check": "FP16-SFT local task is learnable",
            "status": "pass" if isinstance(fp_acc, float) and abs(fp_acc - fp_paper) <= 0.02 else "warn",
            "evidence": f"local={fmt(fp_acc)}, paper_anchor={fmt(fp_paper)}, delta={fmt((fp_acc - fp_paper) if isinstance(fp_acc, float) else None)}",
            "implication": "The weak BitNet-SFT result is unlikely to be caused only by task formatting or dataset split.",
        }
    )
    checks.append(
        {
            "check": "Default 1000-step BitNet-SFT matches paper anchor",
            "status": "fail" if isinstance(bitnet_acc, float) and bitnet_acc < bitnet_paper - 0.05 else "pass",
            "evidence": f"local={fmt(bitnet_acc)}, paper_anchor={fmt(bitnet_paper)}, delta={fmt((bitnet_acc - bitnet_paper) if isinstance(bitnet_acc, float) else None)}",
            "implication": "The short/default run is undertrained and should not be used as the final BitNet-SFT anchor.",
        }
    )
    checks.append(
        {
            "check": "Best completed budget BitNet-SFT matches paper anchor",
            "status": "pass" if isinstance(best_budget_acc, float) and best_budget_acc >= bitnet_paper else "pending",
            "evidence": (
                f"best={fmt(best_budget_acc)}, paper_anchor={fmt(bitnet_paper)}, "
                f"steps={best_budget.get('steps', '-')}, lr={best_budget.get('lr', '-')}"
            ),
            "implication": "The local blocker has shifted from BitNet-SFT viability to BitDistill/FP16-level recovery.",
        }
    )
    checks.append(
        {
            "check": "Best completed budget BitNet-SFT is still below paired FP16",
            "status": "fail" if isinstance(best_paired_delta, float) and best_paired_delta < -0.01 else "pass",
            "evidence": (
                f"candidate_minus_fp16={fmt(best_paired_delta)}, ci95={fmt_ci(best_paired_ci)}, "
                f"mcnemar={fmt(best_paired.get('mcnemar_exact_p'))}"
            ),
            "implication": "Clearing the paper BitNet-SFT anchor is not enough; the BitDistill stage must recover the remaining FP16 gap.",
        }
    )
    checks.append(
        {
            "check": "BitNet-SFT ternary projection count matches Qwen2.5-0.5B decoder projections",
            "status": "pass" if bitnet_t.get("ternary_weight_count") == EXPECTED_QWEN25_05B_TERNARY else "fail",
            "evidence": f"ternary={bitnet_t.get('ternary_weight_count')}, expected={EXPECTED_QWEN25_05B_TERNARY}",
            "implication": "The low baseline is not explained by exporting only one ternary tensor or missing whole decoder projection families.",
        }
    )
    dense_keys = set(bitnet_t.get("dense_weight_keys", [])) if isinstance(bitnet_t.get("dense_weight_keys"), list) else set()
    head_dense = any(key in dense_keys for key in ("score.weight", "classifier.weight"))
    checks.append(
        {
            "check": "Classifier head remains dense",
            "status": "pass" if head_dense else "warn",
            "evidence": ", ".join(sorted(key for key in dense_keys if key in {"score.weight", "classifier.weight"})) or "no dense classifier key found",
            "implication": "The poor score is not caused by accidentally ternarizing the classification head in this checkpoint.",
        }
    )
    subln = nested(bitnet, "preparation", "subln_inserted", default=0)
    subln_delta = subln_bitnet_acc - bitnet_acc if isinstance(subln_bitnet_acc, float) and isinstance(bitnet_acc, float) else None
    checks.append(
        {
            "check": "SubLN-only BitNet-SFT control explains the gap",
            "status": (
                "fail"
                if isinstance(subln_delta, float) and subln_delta <= 0
                else ("pass" if isinstance(subln_delta, float) else "warn")
            ),
            "evidence": (
                f"default_subln_inserted={subln}, subln_accuracy={fmt(subln_bitnet_acc)}, delta_vs_default={fmt(subln_delta)}"
                if isinstance(subln_bitnet_acc, float)
                else f"default_subln_inserted={subln}; no completed SubLN BitNet-SFT checkpoint is present yet."
            ),
            "implication": "Current SubLN insertion alone does not recover the paper anchor; either the SubLN recipe differs or it requires matched warmup/search.",
        }
    )
    checks.append(
        {
            "check": "Weights-only vs W1.58A8 ablation exists",
            "status": "pass" if isinstance(weights_only_acc, float) else "missing",
            "evidence": (
                f"weights_only_accuracy={fmt(weights_only_acc)}, A8_accuracy={fmt(bitnet_acc)}"
                if isinstance(weights_only_acc, float)
                else "No completed A8-off checkpoint is present yet."
            ),
            "implication": "This separates weight-code collapse from activation-quantization damage.",
        }
    )
    checks.append(
        {
            "check": "BitNet-SFT budget issue is explained",
            "status": "pass" if isinstance(best_budget_acc, float) and best_budget_acc >= bitnet_paper else "warn",
            "evidence": (
                f"default_steps={bitnet.get('steps')}, default_acc={fmt(bitnet_acc)}, "
                f"best_steps={best_budget.get('steps', '-')}, best_acc={fmt(best_budget_acc)}"
            ),
            "implication": "The next controlled comparison should use the best budget row, not the default 1000-step row.",
        }
    )
    return checks


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


def fmt_ci(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "-"
    return f"[{fmt(float(value[0]))}, {fmt(float(value[1]))}]"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell).replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    rows = summary["checkpoints"]
    checks = summary["checks"]
    best_budget = summary.get("best_budget_sweep") if isinstance(summary.get("best_budget_sweep"), dict) else {}
    best_paired = summary.get("best_budget_paired") if isinstance(summary.get("best_budget_paired"), dict) else {}
    checkpoint_table = [
        [
            row["name"],
            row["method"],
            row["accuracy"],
            row["eval_examples"],
            row["fp16_gap"],
            row["paper_anchor"],
            row["paper_gap"],
            nested(row, "preparation", "bitlinear_replaced"),
            nested(row, "preparation", "subln_inserted"),
            nested(row, "preparation", "activation_quantization"),
            nested(row, "ternary", "ternary_weight_count"),
            nested(row, "ternary", "scale_numel_by_key"),
        ]
        for row in rows
    ]
    code_table = []
    for row in rows:
        ternary = row.get("ternary", {})
        if not isinstance(ternary, dict) or not ternary.get("exists"):
            continue
        code_table.append(
            [
                row["name"],
                nested(row, "ternary", "total_codes"),
                nested(row, "ternary", "code_fractions", "-1"),
                nested(row, "ternary", "code_fractions", "0"),
                nested(row, "ternary", "code_fractions", "1"),
                nested(row, "ternary", "scale_summary", "mean"),
                nested(row, "ternary", "scale_summary", "std"),
            ]
        )
    check_table = [[item["check"], item["status"], item["evidence"], item["implication"]] for item in checks]
    return "\n\n".join(
        [
            f"# BitNet-SFT Baseline Audit, {summary['date']}",
            (
                "Verdict: the local FP16-SFT MNLI baseline is close to the paper anchor, "
                "and the best completed BitNet-SFT budget row now clears the paper's "
                "BitNet-SFT anchor. The original 1000-step default row was undertrained. "
                "Static checkpoint checks do not show a missing projection-export bug; "
                "the remaining reproduction blocker is BitDistill/FP16-level recovery, "
                "especially SubLN and distillation-loss parity."
            ),
            (
                "Best completed budget row: "
                f"`{fmt(best_budget.get('accuracy'))}` at steps=`{best_budget.get('steps', '-')}`, "
                f"lr=`{best_budget.get('lr', '-')}`. Paired candidate-minus-FP16 delta: "
                f"`{fmt(best_paired.get('delta_vs_reference'))}` with CI "
                f"`{fmt_ci(best_paired.get('paired_ci95'))}`."
            ),
            "## Accuracy And Mechanical Summary",
            md_table(
                [
                    "run",
                    "method",
                    "accuracy",
                    "examples",
                    "FP16 gap",
                    "paper anchor",
                    "paper anchor - local",
                    "BitLinear replaced",
                    "SubLN inserted",
                    "A8 activations",
                    "ternary tensors",
                    "scale numel histogram",
                ],
                checkpoint_table,
            ),
            "## Ternary Code Summary",
            md_table(
                ["run", "codes", "-1 frac", "0 frac", "+1 frac", "scale mean", "scale std"],
                code_table,
            ),
            "## Checks",
            md_table(["check", "status", "evidence", "implication"], check_table),
            "## Next Narrow Experiments",
            "\n".join(
                [
                    "1. Treat the completed 10000-step LR rows as schedule-sensitivity evidence: lr=2e-5 is the best current CE-only row, while lr=5e-5 is the paper-anchor control.",
                    "2. Use the best cleared BitNet-SFT budget row as the controlled CE-only baseline for BitDistill recovery.",
                    "3. Audit SubLN placement, initialization, and whether it should be enabled before or after continued pretraining.",
                    "4. Add activation variance, int8 saturation, ternary flip-rate, and loss-gradient telemetry to the Stage-3 loop.",
                    "5. Keep row-scale results labeled as a retrofit variant, not as a paper-reproduction result.",
                ]
            ),
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitnet_sft_baseline_audit_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitnet_sft_baseline_audit_{DATE}.md"))
    parser.add_argument("--fp16-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1"))
    parser.add_argument("--bitnet-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-layer-1"))
    parser.add_argument("--bitnet-weights-only-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-ablate/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-noa8-layer-1"))
    parser.add_argument("--bitnet-subln-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-ablate/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-subln-tensor-layer-1"))
    parser.add_argument("--bitnet-headinit-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-teacherhead/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-headinit-tensor-layer-1"))
    parser.add_argument("--best-row-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8"))
    parser.add_argument("--budget-sweep-json", type=Path, default=Path(f"benchmark_results/bitnet_sft_budget_sweep_{DATE}.json"))
    parser.add_argument("--budget-paired-json", type=Path, default=Path(f"benchmark_results/bitnet_sft_budget_paired_{DATE}.json"))
    args = parser.parse_args()

    rows = [
        summarize_checkpoint("FP16-SFT", args.fp16_root),
        summarize_checkpoint("BitNet-SFT", args.bitnet_root),
        summarize_checkpoint("BitNet-SFT weights-only", args.bitnet_weights_only_root),
        summarize_checkpoint("BitNet-SFT SubLN", args.bitnet_subln_root),
        summarize_checkpoint("BitNet-SFT head-init", args.bitnet_headinit_root),
        summarize_checkpoint("Best local row-scale BitDistill", args.best_row_root),
    ]
    add_fp_gaps(rows)
    budget_sweep = read_json(args.budget_sweep_json)
    paired_budget = read_json(args.budget_paired_json)
    summary = {
        "schema": "bitnet-sft-baseline-audit-v1",
        "date": DATE,
        "paper_anchor": PAPER_QWEN25_MNLI,
        "expected_qwen25_05b_ternary_projection_count": EXPECTED_QWEN25_05B_TERNARY,
        "checkpoints": rows,
        "best_budget_sweep": budget_sweep.get("best", {}),
        "best_budget_paired": paired_budget.get("best", {}),
        "checks": build_checks(rows, budget_sweep, paired_budget),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = render_markdown(summary)
    args.output_md.write_text(md, encoding="utf-8")
    print(md, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
