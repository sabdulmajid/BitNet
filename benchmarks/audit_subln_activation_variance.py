#!/usr/bin/env python3
"""Audit the immediate activation/logit effect of local SubLN insertion.

This is a read-only diagnostic. It does not train a model. The goal is to make
the SubLN reproduction risk concrete: the local implementation inserts RMSNorm
before attention output and FFN down projections, which is mathematically not an
identity transform at step zero.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_bitdistill import SubLNLinear, add_subln_to_qwen_blocks


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
DEFAULT_LOCAL_MODEL = "checkpoints/qwen2.5-0.5b-fineweb-edu-12/step-1000"
DEFAULT_MODEL = DEFAULT_LOCAL_MODEL if (REPO_ROOT / DEFAULT_LOCAL_MODEL).exists() else "Qwen/Qwen2.5-0.5B"

PROMPTS = [
    "Premise: A scientist adjusts a small radio telescope during a winter field test. Hypothesis: A researcher is working outdoors.",
    "Question: What makes low-bit language model inference difficult on commodity CPUs? Answer:",
    "The committee reviewed the benchmark table and asked whether the ternary model preserved the full-precision decision boundary.",
    "Summarize the following claim in one sentence: blind post-training ternarization collapses dense model weight geometry.",
]


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if 0.0 < abs(value) < 0.0001 or abs(value) >= 1000:
            return f"{value:.6e}"
        return f"{value:.6f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell).replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    x = tensor.detach().float()
    token_rms = torch.sqrt(torch.mean(x.square(), dim=-1))
    return {
        "element_mean": float(x.mean().item()),
        "element_std": float(x.std(unbiased=False).item()),
        "element_absmax": float(x.abs().max().item()),
        "token_rms_mean": float(token_rms.mean().item()),
        "token_rms_std": float(token_rms.std(unbiased=False).item()),
    }


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def aggregate(records: dict[str, list[dict[str, float]]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, rows in records.items():
        out[name] = {
            key: mean([float(row[key]) for row in rows])
            for key in ("element_mean", "element_std", "element_absmax", "token_rms_mean", "token_rms_std")
        }
    return out


def summarize_family(rows: dict[str, dict[str, float]], suffix: str) -> dict[str, float | int | None]:
    matched = [value for name, value in rows.items() if name.endswith(suffix)]
    return {
        "modules": len(matched),
        "token_rms_mean": mean([float(row["token_rms_mean"]) for row in matched]),
        "token_rms_std": mean([float(row["token_rms_std"]) for row in matched]),
        "element_std": mean([float(row["element_std"]) for row in matched]),
        "element_absmax": mean([float(row["element_absmax"]) for row in matched]),
    }


def attach_base_hooks(model: torch.nn.Module, records: dict[str, list[dict[str, float]]]) -> list[Any]:
    handles = []
    for name, module in model.named_modules():
        if name.endswith(".self_attn.o_proj") or name.endswith(".mlp.down_proj"):
            handles.append(module.register_forward_pre_hook(lambda _m, inputs, n=name: records[n].append(tensor_stats(inputs[0]))))
    return handles


def attach_subln_hooks(
    model: torch.nn.Module,
    before_records: dict[str, list[dict[str, float]]],
    after_records: dict[str, list[dict[str, float]]],
) -> list[Any]:
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, SubLNLinear):
            handles.append(
                module.subln.register_forward_pre_hook(
                    lambda _m, inputs, n=name: before_records[n].append(tensor_stats(inputs[0]))
                )
            )
            handles.append(
                module.subln.register_forward_hook(
                    lambda _m, _inputs, output, n=name: after_records[n].append(tensor_stats(output))
                )
            )
    return handles


def remove_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def relative_rms(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((candidate - reference).float())
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    return float((numerator / denominator).item())


def cosine(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    c = candidate.float().reshape(-1)
    r = reference.float().reshape(-1)
    return float(torch.nn.functional.cosine_similarity(c, r, dim=0).item())


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        local_files_only=args.local_files_only,
    )
    model.eval()
    model.to(args.device)
    model.config.use_cache = False

    batch = tokenizer(
        PROMPTS[: args.examples],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    ).to(args.device)

    base_records: dict[str, list[dict[str, float]]] = defaultdict(list)
    base_handles = attach_base_hooks(model, base_records)
    base_output = model(**batch).logits[:, -1, :].detach().cpu()
    remove_hooks(base_handles)

    inserted = add_subln_to_qwen_blocks(model, eps=args.subln_eps)
    before_records: dict[str, list[dict[str, float]]] = defaultdict(list)
    after_records: dict[str, list[dict[str, float]]] = defaultdict(list)
    subln_handles = attach_subln_hooks(model, before_records, after_records)
    subln_output = model(**batch).logits[:, -1, :].detach().cpu()
    remove_hooks(subln_handles)

    base_agg = aggregate(base_records)
    before_agg = aggregate(before_records)
    after_agg = aggregate(after_records)
    top1_base = torch.argmax(base_output, dim=-1)
    top1_subln = torch.argmax(subln_output, dim=-1)

    families: dict[str, dict[str, Any]] = {}
    for suffix in (".self_attn.o_proj", ".mlp.down_proj"):
        families[suffix] = {
            "base_input": summarize_family(base_agg, suffix),
            "subln_input": summarize_family(before_agg, suffix),
            "subln_output": summarize_family(after_agg, suffix),
        }

    rows = []
    for name in sorted(before_agg):
        if name not in base_agg or name not in after_agg:
            continue
        rows.append(
            {
                "module": name,
                "base_input_rms": base_agg[name]["token_rms_mean"],
                "subln_input_rms": before_agg[name]["token_rms_mean"],
                "subln_output_rms": after_agg[name]["token_rms_mean"],
                "base_input_absmax": base_agg[name]["element_absmax"],
                "subln_output_absmax": after_agg[name]["element_absmax"],
            }
        )

    return {
        "schema": "subln-activation-variance-audit-v1",
        "date": DATE,
        "model": args.model,
        "device": args.device,
        "examples": int(batch["input_ids"].shape[0]),
        "max_length": args.max_length,
        "tokens": int(batch["input_ids"].numel()),
        "subln_eps": args.subln_eps,
        "subln_inserted": inserted,
        "families": families,
        "logit_relative_rms": relative_rms(subln_output, base_output),
        "logit_cosine": cosine(subln_output, base_output),
        "last_token_top1_agreement": float((top1_base == top1_subln).float().mean().item()),
        "rows": rows,
        "interpretation": (
            "Local SubLN insertion is not an identity-preserving conversion. It normalizes the tensors entering "
            "o_proj/down_proj to unit RMS and can materially perturb logits before any continued pretraining. "
            "Therefore a short SubLN-only downstream run is not a decisive test of the paper's Stage-1 recipe."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    family_rows = []
    for family, data in summary["families"].items():
        family_rows.append(
            [
                family,
                data["base_input"]["modules"],
                data["base_input"]["token_rms_mean"],
                data["subln_input"]["token_rms_mean"],
                data["subln_output"]["token_rms_mean"],
                data["base_input"]["element_absmax"],
                data["subln_output"]["element_absmax"],
            ]
        )
    module_rows = [
        [
            row["module"],
            row["base_input_rms"],
            row["subln_input_rms"],
            row["subln_output_rms"],
            row["base_input_absmax"],
            row["subln_output_absmax"],
        ]
        for row in summary["rows"][:12]
    ]
    return "\n\n".join(
        [
            f"# SubLN Activation Variance Audit, {summary['date']}",
            summary["interpretation"],
            "",
            f"- Model: `{summary['model']}`.",
            f"- Examples: `{summary['examples']}`; tokens: `{summary['tokens']}`; max length: `{summary['max_length']}`.",
            f"- SubLN modules inserted: `{summary['subln_inserted']}`.",
            f"- Last-token logit relative RMS drift after untrained SubLN insertion: `{fmt(summary['logit_relative_rms'])}`.",
            f"- Last-token logit cosine after untrained SubLN insertion: `{fmt(summary['logit_cosine'])}`.",
            f"- Last-token top-1 agreement: `{fmt(summary['last_token_top1_agreement'])}`.",
            "",
            "## Family Summary",
            md_table(
                [
                    "family",
                    "modules",
                    "base input RMS",
                    "SubLN input RMS",
                    "SubLN output RMS",
                    "base input absmax",
                    "SubLN output absmax",
                ],
                family_rows,
            ),
            "",
            "## First Modules",
            md_table(
                [
                    "module",
                    "base input RMS",
                    "SubLN input RMS",
                    "SubLN output RMS",
                    "base input absmax",
                    "SubLN output absmax",
                ],
                module_rows,
            ),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--subln-eps", type=float, default=1e-5)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/subln_activation_variance_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/subln_activation_variance_{DATE}.md"))
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
