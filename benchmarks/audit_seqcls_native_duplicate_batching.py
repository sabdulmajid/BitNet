#!/usr/bin/env python3
"""Audit duplicate-prompt batching parity for native I2_SR classifiers.

This is narrower than audit_seqcls_native_batching.py. It repeats the exact
same token-ID prompt multiple times in one llama-embedding batch and verifies
whether each batch position returns the same logits as the single-prompt run.
If this fails, tokenization, prompt formatting, and output-row swaps are ruled
out: the remaining issue is position-dependent native runtime drift.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_seqcls_native_i2sr_cpu import DEFAULT_CHECKPOINT, DEFAULT_GGUF, render_prompt, run_native_classifier


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
DEFAULT_CONTROL_GGUFS = [
    (
        "fp_qwen05b_f16",
        Path("models/qwen2.5-0.5b-fp/qwen05b_fp_f16.gguf"),
    ),
    (
        "bitnet_qwen_i2sr_backbone",
        Path(
            "models/seqcls-backbone-i2sr/Qwen-Qwen2.5-0.5B/mnli/"
            "bitdistill-longwarmup-row-layer-8_bitnet_qwen_i2_sr.gguf"
        ),
    ),
]


def maybe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def load_mnli_rows(limit: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    return [dict(row) for row in load_dataset("glue", "mnli")["validation_matched"].select(range(limit))]


def rel_rms(candidate: np.ndarray, reference: np.ndarray) -> float:
    diff = candidate.astype(np.float64) - reference.astype(np.float64)
    denom = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
    return float(np.sqrt(np.mean(diff**2)) / max(denom, 1e-12))


def margin(logits: np.ndarray) -> float:
    values = np.sort(logits.astype(np.float64))
    return float(values[-1] - values[-2]) if values.size >= 2 else float("nan")


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return "[" + ", ".join(fmt(item) for item in value) + "]"
    return str(value).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    position_rows: list[list[Any]] = []
    for target in result["targets"]:
        for position in target["duplicate_positions"]:
            position_rows.append(
                [
                    target["target_index"],
                    position["position"],
                    position["prediction"],
                    position["margin"],
                    position["relative_rms_vs_alone"],
                    position["matches_alone_prediction"],
                    position["logits"],
                ]
            )
    control_rows = []
    for control in result.get("control_models", []):
        summary = control["summary"]
        control_rows.append(
            [
                control["label"],
                summary["all_logits_invariant"],
                summary["all_argmax_invariant"],
                summary["changed_argmax_count"],
                summary["max_relative_rms_vs_alone"],
                control["artifacts"]["gguf"],
            ]
        )
    return "\n\n".join(
        [
            f"# Sequence-Classification Native I2_SR Duplicate-Prompt Batching Audit, {result['date']}",
            (
                "This audit repeats the same rendered token-ID prompt within a single native "
                "llama-embedding batch. A mismatch here cannot be attributed to tokenizer "
                "round-trip differences, different text examples, or output-row swaps."
            ),
            md_table(
                ["field", "value"],
                [
                    ["status", result["status"]],
                    ["targets", result["summary"]["target_indices"]],
                    ["repeat count", result["repeat_count"]],
                    ["same prompt repeated", result["summary"]["same_prompt_repeated"]],
                    ["all logits invariant", result["summary"]["all_logits_invariant"]],
                    ["all predictions invariant", result["summary"]["all_predictions_invariant"]],
                    ["changed prediction count", result["summary"]["changed_prediction_count"]],
                    ["max relative RMS vs alone", result["summary"]["max_relative_rms_vs_alone"]],
                    ["formatting/tokenization ruled out", result["summary"]["formatting_and_tokenization_ruled_out"]],
                    ["ready for batched product benchmark", result["ready_for_batched_product_benchmark"]],
                ],
            ),
            "## Duplicate Positions",
            md_table(
                ["target", "position", "pred", "margin", "rel RMS vs alone", "pred matches alone", "logits"],
                position_rows,
            ),
            "## Control Models",
            md_table(
                ["model", "logits invariant", "argmax invariant", "changed argmax", "max rel RMS", "gguf"],
                control_rows,
            ),
            "## Interpretation",
            result["interpretation"],
        ]
    ) + "\n"


def parse_labeled_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("control GGUF must be LABEL=PATH")
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("control GGUF label cannot be empty")
    return label, Path(path)


def audit_duplicate_model(
    *,
    label: str,
    root: Path,
    gguf: Path,
    binary: Path,
    prompts_by_target: dict[int, str],
    repeat_count: int,
    separator: str,
    threads: int,
    ctx_size: int,
    batch_size: int,
    ubatch_size: int,
    timeout_seconds: int,
    logit_atol: float,
) -> dict[str, Any]:
    target_results: list[dict[str, Any]] = []
    changed_argmax_count = 0
    max_rel = 0.0
    all_logits_invariant = True
    all_argmax_invariant = True

    for target, prompt in prompts_by_target.items():
        duplicate_prompts = [prompt for _ in range(repeat_count)]
        alone_logits, alone_meta = run_native_classifier(
            binary=binary,
            gguf=gguf,
            prompts=[prompt],
            separator=separator,
            threads=threads,
            ctx_size=ctx_size,
            batch_size=batch_size,
            ubatch_size=ubatch_size,
            timeout_seconds=timeout_seconds,
        )
        duplicate_logits, duplicate_meta = run_native_classifier(
            binary=binary,
            gguf=gguf,
            prompts=duplicate_prompts,
            separator=separator,
            threads=threads,
            ctx_size=ctx_size,
            batch_size=batch_size,
            ubatch_size=ubatch_size,
            timeout_seconds=timeout_seconds,
        )

        alone = alone_logits[0].astype(np.float32)
        alone_argmax = int(np.argmax(alone))
        positions: list[dict[str, Any]] = []
        for position, logits in enumerate(duplicate_logits):
            candidate = logits.astype(np.float32)
            current_rel = rel_rms(candidate, alone)
            max_rel = max(max_rel, current_rel)
            logit_match = current_rel <= logit_atol
            argmax = int(np.argmax(candidate))
            argmax_match = argmax == alone_argmax
            all_logits_invariant = all_logits_invariant and logit_match
            all_argmax_invariant = all_argmax_invariant and argmax_match
            if not argmax_match:
                changed_argmax_count += 1
            positions.append(
                {
                    "position": position,
                    "logits": [float(value) for value in candidate.tolist()],
                    "prediction": argmax,
                    "margin": margin(candidate),
                    "relative_rms_vs_alone": current_rel,
                    "matches_alone_logits": logit_match,
                    "matches_alone_prediction": argmax_match,
                }
            )

        target_results.append(
            {
                "target_index": target,
                "prompt_input": "token_ids",
                "alone_logits": [float(value) for value in alone.tolist()],
                "alone_prediction": alone_argmax,
                "alone_margin": margin(alone),
                "duplicate_positions": positions,
                "runtime": {
                    "alone_elapsed_seconds": alone_meta.get("elapsed_seconds"),
                    "duplicate_elapsed_seconds": duplicate_meta.get("elapsed_seconds"),
                    "alone_prompt_eval_tokens_per_second": (alone_meta.get("perf", {}) or {}).get(
                        "prompt_eval_tokens_per_second"
                    ),
                    "duplicate_prompt_eval_tokens_per_second": (duplicate_meta.get("perf", {}) or {}).get(
                        "prompt_eval_tokens_per_second"
                    ),
                },
            }
        )

    return {
        "label": label,
        "artifacts": {"gguf": maybe_relative(gguf, root)},
        "summary": {
            "target_indices": list(prompts_by_target),
            "all_logits_invariant": all_logits_invariant,
            "all_argmax_invariant": all_argmax_invariant,
            "changed_argmax_count": changed_argmax_count,
            "max_relative_rms_vs_alone": max_rel,
        },
        "targets": target_results,
    }


def compact_control_model(model: dict[str, Any]) -> dict[str, Any]:
    """Keep control evidence scalar and reproducible without dumping 896-d vectors."""
    compact_targets: list[dict[str, Any]] = []
    for target in model["targets"]:
        compact_targets.append(
            {
                "target_index": target["target_index"],
                "alone_prediction": target["alone_prediction"],
                "alone_margin": target["alone_margin"],
                "duplicate_positions": [
                    {
                        "position": position["position"],
                        "prediction": position["prediction"],
                        "margin": position["margin"],
                        "relative_rms_vs_alone": position["relative_rms_vs_alone"],
                        "matches_alone_logits": position["matches_alone_logits"],
                        "matches_alone_prediction": position["matches_alone_prediction"],
                    }
                    for position in target["duplicate_positions"]
                ],
            }
        )
    return {
        "label": model["label"],
        "artifacts": model["artifacts"],
        "summary": model["summary"],
        "targets": compact_targets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--embedding-binary", type=Path, default=Path("build-portable-avx2/bin/llama-embedding"))
    parser.add_argument("--targets", type=int, nargs="+", default=[15, 35, 0, 1, 2])
    parser.add_argument("--repeat-count", type=int, default=4)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--separator", default="<#BITNET_DUPLICATE_BATCH_PARITY#>")
    parser.add_argument("--logit-atol", type=float, default=1.0e-4)
    parser.add_argument(
        "--control-gguf",
        action="append",
        default=[f"{label}={path}" for label, path in DEFAULT_CONTROL_GGUFS],
        help="Optional duplicate-batching control model as LABEL=PATH. Repeat to add controls.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"benchmark_results/seqcls_native_duplicate_batching_audit_{DATE}.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"benchmarks/results/seqcls_native_duplicate_batching_audit_{DATE}.md"),
    )
    args = parser.parse_args()

    if args.repeat_count < 2:
        raise SystemExit("--repeat-count must be at least 2")

    root = args.repo_root.resolve()
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else root / args.checkpoint_dir
    gguf = args.gguf if args.gguf.is_absolute() else root / args.gguf
    binary = args.embedding_binary if args.embedding_binary.is_absolute() else root / args.embedding_binary

    from transformers import AutoTokenizer

    rows = load_mnli_rows(max(args.targets) + 1)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)

    prompts_by_target = {
        target: render_prompt(tokenizer, "mnli", rows[target], prompt_input="token_ids") for target in args.targets
    }
    same_prompt_repeated = True
    for prompt in prompts_by_target.values():
        same_prompt_repeated = same_prompt_repeated and len({prompt for _ in range(args.repeat_count)}) == 1

    primary = audit_duplicate_model(
        label="primary_cls_i2sr",
        root=root,
        gguf=gguf,
        binary=binary,
        prompts_by_target=prompts_by_target,
        repeat_count=args.repeat_count,
        separator=args.separator,
        threads=args.threads,
        ctx_size=args.ctx_size,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        timeout_seconds=args.timeout_seconds,
        logit_atol=args.logit_atol,
    )
    control_models = []
    for label, control_path in (parse_labeled_path(item) for item in args.control_gguf):
        control_models.append(
            compact_control_model(
                audit_duplicate_model(
                    label=label,
                    root=root,
                    gguf=control_path if control_path.is_absolute() else root / control_path,
                    binary=binary,
                    prompts_by_target=prompts_by_target,
                    repeat_count=args.repeat_count,
                    separator=args.separator,
                    threads=args.threads,
                    ctx_size=args.ctx_size,
                    batch_size=args.batch_size,
                    ubatch_size=args.ubatch_size,
                    timeout_seconds=args.timeout_seconds,
                    logit_atol=args.logit_atol,
                )
            )
        )

    primary_summary = primary["summary"]
    status = (
        "pass"
        if primary_summary["all_logits_invariant"] and primary_summary["all_argmax_invariant"]
        else "duplicate_batching_parity_mismatch"
    )
    formatting_ruled_out = same_prompt_repeated and status != "pass"
    result: dict[str, Any] = {
        "schema": "seqcls_native_duplicate_batching_audit.v1",
        "date": DATE,
        "status": status,
        "repeat_count": args.repeat_count,
        "artifacts": {
            "checkpoint": maybe_relative(checkpoint_dir, root),
            "gguf": maybe_relative(gguf, root),
            "embedding_binary": maybe_relative(binary, root),
        },
        "summary": {
            "target_indices": args.targets,
            "same_prompt_repeated": same_prompt_repeated,
            "all_logits_invariant": primary_summary["all_logits_invariant"],
            "all_predictions_invariant": primary_summary["all_argmax_invariant"],
            "changed_prediction_count": primary_summary["changed_argmax_count"],
            "max_relative_rms_vs_alone": primary_summary["max_relative_rms_vs_alone"],
            "formatting_and_tokenization_ruled_out": formatting_ruled_out,
        },
        "targets": primary["targets"],
        "control_models": control_models,
        "ready_for_batched_product_benchmark": status == "pass",
        "interpretation": (
            "Duplicate token-ID prompts are invariant across batch positions."
            if status == "pass"
            else (
                "Duplicate token-ID prompts are not invariant across batch positions. Because every "
                "entry in each audited batch is the exact same rendered token-ID prompt, this rules "
                "out tokenizer round-trip differences, text formatting differences, and output-row "
                "swaps as sufficient explanations. Batched sequence-classification throughput must "
                "remain blocked until this native runtime position-dependent drift is fixed."
            )
        ),
    }

    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": status, "summary": result["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
