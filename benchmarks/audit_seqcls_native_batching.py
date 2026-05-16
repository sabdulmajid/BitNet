#!/usr/bin/env python3
"""Audit native I2_SR sequence-classification batching parity.

The token-ID path fixes text round-trip tokenization, but the same prompt must
produce stable logits whether evaluated alone or inside a multi-sequence
embedding batch. This audit targets low-margin MNLI rows that changed under
batching and records whether predictions/logits are invariant.
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


def maybe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def load_mnli_rows(limit: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    return [dict(row) for row in load_dataset("glue", "mnli")["validation_matched"].select(range(limit))]


def margin(logits: np.ndarray) -> float:
    values = np.sort(logits.astype(np.float64))
    return float(values[-1] - values[-2]) if values.size >= 2 else float("nan")


def rel_rms(candidate: np.ndarray, reference: np.ndarray) -> float:
    diff = candidate.astype(np.float64) - reference.astype(np.float64)
    denom = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
    return float(np.sqrt(np.mean(diff**2)) / max(denom, 1e-12))


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
    rows = []
    for target in result["targets"]:
        for case in target["cases"]:
            rows.append(
                [
                    target["target_index"],
                    case["case"],
                    case["target_position"],
                    case["prediction"],
                    case["margin"],
                    case["relative_rms_vs_alone"],
                    case["indices"],
                ]
            )
    return "\n\n".join(
        [
            f"# Sequence-Classification Native I2_SR Batching Audit, {result['date']}",
            (
                "This audit checks whether the native token-ID classifier path is invariant "
                "to embedding prompt batching. It intentionally targets low-margin MNLI rows "
                "that changed between batch-1 and batch-4 samples."
            ),
            md_table(
                ["field", "value"],
                [
                    ["status", result["status"]],
                    ["targets", result["summary"]["target_indices"]],
                    ["all predictions invariant", result["summary"]["all_predictions_invariant"]],
                    ["max relative RMS vs alone", result["summary"]["max_relative_rms_vs_alone"]],
                    ["changed cases", result["summary"]["changed_case_count"]],
                    ["ready for batched product benchmark", result["ready_for_batched_product_benchmark"]],
                ],
            ),
            "## Cases",
            md_table(
                ["target", "case", "target pos", "pred", "margin", "rel RMS vs alone", "indices"],
                rows,
            ),
            "## Interpretation",
            result["interpretation"],
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--embedding-binary", type=Path, default=Path("build-portable-avx2/bin/llama-embedding"))
    parser.add_argument("--targets", type=int, nargs="+", default=[15, 35])
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--separator", default="<#BITNET_BATCH_PARITY#>")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"benchmark_results/seqcls_native_batching_audit_{DATE}.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"benchmarks/results/seqcls_native_batching_audit_{DATE}.md"),
    )
    args = parser.parse_args()

    root = args.repo_root.resolve()
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else root / args.checkpoint_dir
    gguf = args.gguf if args.gguf.is_absolute() else root / args.gguf
    binary = args.embedding_binary if args.embedding_binary.is_absolute() else root / args.embedding_binary

    from transformers import AutoTokenizer

    rows = load_mnli_rows(max(args.targets) + 4)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    prompts = [render_prompt(tokenizer, "mnli", row, prompt_input="token_ids") for row in rows]

    target_results: list[dict[str, Any]] = []
    for target in args.targets:
        raw_cases: list[tuple[str, list[int]]] = [("alone", [target])]
        for pos in range(4):
            start = target - pos
            if start < 0 or start + 4 > len(rows):
                continue
            raw_cases.append((f"pos{pos}", list(range(start, start + 4))))

        case_results: list[dict[str, Any]] = []
        alone_logits: np.ndarray | None = None
        for name, indices in raw_cases:
            logits, meta = run_native_classifier(
                binary=binary,
                gguf=gguf,
                prompts=[prompts[index] for index in indices],
                separator=args.separator,
                threads=args.threads,
                ctx_size=args.ctx_size,
                batch_size=args.batch_size,
                ubatch_size=args.ubatch_size,
                timeout_seconds=args.timeout_seconds,
            )
            target_pos = indices.index(target)
            target_logits = logits[target_pos].astype(np.float32)
            if name == "alone":
                alone_logits = target_logits
            if alone_logits is None:
                raise RuntimeError("alone case must run first")
            case_results.append(
                {
                    "case": name,
                    "indices": indices,
                    "target_position": target_pos,
                    "logits": [float(value) for value in target_logits.tolist()],
                    "prediction": int(np.argmax(target_logits)),
                    "margin": margin(target_logits),
                    "relative_rms_vs_alone": rel_rms(target_logits, alone_logits),
                    "runtime": {
                        "elapsed_seconds": meta.get("elapsed_seconds"),
                        "prompt_eval_tokens_per_second": (meta.get("perf", {}) or {}).get(
                            "prompt_eval_tokens_per_second"
                        ),
                    },
                }
            )
        alone_pred = int(case_results[0]["prediction"])
        target_results.append(
            {
                "target_index": target,
                "alone_prediction": alone_pred,
                "predictions_invariant": all(case["prediction"] == alone_pred for case in case_results),
                "max_relative_rms_vs_alone": max(float(case["relative_rms_vs_alone"]) for case in case_results),
                "cases": case_results,
            }
        )

    changed_cases = [
        (target["target_index"], case["case"])
        for target in target_results
        for case in target["cases"]
        if case["prediction"] != target["alone_prediction"]
    ]
    all_predictions_invariant = not changed_cases
    max_rel = max(float(target["max_relative_rms_vs_alone"]) for target in target_results)
    status = "pass" if all_predictions_invariant and max_rel < 1e-4 else "batching_parity_mismatch"
    result = {
        "schema": "seqcls_native_batching_audit.v1",
        "date": DATE,
        "status": status,
        "artifacts": {
            "checkpoint": maybe_relative(checkpoint_dir, root),
            "gguf": maybe_relative(gguf, root),
            "embedding_binary": maybe_relative(binary, root),
        },
        "summary": {
            "target_indices": args.targets,
            "all_predictions_invariant": all_predictions_invariant,
            "changed_cases": changed_cases,
            "changed_case_count": len(changed_cases),
            "max_relative_rms_vs_alone": max_rel,
        },
        "targets": target_results,
        "ready_for_batched_product_benchmark": status == "pass",
        "interpretation": (
            "Native batched logits are invariant for the audited rows."
            if status == "pass"
            else "Native batched logits are not invariant for the audited rows. Do not promote batched "
            "throughput or full-validation numbers until the llama.cpp sequence embedding path has a "
            "stable batching contract."
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
