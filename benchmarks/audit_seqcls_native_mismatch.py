#!/usr/bin/env python3
"""Audit native I2_SR sequence-classification mismatches against PyTorch.

The native classifier smoke proves that the classifier head can live inside a
GGUF. The first sampled MNLI CPU run still disagrees with saved PyTorch
predictions on rows 7 and 15. This audit keeps the scope narrow: for a small
set of rows, compare token IDs, PyTorch pooled hidden states, sidecar GGUF
hidden states plus dense head, and native GGUF logits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from audit_seqcls_i2sr_hidden_contract import (
    DEFAULT_CHECKPOINT,
    DEFAULT_GGUF as DEFAULT_SIDECAR_GGUF,
    DEFAULT_HEAD,
    cosine,
    head_logits,
    llama_embedding,
    llama_token_ids,
    load_pytorch_model,
    pytorch_probe,
    render_prompt,
    tensor_rel_rms,
)


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
DEFAULT_NATIVE_GGUF = Path(
    "models/seqcls-native-i2sr/"
    "Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8_bitnet_qwen_i2_sr_cls.gguf"
)


def maybe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def load_mnli_rows(indices: list[int]) -> dict[int, dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("glue", "mnli")["validation_matched"]
    return {index: dict(dataset[index]) for index in indices}


def read_prediction_trace(path: Path, max_index: int) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line_number > max_index:
                break
            row = json.loads(line)
            if isinstance(row, dict) and isinstance(row.get("index"), int):
                rows[int(row["index"])] = row
    return rows


def prediction(logits: np.ndarray) -> int:
    return int(np.argmax(logits.astype(np.float64)))


def margin(logits: np.ndarray) -> float:
    values = np.sort(logits.astype(np.float64))
    if values.size < 2:
        return float("nan")
    return float(values[-1] - values[-2])


def list_float(values: np.ndarray) -> list[float]:
    return [float(item) for item in values.astype(np.float32).tolist()]


def token_ids_prompt(ids: list[int]) -> str:
    return "token_ids:" + json.dumps([int(item) for item in ids], separators=(",", ":"))


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
    summary = result["summary"]
    rows = result["rows"]
    table_rows = [
        [
            row["index"],
            row["label"],
            row["predictions"].get("saved_pytorch"),
            row["predictions"].get("pytorch"),
            row["predictions"].get("sidecar"),
            row["predictions"].get("native"),
            row["comparisons"]["token_id_match"],
            row["comparisons"]["hidden_relative_rms"],
            row["comparisons"]["hidden_cosine"],
            row["comparisons"]["pytorch_vs_sidecar_logit_relative_rms"],
            row["comparisons"]["native_vs_sidecar_logit_relative_rms"],
            row["margins"]["native"],
        ]
        for row in rows
    ]
    return "\n\n".join(
        [
            f"# Sequence-Classification Native I2_SR Mismatch Audit, {result['date']}",
            (
                "This audit checks the rows that blocked the native same-artifact classifier "
                "product gate. It compares the PyTorch sequence-classification checkpoint, "
                "the sidecar GGUF backbone plus dense head, and the native classifier-head GGUF "
                "on exactly the same MNLI prompts."
            ),
            md_table(
                ["field", "value"],
                [
                    ["status", result["status"]],
                    ["indices", summary["indices"]],
                    ["token IDs all match", summary["all_token_ids_match"]],
                    ["text roundtrip all token IDs match", summary["text_roundtrip_all_token_ids_match"]],
                    ["prompt input", result["prompt_input"]],
                    ["native/sidecar agreement", summary["native_sidecar_prediction_agreement"]],
                    ["saved/native agreement", summary["saved_native_prediction_agreement"]],
                    ["hidden relative RMS max", summary["hidden_relative_rms_max"]],
                    ["hidden cosine min", summary["hidden_cosine_min"]],
                    ["native-vs-sidecar logit relative RMS max", summary["native_vs_sidecar_logit_relative_rms_max"]],
                ],
            ),
            "## Row Comparisons",
            md_table(
                [
                    "idx",
                    "label",
                    "saved",
                    "torch",
                    "sidecar",
                    "native",
                    "tokens",
                    "hidden rel RMS",
                    "hidden cos",
                    "torch/sidecar logit rel",
                    "native/sidecar logit rel",
                    "native margin",
                ],
                table_rows,
            ),
            "## Interpretation",
            result["interpretation"],
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sidecar-gguf", type=Path, default=DEFAULT_SIDECAR_GGUF)
    parser.add_argument("--native-gguf", type=Path, default=DEFAULT_NATIVE_GGUF)
    parser.add_argument("--classifier-head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--embedding-binary", type=Path, default=Path("build-portable-avx2/bin/llama-embedding"))
    parser.add_argument("--tokenize-binary", type=Path, default=Path("build-portable-avx2/bin/llama-tokenize"))
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 7, 15])
    parser.add_argument(
        "--prompt-input",
        choices=["token_ids", "text_roundtrip"],
        default="token_ids",
        help=(
            "Use direct HF token IDs by default. The text round-trip mode is kept only "
            "to reproduce the earlier tokenizer-boundary failure."
        ),
    )
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--model-dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"benchmark_results/seqcls_native_mismatch_audit_{DATE}.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"benchmarks/results/seqcls_native_mismatch_audit_{DATE}.md"),
    )
    args = parser.parse_args()

    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else root / args.checkpoint_dir
    sidecar_gguf = args.sidecar_gguf if args.sidecar_gguf.is_absolute() else root / args.sidecar_gguf
    native_gguf = args.native_gguf if args.native_gguf.is_absolute() else root / args.native_gguf
    classifier_head = args.classifier_head if args.classifier_head.is_absolute() else root / args.classifier_head
    embedding_binary = args.embedding_binary if args.embedding_binary.is_absolute() else root / args.embedding_binary
    tokenize_binary = args.tokenize_binary if args.tokenize_binary.is_absolute() else root / args.tokenize_binary

    model, tokenizer, metrics = load_pytorch_model(checkpoint_dir, model_dtype=args.model_dtype)
    eval_metrics = metrics.get("eval", {}) if isinstance(metrics.get("eval"), dict) else {}
    prediction_path = Path(eval_metrics.get("prediction_path") or checkpoint_dir / "eval_predictions.jsonl")
    if not prediction_path.is_absolute():
        prediction_path = root / prediction_path
    trace = read_prediction_trace(prediction_path, max(args.indices))
    rows_by_index = load_mnli_rows(args.indices)

    audited_rows: list[dict[str, Any]] = []
    for index in args.indices:
        row = rows_by_index[index]
        text_prompt, hf_ids = render_prompt(tokenizer, row, args.max_seq_len)
        runtime_prompt = token_ids_prompt(hf_ids) if args.prompt_input == "token_ids" else text_prompt
        torch_probe = pytorch_probe(model=model, tokenizer=tokenizer, row=row, max_seq_len=args.max_seq_len)
        text_roundtrip_ids, text_roundtrip_tokenize_meta = llama_token_ids(
            binary=tokenize_binary,
            gguf=sidecar_gguf,
            prompt=text_prompt,
            timeout_seconds=args.timeout_seconds,
        )
        if args.prompt_input == "token_ids":
            llama_ids = hf_ids
            llama_tokenize_meta = {
                "mode": "direct_token_ids",
                "text_roundtrip_token_id_match": hf_ids == text_roundtrip_ids,
                "text_roundtrip": text_roundtrip_tokenize_meta,
            }
        else:
            llama_ids = text_roundtrip_ids
            llama_tokenize_meta = {
                "mode": "text_roundtrip",
                "text_roundtrip_token_id_match": hf_ids == text_roundtrip_ids,
                "text_roundtrip": text_roundtrip_tokenize_meta,
            }
        sidecar_hidden, sidecar_meta = llama_embedding(
            binary=embedding_binary,
            gguf=sidecar_gguf,
            prompt=runtime_prompt,
            threads=args.threads,
            ctx_size=args.ctx_size,
            timeout_seconds=args.timeout_seconds,
        )
        sidecar_head = head_logits(classifier_head, sidecar_hidden)
        native_logits, native_meta = llama_embedding(
            binary=embedding_binary,
            gguf=native_gguf,
            prompt=runtime_prompt,
            threads=args.threads,
            ctx_size=args.ctx_size,
            timeout_seconds=args.timeout_seconds,
        )
        torch_logits = torch_probe["logits"].astype(np.float32)
        sidecar_logits = sidecar_head["logits"].astype(np.float32)
        native_logits = native_logits.astype(np.float32)
        saved = trace.get(index, {})
        saved_scores = (
            np.asarray(saved.get("scores", []), dtype=np.float32)
            if isinstance(saved.get("scores"), list)
            else np.asarray([], dtype=np.float32)
        )
        token_id_match = hf_ids == llama_ids
        audited_rows.append(
            {
                "index": index,
                "label": int(row["label"]),
                "text_prompt": text_prompt,
                "runtime_prompt_kind": args.prompt_input,
                "tokenization": {
                    "hf_ids": hf_ids,
                    "llama_ids": llama_ids,
                    "token_id_match": token_id_match,
                    "text_roundtrip_ids": text_roundtrip_ids,
                    "text_roundtrip_token_id_match": hf_ids == text_roundtrip_ids,
                    "llama_tokenize": llama_tokenize_meta,
                },
                "predictions": {
                    "saved_pytorch": int(saved["prediction"]) if isinstance(saved.get("prediction"), int) else None,
                    "pytorch": prediction(torch_logits),
                    "sidecar": prediction(sidecar_logits),
                    "native": prediction(native_logits),
                },
                "logits": {
                    "saved_pytorch": list_float(saved_scores) if saved_scores.size else [],
                    "pytorch": list_float(torch_logits),
                    "sidecar": list_float(sidecar_logits),
                    "native": list_float(native_logits),
                },
                "margins": {
                    "saved_pytorch": margin(saved_scores) if saved_scores.size else None,
                    "pytorch": margin(torch_logits),
                    "sidecar": margin(sidecar_logits),
                    "native": margin(native_logits),
                },
                "comparisons": {
                    "token_id_match": token_id_match,
                    "pytorch_logits_vs_saved_relative_rms": tensor_rel_rms(torch_logits, saved_scores)
                    if saved_scores.size == torch_logits.size
                    else None,
                    "hidden_relative_rms": tensor_rel_rms(torch_probe["hidden"], sidecar_hidden),
                    "hidden_cosine": cosine(torch_probe["hidden"], sidecar_hidden),
                    "pytorch_vs_sidecar_logit_relative_rms": tensor_rel_rms(torch_logits, sidecar_logits),
                    "native_vs_sidecar_logit_relative_rms": tensor_rel_rms(native_logits, sidecar_logits),
                    "native_vs_pytorch_logit_relative_rms": tensor_rel_rms(native_logits, torch_logits),
                },
                "runtime": {
                    "sidecar_embedding": sidecar_meta,
                    "native_classifier": native_meta,
                },
            }
        )

    all_token_ids_match = all(row["comparisons"]["token_id_match"] for row in audited_rows)
    text_roundtrip_all_token_ids_match = all(
        row["tokenization"]["text_roundtrip_token_id_match"] for row in audited_rows
    )
    native_sidecar_agreement = sum(
        int(row["predictions"]["native"] == row["predictions"]["sidecar"]) for row in audited_rows
    ) / len(audited_rows)
    saved_native_agreement = sum(
        int(row["predictions"]["native"] == row["predictions"]["saved_pytorch"]) for row in audited_rows
    ) / len(audited_rows)
    hidden_rel_values = [float(row["comparisons"]["hidden_relative_rms"]) for row in audited_rows]
    hidden_cos_values = [float(row["comparisons"]["hidden_cosine"]) for row in audited_rows]
    native_sidecar_rel_values = [
        float(row["comparisons"]["native_vs_sidecar_logit_relative_rms"]) for row in audited_rows
    ]
    native_sidecar_logits_match = max(native_sidecar_rel_values) < 1e-5
    status = (
        "runtime_hidden_drift"
        if all_token_ids_match and native_sidecar_logits_match and saved_native_agreement < 1.0
        else "tokenization_or_native_head_mismatch"
        if not all_token_ids_match or not native_sidecar_logits_match
        else "pass"
    )
    interpretation = (
        "Token IDs match on the faithful runtime path and native GGUF logits match the sidecar "
        "GGUF+head path for these rows, so the native classifier-head plumbing is not the source "
        "of the disagreement. The blocker is packed-runtime hidden/logit drift relative to the "
        "PyTorch BitLinear checkpoint. The audit also records whether the older text "
        "decode/re-tokenize path is lossless; when it is not, direct token IDs are required for "
        "sequence-pair evaluation."
        if status == "runtime_hidden_drift"
        else "The mismatch is not yet isolated; inspect tokenization and native-vs-sidecar logit comparisons."
    )
    result = {
        "schema": "seqcls_native_mismatch_audit.v1",
        "date": DATE,
        "status": status,
        "artifacts": {
            "checkpoint": maybe_relative(checkpoint_dir, root),
            "sidecar_gguf": maybe_relative(sidecar_gguf, root),
            "native_gguf": maybe_relative(native_gguf, root),
            "classifier_head": maybe_relative(classifier_head, root),
            "prediction_trace": maybe_relative(prediction_path, root),
        },
        "prompt_input": args.prompt_input,
        "summary": {
            "indices": args.indices,
            "all_token_ids_match": all_token_ids_match,
            "text_roundtrip_all_token_ids_match": text_roundtrip_all_token_ids_match,
            "native_sidecar_prediction_agreement": native_sidecar_agreement,
            "saved_native_prediction_agreement": saved_native_agreement,
            "hidden_relative_rms_max": max(hidden_rel_values),
            "hidden_relative_rms_mean": float(np.mean(hidden_rel_values)),
            "hidden_cosine_min": min(hidden_cos_values),
            "native_vs_sidecar_logit_relative_rms_max": max(native_sidecar_rel_values),
            "native_vs_sidecar_logits_match": native_sidecar_logits_match,
        },
        "rows": audited_rows,
        "interpretation": interpretation,
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
