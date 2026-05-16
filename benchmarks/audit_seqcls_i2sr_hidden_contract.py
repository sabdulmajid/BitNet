#!/usr/bin/env python3
"""Compare PyTorch sequence-classification hidden states with I2_SR embeddings.

This audit narrows the current sidecar classifier mismatch.  The sidecar path
uses llama.cpp embeddings from a packed decoder backbone and applies the dense
sequence-classification head outside llama.cpp.  If token IDs match but pooled
hidden states diverge, the blocker is the runtime/model-state contract rather
than GLUE labels or tokenizer pair formatting.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
DEFAULT_CHECKPOINT = Path(
    "checkpoints/bitdistill-glue-seqcls-longwarmup/"
    "Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8"
)
DEFAULT_GGUF = Path(
    "models/seqcls-backbone-i2sr/"
    "Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8_bitnet_qwen_i2_sr.gguf"
)
DEFAULT_HEAD = Path(
    "models/seqcls-backbone-i2sr/"
    "Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8_score_head.npz"
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def run_command(command: list[str], *, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_seconds)


def parse_llama_ids(stdout: str) -> list[int]:
    line = stdout.strip().splitlines()[-1].strip()
    parsed = ast.literal_eval(line)
    if not isinstance(parsed, list) or not all(isinstance(item, int) for item in parsed):
        raise ValueError(f"unexpected llama-tokenize output: {line!r}")
    return [int(item) for item in parsed]


def tensor_rel_rms(lhs: np.ndarray, rhs: np.ndarray) -> float:
    diff = lhs.astype(np.float64) - rhs.astype(np.float64)
    denom = np.sqrt(np.mean(lhs.astype(np.float64) ** 2))
    return float(np.sqrt(np.mean(diff**2)) / max(denom, 1e-12))


def cosine(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs64 = lhs.astype(np.float64)
    rhs64 = rhs.astype(np.float64)
    denom = float(np.linalg.norm(lhs64) * np.linalg.norm(rhs64))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(lhs64, rhs64) / denom)


def load_first_mnli_row() -> dict[str, Any]:
    from datasets import load_dataset

    return dict(load_dataset("glue", "mnli")["validation_matched"][0])


def render_prompt(tokenizer: Any, row: dict[str, Any], max_seq_len: int) -> tuple[str, list[int]]:
    encoded = tokenizer(
        row["premise"],
        row["hypothesis"],
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=True,
    )
    input_ids = [int(item) for item in encoded["input_ids"]]
    prompt = tokenizer.decode(input_ids, clean_up_tokenization_spaces=False)
    return prompt, input_ids


def load_pytorch_model(checkpoint_dir: Path, *, model_dtype: str) -> tuple[Any, Any, dict[str, Any]]:
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    from train_bitdistill import prepare_bitnet_student

    metrics = read_json(checkpoint_dir / "metrics.json")
    if not metrics:
        raise FileNotFoundError(f"missing metrics.json in {checkpoint_dir}")
    state_path = checkpoint_dir / "custom_state_dict.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"missing custom_state_dict.pt in {checkpoint_dir}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True)

    prep = metrics.get("preparation", {}) if isinstance(metrics.get("preparation"), dict) else {}
    if metrics.get("method") in {"bitnet_sft", "bitdistill"}:
        shim_args = argparse.Namespace(
            use_subln=int(prep.get("subln_inserted", 0) or 0) > 0,
            subln_eps=1e-5,
            master_weight_dtype="fp32",
            scale_mode=metrics.get("scale_mode") or "tensor",
            exclude_linear_regex=metrics.get("exclude_linear_regex") or "score|classifier",
            quant_eps=1e-5,
            activation_quantization=True,
            ternary_init_mode="absmean",
            init_state_dict=None,
            ternary_init_iterations=0,
        )
        prepare_bitnet_student(model, shim_args)

    state = torch.load(state_path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "state dict mismatch: "
            f"missing={incompatible.missing_keys[:10]} unexpected={incompatible.unexpected_keys[:10]}"
        )
    if model_dtype == "bf16":
        model.to(torch.bfloat16)
    elif model_dtype == "fp32":
        model.float()
    else:
        raise ValueError(f"unsupported model_dtype={model_dtype}")
    model.eval()
    return model, tokenizer, metrics


def pytorch_probe(
    *,
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    max_seq_len: int,
) -> dict[str, Any]:
    encoded = tokenizer(
        row["premise"],
        row["hypothesis"],
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
        add_special_tokens=True,
    )
    with torch.inference_mode():
        outputs = model(**encoded, output_hidden_states=True)
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    sequence_index = int(attention_mask.sum().item()) - 1
    pooled_hidden = outputs.hidden_states[-1][0, sequence_index].float().cpu().numpy()
    logits = outputs.logits[0].float().cpu().numpy()
    return {
        "input_ids": [int(item) for item in input_ids.tolist()],
        "attention_mask_sum": int(attention_mask.sum().item()),
        "sequence_index": sequence_index,
        "hidden": pooled_hidden,
        "logits": logits,
        "prediction": int(np.argmax(logits)),
    }


def llama_embedding(
    *,
    binary: Path,
    gguf: Path,
    prompt: str,
    threads: int,
    ctx_size: int,
    timeout_seconds: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    command = [
        str(binary),
        "-m",
        str(gguf),
        "-p",
        prompt,
        "--pooling",
        "last",
        "--attention",
        "causal",
        "--embd-output-format",
        "json",
        "--embd-normalize",
        "-1",
        "-ngl",
        "0",
        "-t",
        str(threads),
        "-c",
        str(ctx_size),
    ]
    result = run_command(command, timeout_seconds=timeout_seconds)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:])
    parsed = json.loads(result.stdout)
    data = parsed.get("data", [])
    if not data:
        raise RuntimeError("llama-embedding returned no data")
    embedding = np.asarray(data[0]["embedding"], dtype=np.float32)
    return embedding, {
        "command": command,
        "returncode": result.returncode,
        "stdout_bytes": len(result.stdout.encode("utf-8")),
        "stderr_bytes": len(result.stderr.encode("utf-8")),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-10:]),
    }


def llama_token_ids(
    *,
    binary: Path,
    gguf: Path,
    prompt: str,
    timeout_seconds: int,
) -> tuple[list[int], dict[str, Any]]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(prompt)
        prompt_path = Path(handle.name)
    try:
        command = [
            str(binary),
            "-m",
            str(gguf),
            "-f",
            str(prompt_path),
            "--no-bos",
            "--ids",
            "--log-disable",
        ]
        result = run_command(command, timeout_seconds=timeout_seconds)
    finally:
        prompt_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:])
    return parse_llama_ids(result.stdout), {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-10:]),
    }


def head_logits(head_path: Path, hidden: np.ndarray) -> dict[str, Any]:
    head = np.load(head_path)
    weight_key = "score_weight" if "score_weight" in head.files else "classifier_weight"
    bias_key = "score_bias" if "score_bias" in head.files else "classifier_bias"
    weight = np.asarray(head[weight_key], dtype=np.float32)
    logits = weight @ hidden.astype(np.float32)
    has_bias = bias_key in head.files
    if has_bias:
        logits = logits + np.asarray(head[bias_key], dtype=np.float32)
    return {
        "weight_key": weight_key,
        "bias_key": bias_key if has_bias else None,
        "weight_shape": list(weight.shape),
        "logits": logits,
        "prediction": int(np.argmax(logits)),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    comparisons = result["comparisons"]
    pytorch = result["pytorch"]
    llama = result["llama"]
    hidden_rel = comparisons["hidden_relative_rms"]
    hidden_cos = comparisons["hidden_cosine"]
    if result["status"] == "pass":
        interpretation = "Token IDs and pooled hidden states match within this audit's tolerance for the sampled MNLI prompt."
    elif comparisons["token_id_match"] and hidden_cos > 0.99 and hidden_rel < 0.2:
        interpretation = (
            "Token IDs match and the packed decoder now follows the PyTorch pooled hidden state closely "
            f"(cosine {hidden_cos:.6f}), but the relative RMS error remains above the strict pass "
            f"threshold ({hidden_rel:.6f}). Treat this as a repaired runtime contract that still needs "
            "full-split validation and native classifier-head support, not a final deployable classifier."
        )
    elif comparisons["token_id_match"]:
        interpretation = (
            "Token IDs matching rules out the tokenizer pair-format path for this sample. "
            "The hidden-state comparison still shows a runtime/model-state mismatch, so the sidecar "
            "is not a deployable classifier yet."
        )
    else:
        interpretation = (
            "Token IDs do not match, so tokenizer or pair-format parity must be fixed before "
            "interpreting hidden-state or classifier differences."
        )
    return "\n\n".join(
        [
            f"# Sequence-Classification I2_SR Hidden-Contract Audit, {result['date']}",
            (
                "This audit tests whether the current sidecar classifier failure is caused by "
                "tokenization or by a deeper runtime-contract mismatch between the PyTorch "
                "sequence-classification checkpoint and the packed llama.cpp backbone."
            ),
            md_table(
                ["field", "value"],
                [
                    ["status", result["status"]],
                    ["label", result["example"]["label"]],
                    ["token IDs match", comparisons["token_id_match"]],
                    ["token count", len(result["tokenization"]["hf_ids"])],
                    ["PyTorch prediction", pytorch["prediction"]],
                    ["llama sidecar prediction", llama["prediction"]],
                    ["hidden relative RMS", comparisons["hidden_relative_rms"]],
                    ["hidden cosine", comparisons["hidden_cosine"]],
                    ["PyTorch hidden norm", comparisons["pytorch_hidden_l2"]],
                    ["llama hidden norm", comparisons["llama_hidden_l2"]],
                    ["logit relative RMS", comparisons["logit_relative_rms"]],
                ],
            ),
            "## Prompt",
            f"`{result['tokenization']['decoded_prompt']}`",
            "## Logits",
            md_table(
                ["source", "logits"],
                [
                    ["PyTorch model", pytorch["logits"]],
                    ["llama embedding + sidecar head", llama["logits"]],
                    ["PyTorch hidden + sidecar head", pytorch["sidecar_head_logits"]],
                ],
            ),
            "## Interpretation",
            interpretation,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--classifier-head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--embedding-binary", type=Path, default=Path("build-portable-avx2/bin/llama-embedding"))
    parser.add_argument("--tokenize-binary", type=Path, default=Path("build-portable-avx2/bin/llama-tokenize"))
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--model-dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/seqcls_i2sr_hidden_contract_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/seqcls_i2sr_hidden_contract_{DATE}.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else root / args.checkpoint_dir
    gguf = args.gguf if args.gguf.is_absolute() else root / args.gguf
    classifier_head = args.classifier_head if args.classifier_head.is_absolute() else root / args.classifier_head
    embedding_binary = args.embedding_binary if args.embedding_binary.is_absolute() else root / args.embedding_binary
    tokenize_binary = args.tokenize_binary if args.tokenize_binary.is_absolute() else root / args.tokenize_binary

    row = load_first_mnli_row()
    model, tokenizer, metrics = load_pytorch_model(checkpoint_dir, model_dtype=args.model_dtype)
    prompt, hf_ids = render_prompt(tokenizer, row, args.max_seq_len)
    torch_probe = pytorch_probe(model=model, tokenizer=tokenizer, row=row, max_seq_len=args.max_seq_len)
    if torch_probe["input_ids"] != hf_ids:
        raise RuntimeError("rendered prompt token IDs differ from PyTorch input IDs")

    llama_ids, llama_tokenize_meta = llama_token_ids(
        binary=tokenize_binary,
        gguf=gguf,
        prompt=prompt,
        timeout_seconds=args.timeout_seconds,
    )
    llama_hidden, llama_embedding_meta = llama_embedding(
        binary=embedding_binary,
        gguf=gguf,
        prompt=prompt,
        threads=args.threads,
        ctx_size=args.ctx_size,
        timeout_seconds=args.timeout_seconds,
    )
    llama_sidecar = head_logits(classifier_head, llama_hidden)
    torch_sidecar = head_logits(classifier_head, torch_probe["hidden"])

    pytorch_logits = torch_probe["logits"].astype(np.float32)
    llama_logits = llama_sidecar["logits"].astype(np.float32)
    token_id_match = hf_ids == llama_ids
    hidden_rel = tensor_rel_rms(torch_probe["hidden"], llama_hidden)
    hidden_cos = cosine(torch_probe["hidden"], llama_hidden)
    logit_rel = tensor_rel_rms(pytorch_logits, llama_logits)
    hidden_contract_pass = token_id_match and hidden_rel < 0.1 and hidden_cos > 0.9
    status = "pass" if hidden_contract_pass else "hidden_contract_mismatch"

    result = {
        "schema": "seqcls_i2sr_hidden_contract.v1",
        "date": DATE,
        "status": status,
        "checkpoint": {
            "path": maybe_relative(checkpoint_dir, root),
            "stored_accuracy": (metrics.get("eval", {}) if isinstance(metrics.get("eval"), dict) else {}).get("accuracy"),
            "scale_mode": metrics.get("scale_mode"),
            "method": metrics.get("method"),
        },
        "artifacts": {
            "gguf": maybe_relative(gguf, root),
            "classifier_head": maybe_relative(classifier_head, root),
            "embedding_binary": maybe_relative(embedding_binary, root),
            "tokenize_binary": maybe_relative(tokenize_binary, root),
            "head_weight_shape": llama_sidecar["weight_shape"],
        },
        "example": {
            "task": "mnli",
            "index": 0,
            "label": int(row["label"]),
            "premise": row["premise"],
            "hypothesis": row["hypothesis"],
        },
        "tokenization": {
            "decoded_prompt": prompt,
            "hf_ids": hf_ids,
            "llama_ids": llama_ids,
            "llama_tokenize": llama_tokenize_meta,
        },
        "pytorch": {
            "model_dtype": args.model_dtype,
            "sequence_index": torch_probe["sequence_index"],
            "attention_mask_sum": torch_probe["attention_mask_sum"],
            "logits": [float(item) for item in pytorch_logits],
            "prediction": torch_probe["prediction"],
            "sidecar_head_logits": [float(item) for item in torch_sidecar["logits"].astype(np.float32)],
            "sidecar_head_prediction": torch_sidecar["prediction"],
        },
        "llama": {
            "embedding": llama_embedding_meta,
            "logits": [float(item) for item in llama_logits],
            "prediction": llama_sidecar["prediction"],
        },
        "comparisons": {
            "token_id_match": token_id_match,
            "hidden_relative_rms": hidden_rel,
            "hidden_cosine": hidden_cos,
            "pytorch_hidden_l2": float(np.linalg.norm(torch_probe["hidden"].astype(np.float64))),
            "llama_hidden_l2": float(np.linalg.norm(llama_hidden.astype(np.float64))),
            "logit_relative_rms": logit_rel,
            "pytorch_logits_vs_sidecar_relative_rms": tensor_rel_rms(pytorch_logits, torch_sidecar["logits"]),
            "pytorch_logits_equal_sidecar_logits": bool(
                np.allclose(pytorch_logits, torch_sidecar["logits"], rtol=1e-2, atol=1e-2)
            ),
        },
        "interpretation": (
            "Tokenization matches but hidden states diverge; fix pooling/final-norm/runtime "
            "semantics before treating the sidecar path as classifier quality evidence."
            if not hidden_contract_pass and token_id_match
            else "Hidden contract passed for this sample."
        ),
    }

    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "comparisons": result["comparisons"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
