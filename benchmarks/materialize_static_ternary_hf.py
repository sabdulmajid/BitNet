#!/usr/bin/env python3
"""Materialize a static ternary export as a dense Hugging Face checkpoint.

This is an intermediate bridge for GGUF conversion. It converts tensors named
`*.ternary_weight` plus `*.weight_scale` back to dense `*.weight` tensors, while
copying non-ternary tensors and tokenizer/config sidecars through unchanged.

It is not a packed format writer. Use it to test whether a GGUF conversion path
can preserve the PyTorch static-ternary checkpoint semantics before investing in
a native I2_S writer for `ternary_state_dict.pt`.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


SIDECAR_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"fp16", "float16", "f16"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "f32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def copy_sidecars(source_dir: Path, output_dir: Path, dtype_name: str, *, untie_word_embeddings: bool) -> None:
    for filename in SIDECAR_FILES:
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)

    config_path = output_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["dtype"] = "float16" if dtype_name in {"fp16", "float16", "f16"} else dtype_name
        config["torch_dtype"] = config["dtype"]
        if untie_word_embeddings:
            config["tie_word_embeddings"] = False
        config_path.write_text(json.dumps(config, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def validate_ternary_codes(key: str, tensor: torch.Tensor) -> None:
    values = torch.unique(tensor.cpu())
    allowed = torch.tensor([-1, 0, 1], dtype=values.dtype)
    invalid = values[~torch.isin(values, allowed)]
    if invalid.numel() > 0:
        raise ValueError(f"{key} contains non-ternary codes: {invalid.tolist()}")


def materialize(input_path: Path, output_dir: Path, dtype: torch.dtype, *, validate_codes: bool) -> dict[str, object]:
    state = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"expected a flat state dict in {input_path}")

    output: dict[str, torch.Tensor] = {}
    ternary_count = 0
    copied_count = 0
    scale_rank_histogram: Counter[str] = Counter()
    ternary_dtype_histogram: Counter[str] = Counter()
    scale_shape_examples: list[dict[str, object]] = []

    for key, tensor in state.items():
        if key.endswith(".weight_scale"):
            continue
        if key.endswith(".ternary_weight"):
            prefix = key[: -len(".ternary_weight")]
            scale_key = f"{prefix}.weight_scale"
            if scale_key not in state:
                raise KeyError(f"missing scale tensor for {key}: {scale_key}")
            scale = state[scale_key].to(dtype=dtype)
            if validate_codes:
                validate_ternary_codes(key, tensor)
            scale_rank_histogram[str(scale.ndim)] += 1
            ternary_dtype_histogram[str(tensor.dtype).replace("torch.", "")] += 1
            if len(scale_shape_examples) < 8:
                scale_shape_examples.append({"key": scale_key, "shape": list(scale.shape)})
            output[f"{prefix}.weight"] = tensor.to(dtype=dtype).mul(scale)
            ternary_count += 1
        else:
            if torch.is_floating_point(tensor):
                output[key] = tensor.to(dtype=dtype)
            else:
                output[key] = tensor.cpu()
            copied_count += 1

    save_file(output, output_dir / "model.safetensors")
    return {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "dtype": str(dtype).replace("torch.", ""),
        "ternary_materialized": ternary_count,
        "copied_tensors": copied_count,
        "output_tensors": len(output),
        "materialized_lm_head": int("lm_head.weight" in output),
        "scale_rank_histogram": dict(sorted(scale_rank_histogram.items())),
        "ternary_dtype_histogram": dict(sorted(ternary_dtype_histogram.items())),
        "scale_shape_examples": scale_shape_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize static ternary weights as dense HF safetensors")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--ternary-state", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", default="float16", choices=["fp16", "float16", "bf16", "bfloat16", "fp32", "float32"])
    parser.add_argument("--expect-ternary-keys", type=int, default=None)
    parser.add_argument("--validate-codes", action="store_true", help="Verify every ternary tensor contains only -1, 0, 1")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ternary_path = args.ternary_state or args.checkpoint_dir / "ternary_state_dict.pt"
    dtype = dtype_from_name(args.dtype)

    manifest = materialize(ternary_path, args.output_dir, dtype, validate_codes=args.validate_codes)
    copy_sidecars(
        args.checkpoint_dir,
        args.output_dir,
        args.dtype,
        untie_word_embeddings=bool(manifest["materialized_lm_head"]),
    )

    if args.expect_ternary_keys is not None and manifest["ternary_materialized"] != args.expect_ternary_keys:
        raise SystemExit(
            f"expected {args.expect_ternary_keys} ternary keys, "
            f"found {manifest['ternary_materialized']}"
        )

    (args.output_dir / "static_ternary_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
