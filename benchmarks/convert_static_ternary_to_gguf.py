#!/usr/bin/env python3
"""Convert a static ternary checkpoint directly to GGUF without HF materialization.

This is a direct source-to-GGUF bridge for `ternary_state_dict.pt`: it loads the
static ternary state, reconstructs effective dense tensors in memory, and then
reuses the vendored llama.cpp Hugging Face GGUF writer for metadata, tokenizer,
tensor-name mapping, and dense GGUF serialization.

It intentionally does not claim to be the final packed ternary writer. `I2_S`
still requires a GGUF quantization step or a future row-scale-aware GGUF type.
The vendored Python writer may silently fall back to F16 for unsupported
quantized tensor shapes, so quantized outtypes are blocked unless explicitly
enabled for experiments.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import torch


SIDECAR_CONFIG = "config.json"


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"fp16", "float16", "f16"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "f32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def load_converter(path: Path) -> ModuleType:
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("llama_convert_hf_to_gguf_static_ternary", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import converter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_architecture(checkpoint_dir: Path) -> str:
    config_path = checkpoint_dir / SIDECAR_CONFIG
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ValueError(f"{config_path} does not contain a non-empty architectures list")
    return str(architectures[0])


def validate_ternary_codes(key: str, tensor: torch.Tensor) -> None:
    values = torch.unique(tensor.cpu())
    allowed = torch.tensor([-1, 0, 1], dtype=values.dtype)
    invalid = values[~torch.isin(values, allowed)]
    if invalid.numel() > 0:
        raise ValueError(f"{key} contains non-ternary codes: {invalid.tolist()}")


def iter_effective_tensors(
    state: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype,
    validate_codes: bool,
    summary: dict[str, Any],
) -> Iterable[tuple[str, torch.Tensor]]:
    scale_rank_histogram: Counter[str] = Counter()
    ternary_dtype_histogram: Counter[str] = Counter()
    copied_dtype_histogram: Counter[str] = Counter()
    scale_shape_examples: list[dict[str, object]] = []
    ternary_count = 0
    copied_count = 0

    for key, tensor in state.items():
        if key.endswith(".weight_scale"):
            continue
        if key.endswith(".ternary_weight"):
            prefix = key[: -len(".ternary_weight")]
            scale_key = f"{prefix}.weight_scale"
            if scale_key not in state:
                raise KeyError(f"missing scale tensor for {key}: {scale_key}")
            if validate_codes:
                validate_ternary_codes(key, tensor)
            scale = state[scale_key].to(dtype=dtype)
            scale_rank_histogram[str(scale.ndim)] += 1
            ternary_dtype_histogram[str(tensor.dtype).replace("torch.", "")] += 1
            if len(scale_shape_examples) < 8:
                scale_shape_examples.append({"key": scale_key, "shape": list(scale.shape)})
            ternary_count += 1
            yield f"{prefix}.weight", tensor.to(dtype=dtype).mul(scale)
        else:
            copied_count += 1
            copied_dtype_histogram[str(tensor.dtype).replace("torch.", "")] += 1
            yield key, tensor.to(dtype=dtype) if torch.is_floating_point(tensor) else tensor.cpu()

    summary.update(
        {
            "ternary_materialized": ternary_count,
            "copied_tensors": copied_count,
            "output_tensors": ternary_count + copied_count,
            "materialized_lm_head": int("lm_head.ternary_weight" in state or "lm_head.weight" in state),
            "scale_rank_histogram": dict(sorted(scale_rank_histogram.items())),
            "ternary_dtype_histogram": dict(sorted(ternary_dtype_histogram.items())),
            "copied_dtype_histogram": dict(sorted(copied_dtype_histogram.items())),
            "scale_shape_examples": scale_shape_examples,
        }
    )


def make_static_model_class(base_cls: type, state: dict[str, torch.Tensor], dtype: torch.dtype, validate_codes: bool, summary: dict[str, Any]) -> type:
    class StaticTernaryModel(base_cls):  # type: ignore[misc, valid-type]
        model_arch = base_cls.model_arch

        def get_tensors(self):  # type: ignore[no-untyped-def]
            yield from iter_effective_tensors(state, dtype=dtype, validate_codes=validate_codes, summary=summary)

    StaticTernaryModel.__name__ = f"StaticTernary{base_cls.__name__}"
    return StaticTernaryModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--ternary-state", type=Path, default=None)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--outtype", choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0"], default="f16")
    parser.add_argument("--materialize-dtype", choices=["fp16", "float16", "bf16", "bfloat16", "fp32", "float32"], default="float16")
    parser.add_argument("--expect-ternary-keys", type=int, default=None)
    parser.add_argument("--converter", type=Path, default=Path("3rdparty/llama.cpp/convert_hf_to_gguf.py"))
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--validate-codes", action="store_true")
    parser.add_argument("--use-temp-file", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-converter-quantized-outtype",
        action="store_true",
        help=(
            "Permit q8_0/tq1_0/tq2_0 output through the Python converter. "
            "Use only for experiments because unsupported shapes can fall back to F16."
        ),
    )
    args = parser.parse_args()

    if args.outtype not in {"f32", "f16", "bf16"} and not args.allow_converter_quantized_outtype:
        raise SystemExit(
            f"{args.outtype} direct output is not enabled by default. "
            "Use dense f16/bf16/f32 here, then run the audited llama-quantize bridge; "
            "or pass --allow-converter-quantized-outtype for an experimental converter test."
        )

    ternary_state = args.ternary_state or args.checkpoint_dir / "ternary_state_dict.pt"
    dtype = dtype_from_name(args.materialize_dtype)
    converter = load_converter(args.converter)
    architecture = load_architecture(args.checkpoint_dir)
    base_cls = converter.Model.from_model_architecture(architecture)
    ftype_map = {
        "f32": converter.gguf.LlamaFileType.ALL_F32,
        "f16": converter.gguf.LlamaFileType.MOSTLY_F16,
        "bf16": converter.gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": converter.gguf.LlamaFileType.MOSTLY_Q8_0,
        "tq1_0": converter.gguf.LlamaFileType.MOSTLY_TQ1_0,
        "tq2_0": converter.gguf.LlamaFileType.MOSTLY_TQ2_0,
    }

    state = torch.load(ternary_state, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"expected a flat state dict in {ternary_state}")

    summary: dict[str, Any] = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "ternary_state": str(ternary_state),
        "architecture": architecture,
        "outfile": str(args.outfile),
        "outtype": args.outtype,
        "materialize_dtype": str(dtype).replace("torch.", ""),
        "converter": str(args.converter),
        "direct_writer_limitations": [
            "This reconstructs effective dense tensors in memory.",
            "It avoids writing an intermediate dense Hugging Face checkpoint.",
            "It is not a packed row-scale I2_S writer.",
            "Quantized Python-converter outtypes are experimental because unsupported shapes can fall back to F16.",
        ],
    }

    model_cls = make_static_model_class(base_cls, state, dtype, args.validate_codes, summary)
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        model = model_cls(
            dir_model=args.checkpoint_dir,
            ftype=ftype_map[args.outtype],
            fname_out=args.outfile,
            use_temp_file=args.use_temp_file,
            eager=True,
            dry_run=args.dry_run,
        )
        model.write()

    if args.expect_ternary_keys is not None and summary.get("ternary_materialized") != args.expect_ternary_keys:
        raise SystemExit(
            f"expected {args.expect_ternary_keys} ternary keys, "
            f"found {summary.get('ternary_materialized')}"
        )
    if args.outfile.exists():
        summary["outfile_size_bytes"] = args.outfile.stat().st_size

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
