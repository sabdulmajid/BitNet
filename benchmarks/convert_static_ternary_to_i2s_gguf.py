#!/usr/bin/env python3
"""Directly pack scalar-scale static ternary checkpoints as I2_S GGUF.

This writer handles the scalar/tensor-scale case only. Row-scale checkpoints
need a compatibility-safe row-scale GGUF type or versioned layout and are
rejected by default.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch


class NamedInt(int):
    """Integer value with the .name attribute expected by llama.cpp's converter."""

    def __new__(cls, value: int, name: str):
        obj = int.__new__(cls, value)
        obj._name = name
        return obj

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"{self._name}({int(self)})"


def load_converter(path: Path) -> ModuleType:
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("llama_convert_hf_to_gguf_i2s", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import converter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_architecture(checkpoint_dir: Path) -> str:
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ValueError(f"{checkpoint_dir / 'config.json'} does not contain architectures")
    return str(architectures[0])


def validate_ternary_codes(key: str, tensor: torch.Tensor) -> None:
    values = torch.unique(tensor.cpu())
    allowed = torch.tensor([-1, 0, 1], dtype=values.dtype)
    invalid = values[~torch.isin(values, allowed)]
    if invalid.numel() > 0:
        raise ValueError(f"{key} contains non-ternary codes: {invalid.tolist()}")


def pack_i2_s_scalar(codes: torch.Tensor, scale: torch.Tensor) -> np.ndarray:
    if codes.ndim != 2:
        raise ValueError(f"I2_S packing expects a 2D weight matrix, got shape {tuple(codes.shape)}")
    if scale.numel() != 1:
        raise ValueError(f"scalar I2_S packing expects one scale, got shape {tuple(scale.shape)}")

    flat = (codes.to(torch.int16).cpu().numpy().reshape(-1) + 1).astype(np.uint8)
    if flat.size % 4 != 0:
        raise ValueError(f"I2_S code count must be divisible by 4, got {flat.size}")
    packed = (
        (flat[0::4] << np.uint8(6))
        | (flat[1::4] << np.uint8(4))
        | (flat[2::4] << np.uint8(2))
        | flat[3::4]
    ).astype(np.uint8)

    output = np.zeros(packed.size + 32, dtype=np.uint8)
    output[: packed.size] = packed
    output[packed.size : packed.size + 4] = np.asarray([float(scale.reshape(-1)[0].item())], dtype=np.float32).view(np.uint8)
    return output


def get_gguf_i2s_types(gguf: Any) -> tuple[Any, Any, bool]:
    has_native_constants = hasattr(gguf.GGMLQuantizationType, "I2_S") and hasattr(gguf.LlamaFileType, "MOSTLY_I2_S")
    i2_s_dtype = getattr(gguf.GGMLQuantizationType, "I2_S", NamedInt(36, "I2_S"))
    mostly_i2_s_ftype = getattr(gguf.LlamaFileType, "MOSTLY_I2_S", NamedInt(40, "MOSTLY_I2_S"))
    return i2_s_dtype, mostly_i2_s_ftype, has_native_constants


def make_i2s_model_class(
    base_cls: type,
    state: dict[str, torch.Tensor],
    converter: ModuleType,
    args: argparse.Namespace,
    summary: dict[str, Any],
    i2_s_dtype: Any,
) -> type:
    gguf = converter.gguf

    class StaticTernaryI2SModel(base_cls):  # type: ignore[misc, valid-type]
        model_arch = base_cls.model_arch

        def prepare_tensors(self):  # type: ignore[no-untyped-def]
            ternary_packed = 0
            copied_tensors = 0
            output_f16 = 0
            row_scale_rejected = 0
            packed_bytes = 0

            for key, tensor in state.items():
                if key.endswith(".weight_scale"):
                    continue

                if key.endswith(".ternary_weight"):
                    prefix = key[: -len(".ternary_weight")]
                    scale_key = f"{prefix}.weight_scale"
                    if scale_key not in state:
                        raise KeyError(f"missing scale tensor for {key}: {scale_key}")
                    if args.validate_codes:
                        validate_ternary_codes(key, tensor)
                    new_name = self.map_tensor_name(f"{prefix}.weight")
                    scale = state[scale_key]
                    if scale.numel() != 1:
                        row_scale_rejected += 1
                        raise ValueError(
                            f"{scale_key} has row or non-scalar scale shape {tuple(scale.shape)}; "
                            "direct I2_S writer currently supports scalar-scale checkpoints only"
                        )
                    if args.keep_output_f16 and new_name == self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT):
                        dense = tensor.to(dtype=torch.float16).mul(scale.to(dtype=torch.float16)).squeeze().numpy()
                        self.gguf_writer.add_tensor(new_name, dense, raw_dtype=gguf.GGMLQuantizationType.F16)
                        output_f16 += 1
                    else:
                        packed = pack_i2_s_scalar(tensor, scale)
                        self.gguf_writer.add_tensor(
                            new_name,
                            packed,
                            raw_shape=tuple(tensor.shape),
                            raw_dtype=i2_s_dtype,
                        )
                        ternary_packed += 1
                        packed_bytes += int(packed.nbytes)
                    continue

                new_name = self.map_tensor_name(key)
                if torch.is_floating_point(tensor):
                    if tensor.ndim <= 1 or new_name.endswith("_norm.weight"):
                        data = tensor.to(dtype=torch.float32).squeeze().numpy()
                        raw_dtype = gguf.GGMLQuantizationType.F32
                    else:
                        data = tensor.to(dtype=torch.float16).squeeze().numpy()
                        raw_dtype = gguf.GGMLQuantizationType.F16
                else:
                    data = tensor.cpu().squeeze().numpy()
                    raw_dtype = None
                self.gguf_writer.add_tensor(new_name, data, raw_dtype=raw_dtype)
                copied_tensors += 1

            summary.update(
                {
                    "ternary_i2s_packed": ternary_packed,
                    "copied_tensors": copied_tensors,
                    "output_f16_tensors": output_f16,
                    "row_scale_rejected": row_scale_rejected,
                    "packed_i2s_bytes": packed_bytes,
                    "output_tensors": ternary_packed + copied_tensors + output_f16,
                }
            )

    StaticTernaryI2SModel.__name__ = f"StaticTernaryI2S{base_cls.__name__}"
    return StaticTernaryI2SModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--ternary-state", type=Path, default=None)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--converter", type=Path, default=Path("3rdparty/llama.cpp/convert_hf_to_gguf.py"))
    parser.add_argument("--expect-ternary-keys", type=int, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--validate-codes", action="store_true")
    parser.add_argument("--keep-output-f16", action="store_true", default=True)
    args = parser.parse_args()

    ternary_state = args.ternary_state or args.checkpoint_dir / "ternary_state_dict.pt"
    converter = load_converter(args.converter)
    i2_s_dtype, mostly_i2_s_ftype, has_native_gguf_constants = get_gguf_i2s_types(converter.gguf)
    architecture = load_architecture(args.checkpoint_dir)
    base_cls = converter.Model.from_model_architecture(architecture)
    state = torch.load(ternary_state, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"expected a flat state dict in {ternary_state}")

    summary: dict[str, Any] = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "ternary_state": str(ternary_state),
        "architecture": architecture,
        "outfile": str(args.outfile),
        "converter": str(args.converter),
        "format_scope": "scalar-scale I2_S only",
        "i2_s_dtype": int(i2_s_dtype),
        "mostly_i2_s_file_type": int(mostly_i2_s_ftype),
        "has_native_gguf_python_constants": has_native_gguf_constants,
        "limitations": [
            "Rejects row-scale checkpoints.",
            "Uses existing tensor-scale I2_S layout.",
            "Keeps output.weight in F16 by default to match llama-quantize policy.",
        ],
    }

    model_cls = make_i2s_model_class(base_cls, state, converter, args, summary, i2_s_dtype)
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        model = model_cls(
            dir_model=args.checkpoint_dir,
            ftype=mostly_i2_s_ftype,
            fname_out=args.outfile,
            eager=True,
        )
        model.write()

    if args.expect_ternary_keys is not None:
        produced = int(summary.get("ternary_i2s_packed", 0)) + int(summary.get("output_f16_tensors", 0))
        if produced != args.expect_ternary_keys:
            raise SystemExit(f"expected {args.expect_ternary_keys} ternary keys, produced {produced}")
    if args.outfile.exists():
        summary["outfile_size_bytes"] = args.outfile.stat().st_size

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
