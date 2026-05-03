#!/usr/bin/env python3
"""Export BitNet-style ternary weights from a saved HF safetensors checkpoint.

This is primarily a repair/export utility for checkpoints whose train-time
forward path used BitLinear but whose saved state dict contains FP master
weights. It writes the same state structure consumed by eval_ternary.py:

    <linear>.ternary_weight: int8 tensor in {-1, 0, 1}
    <linear>.weight_scale:   FP32 tensor scalar or per-row scale

Non-quantized tensors are copied through unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open


DEFAULT_LINEAR_KEY_REGEX = (
    r"(?:"
    r"model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)"
    r"|lm_head"
    r")\.weight$"
)


def safetensor_files(checkpoint_dir: Path, input_file: Path | None) -> list[Path]:
    if input_file is not None:
        return [input_file]

    single = checkpoint_dir / "model.safetensors"
    if single.exists():
        return [single]

    index_file = checkpoint_dir / "model.safetensors.index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"no model.safetensors or model.safetensors.index.json found in {checkpoint_dir}")

    with index_file.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"invalid safetensors index: {index_file}")

    return sorted({checkpoint_dir / shard for shard in weight_map.values()})


def available_backup_path(path: Path) -> Path:
    candidate = path.with_suffix(path.suffix + ".bak")
    index = 1
    while candidate.exists():
        candidate = path.with_suffix(path.suffix + f".bak{index}")
        index += 1
    return candidate


def quantize_weight(weight: torch.Tensor, scale_mode: str, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    weight = weight.float().cpu()
    if scale_mode == "row":
        scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)
    elif scale_mode == "tensor":
        scale = weight.abs().mean().clamp_min(eps).reshape(1)
    else:
        raise ValueError(f"unsupported scale_mode={scale_mode}")
    codes = torch.round(weight / scale).clamp_(-1, 1).to(torch.int8)
    return codes, scale


def export_ternary(
    files: Iterable[Path],
    output_path: Path,
    *,
    linear_key_regex: str,
    scale_mode: str,
    eps: float,
    backup_existing: bool,
    dry_run: bool,
) -> dict[str, int | float]:
    pattern = re.compile(linear_key_regex)
    export: dict[str, torch.Tensor] = {}
    ternary_keys = 0
    ternary_elements = 0
    ternary_neg = 0
    ternary_zero = 0
    ternary_pos = 0
    source_tensors = 0

    for file_path in files:
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                source_tensors += 1
                tensor = handle.get_tensor(key)
                if pattern.match(key):
                    module_name = key[: -len(".weight")]
                    codes, scale = quantize_weight(tensor, scale_mode, eps)
                    export[f"{module_name}.ternary_weight"] = codes
                    export[f"{module_name}.weight_scale"] = scale
                    ternary_keys += 1
                    ternary_elements += codes.numel()
                    ternary_neg += int((codes == -1).sum())
                    ternary_zero += int((codes == 0).sum())
                    ternary_pos += int((codes == 1).sum())
                else:
                    export[key] = tensor.cpu()

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and backup_existing:
            backup_path = available_backup_path(output_path)
            output_path.rename(backup_path)
            print(f"Backed up existing output to {backup_path}", flush=True)
        torch.save(export, output_path)

    frac = lambda count: float(count) / float(ternary_elements) if ternary_elements else 0.0
    return {
        "source_tensors": source_tensors,
        "output_tensors": len(export),
        "ternary_keys": ternary_keys,
        "ternary_elements": ternary_elements,
        "ternary_neg_frac": frac(ternary_neg),
        "ternary_zero_frac": frac(ternary_zero),
        "ternary_pos_frac": frac(ternary_pos),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export W1.58 ternary tensors from model.safetensors")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--linear-key-regex", default=DEFAULT_LINEAR_KEY_REGEX)
    parser.add_argument("--scale-mode", default="tensor", choices=["tensor", "row"])
    parser.add_argument("--quant-eps", type=float, default=1e-5)
    parser.add_argument("--expect-ternary-keys", type=int, default=None)
    parser.add_argument("--backup-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_path = args.output or args.checkpoint_dir / "ternary_state_dict.pt"
    files = safetensor_files(args.checkpoint_dir, args.input_file)
    stats = export_ternary(
        files,
        output_path,
        linear_key_regex=args.linear_key_regex,
        scale_mode=args.scale_mode,
        eps=args.quant_eps,
        backup_existing=args.backup_existing,
        dry_run=args.dry_run,
    )
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}={value:.6f}", flush=True)
        else:
            print(f"{key}={value}", flush=True)

    if args.expect_ternary_keys is not None and stats["ternary_keys"] != args.expect_ternary_keys:
        raise SystemExit(
            f"expected {args.expect_ternary_keys} ternary keys, found {stats['ternary_keys']}"
        )
    if not args.dry_run:
        print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
