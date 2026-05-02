#!/usr/bin/env python3
"""Empirical sanity check for BitNet ternarization on ordinary FP weights.

The formulas here mirror the conversion paths in this repository:

1. utils/preprocess-huggingface-bitnet.py::quant_weight_fp16 and
   utils/convert-hf-to-gguf-bitnet.py::BitnetModel.weight_quant:
       s = 1 / mean(abs(W))
       Wq = clamp(round(W * s), -1, 1) / s

2. utils/convert-hf-to-gguf-bitnet.py::transform_to_tl{1,2} plus
   preprocess_weights_tl{1,2}, and src/ggml-bitnet-mad.cpp::quantize_i2_s:
       scale = max(abs(W))
       packed = sign(W)
   The executable kernels then multiply by that single stored scale.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np


def _try_torch() -> Any | None:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return None
    return torch


def torch_mean_abs_ternary(torch: Any, weight: Any) -> Any:
    """Exact PyTorch operation used by the repo's preprocessing scripts."""
    weight = weight.to(torch.float32)
    s = 1.0 / weight.abs().mean().clamp_(min=1e-5)
    return (weight * s).round().clamp(-1, 1) / s


def numpy_mean_abs_ternary(weight: np.ndarray) -> np.ndarray:
    """NumPy fallback for the same operation."""
    weight = weight.astype(np.float32)
    scale = 1.0 / np.clip(np.mean(np.abs(weight)), 1e-5, None)
    return np.clip(np.round(weight * scale), -1, 1) / scale


def numpy_sign_max_ternary(weight: np.ndarray) -> np.ndarray:
    """Dequantized equivalent of the generic TL/I2 sign + max-scale path."""
    weight = weight.astype(np.float32)
    scale = np.max(np.abs(weight))
    signs = np.where(np.abs(weight) < 1e-6, 0.0, np.sign(weight))
    return signs * scale


def summarize(name: str, weight: np.ndarray, quantized: np.ndarray, x: np.ndarray) -> dict[str, float]:
    y = x @ weight.T
    yq = x @ quantized.T
    delta_w = weight - quantized
    delta_y = y - yq
    rel_w = np.linalg.norm(delta_w, ord="fro") / np.linalg.norm(weight, ord="fro")
    rel_y = np.linalg.norm(delta_y, ord="fro") / np.linalg.norm(y, ord="fro")
    cos_w = float(np.vdot(weight.reshape(-1), quantized.reshape(-1)) /
                  (np.linalg.norm(weight.reshape(-1)) * np.linalg.norm(quantized.reshape(-1))))
    cos_y = float(np.vdot(y.reshape(-1), yq.reshape(-1)) /
                  (np.linalg.norm(y.reshape(-1)) * np.linalg.norm(yq.reshape(-1))))
    zero_frac = float(np.mean(quantized == 0.0))
    return {
        "relative_weight_fro_error": float(rel_w),
        "relative_output_fro_error": float(rel_y),
        "weight_cosine": cos_w,
        "output_cosine": cos_y,
        "zero_fraction": zero_frac,
        "unique_values": float(len(np.unique(quantized))),
    }


def theoretical_mean_abs_mse() -> float:
    """E[(Z - Q(Z))^2] for Z~N(0,1), alpha=E|Z|, Q in {-alpha,0,+alpha}."""
    alpha = math.sqrt(2.0 / math.pi)
    threshold = alpha / 2.0
    phi = math.exp(-0.5 * threshold * threshold) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(threshold / math.sqrt(2.0)))
    tail_prob = 2.0 * (1.0 - cdf)
    tail_abs_moment = 2.0 * phi
    return 1.0 - 2.0 * alpha * tail_abs_moment + alpha * alpha * tail_prob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-features", type=int, default=2048)
    parser.add_argument("--in-features", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch = _try_torch()
    rng = np.random.default_rng(args.seed)

    if torch is not None:
        torch.manual_seed(args.seed)
        w_t = torch.randn(args.out_features, args.in_features, dtype=torch.float16)
        x_t = torch.randn(args.batch, args.in_features, dtype=torch.float16)
        w = w_t.float().cpu().numpy()
        x = x_t.float().cpu().numpy()
        mean_abs_q = torch_mean_abs_ternary(torch, w_t).float().cpu().numpy()
        backend = f"torch {torch.__version__}"
    else:
        w = rng.standard_normal((args.out_features, args.in_features)).astype(np.float16).astype(np.float32)
        x = rng.standard_normal((args.batch, args.in_features)).astype(np.float16).astype(np.float32)
        mean_abs_q = numpy_mean_abs_ternary(w)
        backend = "numpy fallback; torch is not installed"

    sign_max_q = numpy_sign_max_ternary(w)
    mean_abs_stats = summarize("mean_abs", w, mean_abs_q, x)
    sign_max_stats = summarize("sign_max", w, sign_max_q, x)

    print(f"backend: {backend}")
    print(f"shape: W=({args.out_features}, {args.in_features}), X=({args.batch}, {args.in_features})")
    print(f"theoretical_mean_abs_relative_fro_error: {math.sqrt(theoretical_mean_abs_mse()):.6f}")
    for name, stats in (("mean_abs_ternary_repo_formula", mean_abs_stats),
                        ("sign_max_tl_i2_generic_path", sign_max_stats)):
        print(f"\n{name}")
        for key, value in stats.items():
            if key == "unique_values":
                print(f"  {key}: {int(value)}")
            else:
                print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
