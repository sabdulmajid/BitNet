#!/usr/bin/env python3
"""Generate text from an exported W1.58A8 ternary student checkpoint.

The script reconstructs the student architecture with Hugging Face, replaces
the exported BitLinear layers with static ternary linear layers, then runs
standard autoregressive generation. The linear forward path matches the
training/export math:

    y = linear(dequant(round(x / s_x), int8) * s_x,
               ternary_weight * weight_scale,
               bias)

where s_x is the per-token absmax activation scale and ternary_weight is in
{-1, 0, 1}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticTernaryLinear(nn.Module):
    """Inference-only W1.58A8 linear layer.

    We keep ternary codes as int8 buffers and apply dynamic per-token int8
    activation quantization at every forward pass. The matmul is executed via
    dequantized PyTorch tensors here, but the arithmetic representation is the
    same ternary-weight / int8-activation contract used by BitNet CPU kernels.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        eps: float,
        scale_shape: torch.Size | tuple[int, ...] = (1,),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.register_buffer("ternary_weight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.ones(tuple(scale_shape), dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act_scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(self.eps) / 127.0
        q_x = torch.round(x.detach() / act_scale).clamp_(-128, 127)
        x_dequant = (q_x * act_scale).to(x.dtype)
        weight = self.ternary_weight.to(dtype=x.dtype) * self.weight_scale.to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x_dequant, weight, bias)


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent_name = ".".join(parts[:-1])
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, parts[-1], module)


def replace_ternary_linears(
    model: nn.Module,
    ternary_state: dict[str, torch.Tensor],
    *,
    eps: float,
) -> int:
    names = sorted(key[: -len(".ternary_weight")] for key in ternary_state if key.endswith(".ternary_weight"))
    if not names:
        raise ValueError("checkpoint contains no *.ternary_weight tensors")

    replaced = 0
    for name in names:
        original = model.get_submodule(name)
        if not isinstance(original, nn.Linear):
            raise TypeError(f"expected {name} to be nn.Linear in the base model, got {type(original).__name__}")
        has_bias = f"{name}.bias" in ternary_state or original.bias is not None
        scale = ternary_state.get(f"{name}.weight_scale")
        scale_shape = scale.shape if scale is not None else (1,)
        ternary = StaticTernaryLinear(
            original.in_features,
            original.out_features,
            bias=has_bias,
            eps=eps,
            scale_shape=scale_shape,
        )
        set_submodule(model, name, ternary)
        replaced += 1
    return replaced


def resolve_paths(args: argparse.Namespace) -> tuple[str, str, Path]:
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    model_path = args.model or str(checkpoint_dir) if checkpoint_dir else args.model
    tokenizer_path = args.tokenizer or model_path
    ternary_state_path = Path(args.ternary_state) if args.ternary_state else None
    if ternary_state_path is None:
        if checkpoint_dir is None:
            raise ValueError("pass --ternary-state or --checkpoint-dir")
        ternary_state_path = checkpoint_dir / "ternary_state_dict.pt"
    if model_path is None:
        raise ValueError("pass --model or --checkpoint-dir")
    return model_path, tokenizer_path, ternary_state_path


def load_model(args: argparse.Namespace) -> tuple[nn.Module, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path, tokenizer_path, ternary_state_path = resolve_paths(args)
    if not ternary_state_path.exists():
        raise FileNotFoundError(f"ternary checkpoint not found: {ternary_state_path}")

    dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    ternary_state = torch.load(ternary_state_path, map_location="cpu", weights_only=True)
    replaced = replace_ternary_linears(model, ternary_state, eps=args.quant_eps)
    missing, unexpected = model.load_state_dict(ternary_state, strict=False)
    relevant_missing = [key for key in missing if not key.endswith(".weight")]
    if relevant_missing or unexpected:
        raise RuntimeError(
            "ternary checkpoint did not match model; "
            f"missing={relevant_missing[:10]} unexpected={unexpected[:10]}"
        )
    print(f"Loaded {ternary_state_path} and replaced {replaced} Linear modules with StaticTernaryLinear", flush=True)
    return model, tokenizer


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is not available")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an exported ternary BitNet student checkpoint")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory containing config/tokenizer and ternary_state_dict.pt")
    parser.add_argument("--ternary-state", default=None, help="Path to ternary_state_dict.pt")
    parser.add_argument("--model", default=None, help="HF model id/path for the student architecture")
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer id/path; defaults to --model or --checkpoint-dir")
    parser.add_argument("--prompt", default="The future of artificial intelligence lies in...")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--quant-eps", type=float, default=1e-5)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model, tokenizer = load_model(args)
    device = select_device(args.device)
    model.to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True), flush=True)


if __name__ == "__main__":
    main()
