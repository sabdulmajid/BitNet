#!/usr/bin/env python3
"""Directly pack static ternary checkpoints as I2_S/I2_SR GGUF.

This writer handles the scalar/tensor-scale case by default. Row-scale
checkpoints need a compatibility-safe GGUF type or versioned layout and are
therefore rejected unless --row-scale-prototype or --row-scale-qtype=i2_sr is
passed. The i2_sr mode emits the stable row-scale type IDs used by this fork's
promoted I2_SR runtime.
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


def load_config(checkpoint_dir: Path) -> dict[str, Any]:
    return json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))


def classifier_label_count(state: dict[str, torch.Tensor], config: dict[str, Any]) -> int | None:
    for key in ("score.weight", "classifier.weight"):
        tensor = state.get(key)
        if tensor is not None and tensor.ndim == 2:
            return int(tensor.shape[0])
    value = config.get("num_labels")
    return int(value) if isinstance(value, int) and value > 0 else None


def classifier_labels(config: dict[str, Any], label_count: int | None) -> list[str]:
    id2label = config.get("id2label")
    if isinstance(id2label, dict):
        labels: list[str] = []
        for idx in range(label_count or len(id2label)):
            value = id2label.get(str(idx), id2label.get(idx))
            labels.append(str(value) if value is not None else str(idx))
        return labels
    return [str(idx) for idx in range(label_count or 0)]


def normalize_bitdistill_subln_name(name: str) -> str:
    """Map this repo's SubLNLinear wrapper keys to llama.cpp BitNet tensor names."""
    return (
        name.replace(".self_attn.o_proj.proj.", ".self_attn.o_proj.")
        .replace(".self_attn.o_proj.proj", ".self_attn.o_proj")
        .replace(".mlp.down_proj.proj.", ".mlp.down_proj.")
        .replace(".mlp.down_proj.proj", ".mlp.down_proj")
        .replace(".self_attn.o_proj.subln.weight", ".self_attn.attn_sub_norm.weight")
        .replace(".mlp.down_proj.subln.weight", ".mlp.ffn_sub_norm.weight")
    )


def validate_ternary_codes(key: str, tensor: torch.Tensor) -> None:
    values = torch.unique(tensor.cpu())
    allowed = torch.tensor([-1, 0, 1], dtype=values.dtype)
    invalid = values[~torch.isin(values, allowed)]
    if invalid.numel() > 0:
        raise ValueError(f"{key} contains non-ternary codes: {invalid.tolist()}")


def i2_s_logical_rows(codes: torch.Tensor) -> int:
    if codes.ndim not in {2, 3}:
        raise ValueError(f"I2_S packing expects a 2D matrix or 3D merged expert tensor, got shape {tuple(codes.shape)}")
    return int(np.prod(tuple(codes.shape[:-1])))


def pack_i2_s_codes_x86_act(codes: torch.Tensor) -> np.ndarray:
    """Pack BitNet's active x86 I2_S layout.

    `include/gemm-config.h` defines ACT_PARALLEL in this fork, so the x86
    runtime packs each flat row-major group of 128 ternary codes into 32 bytes.
    This is the layout produced by `quantize_i2_s` and consumed by the current
    `ggml_vec_dot_i2_i8_s` kernels.

    Dense tensors are shaped `[out, in]`. Merged MoE expert tensors are shaped
    `[experts, out, in]`; they are packed as expert-major logical rows while
    preserving the 3D GGUF raw shape at the caller.
    """
    _ = i2_s_logical_rows(codes)

    q8 = (codes.to(torch.int16).cpu().numpy() + 1).astype(np.uint8, copy=False)
    qk_i2_s = 128
    if q8.shape[-1] % qk_i2_s != 0:
        raise ValueError(
            "I2_S x86 ACT_PARALLEL packing expects each logical row's input "
            f"dimension to be divisible by {qk_i2_s}, got shape {tuple(codes.shape)}"
        )

    groups = q8.reshape(-1, qk_i2_s)
    packed = (
        (groups[:, 0:32] << np.uint8(6))
        | (groups[:, 32:64] << np.uint8(4))
        | (groups[:, 64:96] << np.uint8(2))
        | groups[:, 96:128]
    ).astype(np.uint8, copy=False)
    return packed.reshape(-1)


def pack_i2_s_scalar(codes: torch.Tensor, scale: torch.Tensor) -> np.ndarray:
    if scale.numel() != 1:
        raise ValueError(f"scalar I2_S packing expects one scale, got shape {tuple(scale.shape)}")

    packed = pack_i2_s_codes_x86_act(codes)

    output = np.zeros(packed.size + 32, dtype=np.uint8)
    output[: packed.size] = packed
    output[packed.size : packed.size + 4] = np.asarray([float(scale.reshape(-1)[0].item())], dtype=np.float32).view(np.uint8)
    return output


def pack_i2_s_row_prototype(codes: torch.Tensor, scale: torch.Tensor) -> np.ndarray:
    rows = i2_s_logical_rows(codes)
    row_scales = scale.reshape(-1)
    if row_scales.numel() != rows:
        raise ValueError(
            f"row-scale I2_S packing expects {rows} logical row scales for "
            f"weight shape {tuple(codes.shape)}, got scale shape {tuple(scale.shape)}"
        )

    packed = pack_i2_s_codes_x86_act(codes)
    if packed.size % 4 != 0:
        raise ValueError(f"row-scale I2_S scale offset must be 4-byte aligned, got {packed.size}")

    scale_bytes = row_scales.to(dtype=torch.float32).cpu().numpy().astype(np.float32, copy=False).view(np.uint8)
    output = np.zeros(packed.size + scale_bytes.size + 32, dtype=np.uint8)
    output[: packed.size] = packed
    output[packed.size : packed.size + scale_bytes.size] = scale_bytes
    return output


def get_gguf_i2s_types(gguf: Any, *, row_scale_qtype: str | None) -> tuple[Any, Any, Any, Any, bool, bool]:
    has_native_i2s_constants = hasattr(gguf.GGMLQuantizationType, "I2_S") and hasattr(gguf.LlamaFileType, "MOSTLY_I2_S")
    has_native_i2sr_constants = hasattr(gguf.GGMLQuantizationType, "I2_SR") and hasattr(gguf.LlamaFileType, "MOSTLY_I2_SR")
    i2_s_dtype = getattr(gguf.GGMLQuantizationType, "I2_S", NamedInt(36, "I2_S"))
    mostly_i2_s_ftype = getattr(gguf.LlamaFileType, "MOSTLY_I2_S", NamedInt(40, "MOSTLY_I2_S"))
    i2_sr_dtype = getattr(gguf.GGMLQuantizationType, "I2_SR", NamedInt(40, "I2_SR"))
    mostly_i2_sr_ftype = getattr(gguf.LlamaFileType, "MOSTLY_I2_SR", NamedInt(41, "MOSTLY_I2_SR"))
    if row_scale_qtype not in {None, "i2_sr"}:
        raise ValueError(f"unsupported row_scale_qtype={row_scale_qtype!r}")
    return (
        i2_s_dtype,
        mostly_i2_s_ftype,
        i2_sr_dtype,
        mostly_i2_sr_ftype,
        has_native_i2s_constants,
        has_native_i2sr_constants,
    )


def make_i2s_model_class(
    base_cls: type,
    state: dict[str, torch.Tensor],
    converter: ModuleType,
    args: argparse.Namespace,
    summary: dict[str, Any],
    i2_s_dtype: Any,
    i2_sr_dtype: Any,
    config: dict[str, Any],
) -> type:
    gguf = converter.gguf
    label_count = classifier_label_count(state, config)
    labels = classifier_labels(config, label_count)

    class StaticTernaryI2SModel(base_cls):  # type: ignore[misc, valid-type]
        model_arch = getattr(args, "_model_arch_override", None) or base_cls.model_arch

        def set_gguf_parameters(self):  # type: ignore[no-untyped-def]
            rope_parameters = self.hparams.get("rope_parameters")
            if self.hparams.get("rope_theta") is None and isinstance(rope_parameters, dict):
                rope_theta = rope_parameters.get("rope_theta")
                if rope_theta is not None:
                    self.hparams["rope_theta"] = rope_theta
                    summary["rope_theta_from_rope_parameters"] = float(rope_theta)
            super().set_gguf_parameters()
            if args.classifier_head_gguf:
                if label_count is None:
                    raise ValueError("--classifier-head-gguf requires score.weight/classifier.weight or config.num_labels")
                if hasattr(self.gguf_writer, "add_classifier_label_count"):
                    self.gguf_writer.add_classifier_label_count(label_count)
                else:
                    self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.classifier_label_count", label_count)
                if hasattr(gguf.PoolingType, "LAST"):
                    self.gguf_writer.add_pooling_type(gguf.PoolingType.LAST)
                else:
                    self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.pooling_type", 3)
                self.gguf_writer.add_string("bitnet.sequence_classification.head", "cls")
                self.gguf_writer.add_string("bitnet.sequence_classification.pooling", "last")
                self.gguf_writer.add_string(
                    "bitnet.sequence_classification.problem_type",
                    str(config.get("problem_type") or "single_label_classification"),
                )
                self.gguf_writer.add_array("bitnet.sequence_classification.labels", labels)
                summary["classifier_label_count"] = label_count
                summary["classifier_labels"] = labels

        def set_vocab(self):  # type: ignore[no-untyped-def]
            if args.synthetic_vocab_for_smoke:
                vocab_size = int(self.hparams.get("vocab_size", 0) or 0)
                if vocab_size <= 0:
                    raise ValueError("synthetic smoke vocab requires config.json vocab_size > 0")
                self.gguf_writer.add_tokenizer_model("gpt2")
                self.gguf_writer.add_tokenizer_pre("default")
                self.gguf_writer.add_token_list([f"<tok{i}>".encode("utf-8") for i in range(vocab_size)])
                self.gguf_writer.add_token_types([int(gguf.TokenType.NORMAL)] * vocab_size)
                if "bos_token_id" in self.hparams:
                    self.gguf_writer.add_bos_token_id(int(self.hparams["bos_token_id"]))
                if "eos_token_id" in self.hparams:
                    self.gguf_writer.add_eos_token_id(int(self.hparams["eos_token_id"]))
                if "pad_token_id" in self.hparams and self.hparams["pad_token_id"] is not None:
                    self.gguf_writer.add_pad_token_id(int(self.hparams["pad_token_id"]))
                summary["synthetic_vocab_for_smoke"] = True
                summary["synthetic_vocab_size"] = vocab_size
                return
            super().set_vocab()

        def prepare_tensors(self):  # type: ignore[no-untyped-def]
            ternary_packed = 0
            copied_tensors = 0
            output_f16 = 0
            row_scale_rejected = 0
            row_scale_packed = 0
            packed_bytes = 0
            classifier_tensors: dict[str, np.ndarray] = {}
            classifier_gguf_tensors: list[str] = []
            try:
                output_tensor_name = self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT)
            except ValueError:
                output_tensor_name = None

            for key, tensor in state.items():
                if key.endswith(".weight_scale"):
                    continue
                if key in {"score.weight", "score.bias", "classifier.weight", "classifier.bias"}:
                    if args.classifier_head_npz is None and not args.classifier_head_gguf:
                        raise ValueError(
                            f"{key} is a sequence-classification head tensor; pass "
                            "--classifier-head-npz, --classifier-head-gguf, or both"
                        )
                    data = tensor.to(dtype=torch.float32).cpu().numpy()
                    if args.classifier_head_npz is not None:
                        classifier_tensors[key.replace(".", "_")] = data
                    if args.classifier_head_gguf:
                        gguf_name = {
                            "score.weight": "cls.weight",
                            "score.bias": "cls.bias",
                            "classifier.weight": "cls.weight",
                            "classifier.bias": "cls.bias",
                        }[key]
                        self.gguf_writer.add_tensor(
                            gguf_name,
                            data,
                            raw_dtype=gguf.GGMLQuantizationType.F32,
                        )
                        classifier_gguf_tensors.append(gguf_name)
                    continue

                if key.endswith(".ternary_weight"):
                    prefix = key[: -len(".ternary_weight")]
                    scale_key = f"{prefix}.weight_scale"
                    if scale_key not in state:
                        raise KeyError(f"missing scale tensor for {key}: {scale_key}")
                    if args.validate_codes:
                        validate_ternary_codes(key, tensor)
                    mapped_prefix = normalize_bitdistill_subln_name(prefix) if args.bitdistill_subln else prefix
                    new_name = self.map_tensor_name(f"{mapped_prefix}.weight")
                    scale = state[scale_key]
                    is_output = args.keep_output_f16 and output_tensor_name is not None and new_name == output_tensor_name
                    if scale.numel() != 1 and not (args.row_scale_prototype or args.row_scale_qtype == "i2_sr"):
                        row_scale_rejected += 1
                        raise ValueError(
                            f"{scale_key} has row or non-scalar scale shape {tuple(scale.shape)}; "
                            "direct packed writer supports row scales only with --row-scale-prototype "
                            "or --row-scale-qtype=i2_sr"
                        )
                    if is_output:
                        dense = tensor.to(dtype=torch.float16).mul(scale.to(dtype=torch.float16)).squeeze().numpy()
                        self.gguf_writer.add_tensor(new_name, dense, raw_dtype=gguf.GGMLQuantizationType.F16)
                        output_f16 += 1
                    else:
                        if scale.numel() == 1:
                            packed = pack_i2_s_scalar(tensor, scale)
                            raw_dtype = i2_s_dtype
                        else:
                            packed = pack_i2_s_row_prototype(tensor, scale)
                            row_scale_packed += 1
                            raw_dtype = i2_sr_dtype if args.row_scale_qtype == "i2_sr" else i2_s_dtype
                        self.gguf_writer.add_tensor(
                            new_name,
                            packed,
                            raw_shape=tuple(tensor.shape),
                            raw_dtype=raw_dtype,
                        )
                        ternary_packed += 1
                        packed_bytes += int(packed.nbytes)
                    continue

                mapped_key = normalize_bitdistill_subln_name(key) if args.bitdistill_subln else key
                new_name = self.map_tensor_name(mapped_key)
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

            if classifier_tensors:
                args.classifier_head_npz.parent.mkdir(parents=True, exist_ok=True)
                np.savez(args.classifier_head_npz, **classifier_tensors)

            summary.update(
                {
                    "ternary_i2s_packed": ternary_packed,
                    "row_scale_i2s_packed": row_scale_packed,
                    "copied_tensors": copied_tensors,
                    "classifier_head_sidecar": str(args.classifier_head_npz) if classifier_tensors else None,
                    "classifier_head_tensors": sorted(classifier_tensors),
                    "classifier_head_gguf": bool(classifier_gguf_tensors),
                    "classifier_head_gguf_tensors": sorted(classifier_gguf_tensors),
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
    parser.add_argument(
        "--source-architecture-alias",
        default=None,
        help=(
            "Use this Hugging Face architecture name when selecting the llama.cpp converter class. "
            "This is useful for Qwen2ForSequenceClassification backbones that share the Qwen2ForCausalLM decoder."
        ),
    )
    parser.add_argument(
        "--classifier-head-npz",
        type=Path,
        default=None,
        help=(
            "Export sequence-classification head tensors such as score.weight to an NPZ sidecar. "
            "The current llama.cpp path can then run the packed backbone in embedding/last-pooling mode while an external evaluator applies the dense head."
        ),
    )
    parser.add_argument(
        "--classifier-head-gguf",
        action="store_true",
        help=(
            "Persist sequence-classification score/classifier head tensors and label metadata directly in GGUF "
            "as cls.weight/cls.bias. This requires the bitnet-qwen runtime classifier path in this fork."
        ),
    )
    parser.add_argument("--validate-codes", action="store_true")
    parser.add_argument("--keep-output-f16", action="store_true", default=True)
    parser.add_argument(
        "--gguf-arch",
        choices=["source", "bitnet-25", "bitnet-qwen"],
        default="source",
        help=(
            "Override the emitted GGUF architecture. Use bitnet-qwen for Qwen "
            "BitDistill checkpoints with SubLN and SiLU/SwiGLU MLP semantics; "
            "use bitnet-25 only for checkpoints trained against the BitNet 2.5 runtime graph."
        ),
    )
    parser.add_argument(
        "--bitdistill-subln",
        action="store_true",
        help="Map SubLNLinear wrapper keys to llama.cpp BitNet attn_sub_norm/ffn_sub_norm tensors.",
    )
    parser.add_argument(
        "--synthetic-vocab-for-smoke",
        action="store_true",
        help=(
            "Use a repo-local built-in tokenizer stub sized to config.json vocab_size. "
            "This is only for tiny smoke checkpoints whose synthetic tokenizer is not deployable."
        ),
    )
    parser.add_argument(
        "--row-scale-prototype",
        action="store_true",
        help="Allow row-scale checkpoints using the experimental per-output-row I2_S layout.",
    )
    parser.add_argument(
        "--row-scale-qtype",
        choices=["i2_sr"],
        default=None,
        help=(
            "Emit the stable row-scale qtype instead of overloading I2_S. "
            "Requires a runtime with stable I2_SR qtype support."
        ),
    )
    args = parser.parse_args()
    if args.row_scale_prototype and args.row_scale_qtype is not None:
        raise SystemExit("--row-scale-prototype and --row-scale-qtype are mutually exclusive")
    if args.classifier_head_gguf and args.gguf_arch != "bitnet-qwen":
        raise SystemExit("--classifier-head-gguf currently requires --gguf-arch bitnet-qwen")

    ternary_state = args.ternary_state or args.checkpoint_dir / "ternary_state_dict.pt"
    converter = load_converter(args.converter)
    (
        i2_s_dtype,
        mostly_i2_s_ftype,
        i2_sr_dtype,
        mostly_i2_sr_ftype,
        has_native_i2s_constants,
        has_native_i2sr_constants,
    ) = get_gguf_i2s_types(converter.gguf, row_scale_qtype=args.row_scale_qtype)
    config = load_config(args.checkpoint_dir)
    architecture = load_architecture(args.checkpoint_dir)
    converter_architecture = args.source_architecture_alias or architecture
    base_cls = converter.Model.from_model_architecture(converter_architecture)
    if args.gguf_arch in {"bitnet-25", "bitnet-qwen"}:
        args._model_arch_override = (
            converter.gguf.MODEL_ARCH.BITNET_QWEN
            if args.gguf_arch == "bitnet-qwen"
            else converter.gguf.MODEL_ARCH.BITNET_25
        )
        if not args.bitdistill_subln:
            raise SystemExit(
                f"--gguf-arch {args.gguf_arch} requires --bitdistill-subln for this repo's SubLN wrapper keys"
            )
    else:
        args._model_arch_override = None
    state = torch.load(ternary_state, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"expected a flat state dict in {ternary_state}")

    summary: dict[str, Any] = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "ternary_state": str(ternary_state),
        "architecture": architecture,
        "converter_architecture": converter_architecture,
        "outfile": str(args.outfile),
        "converter": str(args.converter),
        "format_scope": "scalar-scale I2_S by default; row-scale only with --row-scale-prototype or --row-scale-qtype=i2_sr",
        "i2_s_dtype": int(i2_s_dtype),
        "i2_sr_dtype": int(i2_sr_dtype),
        "mostly_i2_s_file_type": int(mostly_i2_s_ftype),
        "mostly_i2_sr_file_type": int(mostly_i2_sr_ftype),
        "has_native_i2s_gguf_python_constants": has_native_i2s_constants,
        "has_native_i2sr_gguf_python_constants": has_native_i2sr_constants,
        "limitations": [
            "Rejects row-scale checkpoints unless --row-scale-prototype or --row-scale-qtype=i2_sr is passed.",
            "Scalar mode uses the existing tensor-scale I2_S layout.",
            "Row-scale prototype mode requires a matching experimental runtime layout and overloads I2_S.",
            "Row-scale i2_sr mode requires a runtime with a stable I2_SR qtype.",
            "Keeps output.weight in F16 by default to match llama-quantize policy.",
            "Native GGUF classifier heads are supported only for the fork's bitnet-qwen path.",
        ],
        "row_scale_prototype": bool(args.row_scale_prototype),
        "row_scale_qtype": args.row_scale_qtype,
        "gguf_arch": args.gguf_arch,
        "bitdistill_subln": bool(args.bitdistill_subln),
        "synthetic_vocab_for_smoke": bool(args.synthetic_vocab_for_smoke),
        "classifier_head_npz": str(args.classifier_head_npz) if args.classifier_head_npz is not None else None,
        "classifier_head_gguf_requested": bool(args.classifier_head_gguf),
    }

    model_cls = make_i2s_model_class(base_cls, state, converter, args, summary, i2_s_dtype, i2_sr_dtype, config)
    output_ftype = mostly_i2_sr_ftype if args.row_scale_qtype == "i2_sr" else mostly_i2_s_ftype
    summary["output_ftype"] = int(output_ftype)
    summary["output_ftype_name"] = getattr(output_ftype, "name", str(output_ftype))
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        model = model_cls(
            dir_model=args.checkpoint_dir,
            ftype=output_ftype,
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
