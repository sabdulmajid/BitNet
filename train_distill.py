#!/usr/bin/env python3
"""QAT knowledge distillation for ternary W1.58A8 BitNet-style students.

The training path is intentionally self-contained:

* load a frozen FP teacher and an initialized student from Hugging Face,
* replace student nn.Linear modules with BitLinear,
* train with KL(logits) + hidden-state MSE,
* shard student parameters, gradients, and optimizer state with FSDP.

Run a local CPU/GPU sanity check with:

    python train_distill.py --smoke-test --max-steps 2

Run on two GPUs with SLURM via slurm_distill.sh.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class TernaryWeightSTE(torch.autograd.Function):
    """Absmean ternary projection with an identity backward pass.

    Forward:
        alpha = mean(abs(W))
        W_q = alpha * clamp(round(W / alpha), -1, 1)

    Backward:
        dL/dW = dL/dW_q, the straight-through estimator used for QAT.
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor, eps: float, scale_mode_id: int) -> torch.Tensor:  # type: ignore[override]
        if scale_mode_id == 0:
            alpha = weight.detach().abs().mean().clamp_min(eps)
        elif scale_mode_id == 1:
            alpha = weight.detach().abs().mean(dim=1, keepdim=True).clamp_min(eps)
        else:
            raise ValueError(f"unknown scale_mode_id={scale_mode_id}")

        q = torch.round(weight.detach() / alpha).clamp_(-1, 1)
        return q * alpha

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:  # type: ignore[override]
        return grad_output, None, None


class AbsmaxActivationSTE(torch.autograd.Function):
    """Per-token 8-bit absmax activation quantization with STE."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float) -> torch.Tensor:  # type: ignore[override]
        scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(eps) / 127.0
        q = torch.round(x.detach() / scale).clamp_(-128, 127)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        return grad_output, None


class BitLinear(nn.Module):
    """Drop-in nn.Linear replacement for W1.58A8 QAT.

    The module stores high-precision master weights. During forward it uses
    absmean ternary weights and 8-bit absmax activations. The dequantized
    PyTorch matmul is mathematically equivalent to int8 x ternary accumulation
    followed by full-precision activation/weight rescaling.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        master_weight_dtype: torch.dtype = torch.float32,
        scale_mode: str = "tensor",
        eps: float = 1e-5,
        activation_quantization: bool = True,
    ) -> None:
        super().__init__()
        if scale_mode not in {"tensor", "row"}:
            raise ValueError("scale_mode must be 'tensor' or 'row'")
        self.in_features = in_features
        self.out_features = out_features
        self.scale_mode = scale_mode
        self.eps = eps
        self.activation_quantization = activation_quantization
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=master_weight_dtype))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=master_weight_dtype)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        master_weight_dtype: torch.dtype,
        scale_mode: str,
        eps: float,
        activation_quantization: bool = True,
    ) -> "BitLinear":
        module = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            master_weight_dtype=master_weight_dtype,
            scale_mode=scale_mode,
            eps=eps,
            activation_quantization=activation_quantization,
        )
        module.weight.data.copy_(linear.weight.detach().to(master_weight_dtype))
        if linear.bias is not None and module.bias is not None:
            module.bias.data.copy_(linear.bias.detach().to(master_weight_dtype))
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_mode_id = 0 if self.scale_mode == "tensor" else 1
        q_weight = TernaryWeightSTE.apply(self.weight, self.eps, scale_mode_id).to(x.dtype)
        q_x = AbsmaxActivationSTE.apply(x, self.eps) if self.activation_quantization else x
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(q_x, q_weight, bias)


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def iter_parent_child_modules(root: nn.Module) -> Iterator[tuple[str, nn.Module, str, nn.Module]]:
    for parent_name, parent in root.named_modules():
        for child_name, child in parent.named_children():
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            yield full_name, parent, child_name, child


def replace_linear_layers(
    model: nn.Module,
    *,
    master_weight_dtype: torch.dtype,
    scale_mode: str,
    exclude_regex: str,
    eps: float,
    activation_quantization: bool = True,
) -> int:
    pattern = re.compile(exclude_regex) if exclude_regex else None
    replacements: list[tuple[nn.Module, str, BitLinear]] = []
    for full_name, parent, child_name, child in iter_parent_child_modules(model):
        if isinstance(child, nn.Linear):
            if pattern is not None and pattern.search(full_name):
                continue
            bitlinear = BitLinear.from_linear(
                child,
                master_weight_dtype=master_weight_dtype,
                scale_mode=scale_mode,
                eps=eps,
                activation_quantization=activation_quantization,
            )
            replacements.append((parent, child_name, bitlinear))

    for parent, child_name, bitlinear in replacements:
        setattr(parent, child_name, bitlinear)
    return len(replacements)


def mark_untied_output_if_needed(model: nn.Module) -> bool:
    """Keep HF config metadata consistent after replacing `lm_head`.

    Qwen checkpoints declare tied token embeddings. Replacing `lm_head` with a
    separate BitLinear module breaks that tie, so saved artifacts must not ask
    downstream loaders or converters to re-tie the output projection.
    """

    if not hasattr(model, "get_output_embeddings") or not hasattr(model, "config"):
        return False
    try:
        output_embeddings = model.get_output_embeddings()
    except Exception:
        return False
    if isinstance(output_embeddings, BitLinear) and getattr(model.config, "tie_word_embeddings", False):
        model.config.tie_word_embeddings = False
        return True
    return False


def rank0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def log(message: str) -> None:
    if rank0():
        print(message, flush=True)


def setup_distributed() -> tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    rank = dist.get_rank() if dist.is_initialized() else 0
    return device, rank, world_size, distributed


def teardown_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class SyntheticTokenDataset(Dataset[Batch]):
    def __init__(self, *, vocab_size: int, seq_len: int, samples: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(3, vocab_size, (samples, seq_len), generator=generator)
        self.attention_mask = torch.ones(samples, seq_len, dtype=torch.long)

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, index: int) -> Batch:
        return Batch(self.input_ids[index], self.attention_mask[index])


def collate_synthetic(batch: list[Batch]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item.input_ids for item in batch], dim=0),
        "attention_mask": torch.stack([item.attention_mask for item in batch], dim=0),
    }


class PackedTokenDataset(Dataset[dict[str, torch.Tensor]]):
    """Fixed-length causal-LM token blocks packed without padding."""

    def __init__(self, input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [n_blocks, block_size], got {tuple(input_ids.shape)}")
        self.input_ids = input_ids.to(torch.long).contiguous()
        self.attention_mask = torch.ones_like(self.input_ids, dtype=torch.long)

    @property
    def block_size(self) -> int:
        return self.input_ids.shape[1]

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
        }


def build_synthetic_models(args: argparse.Namespace) -> tuple[nn.Module, nn.Module, Iterable[dict[str, torch.Tensor]], None]:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=257,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=args.max_seq_len,
        rms_norm_eps=1e-5,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    teacher = LlamaForCausalLM(config)
    student = LlamaForCausalLM(config)
    student.load_state_dict(teacher.state_dict())
    dataset = SyntheticTokenDataset(
        vocab_size=config.vocab_size,
        seq_len=args.max_seq_len,
        samples=max(args.max_steps * args.per_device_batch_size * args.grad_accum_steps * 2, 8),
        seed=args.seed,
    )
    dataloader = DataLoader(dataset, batch_size=args.per_device_batch_size, shuffle=False, collate_fn=collate_synthetic)
    return teacher, student, dataloader, None


def build_hf_dataloader(
    args: argparse.Namespace,
    tokenizer,
    *,
    rank: int,
    world_size: int,
    distributed: bool,
) -> DataLoader:
    from datasets import load_dataset

    if args.dataset_name is None:
        raise ValueError("--dataset-name is required unless --smoke-test is set")

    log(f"Loading dataset {args.dataset_name}/{args.dataset_config or ''} split={args.dataset_split}")
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
        streaming=args.dataset_streaming,
    )
    if args.dataset_streaming and args.num_train_samples <= 0:
        raise ValueError("--dataset-streaming requires a positive --num-train-samples for finite packing")
    if not args.dataset_streaming and args.num_train_samples > 0:
        dataset = dataset.select(range(min(args.num_train_samples, len(dataset))))
    text_column = args.text_column
    if text_column is None:
        if args.dataset_streaming:
            dataset_iter = iter(dataset)
            first_row = next(dataset_iter, None)
            if first_row is None:
                raise ValueError("streaming dataset yielded no rows")
            text_columns = [name for name, value in first_row.items() if isinstance(value, str)]
            dataset = itertools.chain([first_row], dataset_iter)
        else:
            text_columns = [name for name, feature in dataset.features.items() if getattr(feature, "dtype", None) == "string"]
        if not text_columns:
            raise ValueError("no string column found; pass --text-column")
        text_column = text_columns[0]

    def iter_texts() -> Iterator[str]:
        if args.dataset_streaming:
            rows = itertools.islice(dataset, args.num_train_samples)
            for row in rows:
                text = row.get(text_column)
                if text is not None and text.strip():
                    yield text
        else:
            for text in dataset[text_column]:
                if text is not None and text.strip():
                    yield text

    eos_token_id = tokenizer.eos_token_id
    block_size = args.max_seq_len
    token_buffer: list[int] = []
    blocks: list[list[int]] = []
    batch_texts: list[str] = []
    total_tokens = 0
    text_count = 0

    def encode_pending_texts() -> bool:
        nonlocal batch_texts, token_buffer, total_tokens
        if not batch_texts:
            return False
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
        )
        batch_texts = []
        for ids in encoded["input_ids"]:
            if not ids:
                continue
            token_buffer.extend(ids)
            total_tokens += len(ids)
            if eos_token_id is not None:
                token_buffer.append(eos_token_id)
                total_tokens += 1
            while len(token_buffer) >= block_size:
                blocks.append(token_buffer[:block_size])
                token_buffer = token_buffer[block_size:]
                if args.max_packed_blocks > 0 and len(blocks) >= args.max_packed_blocks:
                    return True
        return False

    reached_block_cap = False
    for text in iter_texts():
        text_count += 1
        batch_texts.append(text)
        if len(batch_texts) >= args.tokenizer_batch_size:
            reached_block_cap = encode_pending_texts()
            if reached_block_cap:
                break
    if not reached_block_cap:
        reached_block_cap = encode_pending_texts()

    if not blocks:
        raise ValueError(
            f"not enough tokens ({total_tokens}) to create one block of length {block_size}; "
            "increase --num-train-samples or reduce --max-seq-len"
        )
    token_tensor = torch.tensor(blocks, dtype=torch.long)
    packed_dataset = PackedTokenDataset(token_tensor)
    if distributed and len(packed_dataset) < world_size * args.per_device_batch_size:
        raise ValueError(
            f"packed dataset has only {len(packed_dataset)} blocks, but distributed drop_last needs at least "
            f"{world_size * args.per_device_batch_size}; increase --num-train-samples or reduce batch size"
        )
    sampler = DistributedSampler(
        packed_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    ) if distributed else None
    log(
        f"Packed {text_count} text rows into {len(packed_dataset)} blocks "
        f"of {packed_dataset.block_size} tokens; dropped {len(token_buffer)} remainder tokens; "
        f"total_tokens={total_tokens}; "
        f"{'stopped at max_packed_blocks=' + str(args.max_packed_blocks) + '; ' if reached_block_cap else ''}"
        f"batch_size_per_rank={args.per_device_batch_size}; sampler="
        f"{'DistributedSampler' if sampler is not None else 'single-process'}"
    )
    return DataLoader(
        packed_dataset,
        batch_size=args.per_device_batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.dataloader_num_workers,
        persistent_workers=args.dataloader_num_workers > 0,
    )


def build_hf_models_and_data(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    distributed: bool,
) -> tuple[nn.Module, nn.Module, DataLoader, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dtype = dtype_from_name(args.model_dtype)
    master_dtype = dtype_from_name(args.master_weight_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, **model_kwargs)
    student_source = args.student_init_model or args.teacher_model
    student_kwargs = dict(model_kwargs)
    student_kwargs["torch_dtype"] = master_dtype
    student = AutoModelForCausalLM.from_pretrained(student_source, **student_kwargs)
    dataloader = build_hf_dataloader(
        args,
        tokenizer,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
    )
    return teacher, student, dataloader, tokenizer


def freeze_teacher(teacher: nn.Module) -> None:
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)


def enable_training_memory_features(model: nn.Module, args: argparse.Namespace) -> None:
    if hasattr(model, "config"):
        model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def get_transformer_layer_classes(model: nn.Module, explicit_names: str) -> set[type[nn.Module]]:
    requested = {name.strip() for name in explicit_names.split(",") if name.strip()}
    classes: set[type[nn.Module]] = set()
    for module in model.modules():
        class_name = module.__class__.__name__
        if requested and class_name in requested:
            classes.add(type(module))
        elif not requested and class_name.endswith(("DecoderLayer", "EncoderLayer", "Block", "Layer")):
            has_transformer_shape = any(hasattr(module, attr) for attr in ("self_attn", "attention", "mlp", "feed_forward"))
            if has_transformer_shape:
                classes.add(type(module))
    return classes


def maybe_wrap_fsdp(student: nn.Module, args: argparse.Namespace, device: torch.device, distributed: bool) -> nn.Module:
    if not args.use_fsdp:
        return student.to(device)
    if device.type != "cuda":
        raise RuntimeError("FSDP training here is configured for CUDA; use --no-use-fsdp for CPU smoke runs")

    transformer_classes = get_transformer_layer_classes(student, args.fsdp_wrap_class_names)
    if transformer_classes:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_classes,
        )
        log("FSDP auto-wrap classes: " + ", ".join(sorted(cls.__name__ for cls in transformer_classes)))
    else:
        auto_wrap_policy = None
        log("FSDP auto-wrap classes: none found; wrapping whole model")

    mp_policy = None
    if args.fsdp_mixed_precision:
        compute_dtype = dtype_from_name(args.model_dtype)
        mp_policy = MixedPrecision(
            param_dtype=compute_dtype,
            reduce_dtype=compute_dtype,
            buffer_dtype=compute_dtype,
        )

    return FSDP(
        student.to(device),
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=args.fsdp_cpu_offload),
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        limit_all_gathers=True,
        sync_module_states=distributed,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(args.warmup_steps, 0)
    total_steps = max(args.max_steps, 1)
    min_lr_ratio = min(max(args.min_lr_ratio, 0.0), 1.0)
    warmup_start_ratio = min(max(args.warmup_start_lr_ratio, 0.0), 1.0)

    def lr_scale(current_step: int) -> float:
        if args.lr_scheduler == "constant":
            return 1.0
        if args.lr_scheduler != "cosine":
            raise ValueError(f"unsupported --lr-scheduler={args.lr_scheduler}")
        if warmup_steps > 0 and current_step < warmup_steps:
            progress = float(current_step + 1) / float(warmup_steps)
            return warmup_start_ratio + (1.0 - warmup_start_ratio) * progress

        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max(float(current_step - warmup_steps + 1) / float(decay_steps), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)


def select_hidden_states(
    student_hidden: tuple[torch.Tensor, ...],
    teacher_hidden: tuple[torch.Tensor, ...],
    mode: str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if mode == "none":
        return []
    if mode == "last":
        return [(student_hidden[-1], teacher_hidden[-1])]
    if mode == "all":
        count = min(len(student_hidden), len(teacher_hidden))
        return [(student_hidden[i], teacher_hidden[i]) for i in range(count)]
    if mode.startswith("every_"):
        stride = int(mode.split("_", 1)[1])
        count = min(len(student_hidden), len(teacher_hidden))
        return [(student_hidden[i], teacher_hidden[i]) for i in range(0, count, stride)]
    raise ValueError("--hidden-state-layers must be one of none,last,all,every_N")


def distillation_loss(
    student_outputs,
    teacher_outputs,
    *,
    attention_mask: torch.Tensor,
    temperature: float,
    kl_weight: float,
    hidden_mse_weight: float,
    hidden_state_layers: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    student_logits = student_outputs.logits.float()
    teacher_logits = teacher_outputs.logits.float()
    mask = attention_mask.bool()

    s_log_probs = F.log_softmax(student_logits[mask] / temperature, dim=-1)
    t_probs = F.softmax(teacher_logits[mask] / temperature, dim=-1)
    kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature**2)

    hidden_pairs = select_hidden_states(student_outputs.hidden_states, teacher_outputs.hidden_states, hidden_state_layers)
    if hidden_pairs:
        hidden_losses = []
        for student_h, teacher_h in hidden_pairs:
            hidden_losses.append(F.mse_loss(student_h.float()[mask], teacher_h.float()[mask]))
        hidden_mse = torch.stack(hidden_losses).mean()
    else:
        hidden_mse = student_logits.new_zeros(())

    total = kl_weight * kl + hidden_mse_weight * hidden_mse
    metrics = {
        "loss": float(total.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "hidden_mse": float(hidden_mse.detach().cpu()),
    }
    return total, metrics


def set_seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def save_student(student: nn.Module, output_dir: Path, tokenizer: object | None, step: int) -> None:
    output_dir = output_dir / f"step-{step}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = student.module if isinstance(student, FSDP) else student
    full_state_dict: dict[str, torch.Tensor] | None = None
    if isinstance(student, FSDP):
        from torch.distributed.fsdp import FullStateDictConfig

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(student, StateDictType.FULL_STATE_DICT, cfg):
            full_state_dict = student.state_dict()
        if rank0() and hasattr(model_to_save, "save_pretrained"):
            model_to_save.save_pretrained(output_dir, state_dict=full_state_dict)
    else:
        full_state_dict = model_to_save.state_dict()
        if rank0() and hasattr(model_to_save, "save_pretrained"):
            model_to_save.save_pretrained(output_dir)
    if rank0():
        if full_state_dict is not None:
            torch.save(build_ternary_state_dict(model_to_save, full_state_dict), output_dir / "ternary_state_dict.pt")
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(output_dir)


def normalize_wrapped_module_name(name: str) -> str:
    """Map wrapper-internal module paths back to their public state_dict names."""

    wrapper_segments = {"_fsdp_wrapped_module", "_checkpoint_wrapped_module"}
    return ".".join(part for part in name.split(".") if part not in wrapper_segments)


def collect_bitlinear_modules(model: nn.Module) -> dict[str, BitLinear]:
    modules: dict[str, BitLinear] = {}
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            modules[name] = module
            modules.setdefault(normalize_wrapped_module_name(name), module)
    return modules


def build_ternary_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Export BitLinear weights as ternary codes plus scales.

    Master weights remain trainable FP tensors during QAT. This checkpoint
    materializes the deployable ternary representation without changing the
    in-memory training model.
    """

    bitlinear_modules = collect_bitlinear_modules(model)
    export: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(".weight"):
            module_name = key[:-len(".weight")]
            normalized_module_name = normalize_wrapped_module_name(module_name)
            module = bitlinear_modules.get(module_name) or bitlinear_modules.get(normalized_module_name)
            if module is not None:
                weight = value.detach().float().cpu()
                if module.scale_mode == "row":
                    scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(module.eps)
                else:
                    scale = weight.abs().mean().clamp_min(module.eps).reshape(1)
                codes = torch.round(weight / scale).clamp_(-1, 1).to(torch.int8)
                export[f"{module_name}.ternary_weight"] = codes
                export[f"{module_name}.weight_scale"] = scale
                continue
        export[key] = value.detach().cpu()
    return export


def train(args: argparse.Namespace) -> None:
    device, rank, world_size, distributed = setup_distributed()
    set_seed(args.seed, rank)

    if args.smoke_test:
        teacher, student, dataloader, tokenizer = build_synthetic_models(args)
    else:
        teacher, student, dataloader, tokenizer = build_hf_models_and_data(
            args,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
        )

    master_dtype = dtype_from_name(args.master_weight_dtype)
    replaced = replace_linear_layers(
        student,
        master_weight_dtype=master_dtype,
        scale_mode=args.scale_mode,
        exclude_regex=args.exclude_linear_regex,
        eps=args.quant_eps,
        activation_quantization=args.activation_quantization,
    )
    if replaced == 0:
        raise RuntimeError("no nn.Linear layers were replaced; check model architecture or exclude regex")
    log(f"Replaced {replaced} student nn.Linear modules with BitLinear")
    if mark_untied_output_if_needed(student):
        log("Set config.tie_word_embeddings=False because lm_head is BitLinear")

    freeze_teacher(teacher)
    enable_training_memory_features(student, args)
    teacher = teacher.to(device)
    student = maybe_wrap_fsdp(student, args, device, distributed)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in student.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer, args)
    log(
        f"LR scheduler={args.lr_scheduler} peak_lr={args.learning_rate:.3e} "
        f"warmup_steps={args.warmup_steps} min_lr_ratio={args.min_lr_ratio:g}"
    )

    student.train()
    autocast_enabled = device.type == "cuda" and args.model_dtype in {"bf16", "fp16", "float16", "bfloat16"}
    autocast_dtype = dtype_from_name(args.model_dtype)
    step = 0
    optimizer.zero_grad(set_to_none=True)
    start = time.time()

    while step < args.max_steps:
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(step)
        for batch_index, batch in enumerate(dataloader):
            batch = move_batch(batch, device)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                    teacher_outputs = teacher(**batch, output_hidden_states=True, use_cache=False)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                student_outputs = student(**batch, output_hidden_states=True, use_cache=False)
                loss, metrics = distillation_loss(
                    student_outputs,
                    teacher_outputs,
                    attention_mask=batch["attention_mask"],
                    temperature=args.temperature,
                    kl_weight=args.kl_weight,
                    hidden_mse_weight=args.hidden_mse_weight,
                    hidden_state_layers=args.hidden_state_layers,
                )
                loss = loss / args.grad_accum_steps

            loss.backward()
            if (batch_index + 1) % args.grad_accum_steps != 0:
                continue

            if args.max_grad_norm > 0:
                if isinstance(student, FSDP):
                    student.clip_grad_norm_(args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            optimizer.step()
            step += 1
            lr_used = optimizer.param_groups[0]["lr"]
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if rank0() and (step == 1 or step % args.log_every_steps == 0):
                elapsed = time.time() - start
                print(
                    f"step={step} loss={metrics['loss']:.6f} kl={metrics['kl']:.6f} "
                    f"hidden_mse={metrics['hidden_mse']:.6f} lr={lr_used:.3e} elapsed={elapsed:.1f}s",
                    flush=True,
                )

            if args.output_dir and args.save_every_steps > 0 and step % args.save_every_steps == 0:
                save_student(student, Path(args.output_dir), tokenizer, step)

            if step >= args.max_steps:
                break

    if args.output_dir and args.save_final:
        save_student(student, Path(args.output_dir), tokenizer, step)
    log(f"training complete: steps={step}")
    teardown_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QAT distillation into BitLinear W1.58A8 student")
    output_dir_was_explicit = "--output-dir" in sys.argv[1:]
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--student-init-model", default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--num-train-samples", type=int, default=50000)
    parser.add_argument("--tokenizer-batch-size", type=int, default=256)
    parser.add_argument("--max-packed-blocks", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--warmup-start-lr-ratio", type=float, default=0.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--hidden-mse-weight", type=float, default=1.0)
    parser.add_argument("--hidden-state-layers", default="last")
    parser.add_argument("--model-dtype", default="bf16", choices=["bf16", "fp16", "float16", "bfloat16", "fp32", "float32"])
    parser.add_argument("--master-weight-dtype", default="fp32", choices=["bf16", "bfloat16", "fp32", "float32"])
    parser.add_argument("--scale-mode", default="tensor", choices=["tensor", "row"])
    parser.add_argument("--quant-eps", type=float, default=1e-5)
    parser.add_argument("--activation-quantization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-linear-regex", default="")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-fsdp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fsdp-cpu-offload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fsdp-mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fsdp-wrap-class-names", default="")
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default="checkpoints/bitnet-distill")
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.use_fsdp = False
        args.dataset_name = None
        args.max_seq_len = min(args.max_seq_len, 64)
        args.per_device_batch_size = max(args.per_device_batch_size, 2)
        args.grad_accum_steps = 1
        if not output_dir_was_explicit:
            args.output_dir = ""
        args.model_dtype = "fp32"
    return args


if __name__ == "__main__":
    train(parse_args())
