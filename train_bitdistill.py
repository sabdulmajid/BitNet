#!/usr/bin/env python3
"""BitDistill-style QAT for task-specific 1.58-bit Qwen reproduction.

This script is intentionally separate from train_distill.py.  The older script
tests generic causal-LM KL/hidden-MSE distillation.  This one implements the
paper-specific ingredients needed for a fair reproduction:

* optional SubLN wrappers before attention output and FFN down projections,
* Stage-2 causal-LM continued pretraining with cross-entropy,
* Stage-3 GLUE sequence-classification fine-tuning,
* logits distillation from a task-tuned FP teacher,
* MiniLM-style Q/K/V attention-relation distillation at one selected layer.

Smoke tests run without network access:

    python train_bitdistill.py --smoke-test --stage continued_pretrain --method bitdistill --max-steps 2
    python train_bitdistill.py --smoke-test --stage task_sft --method bitdistill --max-steps 2
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_distill import (
    BitLinear,
    build_ternary_state_dict,
    dtype_from_name,
    replace_linear_layers,
)


GLUE_SPECS = {
    "mnli": {
        "dataset": ("glue", "mnli"),
        "text_keys": ("premise", "hypothesis"),
        "num_labels": 3,
        "train_split": "train",
        "eval_split": "validation_matched",
        "metric": "accuracy",
    },
    "qnli": {
        "dataset": ("glue", "qnli"),
        "text_keys": ("question", "sentence"),
        "num_labels": 2,
        "train_split": "train",
        "eval_split": "validation",
        "metric": "accuracy",
    },
    "sst2": {
        "dataset": ("glue", "sst2"),
        "text_keys": ("sentence", None),
        "num_labels": 2,
        "train_split": "train",
        "eval_split": "validation",
        "metric": "accuracy",
    },
}

GLUE_LABEL_TEXTS_WORDS = {
    "mnli": [" entailment", " neutral", " contradiction"],
    "qnli": [" entailment", " not_entailment"],
    "sst2": [" negative", " positive"],
}

GLUE_LABEL_TEXTS_LETTERS = {
    "mnli": [" A", " B", " C"],
    "qnli": [" A", " B"],
    "sst2": [" A", " B"],
}

GLUE_LABEL_DESCRIPTIONS = {
    "mnli": ["entailment", "neutral", "contradiction"],
    "qnli": ["entailment", "not_entailment"],
    "sst2": ["negative", "positive"],
}


def glue_label_texts(task_name: str, label_scheme: str) -> list[str]:
    if label_scheme == "words":
        return GLUE_LABEL_TEXTS_WORDS[task_name]
    if label_scheme == "letters":
        return GLUE_LABEL_TEXTS_LETTERS[task_name]
    raise ValueError(f"unsupported label_scheme={label_scheme}")


@dataclass
class StepMetrics:
    loss: float
    ce: float = 0.0
    logit_kd: float = 0.0
    attention_kd: float = 0.0
    weighted_logit_kd: float = 0.0
    weighted_attention_kd: float = 0.0
    lr: float = 0.0


class SyntheticSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, *, samples: int, seq_len: int, vocab_size: int, num_labels: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(3, vocab_size, (samples, seq_len), generator=generator)
        self.attention_mask = torch.ones_like(self.input_ids)
        self.labels = torch.randint(0, num_labels, (samples,), generator=generator)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


class PackedTokenDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError(f"expected [blocks, seq], got {tuple(input_ids.shape)}")
        self.input_ids = input_ids.to(torch.long).contiguous()
        self.attention_mask = torch.ones_like(self.input_ids)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.input_ids[index],
        }


class CausalGlueDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, features: list[dict[str, list[int]]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.features[index]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.ones(len(item["input_ids"]), dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


class SubLNLinear(nn.Module):
    """RMSNorm inserted immediately before a projection layer."""

    def __init__(self, projection: nn.Module, *, eps: float) -> None:
        super().__init__()
        in_features = getattr(projection, "in_features", None)
        if not isinstance(in_features, int):
            raise TypeError(f"projection {projection.__class__.__name__} lacks integer in_features")
        self.subln = nn.RMSNorm(in_features, eps=eps)
        self.proj = projection

    @property
    def in_features(self) -> int:
        return int(getattr(self.proj, "in_features"))

    @property
    def out_features(self) -> int:
        return int(getattr(self.proj, "out_features"))

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return getattr(self.proj, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.subln(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    no_decay = ("bias", "norm", "Norm", "subln")
    decay_params: list[nn.Parameter] = []
    nodecay_params: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(item in name for item in no_decay):
            nodecay_params.append(parameter)
        else:
            decay_params.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )


def make_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(args.warmup_steps, 0)
    total = max(args.max_steps, 1)
    min_ratio = min(max(args.min_lr_ratio, 0.0), 1.0)

    def scale(step: int) -> float:
        if args.lr_scheduler == "constant":
            return 1.0
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        progress = min(max(float(step - warmup + 1) / float(max(total - warmup, 1)), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


def freeze(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def maybe_copy_output_head(student: nn.Module, teacher: nn.Module | None, args: argparse.Namespace) -> dict[str, Any]:
    if teacher is None or not args.init_output_head_from_teacher:
        return {"copied": False}
    copied: list[str] = []
    skipped: dict[str, str] = {}
    for name in ("score", "classifier"):
        student_head = getattr(student, name, None)
        teacher_head = getattr(teacher, name, None)
        if student_head is None and teacher_head is None:
            continue
        if student_head is None or teacher_head is None:
            skipped[name] = "missing_on_one_model"
            continue
        student_state = student_head.state_dict()
        teacher_state = teacher_head.state_dict()
        if student_state.keys() != teacher_state.keys():
            skipped[name] = "state_keys_differ"
            continue
        mismatched = [
            key
            for key in student_state
            if tuple(student_state[key].shape) != tuple(teacher_state[key].shape)
        ]
        if mismatched:
            skipped[name] = "shape_mismatch:" + ",".join(mismatched)
            continue
        student_head.load_state_dict({key: tensor.detach().clone() for key, tensor in teacher_state.items()})
        copied.append(name)
    return {"copied": bool(copied), "heads": copied, "skipped": skipped}


def unwrap_projection(module: nn.Module) -> nn.Module:
    return module.proj if isinstance(module, SubLNLinear) else module


def add_subln_to_qwen_blocks(model: nn.Module, *, eps: float) -> int:
    """Insert SubLN before attention output and FFN down projections.

    This targets Qwen/Qwen2-style blocks but avoids hard-coding the root prefix
    so it also works for sequence-classification wrappers.
    """

    inserted = 0
    for _, module in model.named_modules():
        self_attn = getattr(module, "self_attn", None)
        if self_attn is not None and hasattr(self_attn, "o_proj"):
            current = getattr(self_attn, "o_proj")
            if not isinstance(current, SubLNLinear):
                setattr(self_attn, "o_proj", SubLNLinear(current, eps=eps))
                inserted += 1
        mlp = getattr(module, "mlp", None)
        if mlp is not None and hasattr(mlp, "down_proj"):
            current = getattr(mlp, "down_proj")
            if not isinstance(current, SubLNLinear):
                setattr(mlp, "down_proj", SubLNLinear(current, eps=eps))
                inserted += 1
    return inserted


def prepare_bitnet_student(model: nn.Module, args: argparse.Namespace) -> dict[str, int]:
    subln_inserted = 0
    if args.use_subln:
        subln_inserted = add_subln_to_qwen_blocks(model, eps=args.subln_eps)
    replaced = replace_linear_layers(
        model,
        master_weight_dtype=dtype_from_name(args.master_weight_dtype),
        scale_mode=args.scale_mode,
        exclude_regex=args.exclude_linear_regex,
        eps=args.quant_eps,
    )
    if replaced == 0:
        raise RuntimeError("no nn.Linear modules were replaced with BitLinear")
    return {"subln_inserted": subln_inserted, "bitlinear_replaced": replaced}


def find_qwen_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    layers: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        self_attn = getattr(module, "self_attn", None)
        if self_attn is None:
            continue
        if all(hasattr(self_attn, proj) for proj in ("q_proj", "k_proj", "v_proj")):
            layers.append((name, module))
    return layers


@contextmanager
def capture_qkv(model: nn.Module, *, layer_index: int) -> Iterator[dict[str, torch.Tensor]]:
    layers = find_qwen_layers(model)
    if not layers:
        raise RuntimeError("no Qwen-style layers with q_proj/k_proj/v_proj found")
    if layer_index < 0:
        layer_index = len(layers) + layer_index
    if layer_index < 0 or layer_index >= len(layers):
        raise IndexError(f"distill layer {layer_index} outside 0..{len(layers) - 1}")

    store: dict[str, torch.Tensor] = {}
    _, layer = layers[layer_index]
    handles = []

    def make_hook(key: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            store[key] = tensor

        return hook

    for key in ("q", "k", "v"):
        proj = unwrap_projection(getattr(layer.self_attn, f"{key}_proj"))
        handles.append(proj.register_forward_hook(make_hook(key)))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


def relation_rows(values: torch.Tensor, attention_mask: torch.Tensor, *, split_heads: int, temperature: float) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError(f"expected [batch, seq, channels], got {tuple(values.shape)}")
    batch, seq_len, channels = values.shape
    if channels % split_heads != 0:
        raise ValueError(f"channels={channels} is not divisible by split_heads={split_heads}")
    width = channels // split_heads
    states = values.float().reshape(batch, seq_len, split_heads, width).transpose(1, 2)
    states = F.normalize(states, dim=-1)
    relation = torch.matmul(states, states.transpose(-2, -1)) / max(temperature, 1e-8)
    key_mask = attention_mask[:, None, None, :].bool()
    relation = relation.masked_fill(~key_mask, -1.0e4)
    probs = F.softmax(relation, dim=-1).clamp_min(1.0e-8)
    query_mask = attention_mask[:, None, :].expand(batch, split_heads, seq_len).reshape(-1).bool()
    return probs.reshape(batch * split_heads * seq_len, seq_len)[query_mask]


def attention_relation_distillation_loss(
    student_qkv: dict[str, torch.Tensor],
    teacher_qkv: dict[str, torch.Tensor],
    attention_mask: torch.Tensor,
    *,
    split_heads: int,
    temperature: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for key in ("q", "k", "v"):
        if key not in student_qkv or key not in teacher_qkv:
            raise RuntimeError(f"missing captured {key}-projection state for attention distillation")
        student_rows = relation_rows(student_qkv[key], attention_mask, split_heads=split_heads, temperature=temperature)
        teacher_rows = relation_rows(teacher_qkv[key], attention_mask, split_heads=split_heads, temperature=temperature)
        losses.append(F.kl_div(torch.log(student_rows), teacher_rows, reduction="batchmean", log_target=False))
    return torch.stack(losses).mean()


def logits_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
    temperature_scale: str,
) -> torch.Tensor:
    s_log_prob = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t_prob = F.softmax(teacher_logits.float() / temperature, dim=-1)
    loss = F.kl_div(s_log_prob, t_prob, reduction="batchmean", log_target=False)
    if temperature_scale == "square":
        return loss * (temperature**2)
    if temperature_scale == "none":
        return loss
    raise ValueError(f"unsupported temperature_scale={temperature_scale}")


def causal_logits_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    temperature_scale: str,
) -> torch.Tensor:
    mask = labels.ne(-100)
    if not bool(mask.any()):
        return student_logits.new_zeros(())
    return logits_kd_loss(
        student_logits[mask],
        teacher_logits[mask],
        temperature=temperature,
        temperature_scale=temperature_scale,
    )


def make_tiny_qwen_config(args: argparse.Namespace, *, task: bool):
    from transformers import Qwen2Config

    return Qwen2Config(
        vocab_size=257,
        # Keep the smoke model small but compatible with the active x86 I2_S/I2_SR
        # packer, which requires each logical row width to be divisible by 128.
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max(args.max_seq_len, 64),
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        num_labels=args.num_labels if task else 2,
    )


def build_smoke_models(args: argparse.Namespace, *, task: bool) -> tuple[nn.Module, nn.Module, Any]:
    if task:
        from transformers import Qwen2ForSequenceClassification

        config = make_tiny_qwen_config(args, task=True)
        teacher = Qwen2ForSequenceClassification(config)
        student = copy.deepcopy(teacher)
        tokenizer = None
    else:
        from transformers import Qwen2ForCausalLM

        config = make_tiny_qwen_config(args, task=False)
        teacher = Qwen2ForCausalLM(config)
        student = copy.deepcopy(teacher)
        tokenizer = None
    return teacher, student, tokenizer


def collate_fixed(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}


def collate_causal_lm(batch: list[dict[str, torch.Tensor]], *, pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(int(item["input_ids"].numel()) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for row, item in enumerate(batch):
        length = int(item["input_ids"].numel())
        input_ids[row, :length] = item["input_ids"]
        attention_mask[row, :length] = item["attention_mask"]
        labels[row, :length] = item["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_smoke_loader(args: argparse.Namespace, *, task: bool) -> DataLoader:
    samples = max(args.max_steps * args.per_device_batch_size * args.grad_accum_steps * 2, 8)
    if task:
        dataset = SyntheticSequenceDataset(
            samples=samples,
            seq_len=min(args.max_seq_len, 64),
            vocab_size=257,
            num_labels=args.num_labels,
            seed=args.seed,
        )
    else:
        ids = torch.randint(3, 257, (samples, min(args.max_seq_len, 64)), generator=torch.Generator().manual_seed(args.seed))
        dataset = PackedTokenDataset(ids)
    return DataLoader(dataset, batch_size=args.per_device_batch_size, shuffle=False, collate_fn=collate_fixed)


def build_smoke_causal_glue(args: argparse.Namespace) -> tuple[DataLoader, list[dict[str, Any]]]:
    features: list[dict[str, list[int]]] = []
    generator = torch.Generator().manual_seed(args.seed)
    for _ in range(max(args.max_steps * args.per_device_batch_size * 2, 8)):
        prompt = torch.randint(3, 257, (min(args.max_seq_len, 64) - 2,), generator=generator).tolist()
        label = [int(torch.randint(3, 20, (1,), generator=generator).item())]
        features.append({"input_ids": prompt + label, "labels": [-100] * len(prompt) + label})
    collate = lambda batch: collate_causal_lm(batch, pad_token_id=0)
    rows = [{"sentence": "synthetic", "label": index % 2} for index in range(8)]
    return DataLoader(CausalGlueDataset(features), batch_size=args.per_device_batch_size, shuffle=False, collate_fn=collate), rows


def build_glue_loaders(args: argparse.Namespace, tokenizer: Any) -> tuple[DataLoader, DataLoader]:
    from datasets import load_dataset
    from transformers import DataCollatorWithPadding

    spec = GLUE_SPECS[args.task_name]
    dataset_name, dataset_config = spec["dataset"]
    dataset = load_dataset(dataset_name, dataset_config)
    text_a, text_b = spec["text_keys"]

    def preprocess(batch: dict[str, Any]) -> dict[str, Any]:
        if text_b is None:
            result = tokenizer(batch[text_a], truncation=True, max_length=args.max_seq_len)
        else:
            result = tokenizer(batch[text_a], batch[text_b], truncation=True, max_length=args.max_seq_len)
        result["labels"] = batch["label"]
        return result

    train = dataset[spec["train_split"]]
    eval_ds = dataset[spec["eval_split"]]
    if args.max_train_samples > 0:
        train = train.select(range(min(args.max_train_samples, len(train))))
    if args.max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))
    keep = {"input_ids", "attention_mask", "labels"}
    train = train.map(preprocess, batched=True, remove_columns=[col for col in train.column_names if col not in keep])
    eval_ds = eval_ds.map(preprocess, batched=True, remove_columns=[col for col in eval_ds.column_names if col not in keep])
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=args.pad_to_multiple_of or None)
    return (
        DataLoader(train, batch_size=args.per_device_batch_size, shuffle=True, collate_fn=collator, num_workers=args.dataloader_num_workers),
        DataLoader(eval_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator, num_workers=args.dataloader_num_workers),
    )


def format_glue_prompt(task_name: str, row: dict[str, Any], *, label_scheme: str = "words") -> str:
    options = ""
    answer_field = {
        "mnli": "Relationship",
        "qnli": "Entailment",
        "sst2": "Sentiment",
    }[task_name]
    if label_scheme == "letters":
        descriptions = GLUE_LABEL_DESCRIPTIONS[task_name]
        choices = [f"{chr(ord('A') + index)}={description}" for index, description in enumerate(descriptions)]
        options = "\nOptions: " + "; ".join(choices)
    if task_name == "sst2":
        return f"Sentence: {row['sentence']}{options}\n{answer_field}:"
    if task_name == "qnli":
        return f"Question: {row['question']}\nSentence: {row['sentence']}{options}\n{answer_field}:"
    if task_name == "mnli":
        return f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}{options}\n{answer_field}:"
    raise ValueError(f"unsupported task_name={task_name}")


def encode_causal_glue_row(
    tokenizer: Any,
    task_name: str,
    row: dict[str, Any],
    max_seq_len: int,
    *,
    label_scheme: str,
) -> dict[str, list[int]]:
    label = int(row["label"])
    label_text = glue_label_texts(task_name, label_scheme)[label]
    prompt = format_glue_prompt(task_name, row, label_scheme=label_scheme)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    label_ids = tokenizer(label_text, add_special_tokens=False)["input_ids"]
    if not label_ids:
        raise ValueError(f"label text produced no tokens: {label_text!r}")
    input_ids = prompt_ids + label_ids
    labels = [-100] * len(prompt_ids) + label_ids
    if len(input_ids) > max_seq_len:
        overflow = len(input_ids) - max_seq_len
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
    if all(value == -100 for value in labels):
        raise ValueError("truncation removed all supervised label tokens")
    return {"input_ids": input_ids, "labels": labels}


def build_glue_causal_loaders(args: argparse.Namespace, tokenizer: Any) -> tuple[DataLoader, list[dict[str, Any]]]:
    from datasets import load_dataset

    spec = GLUE_SPECS[args.task_name]
    dataset_name, dataset_config = spec["dataset"]
    dataset = load_dataset(dataset_name, dataset_config)
    train = dataset[spec["train_split"]]
    eval_ds = dataset[spec["eval_split"]]
    if args.max_train_samples > 0:
        train = train.select(range(min(args.max_train_samples, len(train))))
    if args.max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    train_features = [
        encode_causal_glue_row(tokenizer, args.task_name, row, args.max_seq_len, label_scheme=args.label_scheme)
        for row in train
    ]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    collate = lambda batch: collate_causal_lm(batch, pad_token_id=int(pad_id))
    return (
        DataLoader(CausalGlueDataset(train_features), batch_size=args.per_device_batch_size, shuffle=True, collate_fn=collate, num_workers=args.dataloader_num_workers),
        [dict(row) for row in eval_ds],
    )


def build_text_loader(args: argparse.Namespace, tokenizer: Any) -> DataLoader:
    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config or None,
        split=args.dataset_split,
        streaming=args.dataset_streaming,
    )
    if not args.dataset_streaming and args.num_train_samples > 0:
        dataset = dataset.select(range(min(args.num_train_samples, len(dataset))))
    text_column = args.text_column
    if text_column is None:
        if args.dataset_streaming:
            first_row = next(iter(dataset))
            text_column = next(key for key, value in first_row.items() if isinstance(value, str))
            dataset = itertools.chain([first_row], dataset)
        else:
            text_column = next(key for key, feature in dataset.features.items() if getattr(feature, "dtype", None) == "string")

    iterator = itertools.islice(dataset, args.num_train_samples) if args.dataset_streaming and args.num_train_samples > 0 else iter(dataset)
    token_buffer: list[int] = []
    blocks: list[list[int]] = []
    eos = tokenizer.eos_token_id
    pending: list[str] = []

    def flush_pending() -> bool:
        nonlocal pending, token_buffer
        if not pending:
            return False
        encoded = tokenizer(pending, add_special_tokens=False, return_attention_mask=False, truncation=False)
        pending = []
        for ids in encoded["input_ids"]:
            if not ids:
                continue
            token_buffer.extend(ids)
            if eos is not None:
                token_buffer.append(eos)
            while len(token_buffer) >= args.max_seq_len:
                blocks.append(token_buffer[: args.max_seq_len])
                token_buffer = token_buffer[args.max_seq_len :]
                if args.max_packed_blocks > 0 and len(blocks) >= args.max_packed_blocks:
                    return True
        return False

    for row in iterator:
        text = row.get(text_column) if isinstance(row, dict) else None
        if isinstance(text, str) and text.strip():
            pending.append(text)
        if len(pending) >= args.tokenizer_batch_size and flush_pending():
            break
    flush_pending()
    if not blocks:
        raise RuntimeError("no packed training blocks produced")
    ids = torch.tensor(blocks, dtype=torch.long)
    return DataLoader(PackedTokenDataset(ids), batch_size=args.per_device_batch_size, shuffle=True, collate_fn=collate_fixed)


def load_task_models(args: argparse.Namespace) -> tuple[nn.Module | None, nn.Module, Any, DataLoader, DataLoader]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if args.smoke_test:
        teacher, student, tokenizer = build_smoke_models(args, task=True)
        train_loader = build_smoke_loader(args, task=True)
        eval_loader = build_smoke_loader(args, task=True)
        return teacher, student, tokenizer, train_loader, eval_loader

    spec = GLUE_SPECS[args.task_name]
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "num_labels": spec["num_labels"],
        "torch_dtype": dtype_from_name(args.model_dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    student = AutoModelForSequenceClassification.from_pretrained(args.student_model, **model_kwargs)
    student.config.pad_token_id = tokenizer.pad_token_id
    teacher = None
    if args.method == "bitdistill" or args.init_output_head_from_teacher:
        teacher_model = args.teacher_model or args.student_model
        teacher = AutoModelForSequenceClassification.from_pretrained(teacher_model, **model_kwargs)
        teacher.config.pad_token_id = tokenizer.pad_token_id
    train_loader, eval_loader = build_glue_loaders(args, tokenizer)
    return teacher, student, tokenizer, train_loader, eval_loader


def load_causal_task_models(args: argparse.Namespace) -> tuple[nn.Module | None, nn.Module, Any, DataLoader, list[dict[str, Any]]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.smoke_test:
        teacher, student, tokenizer = build_smoke_models(args, task=False)
        train_loader, eval_rows = build_smoke_causal_glue(args)
        return teacher, student, tokenizer, train_loader, eval_rows

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "torch_dtype": dtype_from_name(args.model_dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    student = AutoModelForCausalLM.from_pretrained(args.student_model, **model_kwargs)
    if hasattr(student, "config"):
        student.config.pad_token_id = tokenizer.pad_token_id
    teacher = None
    if args.method == "bitdistill":
        teacher_model = args.teacher_model or args.student_model
        teacher = AutoModelForCausalLM.from_pretrained(teacher_model, **model_kwargs)
        if hasattr(teacher, "config"):
            teacher.config.pad_token_id = tokenizer.pad_token_id
    train_loader, eval_rows = build_glue_causal_loaders(args, tokenizer)
    return teacher, student, tokenizer, train_loader, eval_rows


def load_causal_models(args: argparse.Namespace) -> tuple[nn.Module | None, nn.Module, Any, DataLoader]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.smoke_test:
        teacher, student, tokenizer = build_smoke_models(args, task=False)
        return teacher, student, tokenizer, build_smoke_loader(args, task=False)

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=dtype_from_name(args.model_dtype),
        trust_remote_code=args.trust_remote_code,
    )
    return None, student, tokenizer, build_text_loader(args, tokenizer)


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device, prediction_path: Path | None = None) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    predictions_jsonl: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]
            loss_total += float(outputs.loss.detach().cpu()) * int(labels.numel())
            predictions = logits.argmax(dim=-1)
            labels_cpu = labels.detach().cpu().tolist()
            predictions_cpu = predictions.detach().cpu().tolist()
            scores_cpu = logits.float().detach().cpu().tolist()
            for label, prediction, scores in zip(labels_cpu, predictions_cpu, scores_cpu):
                is_correct = int(prediction) == int(label)
                predictions_jsonl.append(
                    {
                        "index": total,
                        "label": int(label),
                        "prediction": int(prediction),
                        "correct": is_correct,
                        "scores": [float(score) for score in scores],
                    }
                )
                correct += int(is_correct)
                total += 1
    model.train()
    if prediction_path is not None:
        write_jsonl(prediction_path, predictions_jsonl)
    return {
        "accuracy": correct / total if total else 0.0,
        "eval_loss": loss_total / total if total else float("nan"),
        "eval_examples": float(total),
        "prediction_path": str(prediction_path) if prediction_path is not None else "",
        "prediction_schema": "bitdistill-glue-predictions-v1" if prediction_path is not None else "",
    }


def trim_supervised(input_ids: list[int], labels: list[int], max_seq_len: int) -> tuple[list[int], list[int]]:
    if len(input_ids) <= max_seq_len:
        return input_ids, labels
    overflow = len(input_ids) - max_seq_len
    input_ids = input_ids[overflow:]
    labels = labels[overflow:]
    if all(label == -100 for label in labels):
        raise ValueError("truncation removed all supervised tokens")
    return input_ids, labels


def encode_causal_label_candidate(
    tokenizer: Any,
    *,
    prompt: str,
    label_text: str,
    max_seq_len: int,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"] if tokenizer is not None else [3, 4, 5]
    if tokenizer is not None:
        label_ids = tokenizer(label_text, add_special_tokens=False)["input_ids"]
    else:
        label_ids = [6 + (sum(ord(char) for char in label_text) % 31)]
    input_ids = prompt_ids + label_ids
    labels = [-100] * len(prompt_ids) + label_ids
    input_ids, labels = trim_supervised(input_ids, labels, max_seq_len)
    return {"input_ids": input_ids, "labels": labels}


def causal_sequence_scores(logits: torch.Tensor, labels: torch.Tensor, *, score_mode: str) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    token_loss = flat_loss.view_as(shift_labels)
    supervised = shift_labels.ne(-100)
    total_logp = -(token_loss * supervised).sum(dim=1)
    if score_mode == "sum":
        return total_logp
    if score_mode == "mean":
        token_count = supervised.sum(dim=1).clamp_min(1)
        return total_logp / token_count
    raise ValueError(f"unsupported candidate_score={score_mode}")


def maybe_single_token_label_ids(tokenizer: Any, task_name: str, label_scheme: str) -> list[int] | None:
    if tokenizer is None:
        return None
    label_ids: list[int] = []
    for label_text in glue_label_texts(task_name, label_scheme):
        encoded = tokenizer(label_text, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            return None
        label_ids.append(int(encoded[0]))
    return label_ids


def collate_prompt_batch(batch: list[dict[str, torch.Tensor]], *, pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(int(item["input_ids"].numel()) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.empty((len(batch),), dtype=torch.long)
    for row, item in enumerate(batch):
        length = int(item["input_ids"].numel())
        input_ids[row, :length] = item["input_ids"]
        attention_mask[row, :length] = item["attention_mask"]
        labels[row] = item["label"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def evaluate_causal_glue(
    model: nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    prediction_path: Path | None = None,
) -> dict[str, float]:
    model.eval()
    labels = glue_label_texts(args.task_name, args.label_scheme)
    label_token_ids = maybe_single_token_label_ids(tokenizer, args.task_name, args.label_scheme)
    pad_id = 0
    if tokenizer is not None:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
    if label_token_ids is not None:
        context_len = max(1, args.max_seq_len - 1)
        prompt_features: list[dict[str, torch.Tensor]] = []
        for row in rows:
            prompt = (
                format_glue_prompt(args.task_name, row, label_scheme=args.label_scheme)
                if not args.smoke_test
                else "Sentence: synthetic\nSentiment:"
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][-context_len:]
            if not prompt_ids:
                prompt_ids = [int(pad_id)]
            prompt_features.append(
                {
                    "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(prompt_ids), dtype=torch.long),
                    "label": torch.tensor(int(row["label"]), dtype=torch.long),
                }
            )
        loader = DataLoader(
            prompt_features,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_prompt_batch(batch, pad_token_id=int(pad_id)),
        )
        label_index = torch.tensor(label_token_ids, dtype=torch.long, device=device)
        correct = 0
        total = 0
        predictions_jsonl: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, device)
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
                last_positions = batch["attention_mask"].sum(dim=1).sub(1).clamp_min(0)
                next_token_logits = outputs.logits[torch.arange(outputs.logits.shape[0], device=device), last_positions]
                candidate_scores = F.log_softmax(next_token_logits.float(), dim=-1).index_select(dim=-1, index=label_index)
                predictions = candidate_scores.argmax(dim=-1)
                labels_cpu = batch["labels"].detach().cpu().tolist()
                predictions_cpu = predictions.detach().cpu().tolist()
                scores_cpu = candidate_scores.detach().cpu().tolist()
                for label, prediction, scores in zip(labels_cpu, predictions_cpu, scores_cpu):
                    predictions_jsonl.append(
                        {
                            "index": total,
                            "label": int(label),
                            "prediction": int(prediction),
                            "correct": int(prediction) == int(label),
                            "scores": [float(score) for score in scores],
                        }
                    )
                    total += 1
                correct += sum(1 for row in predictions_jsonl[-len(labels_cpu) :] if row["correct"])
        model.train()
        if prediction_path is not None:
            write_jsonl(prediction_path, predictions_jsonl)
        return {
            "accuracy": correct / total if total else 0.0,
            "eval_examples": float(total),
            "causal_eval_mode": "single_forward_single_token_labels",
            "prediction_path": str(prediction_path) if prediction_path is not None else "",
            "prediction_schema": "bitdistill-glue-predictions-v1" if prediction_path is not None else "",
        }

    candidates: list[dict[str, list[int]]] = []
    gold: list[int] = []
    for row in rows:
        prompt = (
            format_glue_prompt(args.task_name, row, label_scheme=args.label_scheme)
            if not args.smoke_test
            else "Sentence: synthetic\nSentiment:"
        )
        gold.append(int(row["label"]))
        for label_text in labels:
            candidates.append(
                encode_causal_label_candidate(
                    tokenizer,
                    prompt=prompt,
                    label_text=label_text,
                    max_seq_len=args.max_seq_len,
                )
            )
    loader = DataLoader(
        CausalGlueDataset(candidates),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_causal_lm(batch, pad_token_id=int(pad_id)),
    )
    scores: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            scores.extend(
                float(value)
                for value in causal_sequence_scores(
                    outputs.logits,
                    batch["labels"],
                    score_mode=args.candidate_score,
                )
                .detach()
                .cpu()
            )
    if len(scores) != len(candidates):
        raise RuntimeError(f"scored {len(scores)} candidates but expected {len(candidates)}")
    correct = 0
    label_count = len(labels)
    predictions_jsonl = []
    for index, true_label in enumerate(gold):
        row_scores = scores[index * label_count : (index + 1) * label_count]
        pred = max(range(label_count), key=lambda label_index: row_scores[label_index])
        correct += int(pred == true_label)
        predictions_jsonl.append(
            {
                "index": index,
                "label": int(true_label),
                "prediction": int(pred),
                "correct": int(pred == true_label),
                "scores": [float(score) for score in row_scores],
            }
        )
    total = len(gold)
    model.train()
    if prediction_path is not None:
        write_jsonl(prediction_path, predictions_jsonl)
    return {
        "accuracy": correct / total if total else 0.0,
        "eval_examples": float(total),
        "causal_eval_mode": "candidate_sequence",
        "prediction_path": str(prediction_path) if prediction_path is not None else "",
        "prediction_schema": "bitdistill-glue-predictions-v1" if prediction_path is not None else "",
    }


def save_outputs(model: nn.Module, tokenizer: Any, args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    if not args.output_dir:
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_model_artifacts:
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(output_dir)
        torch.save(model.state_dict(), output_dir / "custom_state_dict.pt")
        bitlinear_count = sum(1 for module in model.modules() if isinstance(module, BitLinear))
        if bitlinear_count:
            torch.save(build_ternary_state_dict(model, model.state_dict()), output_dir / "ternary_state_dict.pt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def eval_prediction_path(args: argparse.Namespace) -> Path | None:
    if not args.save_eval_predictions or not args.output_dir:
        return None
    return Path(args.output_dir) / "eval_predictions.jsonl"


def save_training_snapshot(
    model: nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    *,
    step: int,
) -> None:
    if not args.output_dir:
        return
    snapshot_dir = Path(args.output_dir) / f"checkpoint-{step}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "config") and hasattr(model.config, "save_pretrained"):
        model.config.save_pretrained(snapshot_dir)
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(snapshot_dir)
    torch.save(model.state_dict(), snapshot_dir / "custom_state_dict.pt")
    bitlinear_count = sum(1 for module in model.modules() if isinstance(module, BitLinear))
    if bitlinear_count:
        torch.save(build_ternary_state_dict(model, model.state_dict()), snapshot_dir / "ternary_state_dict.pt")
    snapshot_metrics = dict(metrics)
    snapshot_metrics["snapshot"] = {"step": step, "complete": False}
    (snapshot_dir / "metrics.json").write_text(json.dumps(snapshot_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_optional_state_dict(model: nn.Module, args: argparse.Namespace) -> dict[str, Any]:
    if not args.init_state_dict:
        return {"loaded": False}
    path = Path(args.init_state_dict)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"{path} did not contain a state dict")
    model_state = model.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    skipped_shape_mismatches: dict[str, dict[str, list[int]]] = {}
    for key, tensor in state.items():
        target = model_state.get(key)
        if target is None:
            compatible_state[key] = tensor
            continue
        if tuple(tensor.shape) != tuple(target.shape):
            skipped_shape_mismatches[key] = {
                "checkpoint": list(tensor.shape),
                "model": list(target.shape),
            }
            continue
        compatible_state[key] = tensor
    incompatible = model.load_state_dict(compatible_state, strict=False)
    return {
        "loaded": True,
        "path": str(path),
        "missing_keys": sorted(incompatible.missing_keys),
        "unexpected_keys": sorted(incompatible.unexpected_keys),
        "skipped_shape_mismatches": skipped_shape_mismatches,
    }


def train_continued_pretrain(args: argparse.Namespace) -> dict[str, Any]:
    _teacher, student, tokenizer, loader = load_causal_models(args)
    if args.method in {"bitnet_sft", "bitdistill"}:
        prep = prepare_bitnet_student(student, args)
    else:
        prep = {"subln_inserted": 0, "bitlinear_replaced": 0}
    state_load = load_optional_state_dict(student, args)
    if hasattr(student, "config"):
        student.config.use_cache = False
    if args.gradient_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    student.to(device)
    optimizer = make_optimizer(student, args)
    scheduler = make_scheduler(optimizer, args)
    student.train()
    step = 0
    start = time.time()
    last = StepMetrics(loss=0.0)
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        for batch_index, batch in enumerate(loader):
            batch = move_batch(batch, device)
            outputs = student(**batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            if (batch_index + 1) % args.grad_accum_steps:
                continue
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            optimizer.step()
            step += 1
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            last = StepMetrics(loss=float((loss * args.grad_accum_steps).detach().cpu()), ce=float((loss * args.grad_accum_steps).detach().cpu()), lr=lr)
            if step == 1 or step % args.log_every_steps == 0:
                print(f"step={step} ce={last.ce:.6f} lr={lr:.3e} elapsed={time.time() - start:.1f}s", flush=True)
            if args.save_every_steps > 0 and step % args.save_every_steps == 0:
                snapshot_metrics = {
                    "stage": args.stage,
                    "method": args.method,
                    "student_model": args.student_model,
                    "scale_mode": args.scale_mode,
                    "exclude_linear_regex": args.exclude_linear_regex,
                    "steps": step,
                    "effective_train_token_presentations": int(step * args.per_device_batch_size * args.grad_accum_steps * args.max_seq_len),
                    "dataset": {
                        "name": args.dataset_name,
                        "config": args.dataset_config,
                        "split": args.dataset_split,
                        "num_train_samples": args.num_train_samples,
                        "max_packed_blocks": args.max_packed_blocks,
                    },
                    "last": last.__dict__,
                    "preparation": prep,
                    "state_load": state_load,
                    "elapsed_seconds": time.time() - start,
                }
                save_training_snapshot(student, tokenizer, args, snapshot_metrics, step=step)
            if step >= args.max_steps:
                break

    metrics = {
        "stage": args.stage,
        "method": args.method,
        "student_model": args.student_model,
        "scale_mode": args.scale_mode,
        "exclude_linear_regex": args.exclude_linear_regex,
        "steps": step,
        "effective_train_token_presentations": int(step * args.per_device_batch_size * args.grad_accum_steps * args.max_seq_len),
        "dataset": {
            "name": args.dataset_name,
            "config": args.dataset_config,
            "split": args.dataset_split,
            "num_train_samples": args.num_train_samples,
            "max_packed_blocks": args.max_packed_blocks,
        },
        "last": last.__dict__,
        "preparation": prep,
        "state_load": state_load,
        "elapsed_seconds": time.time() - start,
        "save_every_steps": args.save_every_steps,
    }
    save_outputs(student, tokenizer, args, metrics)
    return metrics


def train_task(args: argparse.Namespace) -> dict[str, Any]:
    if args.task_format == "causal_lm":
        teacher, student, tokenizer, train_loader, eval_rows = load_causal_task_models(args)
        eval_loader = None
    else:
        teacher, student, tokenizer, train_loader, eval_loader = load_task_models(args)
        eval_rows = []
    if args.method in {"bitnet_sft", "bitdistill"}:
        prep = prepare_bitnet_student(student, args)
    else:
        prep = {"subln_inserted": 0, "bitlinear_replaced": 0}
    state_load = load_optional_state_dict(student, args)
    output_head_init = maybe_copy_output_head(student, teacher, args)
    if args.method != "bitdistill":
        teacher = None
    if args.gradient_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if teacher is not None:
        freeze(teacher)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    student.to(device)
    if teacher is not None:
        teacher.to(device)
    optimizer = make_optimizer(student, args)
    scheduler = make_scheduler(optimizer, args)
    student.train()
    step = 0
    start = time.time()
    last = StepMetrics(loss=0.0)
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        for batch_index, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            if args.method == "bitdistill":
                if teacher is None:
                    raise RuntimeError("BitDistill task stage requires a teacher model")
                need_logit_kd = args.logit_kd_weight != 0.0
                need_attention_kd = args.attention_kd_weight != 0.0
                if need_attention_kd:
                    with torch.no_grad():
                        with capture_qkv(teacher, layer_index=args.distill_layer) as teacher_qkv:
                            teacher_outputs = teacher(**batch)
                    with capture_qkv(student, layer_index=args.distill_layer) as student_qkv:
                        student_outputs = student(**batch)
                else:
                    teacher_qkv = {}
                    student_outputs = student(**batch)
                    teacher_outputs = None
                    if need_logit_kd:
                        with torch.no_grad():
                            teacher_outputs = teacher(**batch)
                ce = student_outputs.loss
                if need_logit_kd and teacher_outputs is None:
                    raise RuntimeError("teacher outputs missing for logits distillation")
                if not need_logit_kd:
                    logit_kd = student_outputs.logits.new_zeros(())
                elif args.task_format == "causal_lm":
                    logit_kd = causal_logits_kd_loss(
                        student_outputs.logits,
                        teacher_outputs.logits,
                        batch["labels"],
                        temperature=args.logit_temperature,
                        temperature_scale=args.logit_kd_temperature_scale,
                    )
                else:
                    logit_kd = logits_kd_loss(
                        student_outputs.logits,
                        teacher_outputs.logits,
                        temperature=args.logit_temperature,
                        temperature_scale=args.logit_kd_temperature_scale,
                    )
                if need_attention_kd:
                    attention_kd = attention_relation_distillation_loss(
                        student_qkv,
                        teacher_qkv,
                        batch["attention_mask"],
                        split_heads=args.attention_split_heads,
                        temperature=args.attention_temperature,
                    )
                else:
                    attention_kd = student_outputs.logits.new_zeros(())
                weighted_logit_kd = args.logit_kd_weight * logit_kd
                weighted_attention_kd = args.attention_kd_weight * attention_kd
                loss = ce + weighted_logit_kd + weighted_attention_kd
            else:
                student_outputs = student(**batch)
                ce = student_outputs.loss
                logit_kd = student_outputs.logits.new_zeros(())
                attention_kd = student_outputs.logits.new_zeros(())
                weighted_logit_kd = student_outputs.logits.new_zeros(())
                weighted_attention_kd = student_outputs.logits.new_zeros(())
                loss = ce

            (loss / args.grad_accum_steps).backward()
            if (batch_index + 1) % args.grad_accum_steps:
                continue
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            optimizer.step()
            step += 1
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            last = StepMetrics(
                loss=float(loss.detach().cpu()),
                ce=float(ce.detach().cpu()),
                logit_kd=float(logit_kd.detach().cpu()),
                attention_kd=float(attention_kd.detach().cpu()),
                weighted_logit_kd=float(weighted_logit_kd.detach().cpu()),
                weighted_attention_kd=float(weighted_attention_kd.detach().cpu()),
                lr=lr,
            )
            if step == 1 or step % args.log_every_steps == 0:
                print(
                    f"step={step} loss={last.loss:.6f} ce={last.ce:.6f} "
                    f"logit_kd={last.logit_kd:.6f} attention_kd={last.attention_kd:.6f} "
                    f"weighted_logit_kd={last.weighted_logit_kd:.6f} "
                    f"weighted_attention_kd={last.weighted_attention_kd:.6f} "
                    f"lr={lr:.3e} elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
            if step >= args.max_steps:
                break

    if args.task_format == "causal_lm":
        eval_metrics = evaluate_causal_glue(student, tokenizer, eval_rows, args, device, prediction_path=eval_prediction_path(args))
    else:
        if eval_loader is None:
            raise RuntimeError("sequence classification evaluation loader missing")
        eval_metrics = evaluate_accuracy(student, eval_loader, device, prediction_path=eval_prediction_path(args))
    metrics = {
        "stage": args.stage,
        "method": args.method,
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "task": args.task_name,
        "task_format": args.task_format,
        "label_scheme": args.label_scheme,
        "candidate_score": args.candidate_score,
        "scale_mode": args.scale_mode,
        "exclude_linear_regex": args.exclude_linear_regex,
        "steps": step,
        "last": last.__dict__,
        "eval": eval_metrics,
        "preparation": prep,
        "state_load": state_load,
        "output_head_init": output_head_init,
        "distill_layer": args.distill_layer,
        "attention_split_heads": args.attention_split_heads,
        "training_budget": {
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "max_seq_len": args.max_seq_len,
            "per_device_batch_size": args.per_device_batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "max_steps": args.max_steps,
            "effective_train_token_presentations_upper_bound": int(
                step * args.per_device_batch_size * args.grad_accum_steps * args.max_seq_len
            ),
        },
        "loss_weights": {
            "logit_kd_weight": args.logit_kd_weight,
            "attention_kd_weight": args.attention_kd_weight,
            "logit_temperature": args.logit_temperature,
            "logit_kd_temperature_scale": args.logit_kd_temperature_scale,
            "attention_temperature": args.attention_temperature,
        },
        "elapsed_seconds": time.time() - start,
    }
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    save_outputs(student, tokenizer, args, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["continued_pretrain", "task_sft"], default="task_sft")
    parser.add_argument("--method", choices=["fp16_sft", "bitnet_sft", "bitdistill"], default="bitdistill")
    parser.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--teacher-model", default="")
    parser.add_argument("--init-state-dict", default="")
    parser.add_argument("--task-name", choices=sorted(GLUE_SPECS), default="sst2")
    parser.add_argument("--task-format", choices=["causal_lm", "sequence_classification"], default="causal_lm")
    parser.add_argument("--label-scheme", choices=["words", "letters"], default="words")
    parser.add_argument("--candidate-score", choices=["sum", "mean"], default="sum")
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--num-train-samples", type=int, default=20000)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--save-eval-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-packed-blocks", type=int, default=0)
    parser.add_argument("--tokenizer-batch-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lr-scheduler", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--model-dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--master-weight-dtype", default="fp32", choices=["bf16", "bfloat16", "fp32", "float32"])
    parser.add_argument("--scale-mode", choices=["tensor", "row"], default="tensor")
    parser.add_argument("--quant-eps", type=float, default=1e-5)
    parser.add_argument("--use-subln", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--subln-eps", type=float, default=1e-5)
    parser.add_argument("--exclude-linear-regex", default="score|classifier")
    parser.add_argument("--logit-temperature", type=float, default=5.0)
    parser.add_argument("--logit-kd-weight", type=float, default=10.0)
    parser.add_argument("--logit-kd-temperature-scale", choices=["none", "square"], default="none")
    parser.add_argument("--attention-temperature", type=float, default=1.0)
    parser.add_argument("--attention-kd-weight", type=float, default=100.0)
    parser.add_argument("--attention-split-heads", type=int, default=8)
    parser.add_argument("--distill-layer", type=int, default=-1)
    parser.add_argument("--init-output-head-from-teacher", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--save-model-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.use_subln is None:
        args.use_subln = args.method == "bitdistill"
    if args.smoke_test:
        args.model_dtype = "fp32"
        args.max_seq_len = min(args.max_seq_len, 64)
        args.max_steps = min(args.max_steps, 2)
        args.per_device_batch_size = min(args.per_device_batch_size, 2)
        args.eval_batch_size = min(args.eval_batch_size, 2)
        args.gradient_checkpointing = False
        args.output_dir = args.output_dir or ""
        args.attention_split_heads = min(args.attention_split_heads, 4)
        args.distill_layer = -1
    if args.method == "bitdistill" and args.stage == "task_sft" and not args.teacher_model and not args.smoke_test:
        raise SystemExit("--teacher-model must point to an FP16-SFT teacher checkpoint for BitDistill task_sft")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.stage == "continued_pretrain":
        metrics = train_continued_pretrain(args)
    else:
        metrics = train_task(args)
    if not args.output_dir:
        print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
