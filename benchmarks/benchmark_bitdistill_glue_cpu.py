#!/usr/bin/env python3
"""CPU task-runtime benchmark for BitDistill GLUE checkpoints.

This script evaluates saved Qwen sequence-classification checkpoints on GLUE
using PyTorch CPU execution.  It is not a packed I2_SR/llama.cpp benchmark; it
answers a narrower question: for the task-specific checkpoints we trained,
what accuracy, latency, examples/sec, and process RSS do we observe when the
checkpoint is reconstructed and run on CPU?
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = datetime.now(timezone.utc).date().isoformat()
TASKS = ["mnli", "qnli", "sst2"]
DEFAULT_RUNS = [
    ("short", "fp16_sft-tensor-layer-1"),
    ("short", "bitnet_sft-tensor-layer-1"),
    ("short", "bitdistill-tensor-layer-1"),
    ("short", "bitdistill-row-layer-1"),
    ("short", "bitdistill-tensor-layer-8"),
    ("longwarmup", "bitdistill-longwarmup-tensor-layer-8"),
    ("longwarmup", "bitdistill-longwarmup-row-layer-8"),
    ("papergamma", "bitdistill-longwarmup-tensor-layer-8"),
    ("papergamma_row", "bitdistill-longwarmup-row-layer-8"),
    ("papergamma_lr1", "bitdistill-longwarmup-tensor-layer-8"),
    ("papergamma_lr5", "bitdistill-longwarmup-tensor-layer-8"),
    ("papergamma_headinit", "bitdistill-longwarmup-tensor-layer-8"),
]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rss_mib() -> float | None:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024.0
    return None


def maxrss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.  This environment is Linux, but
    # keep the fallback so local reruns remain interpretable.
    return float(value) / 1024.0 if value > 10_000_000 else float(value) / 1024.0


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(max(int(round((pct / 100.0) * (len(ordered) - 1))), 0), len(ordered) - 1)
    return ordered[index]


def checkpoint_dir_for(args: argparse.Namespace, task: str, family: str, run: str) -> Path:
    if family == "longwarmup":
        root = args.longwarmup_root
    elif family == "papergamma":
        root = args.paper_hparam_root
    elif family == "papergamma_row":
        root = args.paper_hparam_row_root
    elif family == "papergamma_lr1":
        root = args.paper_hparam_lr1_root
    elif family == "papergamma_lr5":
        root = args.paper_hparam_lr5_root
    elif family == "papergamma_headinit":
        root = args.paper_hparam_headinit_root
    else:
        root = args.short_root
    return root / args.model.replace("/", "-") / task / run


def run_single(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    from train_bitdistill import GLUE_SPECS, prepare_bitnet_student

    torch.set_num_threads(args.threads)
    checkpoint_dir = args.checkpoint_dir
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
    load_start = time.perf_counter()
    model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True)

    method = metrics.get("method")
    prep = metrics.get("preparation", {}) if isinstance(metrics.get("preparation"), dict) else {}
    if method in {"bitnet_sft", "bitdistill"}:
        shim_args = argparse.Namespace(
            use_subln=int(prep.get("subln_inserted", 0) or 0) > 0,
            subln_eps=1e-5,
            master_weight_dtype="fp32",
            scale_mode=metrics.get("scale_mode") or "tensor",
            exclude_linear_regex=metrics.get("exclude_linear_regex") or "score|classifier",
            quant_eps=1e-5,
        )
        prepare_bitnet_student(model, shim_args)

    state = torch.load(state_path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "state dict mismatch: "
            f"missing={incompatible.missing_keys[:10]} unexpected={incompatible.unexpected_keys[:10]}"
        )
    if args.model_dtype == "fp32":
        model.float()
    elif args.model_dtype == "bf16":
        model.to(torch.bfloat16)
    else:
        raise ValueError(f"unsupported model_dtype={args.model_dtype}")
    model.eval()
    load_seconds = time.perf_counter() - load_start
    rss_after_load = rss_mib()

    spec = GLUE_SPECS[args.task_name]
    dataset_name, dataset_config = spec["dataset"]
    text_a, text_b = spec["text_keys"]
    dataset = load_dataset(dataset_name, dataset_config)[spec["eval_split"]]
    if args.max_eval_samples > 0:
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    def preprocess(batch: dict[str, Any]) -> dict[str, Any]:
        if text_b is None:
            encoded = tokenizer(batch[text_a], truncation=True, max_length=args.max_seq_len)
        else:
            encoded = tokenizer(batch[text_a], batch[text_b], truncation=True, max_length=args.max_seq_len)
        encoded["labels"] = batch["label"]
        return encoded

    keep = {"input_ids", "attention_mask", "labels"}
    dataset = dataset.map(preprocess, batched=True, remove_columns=[col for col in dataset.column_names if col not in keep])
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=args.pad_to_multiple_of or None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.warmup_batches:
                break
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

    correct = 0
    total = 0
    latencies: list[float] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            labels = batch["labels"]
            batch_start = time.perf_counter()
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            elapsed = time.perf_counter() - batch_start
            latencies.append(elapsed)
            pred = outputs.logits.float().argmax(dim=-1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
    eval_seconds = time.perf_counter() - start
    rss_after_eval = rss_mib()

    stored_eval = metrics.get("eval", {}) if isinstance(metrics.get("eval"), dict) else {}
    return {
        "task": args.task_name,
        "run": args.run_label,
        "checkpoint_dir": str(checkpoint_dir),
        "method": method,
        "scale_mode": metrics.get("scale_mode"),
        "model_dtype": args.model_dtype,
        "threads": args.threads,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "eval_examples": total,
        "accuracy": correct / total if total else 0.0,
        "stored_full_eval_accuracy": stored_eval.get("accuracy"),
        "stored_full_eval_examples": stored_eval.get("eval_examples"),
        "load_seconds": load_seconds,
        "eval_seconds": eval_seconds,
        "examples_per_second": total / eval_seconds if eval_seconds > 0 else None,
        "mean_batch_seconds": statistics.fmean(latencies) if latencies else None,
        "p50_batch_seconds": percentile(latencies, 50),
        "p95_batch_seconds": percentile(latencies, 95),
        "rss_after_load_mib": rss_after_load,
        "rss_after_eval_mib": rss_after_eval,
        "maxrss_mib": maxrss_mib(),
    }


def child_command(args: argparse.Namespace, task: str, family: str, run: str, output: Path) -> list[str]:
    checkpoint_dir = checkpoint_dir_for(args, task, family, run)
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-run",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--task-name",
        task,
        "--run-label",
        run,
        "--threads",
        str(args.threads),
        "--batch-size",
        str(args.batch_size),
        "--max-seq-len",
        str(args.max_seq_len),
        "--max-eval-samples",
        str(args.max_eval_samples),
        "--warmup-batches",
        str(args.warmup_batches),
        "--pad-to-multiple-of",
        str(args.pad_to_multiple_of),
        "--model-dtype",
        args.model_dtype,
        "--single-output-json",
        str(output),
    ]


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    tmp_dir = args.tmp_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        for family, run in args.runs:
            checkpoint_dir = checkpoint_dir_for(args, task, family, run)
            metrics_path = checkpoint_dir / "metrics.json"
            if not metrics_path.exists():
                rows.append(
                    {
                        "task": task,
                        "run": run,
                        "family": family,
                        "checkpoint_dir": str(checkpoint_dir),
                        "status": "missing",
                    }
                )
                continue
            output = tmp_dir / f"{task}_{family}_{run.replace('/', '_')}.json"
            command = child_command(args, task, family, run, output)
            env = os.environ.copy()
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            try:
                completed = subprocess.run(
                    command,
                    cwd=Path.cwd(),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=args.child_timeout_seconds if args.child_timeout_seconds > 0 else None,
                )
            except subprocess.TimeoutExpired as exc:
                rows.append(
                    {
                        "task": task,
                        "run": run,
                        "family": family,
                        "checkpoint_dir": str(checkpoint_dir),
                        "status": "timeout",
                        "timeout_seconds": args.child_timeout_seconds,
                        "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
                        "stdout_tail": (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else "",
                    }
                )
                continue
            if completed.returncode != 0:
                rows.append(
                    {
                        "task": task,
                        "run": run,
                        "family": family,
                        "checkpoint_dir": str(checkpoint_dir),
                        "status": "failed",
                        "returncode": completed.returncode,
                        "stderr_tail": completed.stderr[-4000:],
                        "stdout_tail": completed.stdout[-1000:],
                    }
                )
                continue
            row = read_json(output)
            row.update({"family": family, "status": "complete"})
            rows.append(row)
    return {
        "schema": "bitdistill-glue-cpu-benchmark-v1",
        "date": DATE,
        "model": args.model,
        "short_root": str(args.short_root),
        "longwarmup_root": str(args.longwarmup_root),
        "paper_hparam_root": str(args.paper_hparam_root),
        "paper_hparam_row_root": str(args.paper_hparam_row_root),
        "tasks": args.tasks,
        "max_eval_samples": args.max_eval_samples,
        "threads": args.threads,
        "batch_size": args.batch_size,
        "model_dtype": args.model_dtype,
        "child_timeout_seconds": args.child_timeout_seconds,
        "note": "PyTorch CPU sequence-classification runtime; not packed I2_SR/llama.cpp inference.",
        "rows": rows,
    }


def parse_runs(values: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for value in values:
        if ":" in value:
            family, run = value.split(":", 1)
        else:
            family, run = "short", value
        valid_families = {
            "short",
            "longwarmup",
            "papergamma",
            "papergamma_row",
            "papergamma_lr1",
            "papergamma_lr5",
            "papergamma_headinit",
        }
        if family not in valid_families:
            raise ValueError(f"run family must be one of {sorted(valid_families)}: {value}")
        parsed.append((family, run))
    return parsed


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    table_rows = [
        [
            row.get("task", "-"),
            row.get("run", "-"),
            row.get("family", "-"),
            row.get("status", "-"),
            fmt(row.get("accuracy")),
            fmt(row.get("stored_full_eval_accuracy")),
            fmt(row.get("eval_examples")),
            fmt(row.get("examples_per_second")),
            fmt(row.get("mean_batch_seconds")),
            fmt(row.get("p95_batch_seconds")),
            fmt(row.get("rss_after_load_mib")),
            fmt(row.get("maxrss_mib")),
            row.get("checkpoint_dir", "-"),
        ]
        for row in summary["rows"]
    ]
    return "\n\n".join(
        [
            f"# BitDistill GLUE CPU Benchmark, {summary['date']}",
            f"Model: `{summary['model']}`.",
            f"Threads: `{summary['threads']}`. Batch size: `{summary['batch_size']}`. Max eval samples: `{summary['max_eval_samples']}`. Dtype: `{summary['model_dtype']}`. Child timeout: `{summary['child_timeout_seconds']}` seconds.",
            "This is PyTorch CPU sequence-classification runtime, not packed `I2_SR`/llama.cpp inference.",
            "## Runs",
            md_table(
                [
                    "task",
                    "run",
                    "family",
                    "status",
                    "accuracy",
                    "stored full accuracy",
                    "examples",
                    "examples/s",
                    "mean batch s",
                    "p95 batch s",
                    "RSS load MiB",
                    "max RSS MiB",
                    "checkpoint",
                ],
                table_rows,
            ),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--short-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls"))
    parser.add_argument("--longwarmup-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup"))
    parser.add_argument("--paper-hparam-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma"))
    parser.add_argument("--paper-hparam-row-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row"))
    parser.add_argument("--paper-hparam-lr1-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr1e-5"))
    parser.add_argument("--paper-hparam-lr5-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr5e-5"))
    parser.add_argument("--paper-hparam-headinit-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-headinit"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[f"{family}:{run}" for family, run in DEFAULT_RUNS],
        help="Run dirs as RUN or FAMILY:RUN, where FAMILY is short or longwarmup.",
    )
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-eval-samples", type=int, default=128)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--model-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--child-timeout-seconds", type=int, default=0)
    parser.add_argument("--tmp-dir", type=Path, default=Path("benchmark_results/bitdistill_glue_cpu_tmp"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_glue_cpu_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_glue_cpu_{DATE}.md"))
    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--task-name", choices=TASKS)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--single-output-json", type=Path)
    args = parser.parse_args()
    args.runs = parse_runs(args.runs)

    if args.single_run:
        if args.checkpoint_dir is None or args.task_name is None or args.single_output_json is None:
            raise SystemExit("--single-run requires --checkpoint-dir, --task-name, and --single-output-json")
        row = run_single(args)
        write_json(args.single_output_json, row)
        print(json.dumps(row, indent=2, sort_keys=True))
        return

    summary = run_parent(args)
    write_json(args.output_json, summary)
    markdown = render_markdown(summary)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
