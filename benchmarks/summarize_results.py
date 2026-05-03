#!/usr/bin/env python3
"""Summarize benchmark JSON/JSONL outputs into Markdown tables."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def summarize_perplexity(pattern: str) -> str:
    rows = []
    for item in sorted(glob.glob(pattern)):
        path = Path(item)
        data = read_json(path)
        rows.append([
            path.stem,
            data.get("model_kind", ""),
            str(data.get("model") or data.get("checkpoint_dir") or ""),
            f"{float(data.get('perplexity', 0.0)):.3f}",
            f"{float(data.get('nll', 0.0)):.4f}",
            f"{float(data.get('tokens_per_second', 0.0)):.2f}",
            str(int(float(data.get("eval_tokens", 0)))),
            f"{int(data.get('max_blocks', 0))}x{int(data.get('max_seq_len', 0))}",
        ])
    if not rows:
        return "No perplexity results found."
    return md_table(
        ["run", "kind", "model_or_checkpoint", "ppl", "nll", "tok/s", "tokens", "blocks"],
        rows,
    )


def summarize_generation(pattern: str) -> str:
    rows = []
    for item in sorted(glob.glob(pattern)):
        path = Path(item)
        records = read_jsonl(path)
        if not records:
            continue
        if "tokens_per_second" not in records[0]:
            continue
        avg_tps = sum(float(record["tokens_per_second"]) for record in records) / len(records)
        rows.append([
            path.stem,
            str(records[0].get("checkpoint_dir", "")),
            str(len(records)),
            f"{avg_tps:.3f}",
            str(records[0].get("device", "")),
            str(records[0].get("dtype", "")),
        ])
    if not rows:
        return "No generation results found."
    return md_table(["run", "checkpoint", "prompts", "avg tok/s", "device", "dtype"], rows)


def bytes_to_gib(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value) / (1024 ** 3):.3f}"
    except (TypeError, ValueError):
        return ""


def summarize_runtime(pattern: str) -> str:
    rows = []
    for item in sorted(glob.glob(pattern)):
        path = Path(item)
        data = read_json(path)
        if "prefill" not in data or "generate" not in data:
            continue
        rows.append([
            path.stem,
            str(data.get("model_kind", "")),
            str(data.get("model") or data.get("checkpoint_dir") or ""),
            str(data.get("device", "")),
            str(data.get("dtype", "")),
            str(data.get("torch_num_threads", "")),
            str(data.get("prompt_tokens", "")),
            str(data.get("generated_tokens_observed", "")),
            f"{float(data.get('prefill', {}).get('tokens_per_second_median', 0.0)):.2f}",
            f"{float(data.get('generate', {}).get('new_tokens_per_second_median_including_prefill', 0.0)):.2f}",
            f"{float(data.get('generate', {}).get('decode_tokens_per_second_estimate', 0.0)):.2f}",
            bytes_to_gib(data.get("rss_after_move_bytes")),
            bytes_to_gib(data.get("model_storage_bytes")),
            bytes_to_gib(data.get("ternary_state_bytes")),
            bytes_to_gib(data.get("checkpoint_safetensors_bytes")),
        ])
    if not rows:
        return "No runtime probe results found."
    return md_table(
        [
            "run",
            "kind",
            "model_or_checkpoint",
            "device",
            "dtype",
            "threads",
            "prompt",
            "new",
            "prefill tok/s",
            "gen tok/s",
            "decode est tok/s",
            "RSS GiB",
            "model GiB",
            "ternary GiB",
            "safetensors GiB",
        ],
        rows,
    )


def summarize_mc(pattern: str) -> str:
    rows = []
    for item in sorted(glob.glob(pattern)):
        path = Path(item)
        data = read_json(path)
        if "accuracy" not in data or "task" not in data:
            continue
        rows.append([
            path.stem,
            str(data.get("task", "")),
            data.get("model_kind", ""),
            str(data.get("model") or data.get("checkpoint_dir") or ""),
            f"{float(data.get('accuracy', 0.0)):.4f}",
            f"{float(data.get('accuracy_norm', 0.0)):.4f}",
            str(int(data.get("limit", 0))),
            f"{float(data.get('examples_per_second', 0.0)):.2f}",
        ])
    if not rows:
        return "No multiple-choice results found."
    return md_table(
        ["run", "task", "kind", "model_or_checkpoint", "acc", "acc_norm", "n", "ex/s"],
        rows,
    )


def summarize_lm_eval(pattern: str) -> str:
    rows = []
    for item in sorted(glob.glob(pattern)):
        path = Path(item)
        data = read_json(path)
        results = data.get("results")
        configs = data.get("configs", {})
        higher_is_better = data.get("higher_is_better", {})
        if not isinstance(results, dict):
            continue
        for task, metrics in sorted(results.items()):
            if not isinstance(metrics, dict):
                continue
            for metric, value in sorted(metrics.items()):
                if metric == "alias" or metric.endswith("_stderr,none"):
                    continue
                if not isinstance(value, (int, float)):
                    continue
                stderr = metrics.get(metric.replace(",none", "_stderr,none"), "")
                hib = ""
                if isinstance(higher_is_better, dict):
                    hib = str(higher_is_better.get(task, {}).get(metric, ""))
                rows.append([
                    path.stem,
                    task,
                    str(configs.get(task, {}).get("num_fewshot", "")) if isinstance(configs, dict) else "",
                    metric,
                    f"{float(value):.4f}",
                    f"{float(stderr):.4f}" if isinstance(stderr, (int, float)) else "",
                    hib,
                ])
    if not rows:
        return "No lm-eval results found."
    return md_table(["run", "task", "fewshot", "metric", "value", "stderr", "higher_is_better"], rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perplexity-glob", default="benchmark_results/perplexity/*.json")
    parser.add_argument("--generation-glob", default="benchmark_results/generation/*.jsonl")
    parser.add_argument("--mc-glob", default="benchmark_results/mc/*.json")
    parser.add_argument("--lm-eval-glob", default="benchmark_results/lm_eval/*.json")
    parser.add_argument("--runtime-glob", default="benchmark_results/runtime/*.json")
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    report = "\n\n".join([
        "# Benchmark Summary",
        "## Perplexity",
        summarize_perplexity(args.perplexity_glob),
        "## Multiple Choice",
        summarize_mc(args.mc_glob),
        "## lm-eval",
        summarize_lm_eval(args.lm_eval_glob),
        "## Runtime",
        summarize_runtime(args.runtime_glob),
        "## Generation",
        summarize_generation(args.generation_glob),
    ])
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
