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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perplexity-glob", default="benchmark_results/perplexity/*.json")
    parser.add_argument("--generation-glob", default="benchmark_results/generation/*.jsonl")
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    report = "\n\n".join([
        "# Benchmark Summary",
        "## Perplexity",
        summarize_perplexity(args.perplexity_glob),
        "## Generation",
        summarize_generation(args.generation_glob),
    ])
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
