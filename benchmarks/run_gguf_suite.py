#!/usr/bin/env python3
"""Run reproducible llama.cpp GGUF smoke, speed, and perplexity benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


def read_models(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("model manifest must be a JSON list")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"manifest item {index} is not an object")
        if "name" not in item or "path" not in item:
            raise ValueError(f"manifest item {index} must contain name and path")
    return data


def run_command(command: list[str], stdout_path: Path, stderr_path: Path, *, skip_existing: bool) -> int:
    if skip_existing and stdout_path.exists() and stderr_path.exists():
        return 0
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    return completed.returncode


def load_bench(path: Path) -> dict[str, dict[str, float | str]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, float | str]] = {}
    for row in rows:
        mode = "prefill" if int(row.get("n_gen", 0)) == 0 else "decode"
        result[mode] = {
            "tok_s": float(row.get("avg_ts", 0.0)),
            "stddev_tok_s": float(row.get("stddev_ts", 0.0)),
            "model_type": str(row.get("model_type", "")),
            "bench_model_bytes": float(row.get("model_size", 0.0)),
            "cpu": str(row.get("cpu_info", "")),
        }
    return result


def parse_ppl(path: Path, stderr_path: Path | None = None) -> dict[str, float | str | None]:
    parts = []
    if path.exists():
        parts.append(path.read_text(encoding="utf-8", errors="replace"))
    if stderr_path is not None and stderr_path.exists():
        parts.append(stderr_path.read_text(encoding="utf-8", errors="replace"))
    text = "\n".join(parts)
    final = re.search(r"Final estimate: PPL = ([^ ]+) \+/- ([^\n]+)", text)
    speed = re.search(
        r"prompt eval time =\s+([0-9.]+) ms /\s+([0-9]+) tokens .*?([0-9.]+) tokens per second",
        text,
    )
    if final:
        ppl_text = final.group(1)
        stderr_text = final.group(2).strip()
        try:
            ppl: float | str = float(ppl_text)
        except ValueError:
            ppl = ppl_text
        try:
            stderr: float | str = float(stderr_text)
        except ValueError:
            stderr = stderr_text
    elif "-nan" in text.lower() or "negative standard deviation" in text.lower():
        ppl = "NaN"
        stderr = "NaN"
    else:
        ppl = None
        stderr = None

    return {
        "ppl": ppl,
        "stderr": stderr,
        "tokens": int(speed.group(2)) if speed else None,
        "tok_s": float(speed.group(3)) if speed else None,
    }


def read_first_line(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return " ".join(text.split())[:240]


def fmt_float(value: object, digits: int = 2) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.{digits}f}"
    return "" if value is None else str(value)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "name",
        "kind",
        "file MiB",
        "prefill tok/s",
        "decode tok/s",
        "PPL",
        "PPL tok/s",
        "smoke",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("name", "")),
                    str(row.get("kind", "")),
                    fmt_float(row.get("file_mib"), 1),
                    fmt_float(row.get("bench", {}).get("prefill", {}).get("tok_s")),
                    fmt_float(row.get("bench", {}).get("decode", {}).get("tok_s")),
                    fmt_float(row.get("perplexity", {}).get("ppl"), 4),
                    fmt_float(row.get("perplexity", {}).get("tok_s")),
                    str(row.get("smoke", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run llama.cpp GGUF benchmarks from a manifest")
    parser.add_argument("--models-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--llama-bin-dir", type=Path, default=Path("build/bin"))
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--perplexity-file", type=Path, default=None)
    parser.add_argument("--ppl-chunks", type=int, default=16)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--smoke-tokens", type=int, default=24)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--no-ppl", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = read_models(args.models_json)
    summary_rows: list[dict[str, Any]] = []

    llama_cli = args.llama_bin_dir / "llama-cli"
    llama_bench = args.llama_bin_dir / "llama-bench"
    llama_ppl = args.llama_bin_dir / "llama-perplexity"

    for item in models:
        name = str(item["name"])
        model_path = Path(str(item["path"]))
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        row: dict[str, Any] = {
            "name": name,
            "kind": str(item.get("kind", "")),
            "path": str(model_path),
            "exists": model_path.exists(),
            "file_mib": model_path.stat().st_size / (1024**2) if model_path.exists() else None,
            "notes": item.get("notes", ""),
        }

        if not model_path.exists():
            row["error"] = "model path does not exist"
            summary_rows.append(row)
            continue

        if not args.no_smoke:
            smoke_out = args.out_dir / f"{stem}.smoke.txt"
            smoke_err = args.out_dir / f"{stem}.smoke.err"
            code = run_command(
                [
                    str(llama_cli),
                    "-m",
                    str(model_path),
                    "-p",
                    args.prompt,
                    "-n",
                    str(args.smoke_tokens),
                    "-t",
                    str(args.threads),
                    "-ngl",
                    "0",
                    "--temp",
                    "0",
                    "--no-display-prompt",
                ],
                smoke_out,
                smoke_err,
                skip_existing=args.skip_existing,
            )
            row["smoke_returncode"] = code
            row["smoke"] = read_first_line(smoke_out)

        if not args.no_bench:
            bench_out = args.out_dir / f"{stem}.bench.json"
            bench_err = args.out_dir / f"{stem}.bench.err"
            code = run_command(
                [
                    str(llama_bench),
                    "-m",
                    str(model_path),
                    "-p",
                    str(args.prompt_tokens),
                    "-n",
                    str(args.gen_tokens),
                    "-t",
                    str(args.threads),
                    "-ngl",
                    "0",
                    "-r",
                    str(args.repetitions),
                    "-o",
                    "json",
                ],
                bench_out,
                bench_err,
                skip_existing=args.skip_existing,
            )
            row["bench_returncode"] = code
            row["bench"] = load_bench(bench_out)

        if not args.no_ppl and args.perplexity_file is not None:
            ppl_out = args.out_dir / f"{stem}.ppl.log"
            ppl_err = args.out_dir / f"{stem}.ppl.err"
            code = run_command(
                [
                    str(llama_ppl),
                    "-m",
                    str(model_path),
                    "-f",
                    str(args.perplexity_file),
                    "-c",
                    str(args.ctx_size),
                    "--chunks",
                    str(args.ppl_chunks),
                    "-t",
                    str(args.threads),
                    "-ngl",
                    "0",
                ],
                ppl_out,
                ppl_err,
                skip_existing=args.skip_existing,
            )
            row["ppl_returncode"] = code
            row["perplexity"] = parse_ppl(ppl_out, ppl_err)

        summary_rows.append(row)

    summary = {
        "models_json": str(args.models_json),
        "out_dir": str(args.out_dir),
        "prompt": args.prompt,
        "perplexity_file": str(args.perplexity_file) if args.perplexity_file else None,
        "ctx_size": args.ctx_size,
        "threads": args.threads,
        "prompt_tokens": args.prompt_tokens,
        "gen_tokens": args.gen_tokens,
        "repetitions": args.repetitions,
        "rows": summary_rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text(markdown_table(summary_rows) + "\n", encoding="utf-8")
    print(markdown_table(summary_rows), flush=True)


if __name__ == "__main__":
    main()
