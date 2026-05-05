#!/usr/bin/env python3
"""Measure llama.cpp GGUF peak RSS with /usr/bin/time -v."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s+([0-9]+)")
ELAPSED_RE = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s+(.+)")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("manifest must be a JSON list")
    return data


def parse_time_stderr(text: str) -> dict[str, Any]:
    rss_match = RSS_RE.search(text)
    elapsed_match = ELAPSED_RE.search(text)
    return {
        "max_rss_kib": int(rss_match.group(1)) if rss_match else None,
        "max_rss_gib": int(rss_match.group(1)) / (1024**2) if rss_match else None,
        "elapsed": elapsed_match.group(1).strip() if elapsed_match else "",
    }


def fmt(value: Any, digits: int = 2) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.{digits}f}"
    return "-" if value is None else str(value)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| name | kind | ctx | file MiB | max RSS GiB | return code |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("name", "")),
                    str(row.get("kind", "")),
                    str(row.get("ctx_size", "")),
                    fmt(row.get("file_mib"), 1),
                    fmt(row.get("max_rss_gib"), 3),
                    str(row.get("returncode", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--llama-bin-dir", type=Path, default=Path("build/bin"))
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--ctx-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    llama_cli = args.llama_bin_dir / "llama-cli"
    rows: list[dict[str, Any]] = []
    ctx_sizes = args.ctx_sizes if args.ctx_sizes is not None else [args.ctx_size]

    for ctx_size in ctx_sizes:
        for item in read_manifest(args.models_json):
            name = str(item["name"])
            model_path = Path(str(item["path"]))
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
            stdout_path = args.out_dir / f"{stem}.ctx{ctx_size}.stdout"
            stderr_path = args.out_dir / f"{stem}.ctx{ctx_size}.stderr"
            command = [
                "/usr/bin/time",
                "-v",
                str(llama_cli),
                "-m",
                str(model_path),
                "-p",
                args.prompt,
                "-n",
                str(args.tokens),
                "-c",
                str(ctx_size),
                "-t",
                str(args.threads),
                "-ngl",
                "0",
                "--temp",
                "0",
                "--no-display-prompt",
            ]

            row: dict[str, Any] = {
                "name": name,
                "kind": str(item.get("kind", "")),
                "path": str(model_path),
                "ctx_size": ctx_size,
                "exists": model_path.exists(),
                "file_mib": model_path.stat().st_size / (1024**2) if model_path.exists() else None,
            }
            if not model_path.exists():
                row["returncode"] = None
                row["error"] = "missing model"
                rows.append(row)
                continue

            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
            row["returncode"] = completed.returncode
            row.update(parse_time_stderr(stderr_path.read_text(encoding="utf-8", errors="replace")))
            rows.append(row)

    summary = {
        "models_json": str(args.models_json),
        "out_dir": str(args.out_dir),
        "llama_bin_dir": str(args.llama_bin_dir),
        "threads": args.threads,
        "ctx_size": args.ctx_size,
        "ctx_sizes": ctx_sizes,
        "tokens": args.tokens,
        "prompt": args.prompt,
        "rows": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    table = markdown_table(rows)
    (args.out_dir / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
