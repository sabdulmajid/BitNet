#!/usr/bin/env python3
"""Gate BitDistill causal-LM I2_SR export and CPU benchmark artifacts.

This is intentionally separate from the GLUE sequence-classification gate.
Sequence-classification heads are evaluated in PyTorch in this fork; packed
llama.cpp I2_SR export is currently valid for causal-LM checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = datetime.now(timezone.utc).date().isoformat()


def model_slug(model: str) -> str:
    return model.replace("/", "-")


def safe_read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def expected_name(model: str, task: str, scale: str, layer: int) -> str:
    safe_layer = str(layer).removeprefix("-")
    return f"{model_slug(model)}_{task}_bitdistill-longwarmup-{scale}-layer-{safe_layer}"


def index_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("name", "")): row for row in rows if isinstance(row, dict)}


def collect_row(args: argparse.Namespace, *, task: str, scale: str) -> dict[str, Any]:
    name = expected_name(args.model, task, scale, args.layer)
    blockers: list[str] = []

    export_summary = safe_read_json(args.results_dir / "export_summary.json")
    if not isinstance(export_summary, dict):
        export_summary = {}
        blockers.append(f"missing export summary {args.results_dir / 'export_summary.json'}")
    exports = index_by_name(export_summary.get("exports", []) if isinstance(export_summary.get("exports"), list) else [])
    export = exports.get(name)
    outfile = Path(str(export.get("outfile", ""))) if export else None
    exported = bool(export and export.get("exists") and outfile and outfile.exists())
    if not export:
        blockers.append("missing export row")
    elif export.get("error"):
        blockers.append(str(export.get("error")))
    elif not exported:
        blockers.append("exported GGUF file is missing")
    if export and not finite_positive(export.get("ternary_keys")):
        blockers.append("ternary key count missing or zero")

    suite_summary = safe_read_json(args.results_dir / "gguf_suite" / "summary.json")
    suite_row: dict[str, Any] = {}
    if args.require_suite:
        if not isinstance(suite_summary, dict):
            blockers.append(f"missing suite summary {args.results_dir / 'gguf_suite' / 'summary.json'}")
        else:
            suite_row = index_by_name(suite_summary.get("rows", []) if isinstance(suite_summary.get("rows"), list) else []).get(name, {})
            if not suite_row:
                blockers.append("missing gguf suite row")
            else:
                bench = suite_row.get("bench", {}) if isinstance(suite_row.get("bench"), dict) else {}
                prefill = bench.get("prefill", {}) if isinstance(bench.get("prefill"), dict) else {}
                decode = bench.get("decode", {}) if isinstance(bench.get("decode"), dict) else {}
                ppl = suite_row.get("perplexity", {}) if isinstance(suite_row.get("perplexity"), dict) else {}
                if suite_row.get("smoke_returncode") != 0:
                    blockers.append("smoke run failed or missing")
                if suite_row.get("bench_returncode") != 0:
                    blockers.append("llama-bench failed or missing")
                if suite_row.get("ppl_returncode") != 0:
                    blockers.append("llama-perplexity failed or missing")
                if not finite_positive(prefill.get("tok_s")):
                    blockers.append("prefill tok/s missing or nonpositive")
                if not finite_positive(decode.get("tok_s")):
                    blockers.append("decode tok/s missing or nonpositive")
                if not finite_number(ppl.get("ppl")):
                    blockers.append("PPL missing or non-finite")

    memory_summary = safe_read_json(args.results_dir / "memory" / "summary.json")
    memory_rows: list[dict[str, Any]] = []
    if args.require_memory:
        if not isinstance(memory_summary, dict):
            blockers.append(f"missing memory summary {args.results_dir / 'memory' / 'summary.json'}")
        else:
            memory_rows = [
                row
                for row in memory_summary.get("rows", [])
                if isinstance(row, dict) and str(row.get("name", "")) == name
            ]
            by_ctx = {int(row.get("ctx_size")): row for row in memory_rows if isinstance(row.get("ctx_size"), int)}
            for ctx in args.ctx_sizes:
                row = by_ctx.get(ctx)
                if not row:
                    blockers.append(f"missing RSS row ctx={ctx}")
                elif row.get("returncode") != 0:
                    blockers.append(f"RSS run failed ctx={ctx}")
                elif not finite_positive(row.get("max_rss_gib")):
                    blockers.append(f"RSS missing ctx={ctx}")

    bench = suite_row.get("bench", {}) if isinstance(suite_row.get("bench"), dict) else {}
    prefill = bench.get("prefill", {}) if isinstance(bench.get("prefill"), dict) else {}
    decode = bench.get("decode", {}) if isinstance(bench.get("decode"), dict) else {}
    ppl = suite_row.get("perplexity", {}) if isinstance(suite_row.get("perplexity"), dict) else {}
    rss_values = [row.get("max_rss_gib") for row in memory_rows if finite_positive(row.get("max_rss_gib"))]

    return {
        "task": task,
        "scale": scale,
        "name": name,
        "exported": exported,
        "ternary_keys": export.get("ternary_keys") if export else None,
        "outfile": str(outfile) if outfile else None,
        "file_mib": suite_row.get("file_mib") or (outfile.stat().st_size / (1024**2) if outfile and outfile.exists() else None),
        "ppl": ppl.get("ppl"),
        "prefill_tok_s": prefill.get("tok_s"),
        "decode_tok_s": decode.get("tok_s"),
        "max_rss_gib": max(rss_values) if rss_values else None,
        "blockers": blockers,
        "complete": not blockers,
    }


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
    rows = [
        [
            row["task"],
            row["scale"],
            fmt(row["complete"]),
            fmt(row["exported"]),
            fmt(row["ternary_keys"]),
            fmt(row["file_mib"]),
            fmt(row["ppl"]),
            fmt(row["prefill_tok_s"]),
            fmt(row["decode_tok_s"]),
            fmt(row["max_rss_gib"]),
            "; ".join(row["blockers"]),
        ]
        for row in summary["rows"]
    ]
    sections = [
        f"# BitDistill Causal I2_SR Export Gate, {summary['date']}",
        f"Results dir: `{summary['results_dir']}`.",
        f"Passed: `{fmt(summary['passed'])}`.",
        "This gate validates packed causal-LM I2_SR artifacts only; it does not validate sequence-classification heads.",
        md_table(
            [
                "task",
                "scale",
                "complete",
                "exported",
                "ternary keys",
                "file MiB",
                "PPL",
                "prefill tok/s",
                "decode tok/s",
                "max RSS GiB",
                "blockers",
            ],
            rows,
        ),
    ]
    blockers = sorted({item for row in summary["rows"] for item in row["blockers"]})
    if blockers:
        sections.extend(["## Blockers", "\n".join(f"- {item}" for item in blockers)])
    return "\n\n".join(sections) + "\n"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(f"benchmark_results/bitdistill-causal-longwarmup-i2sr-{DATE}"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tasks", nargs="+", default=["mnli", "qnli", "sst2"])
    parser.add_argument("--scales", nargs="+", choices=["tensor", "row"], default=["tensor", "row"])
    parser.add_argument("--layer", type=int, default=-8)
    parser.add_argument("--ctx-sizes", type=int, nargs="+", default=[512, 2048, 4096])
    parser.add_argument("--require-suite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_i2sr_export_gate_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_i2sr_export_gate_{DATE}.md"))
    args = parser.parse_args()

    rows = [collect_row(args, task=task, scale=scale) for task in args.tasks for scale in args.scales]
    summary = {
        "date": DATE,
        "results_dir": str(args.results_dir),
        "model": args.model,
        "tasks": args.tasks,
        "scales": args.scales,
        "layer": args.layer,
        "ctx_sizes": args.ctx_sizes,
        "passed": all(row["complete"] for row in rows),
        "rows": rows,
    }
    write_json(args.output_json, summary)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(summary)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
