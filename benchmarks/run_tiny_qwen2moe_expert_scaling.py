#!/usr/bin/env python3
"""Run synthetic Qwen2MoE expert-count/top-k CPU scaling probes.

This is a systems microbenchmark, not a quality benchmark. It checks whether
the vendored llama.cpp runtime can execute tiny routed MoE graphs with different
expert counts and active experts per token, and records throughput/RSS deltas.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_tiny_qwen2moe_fixture import make_tiny_model, parse_rss, run_command, summarize_output


DATE = datetime.now(timezone.utc).date().isoformat()


def run_variant(args: argparse.Namespace, *, num_experts: int, used_experts: int) -> dict[str, Any]:
    label = f"experts{num_experts}_top{used_experts}"
    model_dir = args.model_root / label
    gguf = model_dir / f"tiny-qwen2moe-{label}-f16.gguf"
    out_dir = args.out_dir / label
    hf_model = make_tiny_model(
        model_dir,
        args.tokenizer_model,
        seed=args.seed + num_experts * 100 + used_experts,
        force=args.force,
        num_experts=num_experts,
        num_experts_per_tok=used_experts,
    )
    convert = run_command(
        [
            "python",
            str(args.converter),
            str(model_dir),
            "--outfile",
            str(gguf),
            "--outtype",
            "f16",
        ],
        stdout_path=out_dir / "convert.stdout",
        stderr_path=out_dir / "convert.stderr",
        skip_existing=args.skip_existing and gguf.exists(),
    )
    smoke = {"returncode": None}
    rss_cmd = {"returncode": None}
    runtime = {}
    rss = {}
    if convert["returncode"] == 0:
        smoke = run_command(
            [
                str(args.llama_bin_dir / "llama-cli"),
                "-m",
                str(gguf),
                "-p",
                args.prompt,
                "-n",
                str(args.tokens),
                "-t",
                str(args.threads),
                "-ngl",
                "0",
                "--temp",
                "0",
                "--no-display-prompt",
            ],
            stdout_path=out_dir / "smoke.stdout",
            stderr_path=out_dir / "smoke.stderr",
            skip_existing=args.skip_existing and (out_dir / "smoke.stdout").exists(),
        )
        runtime = summarize_output(out_dir / "smoke.stdout", out_dir / "smoke.stderr")
        rss_cmd = run_command(
            [
                "/usr/bin/time",
                "-v",
                str(args.llama_bin_dir / "llama-cli"),
                "-m",
                str(gguf),
                "-p",
                args.prompt,
                "-n",
                str(args.rss_tokens),
                "-t",
                str(args.threads),
                "-ngl",
                "0",
                "--temp",
                "0",
                "--no-display-prompt",
            ],
            stdout_path=out_dir / "rss.stdout",
            stderr_path=out_dir / "rss.stderr",
            skip_existing=False,
        )
        rss = parse_rss(out_dir / "rss.stderr")

    return {
        "label": label,
        "num_experts": num_experts,
        "num_experts_per_tok": used_experts,
        "hf_model": hf_model,
        "gguf_path": str(gguf),
        "gguf_mib": gguf.stat().st_size / (1024 * 1024) if gguf.exists() else None,
        "commands": {"convert": convert, "smoke": smoke, "rss": rss_cmd},
        "runtime": runtime,
        "rss": rss,
        "passed": convert.get("returncode") == 0 and smoke.get("returncode") == 0 and rss_cmd.get("returncode") == 0,
    }


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary["rows"]:
        runtime = row.get("runtime", {})
        rss = row.get("rss", {})
        rows.append(
            [
                row["label"],
                fmt(row["passed"]),
                fmt(row["num_experts"]),
                fmt(row["num_experts_per_tok"]),
                fmt(row.get("gguf_mib")),
                fmt(runtime.get("model_params_m")),
                fmt(runtime.get("cpu_buffer_mib")),
                fmt(runtime.get("prompt_eval_tok_s")),
                fmt(runtime.get("decode_tok_s")),
                fmt(rss.get("max_rss_mib")),
            ]
        )
    return "\n\n".join(
        [
            f"# Tiny Qwen2MoE Expert Scaling, {summary['date']}",
            f"Overall status: `{'pass' if summary['passed'] else 'fail'}`.",
            "This is a synthetic CPU runtime scaling probe for random tiny Qwen2MoE models. It does not measure model quality, Kimi compatibility, ternary MoE correctness, or router accuracy.",
            md_table(
                [
                    "variant",
                    "pass",
                    "experts",
                    "top-k",
                    "GGUF MiB",
                    "params M",
                    "CPU buffer MiB",
                    "prompt tok/s",
                    "decode tok/s",
                    "RSS MiB",
                ],
                rows,
            ),
            "## Interpretation",
            "A passing row means the converter/runtime can execute that routed graph shape on CPU. Publishable/product claims still require a trained MoE checkpoint, quality evaluation, and a Kimi-specific tensor mapping audit.",
        ]
    ) + "\n"


def parse_variants(values: list[str]) -> list[tuple[int, int]]:
    variants: list[tuple[int, int]] = []
    for value in values:
        left, right = value.split(":", 1)
        variants.append((int(left), int(right)))
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model-root", type=Path, default=Path("models/tiny-qwen2moe-scaling"))
    parser.add_argument("--converter", type=Path, default=Path("3rdparty/llama.cpp/convert_hf_to_gguf.py"))
    parser.add_argument("--llama-bin-dir", type=Path, default=Path("build-portable-avx2/bin"))
    parser.add_argument("--out-dir", type=Path, default=Path(f"benchmark_results/tiny_qwen2moe_expert_scaling_{DATE}"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/tiny_qwen2moe_expert_scaling_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/tiny_qwen2moe_expert_scaling_{DATE}.md"))
    parser.add_argument("--variant", action="append", default=["2:1", "2:2", "4:1", "4:2"])
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--rss-tokens", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rows = [run_variant(args, num_experts=num_experts, used_experts=used) for num_experts, used in parse_variants(args.variant)]
    summary = {
        "schema": "tiny-qwen2moe-expert-scaling-v1",
        "date": DATE,
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
        "does_not_prove": [
            "Kimi model support.",
            "Ternary or row-scale I2_SR MoE support.",
            "Task or language-model quality.",
            "Router accuracy or expert-locality quality.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
