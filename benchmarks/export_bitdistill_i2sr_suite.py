#!/usr/bin/env python3
"""Export BitDistill checkpoints to GGUF and optionally run CPU benchmarks."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import torch


def model_slug(model: str) -> str:
    return model.replace("/", "-")


def count_ternary_keys(path: Path) -> int:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"{path} did not contain a state dict")
    return sum(1 for key in state if key.endswith(".ternary_weight"))


def run_command(command: list[str], *, output: Path | None, skip_existing: bool) -> None:
    print("[run] " + " ".join(command), flush=True)
    if output is not None and skip_existing and output.exists():
        print(f"[skip-existing] {output}", flush=True)
        return
    subprocess.run(command, check=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("checkpoints/bitdistill-glue"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tasks", nargs="+", default=["mnli", "qnli", "sst2"])
    parser.add_argument("--scales", nargs="+", default=["row"], choices=["tensor", "row"])
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--out-model-dir", type=Path, default=Path("models/bitdistill-i2sr"))
    parser.add_argument("--results-dir", type=Path, default=Path("benchmark_results/bitdistill-i2sr-cpu-2026-05-14"))
    parser.add_argument("--llama-bin-dir", type=Path, default=Path("build/bin"))
    parser.add_argument("--converter", type=Path, default=Path("3rdparty/llama.cpp/convert_hf_to_gguf.py"))
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--ppl-chunks", type=int, default=16)
    parser.add_argument("--perplexity-file", type=Path, default=Path("benchmark_results/gguf-ppl/wikitext2_test_excerpt.txt"))
    parser.add_argument("--ctx-sizes", type=int, nargs="+", default=[512, 2048, 4096])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--run-suite", action="store_true")
    parser.add_argument("--run-memory", action="store_true")
    args = parser.parse_args()

    slug = model_slug(args.model)
    manifest: list[dict[str, Any]] = []
    exports: list[dict[str, Any]] = []

    for task in args.tasks:
        for scale in args.scales:
            checkpoint_dir = args.root / slug / task / f"bitdistill-{scale}-layer{args.layer}"
            ternary_state = checkpoint_dir / "ternary_state_dict.pt"
            name = f"{slug}_{task}_bitdistill_{scale}_layer{args.layer}"
            outfile = args.out_model_dir / slug / task / f"{name}_bitnet25_i2sr.gguf"
            summary_json = args.results_dir / "exports" / f"{name}.json"

            if not ternary_state.exists():
                record = {
                    "name": name,
                    "task": task,
                    "scale": scale,
                    "checkpoint_dir": str(checkpoint_dir),
                    "exists": False,
                    "error": f"missing {ternary_state}",
                }
                exports.append(record)
                if args.allow_missing:
                    continue
                raise FileNotFoundError(record["error"])

            ternary_keys = count_ternary_keys(ternary_state)
            command = [
                "python",
                "benchmarks/convert_static_ternary_to_i2s_gguf.py",
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--ternary-state",
                str(ternary_state),
                "--outfile",
                str(outfile),
                "--converter",
                str(args.converter),
                "--gguf-arch",
                "bitnet-25",
                "--bitdistill-subln",
                "--row-scale-qtype",
                "i2_sr",
                "--validate-codes",
                "--expect-ternary-keys",
                str(ternary_keys),
                "--summary-json",
                str(summary_json),
            ]
            outfile.parent.mkdir(parents=True, exist_ok=True)
            run_command(command, output=outfile, skip_existing=args.skip_existing)
            manifest.append(
                {
                    "name": name,
                    "kind": f"bitdistill_{scale}_bitnet25_i2sr",
                    "task": task,
                    "scale": scale,
                    "path": str(outfile),
                }
            )
            exports.append(
                {
                    "name": name,
                    "task": task,
                    "scale": scale,
                    "checkpoint_dir": str(checkpoint_dir),
                    "ternary_keys": ternary_keys,
                    "outfile": str(outfile),
                    "summary_json": str(summary_json),
                    "exists": outfile.exists(),
                }
            )

    manifest_path = args.results_dir / "manifest.json"
    write_json(manifest_path, manifest)
    write_json(args.results_dir / "export_summary.json", {"exports": exports, "manifest": str(manifest_path)})

    if args.run_suite and manifest:
        run_command(
            [
                "python",
                "benchmarks/run_gguf_suite.py",
                "--models-json",
                str(manifest_path),
                "--out-dir",
                str(args.results_dir / "gguf_suite"),
                "--llama-bin-dir",
                str(args.llama_bin_dir),
                "--perplexity-file",
                str(args.perplexity_file),
                "--threads",
                str(args.threads),
                "--prompt-tokens",
                str(args.prompt_tokens),
                "--gen-tokens",
                str(args.gen_tokens),
                "--ppl-chunks",
                str(args.ppl_chunks),
            ],
            output=args.results_dir / "gguf_suite" / "summary.json",
            skip_existing=args.skip_existing,
        )

    if args.run_memory and manifest:
        run_command(
            [
                "python",
                "benchmarks/run_gguf_memory_probe.py",
                "--models-json",
                str(manifest_path),
                "--out-dir",
                str(args.results_dir / "memory"),
                "--llama-bin-dir",
                str(args.llama_bin_dir),
                "--threads",
                str(args.threads),
                "--ctx-sizes",
                *(str(value) for value in args.ctx_sizes),
            ],
            output=args.results_dir / "memory" / "summary.json",
            skip_existing=args.skip_existing,
        )

    print(json.dumps({"exports": exports, "manifest": str(manifest_path)}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
