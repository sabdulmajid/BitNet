#!/usr/bin/env python3
"""Build GGUF artifacts from a static ternary checkpoint via the dense bridge.

This is an orchestration wrapper around the current validated bridge:

1. materialize `ternary_state_dict.pt` to a temporary/dense HF checkpoint,
2. convert that HF checkpoint to F16 GGUF,
3. quantize the GGUF to TQ2_0 and I2_S,
4. write a benchmark manifest, and optionally
5. run the GGUF benchmark suite plus evidence audit.

It is intentionally not described as a direct GGUF writer. The direct writer is
still a research/engineering gap; this script makes the present bridge
reproducible and auditable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CommandRecord:
    label: str
    command: list[str]
    skipped: bool = False


def run_command(record: CommandRecord, *, dry_run: bool) -> None:
    print(f"[{record.label}] {' '.join(record.command)}", flush=True)
    if dry_run or record.skipped:
        return
    subprocess.run(record.command, check=True)


def maybe_run(
    records: list[CommandRecord],
    label: str,
    command: list[str],
    *,
    output: Path | None,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    skipped = bool(output is not None and skip_existing and output.exists())
    record = CommandRecord(label, command, skipped=skipped)
    records.append(record)
    run_command(record, dry_run=dry_run)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--ternary-state", type=Path, default=None)
    parser.add_argument("--expect-ternary-keys", type=int, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--out-model-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--llama-bin-dir", type=Path, default=Path("build-portable-avx2/bin"))
    parser.add_argument("--llama-converter", type=Path, default=Path("3rdparty/llama.cpp/convert_hf_to_gguf.py"))
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--dtype", default="float16", choices=["fp16", "float16", "bf16", "bfloat16", "fp32", "float32"])
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--ppl-chunks", type=int, default=16)
    parser.add_argument("--perplexity-file", type=Path, default=Path("benchmark_results/gguf-ppl/wikitext2_test_excerpt.txt"))
    parser.add_argument("--fp-f16", type=Path, default=Path("models/qwen2.5-1.5b-fp/qwen15b_fp_f16.gguf"))
    parser.add_argument("--fp-q8", type=Path, default=Path("models/qwen2.5-1.5b-fp/qwen15b_fp_q8_0.gguf"))
    parser.add_argument("--fp-q4", type=Path, default=Path("models/qwen2.5-1.5b-fp/qwen15b_fp_q4_k_m.gguf"))
    parser.add_argument("--run-suite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-validate-codes", action="store_true")
    args = parser.parse_args()

    out_model_dir = args.out_model_dir
    results_dir = args.results_dir or Path("benchmark_results") / f"gguf-{args.run_label}"
    hf_dense_dir = out_model_dir / "hf_f16"
    f16_gguf = out_model_dir / f"{args.run_label}_f16.gguf"
    tq2_gguf = out_model_dir / f"{args.run_label}_tq2_0.gguf"
    i2s_gguf = out_model_dir / f"{args.run_label}_i2_s_t1.gguf"
    manifest_path = out_model_dir / f"{args.run_label}_manifest.json"
    bridge_summary_path = out_model_dir / f"{args.run_label}_bridge_summary.json"

    if not args.dry_run:
        out_model_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    records: list[CommandRecord] = []
    materialize_command = [
        "python",
        "benchmarks/materialize_static_ternary_hf.py",
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--output-dir",
        str(hf_dense_dir),
        "--dtype",
        args.dtype,
        "--expect-ternary-keys",
        str(args.expect_ternary_keys),
    ]
    if args.ternary_state is not None:
        materialize_command.extend(["--ternary-state", str(args.ternary_state)])
    if not args.no_validate_codes:
        materialize_command.append("--validate-codes")
    maybe_run(records, "materialize", materialize_command, output=hf_dense_dir / "model.safetensors", skip_existing=args.skip_existing, dry_run=args.dry_run)

    maybe_run(
        records,
        "convert-f16-gguf",
        [
            "python",
            str(args.llama_converter),
            str(hf_dense_dir),
            "--outfile",
            str(f16_gguf),
            "--outtype",
            "f16",
        ],
        output=f16_gguf,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )

    for qtype, output in (("TQ2_0", tq2_gguf), ("I2_S", i2s_gguf)):
        maybe_run(
            records,
            f"quantize-{qtype.lower()}",
            [
                "python",
                "benchmarks/quantize_gguf_safe.py",
                "--llama-quantize",
                str(args.llama_bin_dir / "llama-quantize"),
                "--input",
                str(f16_gguf),
                "--output",
                str(output),
                "--type",
                qtype,
                "--threads",
                str(args.threads),
            ],
            output=output,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )

    manifest = [
        {"name": "qwen15b_fp_f16", "kind": "fp_reference", "path": str(args.fp_f16)},
        {"name": "qwen15b_fp_q8_0", "kind": "llama_q8", "path": str(args.fp_q8)},
        {"name": "qwen15b_fp_q4_k_m", "kind": "llama_q4", "path": str(args.fp_q4)},
        {"name": f"{args.run_label}_f16", "kind": "static_ternary_materialized", "path": str(f16_gguf)},
        {"name": f"{args.run_label}_tq2_0", "kind": "static_ternary_tq2", "path": str(tq2_gguf)},
        {"name": f"{args.run_label}_i2_s", "kind": "static_ternary_i2s_single_thread_quant", "path": str(i2s_gguf)},
    ]
    if not args.dry_run:
        write_json(manifest_path, manifest)
    print(f"[manifest] {manifest_path}", flush=True)

    if args.run_suite:
        maybe_run(
            records,
            "run-gguf-suite",
            [
                "python",
                "benchmarks/run_gguf_suite.py",
                "--models-json",
                str(manifest_path),
                "--out-dir",
                str(results_dir),
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
            output=results_dir / "summary.json",
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
        maybe_run(
            records,
            "audit-gguf-suite",
            [
                "python",
                "benchmarks/audit_evidence.py",
                "--gguf-summary",
                f"{args.run_label}={results_dir / 'summary.json'}:6",
                "--output-md",
                str(results_dir / "audit.md"),
            ],
            output=results_dir / "audit.md",
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )

    bridge_summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "ternary_state": str(args.ternary_state or args.checkpoint_dir / "ternary_state_dict.pt"),
        "expect_ternary_keys": args.expect_ternary_keys,
        "run_label": args.run_label,
        "out_model_dir": str(out_model_dir),
        "results_dir": str(results_dir),
        "manifest": str(manifest_path),
        "artifacts": {
            "hf_dense_dir": str(hf_dense_dir),
            "f16_gguf": str(f16_gguf),
            "tq2_gguf": str(tq2_gguf),
            "i2s_gguf": str(i2s_gguf),
        },
        "bridge_limitations": [
            "This materializes dense HF tensors before GGUF conversion.",
            "It is not a direct ternary_state_dict.pt GGUF writer.",
            "I2_S quantization is routed through quantize_gguf_safe.py, which forces one writer thread unless explicitly overridden there.",
        ],
        "commands": [
            {"label": record.label, "command": record.command, "skipped": record.skipped}
            for record in records
        ],
    }
    if not args.dry_run:
        write_json(bridge_summary_path, bridge_summary)
    print(json.dumps(bridge_summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
