#!/usr/bin/env python3
"""Prove that TL2_SR accepts a matching kernel layout and rejects a mismatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def run_loader(binary: Path, artifact: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(binary),
            "-m",
            str(artifact),
            "-p",
            "layout guard",
            "--pooling",
            "last",
            "--attention",
            "causal",
            "--embd-output-format",
            "json",
            "--embd-normalize",
            "-1",
            "-ngl",
            "0",
            "-t",
            "1",
            "-c",
            "32",
            "-b",
            "32",
            "-ub",
            "32",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def render_markdown(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# TL2_SR Layout Guard",
            "",
            f"Generated: `{result['created_utc']}`. Status: **{result['status']}**.",
            "",
            "| case | exit code | expected | result |",
            "| --- | ---: | --- | --- |",
            f"| matching BM64 runtime | {result['matching']['returncode']} | accept | {'pass' if result['matching']['passed'] else 'fail'} |",
            f"| mismatched BM32 runtime | {result['mismatched']['returncode']} | reject | {'pass' if result['mismatched']['passed'] else 'fail'} |",
            "",
            "The mismatch is accepted as evidence only when the loader exits nonzero and reports",
            "`TL2_SR kernel-layout mismatch` with the artifact and runtime fingerprints.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "models/seqcls-native-tl2sr/Qwen-Qwen2.5-0.5B/mnli/"
            "bitdistill-longwarmup-row-layer-8_bitnet_qwen_tl2_sr_bm64_cls.gguf"
        ),
    )
    parser.add_argument(
        "--matching-binary",
        type=Path,
        default=Path("build-qwen05b-tl2sr-bm64/bin/llama-embedding"),
    )
    parser.add_argument(
        "--mismatched-binary",
        type=Path,
        default=Path("build-qwen05b-tl2sr-bm32/bin/llama-embedding"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmark_results/tl2sr_layout_guard_2026-09-04.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/tl2sr_layout_guard_2026-09-04.md"),
    )
    args = parser.parse_args()

    for path in (args.artifact, args.matching_binary, args.mismatched_binary):
        if not path.is_file():
            raise FileNotFoundError(path)

    matching = run_loader(args.matching_binary, args.artifact)
    mismatched = run_loader(args.mismatched_binary, args.artifact)
    mismatch_marker = "TL2_SR kernel-layout mismatch"
    matching_passed = matching.returncode == 0 and mismatch_marker not in matching.stderr
    mismatched_passed = mismatched.returncode != 0 and mismatch_marker in mismatched.stderr
    result = {
        "schema": "tl2sr-layout-guard-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if matching_passed and mismatched_passed else "fail",
        "artifact": file_identity(args.artifact),
        "matching_binary": file_identity(args.matching_binary),
        "mismatched_binary": file_identity(args.mismatched_binary),
        "matching": {
            "returncode": matching.returncode,
            "passed": matching_passed,
            "stderr_tail": matching.stderr.splitlines()[-20:],
        },
        "mismatched": {
            "returncode": mismatched.returncode,
            "passed": mismatched_passed,
            "stderr_tail": mismatched.stderr.splitlines()[-20:],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
