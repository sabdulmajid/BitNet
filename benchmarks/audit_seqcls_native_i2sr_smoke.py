#!/usr/bin/env python3
"""Audit native packed I2_SR sequence-classification GGUF smoke output."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
DEFAULT_GGUF = Path(
    "models/seqcls-native-i2sr/"
    "Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8_bitnet_qwen_i2_sr_cls.gguf"
)
DEFAULT_CONVERSION = Path(f"benchmark_results/seqcls_native_i2sr_gguf_{DATE}.json")
DEFAULT_SIDECAR = Path(f"benchmark_results/seqcls_backbone_i2sr_smoke_{DATE}.json")
DEFAULT_PROMPT = "Premise: A person is riding a bicycle. Hypothesis: Someone is outdoors."
PROMPT_EVAL_RE = re.compile(
    r"prompt eval time =\s+(?P<ms>[0-9.]+) ms /\s+(?P<tokens>[0-9]+) tokens.*?"
    r"(?P<tps>[0-9.]+) tokens per second"
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def parse_stdout(stdout: str) -> list[float]:
    parsed = json.loads(stdout)
    rows = parsed.get("data", [])
    if not rows:
        raise ValueError("llama-embedding JSON output has no data rows")
    embedding = rows[0].get("embedding")
    if not isinstance(embedding, list):
        raise TypeError("llama-embedding JSON row does not contain an embedding array")
    return [float(value) for value in embedding]


def parse_perf(stderr: str) -> dict[str, Any]:
    match = PROMPT_EVAL_RE.search(stderr)
    if not match:
        return {}
    return {
        "prompt_eval_ms": float(match.group("ms")),
        "prompt_eval_tokens": int(match.group("tokens")),
        "prompt_eval_tokens_per_second": float(match.group("tps")),
    }


def rel_rms(candidate: list[float], reference: list[float]) -> float | None:
    if len(candidate) != len(reference) or not candidate:
        return None
    numerator = sum((a - b) ** 2 for a, b in zip(candidate, reference, strict=True))
    denominator = sum(b * b for b in reference)
    if denominator <= 0:
        return None
    return math.sqrt(numerator / denominator)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return "[" + ", ".join(fmt(item) for item in value) + "]"
    return str(value).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            f"# Sequence-Classification Native I2_SR Smoke, {result['date']}",
            (
                "This is a single-prompt native GGUF smoke test. It proves that the packed "
                "BitNet-Qwen artifact can carry the dense classifier head and emit finite "
                "classifier logits without an NPZ sidecar. It is not a full GLUE quality benchmark."
            ),
            md_table(
                ["field", "value"],
                [
                    ["status", result["status"]],
                    ["single artifact", result["single_artifact"]],
                    ["logit count", result["logit_count"]],
                    ["prediction", result["prediction"]],
                    ["sidecar prediction", result["sidecar_prediction"]],
                    ["max abs logit delta", result["max_abs_logit_delta"]],
                    ["relative RMS logit delta", result["relative_rms_logit_delta"]],
                    ["prompt tok/s", result.get("runtime", {}).get("prompt_eval_tokens_per_second")],
                    ["full validation complete", result["full_validation_complete"]],
                    ["ready to productize", result["ready_to_productize"]],
                ],
            ),
            "## Verdict",
            result["verdict"],
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--conversion-json", type=Path, default=DEFAULT_CONVERSION)
    parser.add_argument("--sidecar-smoke-json", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--embedding-binary", type=Path, default=Path("build-portable-avx2/bin/llama-embedding"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--ctx-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/seqcls_native_i2sr_smoke_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/seqcls_native_i2sr_smoke_{DATE}.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    gguf = args.gguf if args.gguf.is_absolute() else root / args.gguf
    binary = args.embedding_binary if args.embedding_binary.is_absolute() else root / args.embedding_binary
    conversion_json = args.conversion_json if args.conversion_json.is_absolute() else root / args.conversion_json
    sidecar_json = args.sidecar_smoke_json if args.sidecar_smoke_json.is_absolute() else root / args.sidecar_smoke_json

    command = [
        str(binary),
        "-m",
        str(gguf),
        "-p",
        args.prompt,
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
        str(args.threads),
        "-c",
        str(args.ctx_size),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=300)
    elapsed = time.perf_counter() - started
    if completed.returncode == 0:
        logits = parse_stdout(completed.stdout)
    else:
        logits = []

    conversion = read_json(conversion_json)
    sidecar = read_json(sidecar_json)
    sidecar_logits = [
        float(value)
        for value in sidecar.get("classifier_head_application", {}).get("logits", [])
        if isinstance(value, (int, float))
    ]
    max_abs_delta = (
        max(abs(a - b) for a, b in zip(logits, sidecar_logits, strict=True))
        if logits and len(logits) == len(sidecar_logits)
        else None
    )
    relative = rel_rms(logits, sidecar_logits)
    prediction = max(range(len(logits)), key=logits.__getitem__) if logits else None
    sidecar_prediction = sidecar.get("classifier_head_application", {}).get("prediction")
    passed = (
        completed.returncode == 0
        and conversion.get("classifier_head_gguf") is True
        and conversion.get("classifier_head_sidecar") is None
        and len(logits) == int(conversion.get("classifier_label_count", 0) or 0)
        and all(math.isfinite(value) for value in logits)
        and relative is not None
        and relative < 1e-5
        and prediction == sidecar_prediction
    )
    result = {
        "schema": "seqcls-native-i2sr-smoke-v1",
        "date": DATE,
        "status": "pass" if passed else "fail",
        "single_artifact": conversion.get("classifier_head_gguf") is True and conversion.get("classifier_head_sidecar") is None,
        "gguf": maybe_relative(gguf, root),
        "conversion_json": maybe_relative(conversion_json, root),
        "sidecar_smoke_json": maybe_relative(sidecar_json, root),
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "runtime": parse_perf(completed.stderr),
        "logits": logits,
        "sidecar_logits": sidecar_logits,
        "logit_count": len(logits),
        "prediction": prediction,
        "sidecar_prediction": sidecar_prediction,
        "max_abs_logit_delta": max_abs_delta,
        "relative_rms_logit_delta": relative,
        "full_validation_complete": False,
        "ready_to_productize": False,
        "stderr_tail": completed.stderr[-4000:],
        "verdict": (
            "Native single-artifact classifier-head execution works for this smoke prompt, "
            "but the product gate remains blocked until full MNLI validation, batching parity, "
            "RSS, and throughput are measured from this same GGUF."
            if passed
            else "Native classifier-head smoke failed; inspect stderr_tail before promoting this path."
        ),
    }

    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
