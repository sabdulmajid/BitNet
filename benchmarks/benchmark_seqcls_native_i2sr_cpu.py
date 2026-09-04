#!/usr/bin/env python3
"""Evaluate a native sequence-classification GGUF on CPU.

The evaluator accepts F16, conventional quantized, and I2_SR artifacts so
same-checkpoint format comparisons share one task, prompt, and runtime harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import resource
import shlex
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
TASK_SPECS = {
    "mnli": {
        "dataset": ("glue", "mnli"),
        "eval_split": "validation_matched",
        "text_keys": ("premise", "hypothesis"),
        "expected_examples": 9815,
    },
    "qnli": {
        "dataset": ("glue", "qnli"),
        "eval_split": "validation",
        "text_keys": ("question", "sentence"),
        "expected_examples": 5463,
    },
    "sst2": {
        "dataset": ("glue", "sst2"),
        "eval_split": "validation",
        "text_keys": ("sentence", None),
        "expected_examples": 872,
    },
}
DEFAULT_CHECKPOINT = Path(
    "checkpoints/bitdistill-glue-seqcls-longwarmup/"
    "Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8"
)
DEFAULT_GGUF = Path(
    "models/seqcls-native-i2sr/"
    "Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8_bitnet_qwen_i2_sr_cls.gguf"
)
PROMPT_EVAL_RE = re.compile(
    r"prompt eval time =\s+(?P<ms>[0-9.]+) ms /\s+(?P<tokens>[0-9]+) tokens.*?"
    r"(?P<tps>[0-9.]+) tokens per second"
)
LOAD_RE = re.compile(r"load time =\s+(?P<ms>[0-9.]+) ms")
TOTAL_RE = re.compile(r"total time =\s+(?P<ms>[0-9.]+) ms")


def run_clean(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_prediction_trace(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit and len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


def read_progress_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_batching_audit(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "status": "missing",
            "ready_for_batched_product_benchmark": False,
            "ready_for_sequence_isolated_product_benchmark": False,
        }
    data = read_json(path)
    return {
        "path": str(path),
        "exists": True,
        "status": data.get("status"),
        "ready_for_batched_product_benchmark": data.get("ready_for_batched_product_benchmark") is True,
        "ready_for_sequence_isolated_product_benchmark": data.get(
            "ready_for_sequence_isolated_product_benchmark"
        )
        is True,
        "summary": data.get("summary", {}) if isinstance(data.get("summary"), dict) else {},
    }


def maybe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": maybe_relative(resolved, root),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def git_identity(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        revision = git("rev-parse", "HEAD")
        status = git("status", "--short", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return {
            "path": display_path or str(path),
            "revision": None,
            "tracked_files_dirty": None,
        }
    return {
        "path": display_path or str(path.resolve()),
        "revision": revision,
        "tracked_files_dirty": bool(status),
    }


def normalize_ldd_output(value: str) -> str:
    """Remove randomized load addresses while preserving library identities."""
    return re.sub(r"\(0x[0-9a-fA-F]+\)", "(0xADDR)", value.strip())


def normalize_repo_paths(value: str, root: Path) -> str:
    """Replace the local checkout prefix in public provenance text."""
    return value.replace(str(root.resolve()), "$REPO_ROOT")


def runtime_build_contract(binary: Path, root: Path) -> dict[str, Any]:
    """Fingerprint the executable, shared libraries, flags, and kernel sources.

    Hashing only an ELF executable is insufficient for these shared-library
    builds: changing libggml.so can alter both correctness and throughput while
    leaving the launcher unchanged.
    """
    build_dir = binary.resolve().parent.parent
    cmake_cache = build_dir / "CMakeCache.txt"
    compile_commands = build_dir / "compile_commands.json"
    option_names = {
        "BITNET_ARM_TL1",
        "BITNET_X86_TL2",
        "BUILD_SHARED_LIBS",
        "CMAKE_BUILD_TYPE",
        "CMAKE_C_COMPILER",
        "CMAKE_CXX_COMPILER",
        "GGML_AVX",
        "GGML_AVX2",
        "GGML_AVX512",
        "GGML_AVX512_BF16",
        "GGML_AVX512_VBMI",
        "GGML_AVX512_VNNI",
        "GGML_FMA",
        "GGML_NATIVE",
        "GGML_OPENMP",
    }
    cmake_options: dict[str, str] = {}
    if cmake_cache.exists():
        for line in cmake_cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line:
                continue
            key_and_type, value = line.split("=", 1)
            key = key_and_type.split(":", 1)[0]
            if key in option_names:
                cmake_options[key] = value

    linked_libraries: list[dict[str, Any]] = []
    ldd_text = ""
    try:
        ldd_result = subprocess.run(
            ["ldd", str(binary.resolve())],
            check=True,
            capture_output=True,
            text=True,
        )
        ldd_text = normalize_ldd_output(ldd_result.stdout)
        seen: set[Path] = set()
        for line in ldd_text.splitlines():
            match = re.search(r"(?:=>\s+)?(/\S+)", line)
            if not match:
                continue
            library = Path(match.group(1)).resolve()
            if library in seen or not library.is_file():
                continue
            seen.add(library)
            record: dict[str, Any] = {
                "path": maybe_relative(library, root),
                "size_bytes": library.stat().st_size,
            }
            try:
                library.relative_to(build_dir)
            except ValueError:
                record["sha256"] = None
            else:
                record["sha256"] = sha256_file(library)
            linked_libraries.append(record)
    except (OSError, subprocess.CalledProcessError) as exc:
        ldd_text = f"unavailable: {exc}"

    source_candidates = [
        root / "src/ggml-bitnet-mad.cpp",
        root / "include/gemm-config.h",
        root / "3rdparty/llama.cpp/ggml/src/ggml.c",
        root / "3rdparty/llama.cpp/src/llama.cpp",
        root / "3rdparty/llama.cpp/examples/embedding/embedding.cpp",
    ]
    source_files = [file_identity(path, root) for path in source_candidates if path.is_file()]
    compile_units: list[dict[str, Any]] = []
    if compile_commands.exists():
        try:
            entries = json.loads(compile_commands.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            entries = []
        wanted = {str(path.resolve()) for path in source_candidates}
        for entry in entries:
            source = str(Path(entry.get("file", "")).resolve())
            if source not in wanted:
                continue
            command = entry.get("command")
            if not isinstance(command, str) and isinstance(entry.get("arguments"), list):
                command = shlex.join(str(value) for value in entry["arguments"])
            normalized_command = normalize_repo_paths(str(command), root)
            compile_units.append(
                {
                    "file": maybe_relative(Path(source), root),
                    "command": normalized_command,
                    "command_sha256": hashlib.sha256(normalized_command.encode("utf-8")).hexdigest(),
                }
            )

    repositories = {
        "bitnet": git_identity(root, display_path="."),
        "llama_cpp": git_identity(root / "3rdparty/llama.cpp", display_path="3rdparty/llama.cpp"),
    }
    contract: dict[str, Any] = {
        "build_dir": maybe_relative(build_dir, root),
        "cmake_options": cmake_options,
        "cmake_cache": file_identity(cmake_cache, root) if cmake_cache.exists() else None,
        "compile_commands": file_identity(compile_commands, root) if compile_commands.exists() else None,
        "compile_units": compile_units,
        "linked_libraries": linked_libraries,
        "ldd": normalize_repo_paths(ldd_text, root),
        "source_files": source_files,
        "repositories": repositories,
    }
    fingerprint = {
        "cmake_options": cmake_options,
        "cmake_cache_sha256": contract["cmake_cache"]["sha256"] if contract["cmake_cache"] else None,
        "compile_commands_sha256": (
            contract["compile_commands"]["sha256"] if contract["compile_commands"] else None
        ),
        "compile_units": [
            {"file": row["file"], "command_sha256": row["command_sha256"]}
            for row in compile_units
        ],
        "linked_libraries": linked_libraries,
        "source_files": source_files,
    }
    canonical = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    contract["fingerprint_components"] = fingerprint
    contract["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return contract


def cpu_environment(threads: int) -> dict[str, Any]:
    cpuinfo = Path("/proc/cpuinfo")
    processors = 0
    model_names: dict[str, int] = {}
    flags: set[str] = set()
    physical_cores: set[tuple[str, str]] = set()
    physical_id = ""
    core_id = ""
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines() + [""]:
            if not line.strip():
                if physical_id or core_id:
                    physical_cores.add((physical_id, core_id))
                physical_id = ""
                core_id = ""
                continue
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            if key == "processor":
                processors += 1
            elif key == "model name":
                model_names[value] = model_names.get(value, 0) + 1
            elif key == "flags":
                flags.update(value.split())
            elif key == "physical id":
                physical_id = value
            elif key == "core id":
                core_id = value
    model_name = max(model_names, key=model_names.get) if model_names else ""
    relevant_flags = ("avx512f", "avx512dq", "avx512bw", "avx512vl", "avx2", "fma", "bmi2")
    governor_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": model_name,
        "logical_cpus_os": os.cpu_count(),
        "logical_cpus_cpuinfo": processors or None,
        "physical_cores_cpuinfo": len(physical_cores) if physical_cores else None,
        "requested_threads": threads,
        "process_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "scaling_governor": governor_path.read_text(encoding="utf-8").strip()
        if governor_path.exists()
        else None,
        "isa_flags": {flag: flag in flags for flag in relevant_flags},
    }


def load_rows(task: str, limit: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    spec = TASK_SPECS[task]
    dataset_name, dataset_config = spec["dataset"]
    dataset = load_dataset(dataset_name, dataset_config)[spec["eval_split"]]
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [dict(row) for row in dataset]


def render_prompt(tokenizer: Any, task: str, row: dict[str, Any], *, prompt_input: str) -> str:
    spec = TASK_SPECS[task]
    text_a, text_b = spec["text_keys"]
    if text_b is None:
        encoded = tokenizer(row[text_a], truncation=True, max_length=512, add_special_tokens=True)
    else:
        encoded = tokenizer(row[text_a], row[text_b], truncation=True, max_length=512, add_special_tokens=True)
    if prompt_input == "token_ids":
        return "token_ids:" + json.dumps([int(item) for item in encoded["input_ids"]], separators=(",", ":"))
    if prompt_input != "text_roundtrip":
        raise ValueError(f"unsupported prompt_input={prompt_input!r}")
    return tokenizer.decode(encoded["input_ids"], clean_up_tokenization_spaces=False)


def parse_perf(stderr: str) -> dict[str, Any]:
    prompt_match = PROMPT_EVAL_RE.search(stderr)
    load_match = LOAD_RE.search(stderr)
    total_match = TOTAL_RE.search(stderr)
    return {
        "load_time_ms": float(load_match.group("ms")) if load_match else None,
        "prompt_eval_ms": float(prompt_match.group("ms")) if prompt_match else None,
        "prompt_eval_tokens": int(prompt_match.group("tokens")) if prompt_match else None,
        "prompt_eval_tokens_per_second": float(prompt_match.group("tps")) if prompt_match else None,
        "total_time_ms": float(total_match.group("ms")) if total_match else None,
    }


def run_native_classifier(
    *,
    binary: Path,
    gguf: Path,
    prompts: list[str],
    separator: str,
    threads: int,
    ctx_size: int,
    batch_size: int,
    ubatch_size: int,
    timeout_seconds: int,
    embedding_sequential: bool = False,
    cpu_affinity: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if any(separator in prompt for prompt in prompts):
        raise ValueError("separator occurs inside a prompt")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        prompt_file = Path(handle.name)
        handle.write(separator.join(prompts))
    command = [
        str(binary),
        "-m",
        str(gguf),
        "-f",
        str(prompt_file),
        "--embd-separator",
        separator,
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
        str(threads),
        "-c",
        str(ctx_size),
        "-b",
        str(batch_size),
        "-ub",
        str(ubatch_size),
    ]
    if embedding_sequential:
        command.append("--embd-sequential")
    if cpu_affinity:
        command = ["taskset", "-c", cpu_affinity, *command]
    before_rss_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    started = time.perf_counter()
    try:
        result = run_clean(command, timeout=timeout_seconds)
    finally:
        try:
            prompt_file.unlink()
        except FileNotFoundError:
            pass
    elapsed = time.perf_counter() - started
    after_rss_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:])
    parsed = json.loads(result.stdout)
    logits = [row["embedding"] for row in parsed.get("data", [])]
    if len(logits) != len(prompts):
        raise RuntimeError(f"expected {len(prompts)} logits rows, got {len(logits)}")
    meta = {
        "command": command,
        "elapsed_seconds": elapsed,
        "stdout_bytes": len(result.stdout.encode("utf-8")),
        "stderr_bytes": len(result.stderr.encode("utf-8")),
        "perf": parse_perf(result.stderr),
        "embedding_sequential": embedding_sequential,
        "cpu_affinity": cpu_affinity,
        "child_peak_rss_kib_before": before_rss_kib,
        "child_peak_rss_kib_after": after_rss_kib,
        "stderr_tail": result.stderr[-4000:],
    }
    return np.asarray(logits, dtype=np.float32), meta


def append_progress_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def accuracy_ci_wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    phat = correct / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    half = z * ((phat * (1.0 - phat) / total + z * z / (4.0 * total * total)) ** 0.5) / denom
    return [center - half, center + half]


def summarize_agreement(predictions: list[int], labels: list[int], trace: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(int(pred == label) for pred, label in zip(predictions, labels, strict=False))
    total = len(labels)
    trace_predictions = [
        int(row["prediction"])
        for row in trace[:total]
        if isinstance(row, dict) and isinstance(row.get("prediction"), int)
    ]
    trace_labels = [int(row["label"]) for row in trace[:total] if isinstance(row, dict) and isinstance(row.get("label"), int)]
    trace_scores = [
        [float(value) for value in row["scores"]]
        for row in trace[:total]
        if isinstance(row, dict) and isinstance(row.get("scores"), list)
    ]
    trace_agreement = None
    trace_label_match = None
    if len(trace_predictions) == total:
        trace_agreement = sum(int(a == b) for a, b in zip(predictions, trace_predictions, strict=True)) / total
    if len(trace_labels) == total:
        trace_label_match = sum(int(a == b) for a, b in zip(labels, trace_labels, strict=True)) / total
    trace_mismatches = [
        idx
        for idx, (pred, trace_pred) in enumerate(zip(predictions, trace_predictions, strict=False))
        if pred != trace_pred
    ]
    return {
        "examples": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "accuracy_ci95_wilson": accuracy_ci_wilson(correct, total),
        "saved_trace_predictions": len(trace_predictions),
        "saved_trace_scores": len(trace_scores),
        "agreement_with_saved_pytorch_predictions": trace_agreement,
        "disagreements_with_saved_pytorch_predictions": len(trace_mismatches)
        if len(trace_predictions) == total
        else None,
        "first_saved_prediction_mismatch_indices": trace_mismatches[:20]
        if len(trace_predictions) == total
        else [],
        "label_agreement_with_saved_trace": trace_label_match,
    }


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
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    runtime = result["runtime"]
    checkpoint = result["checkpoint"]
    return "\n\n".join(
        [
            f"# Sequence-Classification Native CPU Benchmark, {result['date']}",
            (
                "This benchmark evaluates one native GGUF artifact that contains the model "
                "backbone and dense classifier head. It is the same-artifact runtime path, but it "
                "is not product-ready unless full validation, runtime parity, RSS, and throughput "
                "gates pass."
            ),
            md_table(
                ["field", "value"],
                [
                    ["status", result["status"]],
                    ["task", result["task"]],
                    ["CPU", result["hardware"]["cpu_model"]],
                    ["threads", result["hardware"]["requested_threads"]],
                    ["CPU affinity", result["cpu_affinity"]],
                    ["GGUF MiB", result["artifacts"]["gguf_size_bytes"] / (1024 * 1024)],
                    ["GGUF SHA256", result["artifacts"]["gguf_sha256"]],
                    ["runtime build SHA256", result["runtime_build"]["sha256"]],
                    ["embedding binary SHA256", result["artifacts"]["embedding_binary_sha256"]],
                    ["BitNet revision", result["runtime_build"]["repositories"]["bitnet"]["revision"]],
                    ["llama.cpp revision", result["runtime_build"]["repositories"]["llama_cpp"]["revision"]],
                    ["examples", summary["examples"]],
                    ["expected examples", result["expected_examples"]],
                    ["full validation", result["full_validation_complete"]],
                    ["accuracy", summary["accuracy"]],
                    ["accuracy CI95", summary["accuracy_ci95_wilson"]],
                    ["stored PyTorch accuracy", checkpoint.get("stored_accuracy")],
                    ["agreement with saved PyTorch predictions", summary["agreement_with_saved_pytorch_predictions"]],
                    ["label agreement with saved trace", summary["label_agreement_with_saved_trace"]],
                    ["prompt input", result["prompt_input"]],
                    ["prompt batch size", result["prompt_batch_size"]],
                    ["embedding sequential", result["embedding_sequential"]],
                    ["batching parity ready", result["batching_parity_ready"]],
                    ["sequence-isolated parity ready", result["sequence_isolated_parity_ready"]],
                    ["runtime parity ready", result["runtime_parity_ready"]],
                    ["llama batch size", result["batch_size"]],
                    ["ubatch size", result["ubatch_size"]],
                    ["wall seconds", runtime["wall_seconds"]],
                    ["examples/sec", runtime["examples_per_second"]],
                    ["tokens/sec", runtime["prompt_eval_tokens_per_second"]],
                    ["child peak RSS MiB", runtime["child_peak_rss_mib"]],
                    ["ready to productize", result["ready_to_productize"]],
                ],
            ),
            "## Interpretation",
            result["verdict"],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--task", choices=sorted(TASK_SPECS), default="mnli")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--embedding-binary", type=Path, default=Path("build-portable-avx2/bin/llama-embedding"))
    parser.add_argument("--max-samples", type=int, default=512, help="0 means full validation split")
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=1,
        help=(
            "Number of prompts to pass to one llama-embedding process. Keep at 1 until "
            "multi-prompt classifier batching is proven stable."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument(
        "--embedding-sequential",
        action="store_true",
        help=(
            "Pass --embd-sequential to llama-embedding. This evaluates each prompt as its own "
            "sequence inside one loaded process, avoiding known multi-prompt I2_SR drift while "
            "still amortizing model load over --prompt-batch-size prompts."
        ),
    )
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument(
        "--cpu-affinity",
        default=None,
        help="Optional taskset CPU list, for example 0-11 for one thread per physical core.",
    )
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--separator", default="<#BITNET_NATIVE_EVAL_SEP#>")
    parser.add_argument(
        "--batching-audit-json",
        type=Path,
        default=Path(f"benchmark_results/seqcls_native_batching_audit_{DATE}.json"),
        help=(
            "Native batching parity audit. Product readiness remains false unless "
            "this audit exists and marks batched classifier inference ready."
        ),
    )
    parser.add_argument(
        "--prompt-input",
        choices=["token_ids", "text_roundtrip"],
        default="token_ids",
        help=(
            "Use direct HF token IDs by default. Decoding pair token IDs back to text is not "
            "lossless for Qwen BPE at some sentence-pair boundaries."
        ),
    )
    parser.add_argument(
        "--progress-jsonl",
        type=Path,
        default=None,
        help="Optional per-example progress trace. Enables partial evidence for long CPU validation runs.",
    )
    parser.add_argument(
        "--resume-progress",
        action="store_true",
        help="Resume from a contiguous --progress-jsonl trace instead of starting from example 0.",
    )
    parser.add_argument("--progress-every", type=int, default=64)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"benchmark_results/seqcls_native_i2sr_cpu_mnli_512_{DATE}.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"benchmarks/results/seqcls_native_i2sr_cpu_mnli_512_{DATE}.md"),
    )
    args = parser.parse_args()
    if args.prompt_batch_size <= 0:
        raise SystemExit("--prompt-batch-size must be positive")

    root = args.repo_root.resolve()
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else root / args.checkpoint_dir
    gguf = args.gguf if args.gguf.is_absolute() else root / args.gguf
    embedding_binary = args.embedding_binary if args.embedding_binary.is_absolute() else root / args.embedding_binary

    from transformers import AutoTokenizer

    metrics = read_json(checkpoint_dir / "metrics.json")
    eval_metrics = metrics.get("eval", {}) if isinstance(metrics.get("eval"), dict) else {}
    prediction_trace_path = eval_metrics.get("prediction_path") or str(checkpoint_dir / "eval_predictions.jsonl")
    prediction_trace = read_prediction_trace(root / prediction_trace_path, args.max_samples)
    batching_audit_path = args.batching_audit_json if args.batching_audit_json.is_absolute() else root / args.batching_audit_json
    batching_audit = load_batching_audit(batching_audit_path)
    batching_parity_ready = batching_audit.get("ready_for_batched_product_benchmark") is True
    sequence_isolated_ready = batching_audit.get("ready_for_sequence_isolated_product_benchmark") is True
    runtime_parity_ready = sequence_isolated_ready if args.embedding_sequential else batching_parity_ready
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    rows = load_rows(args.task, args.max_samples)
    prompts = [render_prompt(tokenizer, args.task, row, prompt_input=args.prompt_input) for row in rows]
    labels = [int(row["label"]) for row in rows]

    progress_path = None
    progress_rows: list[dict[str, Any]] = []
    if args.progress_jsonl is not None:
        progress_path = args.progress_jsonl if args.progress_jsonl.is_absolute() else root / args.progress_jsonl
        if args.resume_progress:
            raw_progress = read_progress_trace(progress_path)
            for expected_index, row in enumerate(raw_progress):
                if int(row.get("index", -1)) != expected_index:
                    break
                if not isinstance(row.get("logits"), list):
                    break
                progress_rows.append(row)
        else:
            try:
                progress_path.unlink()
            except FileNotFoundError:
                pass

    logits_rows: list[np.ndarray] = [
        np.asarray(row["logits"], dtype=np.float32)
        for row in progress_rows
    ]
    metas: list[dict[str, Any]] = []
    load_average_start = list(os.getloadavg())
    started = time.perf_counter()
    start_index = len(logits_rows)
    if start_index:
        print(f"resumed native seqcls progress at {start_index}/{len(prompts)} examples", flush=True)
    for start in range(start_index, len(prompts), args.prompt_batch_size):
        batch_prompts = prompts[start : start + args.prompt_batch_size]
        batch_logits, meta = run_native_classifier(
            binary=embedding_binary,
            gguf=gguf,
            prompts=batch_prompts,
            separator=args.separator,
            threads=args.threads,
            ctx_size=args.ctx_size,
            batch_size=args.batch_size,
            ubatch_size=args.ubatch_size,
            timeout_seconds=args.timeout_seconds,
            embedding_sequential=args.embedding_sequential,
            cpu_affinity=args.cpu_affinity,
        )
        metas.append(meta)
        progress_batch_rows: list[dict[str, Any]] = []
        for offset, row_logits in enumerate(batch_logits):
            index = start + offset
            logits_rows.append(row_logits.astype(np.float32))
            progress_batch_rows.append(
                {
                    "index": index,
                    "label": labels[index],
                    "prediction": int(np.argmax(row_logits)),
                    "saved_pytorch_prediction": prediction_trace[index].get("prediction")
                    if index < len(prediction_trace)
                    else None,
                    "logits": [float(value) for value in row_logits.tolist()],
                    "elapsed_seconds": time.perf_counter() - started,
                    "prompt_batch_size": args.prompt_batch_size,
                    "prompt_input": args.prompt_input,
                    "embedding_sequential": args.embedding_sequential,
                }
            )
        if progress_path is not None:
            append_progress_rows(progress_path, progress_batch_rows)
        completed = len(logits_rows)
        if args.progress_every > 0 and (completed == len(prompts) or completed % args.progress_every == 0):
            partial_predictions = [int(np.argmax(row)) for row in logits_rows]
            correct = sum(int(pred == label) for pred, label in zip(partial_predictions, labels, strict=False))
            trace_predictions = [
                int(row["prediction"])
                for row in prediction_trace[:completed]
                if isinstance(row, dict) and isinstance(row.get("prediction"), int)
            ]
            agreement = None
            if len(trace_predictions) == completed:
                agreement = sum(
                    int(pred == trace_pred)
                    for pred, trace_pred in zip(partial_predictions, trace_predictions, strict=True)
                ) / completed
            elapsed = time.perf_counter() - started
            print(
                "native seqcls progress "
                f"{completed}/{len(prompts)} acc_so_far={correct / completed:.6f} "
                f"agreement_so_far={agreement if agreement is not None else 'NA'} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    wall_seconds = time.perf_counter() - started
    load_average_end = list(os.getloadavg())
    logits = np.stack(logits_rows, axis=0) if logits_rows else np.zeros((0, 3), dtype=np.float32)
    predictions = [int(x) for x in np.argmax(logits, axis=-1)]
    summary = summarize_agreement(predictions, labels, prediction_trace)
    expected_examples = int(TASK_SPECS[args.task]["expected_examples"])
    full_validation_complete = args.max_samples == 0 and summary["examples"] == expected_examples
    prompt_eval_tokens = sum(int(meta["perf"].get("prompt_eval_tokens") or 0) for meta in metas)
    prompt_eval_ms = sum(float(meta["perf"].get("prompt_eval_ms") or 0.0) for meta in metas)
    load_time_ms = sum(float(meta["perf"].get("load_time_ms") or 0.0) for meta in metas)
    total_time_ms = sum(float(meta["perf"].get("total_time_ms") or 0.0) for meta in metas)
    child_peak_rss_kib = max((meta.get("child_peak_rss_kib_after") or 0) for meta in metas) if metas else None
    processed_this_run = len(rows) - start_index
    runtime = {
        "wall_seconds": wall_seconds,
        "examples_per_second": processed_this_run / wall_seconds if wall_seconds > 0 else None,
        "evaluated_examples_this_process": processed_this_run,
        "resumed_examples": start_index,
        "total_examples_in_result": len(rows),
        "subprocesses": len(metas),
        "load_time_ms": load_time_ms,
        "prompt_eval_ms": prompt_eval_ms,
        "prompt_eval_tokens": prompt_eval_tokens,
        "prompt_eval_tokens_per_second": prompt_eval_tokens / (prompt_eval_ms / 1000.0)
        if prompt_eval_ms > 0
        else None,
        "total_time_ms": total_time_ms,
        "child_peak_rss_kib": child_peak_rss_kib,
        "child_peak_rss_mib": child_peak_rss_kib / 1024.0
        if isinstance(child_peak_rss_kib, (int, float))
        else None,
        "host_load_average_start": load_average_start,
        "host_load_average_end": load_average_end,
    }
    trace_agreement = summary["agreement_with_saved_pytorch_predictions"]
    status = "pass"
    if summary["examples"] != len(rows) or not isinstance(summary["accuracy"], float) or not math.isfinite(summary["accuracy"]):
        status = "fail"
    elif trace_agreement is not None and trace_agreement < 0.95:
        status = "quality_mismatch"
    elif not full_validation_complete:
        status = "sample_only"
    ready_to_productize = (
        status == "pass"
        and full_validation_complete
        and runtime_parity_ready
        and trace_agreement is not None
        and trace_agreement >= 0.99
        and runtime["child_peak_rss_mib"] is not None
        and runtime["examples_per_second"] is not None
        and runtime["examples_per_second"] > 0
    )
    product_gate_blockers: list[str] = []
    if status != "pass":
        product_gate_blockers.append(f"status={status}")
    if not full_validation_complete:
        product_gate_blockers.append("full_validation_incomplete")
    if not runtime_parity_ready:
        product_gate_blockers.append("runtime_parity_not_ready")
    if trace_agreement is None:
        product_gate_blockers.append("missing_saved_pytorch_agreement")
    elif trace_agreement < 0.99:
        product_gate_blockers.append(f"saved_pytorch_agreement={trace_agreement:.6f}<0.99")
    if runtime["child_peak_rss_mib"] is None:
        product_gate_blockers.append("missing_rss")
    if runtime["examples_per_second"] is None or runtime["examples_per_second"] <= 0:
        product_gate_blockers.append("missing_positive_throughput")
    prediction_json = json.dumps(predictions, separators=(",", ":"))
    label_json = json.dumps(labels, separators=(",", ":"))
    build_contract = runtime_build_contract(embedding_binary, root)
    result = {
        "schema": "seqcls_native_cpu.v2",
        "date": DATE,
        "status": status,
        "task": args.task,
        "max_samples": args.max_samples,
        "expected_examples": expected_examples,
        "full_validation_complete": full_validation_complete,
        "ready_to_productize": ready_to_productize,
        "product_gate_blockers": product_gate_blockers,
        "batching_parity_ready": batching_parity_ready,
        "sequence_isolated_parity_ready": sequence_isolated_ready,
        "runtime_parity_ready": runtime_parity_ready,
        "batching_audit": batching_audit,
        "progress": {
            "path": maybe_relative(progress_path, root) if progress_path is not None else None,
            "resume_requested": args.resume_progress,
            "resumed_examples": start_index,
            "written_examples": len(logits_rows),
        },
        "prompt_input": args.prompt_input,
        "prompt_batch_size": args.prompt_batch_size,
        "embedding_sequential": args.embedding_sequential,
        "cpu_affinity": args.cpu_affinity,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
        "checkpoint": {
            "path": maybe_relative(checkpoint_dir, root),
            "stored_accuracy": eval_metrics.get("accuracy"),
            "stored_eval_examples": eval_metrics.get("eval_examples"),
            "prediction_trace": prediction_trace_path,
        },
        "artifacts": {
            "gguf": maybe_relative(gguf, root),
            "gguf_size_bytes": gguf.stat().st_size,
            "gguf_sha256": sha256_file(gguf),
            "embedding_binary": maybe_relative(embedding_binary, root),
            "embedding_binary_size_bytes": embedding_binary.stat().st_size,
            "embedding_binary_sha256": sha256_file(embedding_binary),
            "benchmark_script": maybe_relative(Path(__file__).resolve(), root),
            "benchmark_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "runtime_build": build_contract,
        "hardware": cpu_environment(args.threads),
        "summary": summary,
        "runtime": runtime,
        "predictions": predictions,
        "prediction_sha256": hashlib.sha256(prediction_json.encode("utf-8")).hexdigest(),
        "labels": labels,
        "label_sha256": hashlib.sha256(label_json.encode("utf-8")).hexdigest(),
        "sample_predictions": [
            {
                "index": idx,
                "label": labels[idx],
                "prediction": predictions[idx],
                "saved_pytorch_prediction": prediction_trace[idx].get("prediction")
                if idx < len(prediction_trace)
                else None,
                "logits": [float(value) for value in logits[idx].tolist()],
            }
            for idx in range(min(20, len(predictions)))
        ],
        "limitations": [
            "This uses llama-embedding JSON output as the classifier-logit transport.",
            "The faithful path uses direct token IDs because text decode/re-tokenize is not lossless for all Qwen pair prompts.",
            "The result is product-ready only after full validation, runtime parity, quality agreement, RSS, and throughput gates pass.",
            "The child RSS value is a process-level peak from resource.getrusage on Linux.",
        ],
        "verdict": (
            "Native same-artifact classifier validation passed the configured product gate."
            if ready_to_productize
            else "Native same-artifact classifier execution is measurable, but the product gate remains blocked by: "
            + ", ".join(product_gate_blockers)
            + ". Multi-prompt batched execution remains blocked by position-dependent drift in the I2_SR path; sequence-isolated mode is "
            "a separate mitigation."
        ),
        "stderr_tail": metas[-1]["stderr_tail"] if metas else "",
    }

    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": summary, "runtime": runtime}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
