#!/usr/bin/env python3
"""Audit the gap between sequence-classification quality and packed CPU runtime."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
DEFAULT_MODEL_SLUG = "Qwen-Qwen2.5-0.5B"
TASKS = ("mnli", "qnli", "sst2")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def git_output(args: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=cwd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return "git_unavailable"
    if result.returncode != 0:
        return result.stderr.strip() or f"git exited {result.returncode}"
    return result.stdout.strip()


def checkpoint_records(root: Path, model_slug: str) -> list[dict[str, Any]]:
    base = root / model_slug
    records: list[dict[str, Any]] = []
    for config_path in sorted(base.glob("*/*/config.json")):
        task = config_path.parent.parent.name
        run = config_path.parent.name
        config = read_json(config_path)
        architectures = config.get("architectures") if isinstance(config.get("architectures"), list) else []
        architecture = str(architectures[0]) if architectures else ""
        records.append(
            {
                "task": task,
                "run": run,
                "checkpoint_dir": str(config_path.parent),
                "architecture": architecture,
                "causal_export_compatible": architecture.endswith("ForCausalLM"),
                "sequence_classification": architecture.endswith("ForSequenceClassification"),
                "ternary_state": (config_path.parent / "ternary_state_dict.pt").exists(),
                "metrics": (config_path.parent / "metrics.json").exists(),
                "predictions": (config_path.parent / "eval_predictions.jsonl").exists(),
            }
        )
    return records


def load_causal_export_summary(path: Path) -> dict[str, Any]:
    data = read_json(path)
    exports = data.get("exports", []) if isinstance(data.get("exports"), list) else []
    exported = [row for row in exports if isinstance(row, dict) and row.get("exists")]
    by_qtype = Counter(str(row.get("export_qtype")) for row in exported)
    return {
        "path": str(path),
        "exists": path.exists(),
        "exports": len(exports),
        "exported": len(exported),
        "by_qtype": dict(sorted(by_qtype.items())),
    }


def load_causal_quality_summary(path: Path) -> dict[str, Any]:
    data = read_json(path)
    verdicts = data.get("verdicts", []) if isinstance(data.get("verdicts"), list) else []
    fp_gaps = [
        row.get("fp_minus_bitdistill")
        for row in verdicts
        if isinstance(row, dict) and isinstance(row.get("fp_minus_bitdistill"), (int, float))
    ]
    return {
        "path": str(path),
        "exists": path.exists(),
        "passed": bool(data.get("passed")),
        "tasks": len(verdicts),
        "max_fp_minus_bitdistill": max(fp_gaps) if fp_gaps else None,
        "min_fp_minus_bitdistill": min(fp_gaps) if fp_gaps else None,
    }


def load_sidecar_smoke(path: Path) -> dict[str, Any]:
    data = read_json(path)
    files = data.get("files", {}) if isinstance(data.get("files"), dict) else {}
    gguf = files.get("gguf", {}) if isinstance(files.get("gguf"), dict) else {}
    head = files.get("classifier_head", {}) if isinstance(files.get("classifier_head"), dict) else {}
    runtime = data.get("runtime_smoke", {}) if isinstance(data.get("runtime_smoke"), dict) else {}
    logits = data.get("classifier_head_application", {}) if isinstance(data.get("classifier_head_application"), dict) else {}
    checkpoint = data.get("checkpoint", {}) if isinstance(data.get("checkpoint"), dict) else {}
    return {
        "path": str(path),
        "exists": path.exists(),
        "status": data.get("status"),
        "passed": data.get("status") == "prototype_smoke_passed",
        "checkpoint_accuracy": checkpoint.get("accuracy"),
        "checkpoint_eval_examples": checkpoint.get("eval_examples"),
        "gguf_size_mib": gguf.get("size_mib"),
        "classifier_head_size_bytes": head.get("size_bytes"),
        "runtime_returncode": runtime.get("returncode"),
        "embedding_shape": logits.get("embedding_shape"),
        "head_shape": logits.get("weight_shape"),
        "finite_logits": logits.get("finite_logits"),
    }


def load_sidecar_cpu_benchmark(path: Path) -> dict[str, Any]:
    data = read_json(path)
    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
    runtime = data.get("runtime", {}) if isinstance(data.get("runtime"), dict) else {}
    return {
        "path": str(path),
        "exists": path.exists(),
        "status": data.get("status"),
        "task": data.get("task"),
        "examples": summary.get("examples"),
        "accuracy": summary.get("accuracy"),
        "agreement_with_saved_pytorch_predictions": summary.get("agreement_with_saved_pytorch_predictions"),
        "examples_per_second": runtime.get("examples_per_second"),
    }


def load_hidden_contract(path: Path) -> dict[str, Any]:
    data = read_json(path)
    comparisons = data.get("comparisons", {}) if isinstance(data.get("comparisons"), dict) else {}
    return {
        "path": str(path),
        "exists": path.exists(),
        "status": data.get("status"),
        "token_id_match": comparisons.get("token_id_match"),
        "hidden_relative_rms": comparisons.get("hidden_relative_rms"),
        "hidden_cosine": comparisons.get("hidden_cosine"),
        "pytorch_hidden_l2": comparisons.get("pytorch_hidden_l2"),
        "llama_hidden_l2": comparisons.get("llama_hidden_l2"),
        "logit_relative_rms": comparisons.get("logit_relative_rms"),
        "pytorch_logits_equal_sidecar_logits": comparisons.get("pytorch_logits_equal_sidecar_logits"),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    arch_counts = Counter(str(row.get("architecture")) for row in records)
    tasks = sorted({str(row.get("task")) for row in records if row.get("task")})
    return {
        "configs": len(records),
        "tasks": tasks,
        "ternary_checkpoints": sum(1 for row in records if row.get("ternary_state")),
        "metrics": sum(1 for row in records if row.get("metrics")),
        "prediction_traces": sum(1 for row in records if row.get("predictions")),
        "causal_export_compatible": sum(1 for row in records if row.get("causal_export_compatible")),
        "sequence_classification": sum(1 for row in records if row.get("sequence_classification")),
        "architectures": dict(sorted(arch_counts.items())),
    }


def render_markdown(result: dict[str, Any]) -> str:
    smoke = result["seqcls_sidecar_smoke"]
    sidecar_cpu = result["seqcls_sidecar_cpu_benchmark"]
    hidden_contract = result["seqcls_hidden_contract"]
    headline_rows = [
        ["status", result["status"]],
        ["same artifact quality+CPU ready", result["same_artifact_quality_cpu_ready"]],
        ["sidecar prototype smoke", smoke["status"]],
        ["sidecar sampled CPU quality", sidecar_cpu["status"]],
        ["sidecar hidden contract", hidden_contract["status"]],
        ["seqcls configs", result["sequence_classification"]["configs"]],
        ["seqcls causal-export compatible", result["sequence_classification"]["causal_export_compatible"]],
        ["causal runtime configs", result["causal_runtime"]["configs"]],
        ["causal GGUF exports", result["causal_export_summary"]["exported"]],
        ["llama.cpp fork remote", result["llama_cpp"]["origin"]],
        ["llama.cpp worktree clean", result["llama_cpp"]["clean"]],
    ]
    seq_arch_rows = [[arch, count] for arch, count in result["sequence_classification"]["architectures"].items()]
    causal_arch_rows = [[arch, count] for arch, count in result["causal_runtime"]["architectures"].items()]
    return "\n\n".join(
        [
            f"# Sequence-Classification Runtime Gap Audit, {result['date']}",
            "This audit separates the best GLUE quality path from the packed CPU runtime path.",
            md_table(["field", "value"], headline_rows),
            "## Sequence-Classification Quality Path",
            md_table(["architecture", "count"], seq_arch_rows),
            (
                "These checkpoints are the strict GLUE reproduction artifacts. They use "
                "`Qwen2ForSequenceClassification`. The standard causal export path still does "
                "not implement a native sequence-classification head, but the sidecar smoke below "
                "shows that a packed decoder backbone plus dense score-head sidecar is now loadable."
            ),
            "## Sidecar Prototype",
            md_table(
                ["field", "value"],
                [
                    ["status", smoke["status"]],
                    ["checkpoint accuracy", smoke["checkpoint_accuracy"]],
                    ["checkpoint eval examples", smoke["checkpoint_eval_examples"]],
                    ["GGUF MiB", smoke["gguf_size_mib"]],
                    ["head bytes", smoke["classifier_head_size_bytes"]],
                    ["runtime return code", smoke["runtime_returncode"]],
                    ["embedding shape", smoke["embedding_shape"]],
                    ["head shape", smoke["head_shape"]],
                    ["finite logits", smoke["finite_logits"]],
                    ["sampled CPU status", sidecar_cpu["status"]],
                    ["sampled examples", sidecar_cpu["examples"]],
                    ["sampled accuracy", sidecar_cpu["accuracy"]],
                    ["agreement with saved PyTorch predictions", sidecar_cpu["agreement_with_saved_pytorch_predictions"]],
                    ["sampled examples/sec", sidecar_cpu["examples_per_second"]],
                    ["token IDs match", hidden_contract["token_id_match"]],
                    ["hidden relative RMS", hidden_contract["hidden_relative_rms"]],
                    ["hidden cosine", hidden_contract["hidden_cosine"]],
                    ["logit relative RMS", hidden_contract["logit_relative_rms"]],
                ],
            ),
            "## Causal Runtime Path",
            md_table(["architecture", "count"], causal_arch_rows),
            (
                "These checkpoints are export-compatible with the current GGUF/I2_SR path, "
                "but they are not the same artifacts as the sequence-classification quality branch."
            ),
            "## Required Runtime Work",
            md_table(
                ["item", "status"],
                [
                    ["Backbone GGUF + dense head sidecar smoke", "prototype implemented"],
                    ["Sampled sidecar CPU quality agreement", "failing"],
                    ["Tokenizer pair formatting parity", "passes for audited MNLI sample"],
                    ["PyTorch pooled hidden state matches llama.cpp embedding", "failing"],
                    ["GGUF writer persists classifier/score head tensors and label metadata", "not implemented"],
                    ["llama.cpp pools the last non-padding token for Qwen sequence classification", "not implemented"],
                    ["CPU evaluator reports GLUE accuracy from the packed classifier artifact", "not implemented"],
                    ["Quality, RSS, and throughput measured on the same deployed artifact", "blocked"],
                ],
            ),
            "## Interpretation",
            (
                "The current repository has a PyTorch quality proof path and a causal GGUF runtime proof path. "
                "It now also has a prototype sequence-classification backbone smoke through I2_SR plus an external "
                "dense head sidecar. The sampled sidecar CPU quality probe currently disagrees with saved PyTorch "
                "predictions. The hidden-contract audit narrows the issue: token IDs match for the first MNLI "
                "sample, but the llama.cpp embedding has high relative RMS error and near-zero cosine versus the "
                "PyTorch pooled hidden state. This is a runtime/model-state mismatch, not a deployable classifier."
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-slug", default=DEFAULT_MODEL_SLUG)
    parser.add_argument("--seqcls-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls"))
    parser.add_argument("--causal-root", type=Path, default=Path("checkpoints/bitdistill-glue-causal-longwarmup-densehead"))
    parser.add_argument(
        "--causal-export-summary",
        type=Path,
        default=Path("benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-15/export_summary.json"),
    )
    parser.add_argument(
        "--causal-quality-summary",
        type=Path,
        default=Path("benchmark_results/bitdistill_causal_longwarmup_densehead_summary_2026-05-15.json"),
    )
    parser.add_argument(
        "--seqcls-sidecar-smoke",
        type=Path,
        default=Path(f"benchmark_results/seqcls_backbone_i2sr_smoke_{DATE}.json"),
    )
    parser.add_argument(
        "--seqcls-sidecar-cpu-benchmark",
        type=Path,
        default=Path(f"benchmark_results/seqcls_i2sr_sidecar_cpu_mnli_64_{DATE}.json"),
    )
    parser.add_argument(
        "--seqcls-hidden-contract",
        type=Path,
        default=Path(f"benchmark_results/seqcls_i2sr_hidden_contract_{DATE}.json"),
    )
    parser.add_argument("--llama-cpp", type=Path, default=Path("3rdparty/llama.cpp"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/seqcls_runtime_gap_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/seqcls_runtime_gap_{DATE}.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    seqcls_root = args.seqcls_root if args.seqcls_root.is_absolute() else root / args.seqcls_root
    causal_root = args.causal_root if args.causal_root.is_absolute() else root / args.causal_root
    llama_cpp = args.llama_cpp if args.llama_cpp.is_absolute() else root / args.llama_cpp

    seqcls_records = checkpoint_records(seqcls_root, args.model_slug)
    causal_records = checkpoint_records(causal_root, args.model_slug)
    seqcls_summary = summarize_records(seqcls_records)
    causal_summary = summarize_records(causal_records)
    export_summary = load_causal_export_summary(root / args.causal_export_summary)
    causal_quality = load_causal_quality_summary(root / args.causal_quality_summary)
    sidecar_smoke = load_sidecar_smoke(root / args.seqcls_sidecar_smoke)
    sidecar_cpu = load_sidecar_cpu_benchmark(root / args.seqcls_sidecar_cpu_benchmark)
    hidden_contract = load_hidden_contract(root / args.seqcls_hidden_contract)
    same_artifact_ready = (
        seqcls_summary["causal_export_compatible"] > 0
        and export_summary["exported"] > 0
        and causal_quality["passed"]
    )
    if same_artifact_ready:
        status = "ready"
    elif sidecar_smoke["passed"]:
        status = "sidecar_prototype_available_native_runtime_blocked"
    else:
        status = "blocked_by_classifier_runtime"
    result = {
        "schema": "seqcls_runtime_gap.v1",
        "date": DATE,
        "status": status,
        "same_artifact_quality_cpu_ready": same_artifact_ready,
        "sequence_classification": seqcls_summary,
        "causal_runtime": causal_summary,
        "causal_export_summary": export_summary,
        "causal_quality_summary": causal_quality,
        "seqcls_sidecar_smoke": sidecar_smoke,
        "seqcls_sidecar_cpu_benchmark": sidecar_cpu,
        "seqcls_hidden_contract": hidden_contract,
        "exporter_rejects_non_causal": "architecture.endswith(\"ForCausalLM\")"
        in (root / "benchmarks/export_bitdistill_i2sr_suite.py").read_text(encoding="utf-8"),
        "llama_cpp": {
            "path": str(llama_cpp.relative_to(root)),
            "origin": git_output(["remote", "get-url", "origin"], llama_cpp),
            "clean": git_output(["status", "--short"], llama_cpp) == "",
        },
        "records": {
            "sequence_classification": seqcls_records,
            "causal_runtime": causal_records,
        },
    }

    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(result) + "\n", encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
