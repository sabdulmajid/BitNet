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


def load_arch_contract(path: Path) -> dict[str, Any]:
    data = read_json(path)
    checks = data.get("checks", {}) if isinstance(data.get("checks"), dict) else {}
    runtime = data.get("runtime_source", {}) if isinstance(data.get("runtime_source"), dict) else {}
    qwen = data.get("bitnet_qwen_contract", {}) if isinstance(data.get("bitnet_qwen_contract"), dict) else {}
    biases = data.get("checkpoint_biases", {}) if isinstance(data.get("checkpoint_biases"), dict) else {}
    config = data.get("checkpoint_config", {}) if isinstance(data.get("checkpoint_config"), dict) else {}
    return {
        "path": str(path),
        "exists": path.exists(),
        "status": data.get("status"),
        "hidden_act": config.get("hidden_act"),
        "bitnet25_ffn_activation": runtime.get("bitnet25_ffn_activation"),
        "bitnet_qwen_available": qwen.get("available"),
        "bitnet_qwen_ffn_activation": qwen.get("ffn_activation"),
        "bitnet_qwen_loader_has_qkv_bias": qwen.get("loader_has_qkv_bias"),
        "projection_bias_count": biases.get("projection_bias_count"),
        "activation_mismatch": checks.get("activation_mismatch"),
        "plain_bitnet_bias_contract_gap": checks.get("plain_bitnet_bias_contract_gap"),
    }


def load_native_smoke(path: Path) -> dict[str, Any]:
    data = read_json(path)
    runtime = data.get("runtime", {}) if isinstance(data.get("runtime"), dict) else {}
    return {
        "path": str(path),
        "exists": path.exists(),
        "status": data.get("status"),
        "passed": data.get("status") == "pass",
        "single_artifact": data.get("single_artifact"),
        "logit_count": data.get("logit_count"),
        "prediction": data.get("prediction"),
        "sidecar_prediction": data.get("sidecar_prediction"),
        "relative_rms_logit_delta": data.get("relative_rms_logit_delta"),
        "full_validation_complete": data.get("full_validation_complete"),
        "ready_to_productize": data.get("ready_to_productize"),
        "prompt_eval_tokens_per_second": runtime.get("prompt_eval_tokens_per_second"),
    }


def load_native_cpu_benchmark(path: Path) -> dict[str, Any]:
    data = read_json(path)
    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
    runtime = data.get("runtime", {}) if isinstance(data.get("runtime"), dict) else {}
    return {
        "path": str(path),
        "exists": path.exists(),
        "status": data.get("status"),
        "task": data.get("task"),
        "prompt_input": data.get("prompt_input"),
        "examples": summary.get("examples"),
        "accuracy": summary.get("accuracy"),
        "agreement_with_saved_pytorch_predictions": summary.get("agreement_with_saved_pytorch_predictions"),
        "examples_per_second": runtime.get("examples_per_second"),
        "child_peak_rss_mib": runtime.get("child_peak_rss_mib"),
        "full_validation_complete": data.get("full_validation_complete"),
        "ready_to_productize": data.get("ready_to_productize"),
        "batching_parity_ready": data.get("batching_parity_ready"),
    }


def load_native_batching_audit(path: Path) -> dict[str, Any]:
    data = read_json(path)
    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
    return {
        "path": str(path),
        "exists": path.exists(),
        "status": data.get("status"),
        "all_predictions_invariant": summary.get("all_predictions_invariant"),
        "changed_case_count": summary.get("changed_case_count"),
        "max_relative_rms_vs_alone": summary.get("max_relative_rms_vs_alone"),
        "ready_for_batched_product_benchmark": data.get("ready_for_batched_product_benchmark"),
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
    arch_contract = result["seqcls_arch_contract"]
    native_smoke = result["seqcls_native_smoke"]
    native_cpu = result["seqcls_native_cpu_benchmark"]
    native_batching = result["seqcls_native_batching_audit"]
    headline_rows = [
        ["status", result["status"]],
        ["same artifact quality+CPU ready", result["same_artifact_quality_cpu_ready"]],
        ["sidecar prototype smoke", smoke["status"]],
        ["native GGUF classifier smoke", native_smoke["status"]],
        ["native CPU quality", native_cpu["status"]],
        ["native CPU benchmark path", native_cpu["path"]],
        ["sidecar sampled CPU quality", sidecar_cpu["status"]],
        ["sidecar hidden contract", hidden_contract["status"]],
        ["sidecar architecture contract", arch_contract["status"]],
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
                "`Qwen2ForSequenceClassification`. The standard causal export path is still not a "
                "full classifier evaluator, but the native smoke below shows that a Qwen-compatible "
                "packed GGUF can now carry the dense score head and emit classifier logits."
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
                    ["checkpoint hidden_act", arch_contract["hidden_act"]],
                    ["bitnet-25 FFN activation", arch_contract["bitnet25_ffn_activation"]],
                    ["bitnet-qwen contract available", arch_contract["bitnet_qwen_available"]],
                    ["bitnet-qwen FFN activation", arch_contract["bitnet_qwen_ffn_activation"]],
                    ["bitnet-qwen has Q/K/V bias slots", arch_contract["bitnet_qwen_loader_has_qkv_bias"]],
                    ["Q/K/V projection bias tensors", arch_contract["projection_bias_count"]],
                ],
            ),
            "## Native GGUF Classifier Smoke",
            md_table(
                ["field", "value"],
                [
                    ["status", native_smoke["status"]],
                    ["single artifact", native_smoke["single_artifact"]],
                    ["logit count", native_smoke["logit_count"]],
                    ["prediction", native_smoke["prediction"]],
                    ["sidecar prediction", native_smoke["sidecar_prediction"]],
                    ["relative RMS logit delta", native_smoke["relative_rms_logit_delta"]],
                    ["prompt tok/s", native_smoke["prompt_eval_tokens_per_second"]],
                    ["full validation complete", native_smoke["full_validation_complete"]],
                    ["ready to productize", native_smoke["ready_to_productize"]],
                ],
            ),
            "## Native GGUF CPU Benchmark",
            md_table(
                ["field", "value"],
                [
                    ["path", native_cpu["path"]],
                    ["status", native_cpu["status"]],
                    ["task", native_cpu["task"]],
                    ["prompt input", native_cpu["prompt_input"]],
                    ["examples", native_cpu["examples"]],
                    ["accuracy", native_cpu["accuracy"]],
                    ["agreement with saved PyTorch predictions", native_cpu["agreement_with_saved_pytorch_predictions"]],
                    ["examples/sec", native_cpu["examples_per_second"]],
                    ["child peak RSS MiB", native_cpu["child_peak_rss_mib"]],
                    ["full validation complete", native_cpu["full_validation_complete"]],
                    ["batching parity ready", native_cpu["batching_parity_ready"]],
                    ["ready to productize", native_cpu["ready_to_productize"]],
                ],
            ),
            "## Native GGUF Batching Audit",
            md_table(
                ["field", "value"],
                [
                    ["status", native_batching["status"]],
                    ["all predictions invariant", native_batching["all_predictions_invariant"]],
                    ["changed cases", native_batching["changed_case_count"]],
                    ["max relative RMS vs alone", native_batching["max_relative_rms_vs_alone"]],
                    ["ready for batched product benchmark", native_batching["ready_for_batched_product_benchmark"]],
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
                    ["Sampled sidecar CPU quality agreement", "prototype improved; needs full validation"],
                    ["Tokenizer pair formatting parity", "requires direct token-ID input for Qwen pair prompts"],
                    ["PyTorch pooled hidden state matches llama.cpp embedding", "near pass on audited sample, not exact"],
                    ["Packed graph matches Qwen2 SiLU/SwiGLU semantics", "implemented via bitnet-qwen"],
                    ["Packed loader supports Qwen2 Q/K/V projection biases", "implemented via bitnet-qwen"],
                    ["GGUF writer persists classifier/score head tensors and label metadata", "single-prompt smoke implemented"],
                    ["llama.cpp pools and applies the Qwen sequence-classification head", "single-prompt smoke implemented"],
                    [
                        "CPU evaluator reports GLUE accuracy from the packed classifier artifact",
                        (
                            "full token-ID MNLI validation implemented"
                            if native_cpu.get("full_validation_complete") is True
                            else "64-row token-ID sample implemented"
                        ),
                    ],
                    ["Batched embedding/classifier parity", "blocked: audited rows change logits/predictions by batch position"],
                    [
                        "Quality, RSS, and throughput measured on the same deployed artifact",
                        (
                            "full single-prompt validation measured; product still blocked by batching parity"
                            if native_cpu.get("full_validation_complete") is True
                            else "single-prompt sample only; full validation blocked"
                        ),
                    ],
                ],
            ),
            "## Interpretation",
            interpretation(result),
        ]
    )


def interpretation(result: dict[str, Any]) -> str:
    native_cpu = result["seqcls_native_cpu_benchmark"]
    native_batching = result["seqcls_native_batching_audit"]
    if native_cpu.get("full_validation_complete") is True:
        return (
            "The current repository now has full-split native CPU validation for one packed "
            "`bitnet-qwen` sequence-classification artifact. The run uses direct token IDs and "
            f"reports MNLI accuracy `{fmt(native_cpu.get('accuracy'))}`, saved-PyTorch prediction "
            f"agreement `{fmt(native_cpu.get('agreement_with_saved_pytorch_predictions'))}`, "
            f"`{fmt(native_cpu.get('examples_per_second'))}` examples/sec, and "
            f"`{fmt(native_cpu.get('child_peak_rss_mib'))}` MiB child peak RSS. This is useful "
            "runtime-fidelity evidence for the checkpoint, not a product-ready classifier: batching "
            f"parity remains `{native_batching.get('status')}`, and the checkpoint accuracy is still "
            "well below the FP16 task model."
        )
    return (
        "The current repository has a PyTorch quality proof path and a causal GGUF runtime proof path. "
        "It now also has a prototype sequence-classification backbone path through `bitnet-qwen` I2_SR plus "
        "an external dense head sidecar, and a native single-artifact GGUF smoke that matches the sidecar "
        "logits for one prompt. A 64-row native CPU sample using direct token IDs is measurable and reaches "
        "high agreement with saved PyTorch predictions, but it remains sample-only and still has residual "
        "packed-runtime drift. A separate batching audit shows that logits can change with sequence position "
        "inside a multi-prompt embedding batch, so batched throughput must not be promoted. Full-split CPU "
        "quality, batching parity, RSS, and throughput have not been measured on a faithful native artifact."
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
        default=Path(f"benchmark_results/seqcls_i2sr_sidecar_cpu_mnli_128_{DATE}.json"),
    )
    parser.add_argument(
        "--seqcls-hidden-contract",
        type=Path,
        default=Path(f"benchmark_results/seqcls_i2sr_hidden_contract_{DATE}.json"),
    )
    parser.add_argument(
        "--seqcls-arch-contract",
        type=Path,
        default=Path(f"benchmark_results/seqcls_i2sr_arch_contract_{DATE}.json"),
    )
    parser.add_argument(
        "--seqcls-native-smoke",
        type=Path,
        default=Path(f"benchmark_results/seqcls_native_i2sr_smoke_{DATE}.json"),
    )
    parser.add_argument(
        "--seqcls-native-cpu-benchmark",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--seqcls-native-batching-audit",
        type=Path,
        default=Path(f"benchmark_results/seqcls_native_batching_audit_{DATE}.json"),
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
    arch_contract = load_arch_contract(root / args.seqcls_arch_contract)
    native_smoke = load_native_smoke(root / args.seqcls_native_smoke)
    if args.seqcls_native_cpu_benchmark is None:
        full_native_cpu = root / f"benchmark_results/seqcls_native_i2sr_cpu_mnli_full_token_ids_{DATE}.json"
        sample_native_cpu = root / f"benchmark_results/seqcls_native_i2sr_cpu_mnli_64_token_ids_{DATE}.json"
        native_cpu_path = full_native_cpu if full_native_cpu.exists() else sample_native_cpu
    else:
        native_cpu_path = args.seqcls_native_cpu_benchmark
        native_cpu_path = native_cpu_path if native_cpu_path.is_absolute() else root / native_cpu_path
    native_cpu = load_native_cpu_benchmark(native_cpu_path)
    native_batching = load_native_batching_audit(root / args.seqcls_native_batching_audit)
    same_artifact_ready = (
        seqcls_summary["causal_export_compatible"] > 0
        and export_summary["exported"] > 0
        and causal_quality["passed"]
    )
    if same_artifact_ready:
        status = "ready"
    elif (
        native_cpu.get("status") == "pass"
        and native_cpu.get("full_validation_complete") is True
        and native_cpu.get("ready_to_productize") is False
    ):
        status = "native_classifier_full_validation_batching_blocked"
    elif native_cpu.get("status") in {"sample_only", "quality_mismatch"}:
        status = "native_classifier_sample_available_full_validation_blocked"
    elif native_smoke["passed"]:
        status = "native_classifier_smoke_available_full_validation_blocked"
    elif (
        sidecar_smoke["passed"]
        and isinstance(sidecar_cpu.get("agreement_with_saved_pytorch_predictions"), (int, float))
        and sidecar_cpu["agreement_with_saved_pytorch_predictions"] >= 0.9
    ):
        status = "sidecar_qwen_contract_available_native_head_blocked"
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
        "seqcls_arch_contract": arch_contract,
        "seqcls_native_smoke": native_smoke,
        "seqcls_native_cpu_benchmark": native_cpu,
        "seqcls_native_batching_audit": native_batching,
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
