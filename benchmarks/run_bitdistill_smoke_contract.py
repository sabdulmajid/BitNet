#!/usr/bin/env python3
"""Run fast CPU smoke checks for the BitDistill training entrypoint."""

from __future__ import annotations

import os
import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()


def run_command(command: list[str], *, cwd: Path, timeout: int = 180) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "returncode": proc.returncode,
        "output_tail": proc.stdout[-6000:],
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def qkv_split_is_finite(metrics: dict[str, Any]) -> bool:
    return all(
        finite(metrics.get(key))
        for key in (
            "attention_q_kd",
            "attention_k_kd",
            "attention_v_kd",
            "weighted_attention_q_kd",
            "weighted_attention_k_kd",
            "weighted_attention_v_kd",
        )
    )


def qkv_weighted_sum_matches(metrics: dict[str, Any], *, tolerance: float = 1e-4) -> bool:
    if not finite(metrics.get("weighted_attention_kd")):
        return False
    pieces = [
        metrics.get("weighted_attention_q_kd"),
        metrics.get("weighted_attention_k_kd"),
        metrics.get("weighted_attention_v_kd"),
    ]
    if not all(finite(value) for value in pieces):
        return False
    return abs(sum(float(value) for value in pieces) - float(metrics["weighted_attention_kd"])) <= tolerance


def activation_telemetry_is_finite(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    activation = rows[-1].get("activation_quantization")
    if not isinstance(activation, dict):
        return False
    return (
        int(activation.get("activation_quantized_modules") or 0) > 0
        and int(activation.get("total_values") or 0) > 0
        and finite(activation.get("clipped_fraction"))
        and finite(activation.get("int8_edge_fraction"))
        and finite(activation.get("scale_mean"))
        and finite(activation.get("absmax_mean"))
    )


def quantization_dynamics_is_finite(rows: list[dict[str, Any]]) -> bool:
    if len(rows) < 2:
        return False
    dynamics = rows[-1].get("quantization_dynamics")
    if not isinstance(dynamics, dict):
        return False
    return (
        dynamics.get("has_previous") is True
        and int(dynamics.get("tracked_modules") or 0) > 0
        and int(dynamics.get("compared_modules") or 0) > 0
        and int(dynamics.get("sampled_code_values") or 0) > 0
        and finite(dynamics.get("flip_fraction"))
        and finite(dynamics.get("scale_abs_delta_mean"))
        and finite(dynamics.get("scale_abs_delta_max"))
    )


def inspect_ternary_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        return {"exists": True, "valid": False, "error": "not a state dict"}
    code_keys = sorted(key for key in state if key.endswith(".ternary_weight"))
    scale_keys = sorted(key for key in state if key.endswith(".weight_scale"))
    invalid_code_keys: list[str] = []
    missing_scales: list[str] = []
    tensor_scale_keys = 0
    row_scale_keys = 0
    for key in code_keys:
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            invalid_code_keys.append(key)
            continue
        invalid_values = tensor[~torch.isin(tensor, torch.tensor([-1, 0, 1], dtype=tensor.dtype))]
        if invalid_values.numel() > 0:
            invalid_code_keys.append(key)
        prefix = key[: -len(".ternary_weight")]
        scale = state.get(f"{prefix}.weight_scale")
        if not isinstance(scale, torch.Tensor):
            missing_scales.append(prefix)
            continue
        if tuple(scale.shape) == (1,):
            tensor_scale_keys += 1
        elif scale.ndim == 2 and scale.shape[0] == tensor.shape[0] and scale.shape[1] == 1:
            row_scale_keys += 1
    return {
        "exists": True,
        "valid": not invalid_code_keys and not missing_scales and len(code_keys) == len(scale_keys),
        "code_keys": len(code_keys),
        "scale_keys": len(scale_keys),
        "tensor_scale_keys": tensor_scale_keys,
        "row_scale_keys": row_scale_keys,
        "invalid_code_keys": invalid_code_keys,
        "missing_scales": missing_scales,
    }


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, evidence: str, blocker: str = "") -> None:
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence, "blocker": "" if passed else blocker})


def build_report(result: dict[str, Any]) -> str:
    rows = [
        [
            check["name"],
            "pass" if check["passed"] else "fail",
            str(check["evidence"]).replace("\n", " "),
            str(check.get("blocker", "")).replace("\n", " "),
        ]
        for check in result["checks"]
    ]
    lines = [
        f"# BitDistill Smoke Contract, {result['date']}",
        f"Overall status: `{'pass' if result['passed'] else 'fail'}`.",
        f"Work dir: `{result['work_dir']}`.",
        "",
        "GGUF export checks use a smoke-only synthetic tokenizer stub. They validate packed tensor emission, row-scale `I2_SR` metadata, and SubLN key mapping; they do not validate text generation quality.",
        "",
        "| check | status | evidence | blocker |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--work-dir", type=Path, default=Path(f"benchmark_results/bitdistill-smoke-contract-{DATE}"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_smoke_contract_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_smoke_contract_{DATE}.md"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    work_dir = args.work_dir
    if work_dir.exists():
        if "bitdistill-smoke-contract" not in work_dir.name:
            raise SystemExit(f"refusing to remove unexpected work dir: {work_dir}")
        # After-any and after-ok postprocess jobs can race on this shared
        # smoke directory. Treat a concurrently removed path as already clean.
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    commands = {
        "help": [py, "train_bitdistill.py", "--help"],
        "py_compile": [py, "-m", "py_compile", "train_bitdistill.py"],
        "continued_pretrain": [
            py,
            "train_bitdistill.py",
            "--smoke-test",
            "--stage",
            "continued_pretrain",
            "--method",
            "bitdistill",
            "--max-steps",
            "2",
            "--device",
            "cpu",
            "--per-device-batch-size",
            "1",
            "--max-seq-len",
            "32",
            "--log-every-steps",
            "1",
            "--output-dir",
            str(work_dir / "continued_pretrain"),
        ],
        "continued_pretrain_row": [
            py,
            "train_bitdistill.py",
            "--smoke-test",
            "--stage",
            "continued_pretrain",
            "--method",
            "bitdistill",
            "--scale-mode",
            "row",
            "--max-steps",
            "2",
            "--device",
            "cpu",
            "--per-device-batch-size",
            "1",
            "--max-seq-len",
            "32",
            "--log-every-steps",
            "1",
            "--output-dir",
            str(work_dir / "continued_pretrain_row"),
        ],
        "task_sft": [
            py,
            "train_bitdistill.py",
            "--smoke-test",
            "--stage",
            "task_sft",
            "--method",
            "bitdistill",
            "--max-steps",
            "2",
            "--device",
            "cpu",
            "--per-device-batch-size",
            "1",
            "--eval-batch-size",
            "1",
            "--max-seq-len",
            "32",
            "--log-every-steps",
            "1",
            "--telemetry-every-steps",
            "1",
            "--telemetry-component-grad-norms",
            "--output-dir",
            str(work_dir / "task_sft"),
        ],
        "task_sft_row": [
            py,
            "train_bitdistill.py",
            "--smoke-test",
            "--stage",
            "task_sft",
            "--method",
            "bitdistill",
            "--scale-mode",
            "row",
            "--max-steps",
            "2",
            "--device",
            "cpu",
            "--per-device-batch-size",
            "1",
            "--eval-batch-size",
            "1",
            "--max-seq-len",
            "32",
            "--log-every-steps",
            "1",
            "--telemetry-every-steps",
            "1",
            "--telemetry-component-grad-norms",
            "--output-dir",
            str(work_dir / "task_sft_row"),
        ],
    }
    runs = {name: run_command(command, cwd=repo_root) for name, command in commands.items()}
    continued = read_json(work_dir / "continued_pretrain" / "metrics.json")
    continued_row = read_json(work_dir / "continued_pretrain_row" / "metrics.json")
    task = read_json(work_dir / "task_sft" / "metrics.json")
    row_task = read_json(work_dir / "task_sft_row" / "metrics.json")
    task_telemetry = read_jsonl(work_dir / "task_sft" / "telemetry.jsonl")
    row_task_telemetry = read_jsonl(work_dir / "task_sft_row" / "telemetry.jsonl")
    continued_ternary = inspect_ternary_state(work_dir / "continued_pretrain" / "ternary_state_dict.pt")
    continued_row_ternary = inspect_ternary_state(work_dir / "continued_pretrain_row" / "ternary_state_dict.pt")
    task_ternary = inspect_ternary_state(work_dir / "task_sft" / "ternary_state_dict.pt")
    row_task_ternary = inspect_ternary_state(work_dir / "task_sft_row" / "ternary_state_dict.pt")

    export_commands = {
        "continued_pretrain_i2s_export": [
            py,
            "benchmarks/convert_static_ternary_to_i2s_gguf.py",
            "--checkpoint-dir",
            str(work_dir / "continued_pretrain"),
            "--ternary-state",
            str(work_dir / "continued_pretrain" / "ternary_state_dict.pt"),
            "--outfile",
            str(work_dir / "continued_pretrain_i2s_smoke.gguf"),
            "--converter",
            "3rdparty/llama.cpp/convert_hf_to_gguf.py",
            "--gguf-arch",
            "bitnet-25",
            "--bitdistill-subln",
            "--synthetic-vocab-for-smoke",
            "--validate-codes",
            "--expect-ternary-keys",
            str(continued_ternary.get("code_keys", 0)),
            "--summary-json",
            str(work_dir / "continued_pretrain_i2s_smoke.json"),
        ],
        "continued_pretrain_row_i2sr_export": [
            py,
            "benchmarks/convert_static_ternary_to_i2s_gguf.py",
            "--checkpoint-dir",
            str(work_dir / "continued_pretrain_row"),
            "--ternary-state",
            str(work_dir / "continued_pretrain_row" / "ternary_state_dict.pt"),
            "--outfile",
            str(work_dir / "continued_pretrain_row_i2sr_smoke.gguf"),
            "--converter",
            "3rdparty/llama.cpp/convert_hf_to_gguf.py",
            "--gguf-arch",
            "bitnet-25",
            "--bitdistill-subln",
            "--synthetic-vocab-for-smoke",
            "--validate-codes",
            "--row-scale-qtype",
            "i2_sr",
            "--expect-ternary-keys",
            str(continued_row_ternary.get("code_keys", 0)),
            "--summary-json",
            str(work_dir / "continued_pretrain_row_i2sr_smoke.json"),
        ],
    }
    export_runs = {name: run_command(command, cwd=repo_root) for name, command in export_commands.items()}
    continued_i2s_export = read_json(work_dir / "continued_pretrain_i2s_smoke.json")
    continued_row_i2sr_export = read_json(work_dir / "continued_pretrain_row_i2sr_smoke.json")

    checks: list[dict[str, Any]] = []
    train_source = (repo_root / "train_bitdistill.py").read_text(encoding="utf-8", errors="replace")
    add_check(
        checks,
        "attention relation KD uses L2-normalized states",
        "F.normalize(states, dim=-1)" in train_source and "relation = torch.matmul(states" in train_source,
        "F.normalize before relation matmul",
        "attention relation KD is missing the MiniLM/BitDistill state normalization step",
    )
    add_check(
        checks,
        "attention relation KD sums Q/K/V losses by default",
        '--attention-qkv-reduction", choices=["sum", "mean"], default="sum"' in train_source
        and 'if qkv_reduction == "sum":' in train_source,
        "default attention_qkv_reduction=sum",
        "attention relation KD default no longer matches the BitDistill Q/K/V summation",
    )
    add_check(
        checks,
        "attention relation KD exposes Q/K/V components",
        all(
            snippet in train_source
            for snippet in (
                "attention_q_kd",
                "attention_k_kd",
                "attention_v_kd",
                "weighted_attention_q_kd",
                "weighted_attention_k_kd",
                "weighted_attention_v_kd",
            )
        ),
        "raw and weighted Q/K/V attention KD fields are present",
        "attention KD telemetry cannot isolate Q, K, and V relation terms",
    )
    add_check(
        checks,
        "BitLinear activation quantization telemetry is implemented",
        all(
            snippet in train_source
            for snippet in (
                "capture_bitlinear_activation_quantization",
                "clipped_fraction",
                "int8_edge_fraction",
                "activation_quantization",
            )
        ),
        "activation A8 clipping, edge occupancy, scale, and absmax fields are present",
        "activation quantization telemetry cannot diagnose A8 saturation",
    )
    add_check(
        checks,
        "BitLinear quantization dynamics telemetry is implemented",
        all(
            snippet in train_source
            for snippet in (
                "BitLinearDynamicsTracker",
                "quantization_dynamics",
                "flip_fraction",
                "scale_abs_delta_mean",
            )
        ),
        "sampled ternary flip-rate and scale-drift fields are present",
        "quantization telemetry cannot diagnose ternary code motion or scale drift",
    )
    for name, run in runs.items():
        add_check(checks, f"{name} command exits zero", run["returncode"] == 0, f"returncode={run['returncode']}", "command failed")

    continued_prep = continued.get("preparation", {}) if isinstance(continued.get("preparation"), dict) else {}
    continued_last = continued.get("last", {}) if isinstance(continued.get("last"), dict) else {}
    add_check(checks, "continued-pretrain writes metrics", bool(continued), str(work_dir / "continued_pretrain" / "metrics.json"), "missing metrics")
    add_check(checks, "continued-pretrain takes two steps", continued.get("steps") == 2, f"steps={continued.get('steps')}", "unexpected step count")
    add_check(
        checks,
        "continued-pretrain uses BitLinear and SubLN",
        int(continued_prep.get("bitlinear_replaced", 0)) > 0 and int(continued_prep.get("subln_inserted", 0)) > 0,
        f"bitlinear={continued_prep.get('bitlinear_replaced')}, subln={continued_prep.get('subln_inserted')}",
        "BitNet preparation did not run",
    )
    add_check(checks, "continued-pretrain CE is finite", finite(continued_last.get("ce")), f"ce={continued_last.get('ce')}", "non-finite CE")
    add_check(
        checks,
        "continued-pretrain ternary export is valid",
        continued_ternary.get("valid") is True and continued_ternary.get("code_keys") == continued_prep.get("bitlinear_replaced"),
        f"codes={continued_ternary.get('code_keys')}, scales={continued_ternary.get('scale_keys')}, bitlinear={continued_prep.get('bitlinear_replaced')}",
        "invalid ternary_state_dict export",
    )
    continued_row_prep = continued_row.get("preparation", {}) if isinstance(continued_row.get("preparation"), dict) else {}
    continued_row_last = continued_row.get("last", {}) if isinstance(continued_row.get("last"), dict) else {}
    add_check(checks, "row continued-pretrain writes metrics", bool(continued_row), str(work_dir / "continued_pretrain_row" / "metrics.json"), "missing metrics")
    add_check(checks, "row continued-pretrain takes two steps", continued_row.get("steps") == 2, f"steps={continued_row.get('steps')}", "unexpected step count")
    add_check(
        checks,
        "row continued-pretrain uses BitLinear and SubLN",
        int(continued_row_prep.get("bitlinear_replaced", 0)) > 0 and int(continued_row_prep.get("subln_inserted", 0)) > 0,
        f"bitlinear={continued_row_prep.get('bitlinear_replaced')}, subln={continued_row_prep.get('subln_inserted')}",
        "BitNet preparation did not run",
    )
    add_check(checks, "row continued-pretrain CE is finite", finite(continued_row_last.get("ce")), f"ce={continued_row_last.get('ce')}", "non-finite CE")
    add_check(
        checks,
        "row continued-pretrain ternary export is valid",
        continued_row_ternary.get("valid") is True
        and continued_row_ternary.get("code_keys") == continued_row_prep.get("bitlinear_replaced")
        and continued_row_ternary.get("row_scale_keys") == continued_row_ternary.get("code_keys"),
        f"codes={continued_row_ternary.get('code_keys')}, tensor_scales={continued_row_ternary.get('tensor_scale_keys')}, row_scales={continued_row_ternary.get('row_scale_keys')}",
        "invalid row-scale ternary_state_dict export",
    )

    for name, run in export_runs.items():
        add_check(checks, f"{name} command exits zero", run["returncode"] == 0, f"returncode={run['returncode']}", "GGUF export command failed")
    add_check(
        checks,
        "continued-pretrain tensor GGUF export maps SubLN",
        (work_dir / "continued_pretrain_i2s_smoke.gguf").exists()
        and continued_i2s_export.get("bitdistill_subln") is True
        and continued_i2s_export.get("ternary_i2s_packed", 0) > 0,
        f"packed={continued_i2s_export.get('ternary_i2s_packed')}, outfile={continued_i2s_export.get('outfile')}",
        "tensor-scale BitDistill GGUF export did not produce a packed file",
    )
    add_check(
        checks,
        "row continued-pretrain I2_SR GGUF export maps SubLN and row scales",
        (work_dir / "continued_pretrain_row_i2sr_smoke.gguf").exists()
        and continued_row_i2sr_export.get("bitdistill_subln") is True
        and continued_row_i2sr_export.get("row_scale_qtype") == "i2_sr"
        and continued_row_i2sr_export.get("row_scale_i2s_packed", 0) > 0,
        f"row_packed={continued_row_i2sr_export.get('row_scale_i2s_packed')}, outfile={continued_row_i2sr_export.get('outfile')}",
        "row-scale BitDistill I2_SR GGUF export did not produce a packed file",
    )

    task_prep = task.get("preparation", {}) if isinstance(task.get("preparation"), dict) else {}
    task_last = task.get("last", {}) if isinstance(task.get("last"), dict) else {}
    task_eval = task.get("eval", {}) if isinstance(task.get("eval"), dict) else {}
    task_predictions = read_jsonl(work_dir / "task_sft" / "eval_predictions.jsonl")
    add_check(checks, "task-sft writes metrics", bool(task), str(work_dir / "task_sft" / "metrics.json"), "missing metrics")
    add_check(checks, "task-sft takes two steps", task.get("steps") == 2, f"steps={task.get('steps')}", "unexpected step count")
    add_check(
        checks,
        "task-sft uses BitLinear and SubLN",
        int(task_prep.get("bitlinear_replaced", 0)) > 0 and int(task_prep.get("subln_inserted", 0)) > 0,
        f"bitlinear={task_prep.get('bitlinear_replaced')}, subln={task_prep.get('subln_inserted')}",
        "BitNet preparation did not run",
    )
    add_check(checks, "task-sft logits KD is finite", finite(task_last.get("weighted_logit_kd")), f"weighted_logit_kd={task_last.get('weighted_logit_kd')}", "non-finite logits KD")
    add_check(checks, "task-sft attention KD is finite", finite(task_last.get("weighted_attention_kd")), f"weighted_attention_kd={task_last.get('weighted_attention_kd')}", "non-finite attention KD")
    add_check(
        checks,
        "task-sft Q/K/V attention KD split is finite",
        qkv_split_is_finite(task_last),
        (
            f"q={task_last.get('attention_q_kd')}, k={task_last.get('attention_k_kd')}, "
            f"v={task_last.get('attention_v_kd')}"
        ),
        "missing or non-finite Q/K/V attention KD split",
    )
    add_check(
        checks,
        "task-sft weighted Q/K/V split sums to aggregate",
        qkv_weighted_sum_matches(task_last),
        (
            f"sum={sum(float(task_last.get(key, 0.0)) for key in ('weighted_attention_q_kd', 'weighted_attention_k_kd', 'weighted_attention_v_kd'))}, "
            f"aggregate={task_last.get('weighted_attention_kd')}"
        ),
        "weighted Q/K/V attention KD terms do not match aggregate attention KD",
    )
    add_check(
        checks,
        "task-sft records paper-style Q/K/V reduction",
        task.get("loss_weights", {}).get("attention_qkv_reduction") == "sum",
        f"attention_qkv_reduction={task.get('loss_weights', {}).get('attention_qkv_reduction')}",
        "task metrics do not record paper-style Q/K/V reduction",
    )
    add_check(checks, "task-sft eval accuracy is finite", finite(task_eval.get("accuracy")), f"accuracy={task_eval.get('accuracy')}", "non-finite accuracy")
    add_check(
        checks,
        "task-sft writes per-example predictions",
        len(task_predictions) == int(task_eval.get("eval_examples", -1)),
        f"predictions={len(task_predictions)}, eval_examples={task_eval.get('eval_examples')}",
        "missing or incomplete eval_predictions.jsonl",
    )
    add_check(
        checks,
        "task-sft tensor-scale ternary export is valid",
        task_ternary.get("valid") is True
        and task_ternary.get("code_keys") == task_prep.get("bitlinear_replaced")
        and task_ternary.get("tensor_scale_keys") == task_ternary.get("code_keys"),
        f"codes={task_ternary.get('code_keys')}, tensor_scales={task_ternary.get('tensor_scale_keys')}, row_scales={task_ternary.get('row_scale_keys')}",
        "invalid tensor-scale ternary export",
    )
    add_check(
        checks,
        "task-sft activation telemetry is finite",
        activation_telemetry_is_finite(task_telemetry),
        f"telemetry_rows={len(task_telemetry)}, last={task_telemetry[-1].get('activation_quantization') if task_telemetry else {}}",
        "missing or non-finite task-sft activation quantization telemetry",
    )
    add_check(
        checks,
        "task-sft quantization dynamics telemetry is finite",
        quantization_dynamics_is_finite(task_telemetry),
        f"telemetry_rows={len(task_telemetry)}, last={task_telemetry[-1].get('quantization_dynamics') if task_telemetry else {}}",
        "missing or non-finite task-sft quantization dynamics telemetry",
    )

    row_task_prep = row_task.get("preparation", {}) if isinstance(row_task.get("preparation"), dict) else {}
    row_task_last = row_task.get("last", {}) if isinstance(row_task.get("last"), dict) else {}
    row_task_eval = row_task.get("eval", {}) if isinstance(row_task.get("eval"), dict) else {}
    row_task_predictions = read_jsonl(work_dir / "task_sft_row" / "eval_predictions.jsonl")
    add_check(checks, "row task-sft writes metrics", bool(row_task), str(work_dir / "task_sft_row" / "metrics.json"), "missing metrics")
    add_check(checks, "row task-sft takes two steps", row_task.get("steps") == 2, f"steps={row_task.get('steps')}", "unexpected step count")
    add_check(
        checks,
        "row task-sft uses BitLinear and SubLN",
        int(row_task_prep.get("bitlinear_replaced", 0)) > 0 and int(row_task_prep.get("subln_inserted", 0)) > 0,
        f"bitlinear={row_task_prep.get('bitlinear_replaced')}, subln={row_task_prep.get('subln_inserted')}",
        "BitNet preparation did not run",
    )
    add_check(checks, "row task-sft logits KD is finite", finite(row_task_last.get("weighted_logit_kd")), f"weighted_logit_kd={row_task_last.get('weighted_logit_kd')}", "non-finite logits KD")
    add_check(checks, "row task-sft attention KD is finite", finite(row_task_last.get("weighted_attention_kd")), f"weighted_attention_kd={row_task_last.get('weighted_attention_kd')}", "non-finite attention KD")
    add_check(
        checks,
        "row task-sft Q/K/V attention KD split is finite",
        qkv_split_is_finite(row_task_last),
        (
            f"q={row_task_last.get('attention_q_kd')}, k={row_task_last.get('attention_k_kd')}, "
            f"v={row_task_last.get('attention_v_kd')}"
        ),
        "missing or non-finite row Q/K/V attention KD split",
    )
    add_check(
        checks,
        "row task-sft weighted Q/K/V split sums to aggregate",
        qkv_weighted_sum_matches(row_task_last),
        (
            f"sum={sum(float(row_task_last.get(key, 0.0)) for key in ('weighted_attention_q_kd', 'weighted_attention_k_kd', 'weighted_attention_v_kd'))}, "
            f"aggregate={row_task_last.get('weighted_attention_kd')}"
        ),
        "weighted row Q/K/V attention KD terms do not match aggregate attention KD",
    )
    add_check(
        checks,
        "row task-sft records paper-style Q/K/V reduction",
        row_task.get("loss_weights", {}).get("attention_qkv_reduction") == "sum",
        f"attention_qkv_reduction={row_task.get('loss_weights', {}).get('attention_qkv_reduction')}",
        "row task metrics do not record paper-style Q/K/V reduction",
    )
    add_check(checks, "row task-sft eval accuracy is finite", finite(row_task_eval.get("accuracy")), f"accuracy={row_task_eval.get('accuracy')}", "non-finite accuracy")
    add_check(
        checks,
        "row task-sft writes per-example predictions",
        len(row_task_predictions) == int(row_task_eval.get("eval_examples", -1)),
        f"predictions={len(row_task_predictions)}, eval_examples={row_task_eval.get('eval_examples')}",
        "missing or incomplete eval_predictions.jsonl",
    )
    add_check(
        checks,
        "row task-sft row-scale ternary export is valid",
        row_task_ternary.get("valid") is True
        and row_task_ternary.get("code_keys") == row_task_prep.get("bitlinear_replaced")
        and row_task_ternary.get("row_scale_keys") == row_task_ternary.get("code_keys"),
        f"codes={row_task_ternary.get('code_keys')}, tensor_scales={row_task_ternary.get('tensor_scale_keys')}, row_scales={row_task_ternary.get('row_scale_keys')}",
        "invalid row-scale ternary export",
    )
    add_check(
        checks,
        "row task-sft activation telemetry is finite",
        activation_telemetry_is_finite(row_task_telemetry),
        f"telemetry_rows={len(row_task_telemetry)}, last={row_task_telemetry[-1].get('activation_quantization') if row_task_telemetry else {}}",
        "missing or non-finite row task-sft activation quantization telemetry",
    )
    add_check(
        checks,
        "row task-sft quantization dynamics telemetry is finite",
        quantization_dynamics_is_finite(row_task_telemetry),
        f"telemetry_rows={len(row_task_telemetry)}, last={row_task_telemetry[-1].get('quantization_dynamics') if row_task_telemetry else {}}",
        "missing or non-finite row task-sft quantization dynamics telemetry",
    )

    result = {
        "schema": "bitdistill-smoke-contract-v1",
        "date": DATE,
        "work_dir": str(work_dir),
        "passed": all(check["passed"] for check in checks),
        "check_count": len(checks),
        "failed": [check["name"] for check in checks if not check["passed"]],
        "checks": checks,
        "runs": runs,
        "export_runs": export_runs,
        "continued_pretrain_metrics": continued,
        "continued_pretrain_row_metrics": continued_row,
        "task_sft_metrics": task,
        "task_sft_row_metrics": row_task,
        "task_sft_telemetry": task_telemetry,
        "task_sft_row_telemetry": row_task_telemetry,
        "continued_pretrain_ternary": continued_ternary,
        "continued_pretrain_row_ternary": continued_row_ternary,
        "task_sft_ternary": task_ternary,
        "task_sft_row_ternary": row_task_ternary,
        "continued_pretrain_i2s_export": continued_i2s_export,
        "continued_pretrain_row_i2sr_export": continued_row_i2sr_export,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_report(result), encoding="utf-8")
    print(build_report(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
