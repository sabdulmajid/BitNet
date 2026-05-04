#!/usr/bin/env python3
"""Audit benchmark and checkpoint artifacts before citing public results.

The goal is to make the evidence trail mechanical: this script reads artifacts
from disk, checks that expected files and sample counts exist, and emits a
compact Markdown report. It intentionally does not train or evaluate models.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SELECTED_LM_EVAL_METRICS = {
    "arc_challenge": "acc_norm",
    "arc_easy": "acc_norm",
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
    "boolq": "acc",
    "copa": "acc",
    "openbookqa": "acc_norm",
    "sciq": "acc_norm",
    "truthfulqa_mc1": "acc",
}


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path
    expected_ternary: int | None
    expected_scales: int | None
    expected_scale_rank: int | None


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def parse_label_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"expected LABEL=path, got {spec!r}")
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"empty label in {spec!r}")
    return label, Path(path)


def parse_checkpoint(spec: str) -> CheckpointSpec:
    label, raw_path = parse_label_path(spec)
    parts = str(raw_path).split(":")
    path = Path(parts[0])
    expected_ternary = int(parts[1]) if len(parts) > 1 and parts[1] else None
    expected_scales = int(parts[2]) if len(parts) > 2 and parts[2] else expected_ternary
    expected_scale_rank = None
    if len(parts) > 3 and parts[3]:
        rank_aliases = {"scalar": 1, "row": 2, "none": None}
        expected_scale_rank = rank_aliases[parts[3]] if parts[3] in rank_aliases else int(parts[3])
    return CheckpointSpec(label, path, expected_ternary, expected_scales, expected_scale_rank)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(task_results: dict[str, Any], metric: str) -> float:
    for key in (metric, f"{metric},none"):
        value = task_results.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    raise KeyError(f"metric {metric!r} not found")


def audit_checkpoint(specs: list[CheckpointSpec]) -> str:
    if not specs:
        return "No checkpoint specs supplied."

    import torch

    rows: list[list[str]] = []
    for spec in specs:
        if not spec.path.exists():
            rows.append([spec.label, str(spec.path), "MISSING", "-", "-", "-", "-", "-", "-"])
            continue
        state = torch.load(spec.path, map_location="cpu", weights_only=True)
        ternary_keys = [key for key in state if key.endswith(".ternary_weight")]
        scale_keys = [key for key in state if key.endswith(".weight_scale")]
        first_scale_shape = ""
        first_ternary_values = ""
        config_tie = ""
        metadata_note = "-"
        if scale_keys:
            first_scale_shape = str(tuple(state[scale_keys[0]].shape))
        if ternary_keys:
            values = torch.unique(state[ternary_keys[0]].cpu()).tolist()
            first_ternary_values = ",".join(str(int(value)) for value in values)
        config_path = spec.path.parent / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config_tie_value = config.get("tie_word_embeddings", "")
            config_tie = str(config_tie_value)
            if "lm_head.ternary_weight" in state and config_tie_value is True:
                metadata_note = "lm_head ternary but config tie_word_embeddings=true"

        ok = True
        if spec.expected_ternary is not None:
            ok = ok and len(ternary_keys) == spec.expected_ternary
        if spec.expected_scales is not None:
            ok = ok and len(scale_keys) == spec.expected_scales
        if spec.expected_scale_rank is not None and scale_keys:
            ok = ok and len(tuple(state[scale_keys[0]].shape)) == spec.expected_scale_rank
        ok = ok and set(first_ternary_values.split(",")) <= {"-1", "0", "1"}
        status = "FAIL"
        if ok:
            status = "WARN" if metadata_note != "-" else "PASS"
        rows.append([
            spec.label,
            str(spec.path),
            status,
            str(len(ternary_keys)),
            str(len(scale_keys)),
            first_scale_shape,
            first_ternary_values,
            config_tie,
            metadata_note,
        ])
    return md_table(
        [
            "label",
            "path",
            "status",
            "ternary keys",
            "scale keys",
            "first scale",
            "first codes",
            "config tie",
            "metadata note",
        ],
        rows,
    )


def audit_lm_eval(specs: list[tuple[str, Path]]) -> str:
    if not specs:
        return "No lm-eval specs supplied."
    rows: list[list[str]] = []
    for label, path in specs:
        if not path.exists():
            rows.append([label, str(path), "MISSING", "-", "-", "-", "-"])
            continue
        data = load_json(path)
        results = data.get("results", {})
        samples = data.get("samples", {})
        values: list[float] = []
        missing: list[str] = []
        sample_count = 0
        for task, metric in SELECTED_LM_EVAL_METRICS.items():
            task_result = results.get(task)
            if not isinstance(task_result, dict):
                missing.append(task)
                continue
            values.append(metric_value(task_result, metric))
            task_samples = samples.get(task, [])
            if isinstance(task_samples, list):
                sample_count += len(task_samples)
        rows.append([
            label,
            str(path),
            "PASS" if not missing else "FAIL",
            str(len(results)) if isinstance(results, dict) else "0",
            str(sample_count),
            f"{sum(values) / len(values):.6f}" if values else "-",
            ",".join(missing) if missing else "-",
        ])
    return md_table(["label", "path", "status", "tasks", "samples", "selected mean", "missing"], rows)


def audit_perplexity(specs: list[tuple[str, Path]]) -> str:
    if not specs:
        return "No perplexity specs supplied."
    rows: list[list[str]] = []
    for label, path in specs:
        if not path.exists():
            rows.append([label, str(path), "MISSING", "-", "-", "-", "-"])
            continue
        data = load_json(path)
        rows.append([
            label,
            str(path),
            "PASS" if "perplexity" in data and "eval_tokens" in data else "FAIL",
            data.get("model_kind", ""),
            f"{float(data.get('perplexity', 0.0)):.3f}",
            f"{float(data.get('nll', 0.0)):.4f}",
            str(int(float(data.get("eval_tokens", 0)))),
        ])
    return md_table(["label", "path", "status", "kind", "ppl", "nll", "tokens"], rows)


def audit_mc(specs: list[tuple[str, Path]]) -> str:
    if not specs:
        return "No multiple-choice specs supplied."
    rows: list[list[str]] = []
    required = {"task", "accuracy", "accuracy_norm", "limit", "model_kind"}
    for label, path in specs:
        if not path.exists():
            rows.append([label, str(path), "MISSING", "-", "-", "-", "-", "-"])
            continue
        data = load_json(path)
        missing = sorted(required - set(data))
        rows.append([
            label,
            str(path),
            "PASS" if not missing else "FAIL",
            str(data.get("task", "")),
            str(data.get("model_kind", "")),
            f"{float(data.get('accuracy', 0.0)):.4f}",
            f"{float(data.get('accuracy_norm', 0.0)):.4f}",
            str(int(float(data.get("limit", 0)))),
        ])
    return md_table(["label", "path", "status", "task", "kind", "acc", "acc_norm", "n"], rows)


def audit_runtime(specs: list[tuple[str, Path]]) -> str:
    if not specs:
        return "No runtime specs supplied."
    rows: list[list[str]] = []
    required = {"model_kind", "device", "dtype", "torch_num_threads", "prompt_tokens", "max_new_tokens", "prefill", "generate"}
    gib = 1024**3
    for label, path in specs:
        if not path.exists():
            rows.append([label, str(path), "MISSING", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"])
            continue
        data = load_json(path)
        missing = sorted(required - set(data))
        prefill = data.get("prefill", {})
        generate = data.get("generate", {})
        rss = float(data.get("rss_after_move_bytes", 0.0)) / gib
        model_size = float(data.get("model_storage_bytes", 0.0)) / gib
        ternary_size = float(data.get("ternary_state_bytes", 0.0)) / gib
        rows.append([
            label,
            str(path),
            "PASS" if not missing else "FAIL",
            str(data.get("model_kind", "")),
            str(data.get("device", "")),
            str(data.get("dtype", "")),
            str(data.get("torch_num_threads", "")),
            str(data.get("prompt_tokens", "")),
            str(data.get("max_new_tokens", "")),
            f"{float(prefill.get('tokens_per_second_median', 0.0)):.2f}",
            f"{float(generate.get('new_tokens_per_second_median_including_prefill', 0.0)):.2f}",
            f"{float(generate.get('decode_tokens_per_second_estimate', 0.0)):.2f}",
            f"{rss:.3f}",
            f"{model_size:.3f}",
            f"{ternary_size:.3f}",
        ])
    return md_table(
        ["label", "path", "status", "kind", "device", "dtype", "threads", "prompt", "new", "prefill tok/s", "gen tok/s", "decode tok/s", "RSS GiB", "model GiB", "ternary GiB"],
        rows,
    )


def audit_math(specs: list[tuple[str, Path]]) -> str:
    if not specs:
        return "No math audit specs supplied."
    rows: list[list[str]] = []
    for label, path in specs:
        if not path.exists():
            rows.append([label, str(path), "MISSING", "-", "-", "-", "-"])
            continue
        data = load_json(path)
        aggregate = data.get("aggregate", {})
        mean_abs = aggregate.get("mean_abs_ternary_repo_formula", {})
        sign_max = aggregate.get("sign_max_tl_i2_generic_path", {})
        rows.append([
            label,
            str(path),
            "PASS" if mean_abs and sign_max else "FAIL",
            str(data.get("trials", "")),
            f"{float(data.get('theoretical_mean_abs_relative_fro_error', 0.0)):.6f}",
            f"{float(mean_abs.get('relative_output_fro_error', {}).get('mean', 0.0)):.6f}",
            f"{float(sign_max.get('relative_output_fro_error', {}).get('mean', 0.0)):.6f}",
        ])
    return md_table(
        ["label", "path", "status", "trials", "theory rel Fro", "repo rel output", "sign/max rel output"],
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[], help="LABEL=path[:expected_ternary[:expected_scales[:scalar|row]]]")
    parser.add_argument("--lm-eval", action="append", default=[], help="LABEL=path.json")
    parser.add_argument("--perplexity", action="append", default=[], help="LABEL=path.json")
    parser.add_argument("--mc", action="append", default=[], help="LABEL=path.json")
    parser.add_argument("--runtime", action="append", default=[], help="LABEL=path.json")
    parser.add_argument("--math", action="append", default=[], help="LABEL=path.json")
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    checkpoints = [parse_checkpoint(spec) for spec in args.checkpoint]
    lm_eval = [parse_label_path(spec) for spec in args.lm_eval]
    perplexity = [parse_label_path(spec) for spec in args.perplexity]
    mc_specs = [parse_label_path(spec) for spec in args.mc]
    runtime = [parse_label_path(spec) for spec in args.runtime]
    math_specs = [parse_label_path(spec) for spec in args.math]

    report = "\n\n".join([
        "# Evidence Audit",
        "## Ternary Checkpoints",
        audit_checkpoint(checkpoints),
        "## lm-eval Artifacts",
        audit_lm_eval(lm_eval),
        "## Perplexity Artifacts",
        audit_perplexity(perplexity),
        "## Multiple-Choice Artifacts",
        audit_mc(mc_specs),
        "## Runtime Artifacts",
        audit_runtime(runtime),
        "## PTQ Math Artifacts",
        audit_math(math_specs),
    ])

    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
