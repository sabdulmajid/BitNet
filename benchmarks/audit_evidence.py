#!/usr/bin/env python3
"""Audit benchmark and checkpoint artifacts before citing public results.

The goal is to make the evidence trail mechanical: this script reads artifacts
from disk, checks that expected files and sample counts exist, and emits a
compact Markdown report. It intentionally does not train or evaluate models.
"""

from __future__ import annotations

import argparse
import json
import math
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
    expected_config_tie: bool | None


@dataclass(frozen=True)
class GGUFSummarySpec:
    label: str
    path: Path
    expected_rows: int | None
    max_i2s_reference_ratio: float


@dataclass(frozen=True)
class ThreadScalingSpec:
    label: str
    path: Path
    expected_threads: tuple[int, ...]


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


def parse_label_path_count(spec: str) -> GGUFSummarySpec:
    label, raw_path = parse_label_path(spec)
    parts = str(raw_path).split(":")
    path = Path(parts[0])
    expected_count = int(parts[1]) if len(parts) > 1 and parts[1] else None
    max_i2s_reference_ratio = float(parts[2]) if len(parts) > 2 and parts[2] else 10.0
    return GGUFSummarySpec(label, path, expected_count, max_i2s_reference_ratio)


def parse_thread_scaling(spec: str) -> ThreadScalingSpec:
    label, raw_path = parse_label_path(spec)
    parts = str(raw_path).split(":")
    path = Path(parts[0])
    expected_threads = ()
    if len(parts) > 1 and parts[1]:
        expected_threads = tuple(int(value) for value in parts[1].split(",") if value)
    return ThreadScalingSpec(label, path, expected_threads)


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
    expected_config_tie = None
    if len(parts) > 4 and parts[4]:
        tie_aliases = {"tie_true": True, "tie_false": False, "true": True, "false": False, "tie_any": None, "none": None}
        if parts[4] not in tie_aliases:
            raise ValueError(f"unknown config tie expectation {parts[4]!r}")
        expected_config_tie = tie_aliases[parts[4]]
    return CheckpointSpec(label, path, expected_ternary, expected_scales, expected_scale_rank, expected_config_tie)


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
        config_tie_value: Any = None
        metadata_notes: list[str] = []
        metadata_warn = False
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
                metadata_warn = True
                metadata_notes.append("lm_head ternary but config tie_word_embeddings=true")

        ok = True
        if spec.expected_ternary is not None:
            ok = ok and len(ternary_keys) == spec.expected_ternary
        if spec.expected_scales is not None:
            ok = ok and len(scale_keys) == spec.expected_scales
        if spec.expected_scale_rank is not None and scale_keys:
            ok = ok and len(tuple(state[scale_keys[0]].shape)) == spec.expected_scale_rank
        if spec.expected_config_tie is not None:
            if not config_path.exists():
                ok = False
                metadata_notes.append(f"missing config.json; expected tie_word_embeddings={spec.expected_config_tie}")
            elif config_tie_value is not spec.expected_config_tie:
                ok = False
                metadata_notes.append(
                    f"expected tie_word_embeddings={spec.expected_config_tie}, got {config_tie_value}"
                )
        ok = ok and set(first_ternary_values.split(",")) <= {"-1", "0", "1"}
        metadata_note = "; ".join(metadata_notes) if metadata_notes else "-"
        status = "FAIL"
        if ok:
            status = "WARN" if metadata_warn else "PASS"
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
        row_mean_abs = aggregate.get("row_mean_abs_ternary_qat_formula", {})
        sign_max = aggregate.get("sign_max_tl_i2_generic_path", {})
        rows.append([
            label,
            str(path),
            "PASS" if mean_abs and sign_max else "FAIL",
            str(data.get("trials", "")),
            f"{float(data.get('theoretical_mean_abs_relative_fro_error', 0.0)):.6f}",
            f"{float(mean_abs.get('relative_output_fro_error', {}).get('mean', 0.0)):.6f}",
            (
                f"{float(row_mean_abs.get('relative_output_fro_error', {}).get('mean', 0.0)):.6f}"
                if row_mean_abs
                else "-"
            ),
            f"{float(sign_max.get('relative_output_fro_error', {}).get('mean', 0.0)):.6f}",
        ])
    return md_table(
        [
            "label",
            "path",
            "status",
            "trials",
            "theory rel Fro",
            "repo tensor rel output",
            "repo row rel output",
            "sign/max rel output",
        ],
        rows,
    )


def audit_gguf_summary(specs: list[GGUFSummarySpec]) -> str:
    if not specs:
        return "No GGUF summary specs supplied."
    rows_out: list[list[str]] = []
    for spec in specs:
        label = spec.label
        path = spec.path
        if not path.exists():
            rows_out.append([label, str(path), "MISSING", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"])
            continue
        data = load_json(path)
        rows = data.get("rows", [])
        if not isinstance(rows, list):
            rows_out.append([label, str(path), "FAIL", "0", "-", "rows is not a list", "-", "-", "-", "-", "-", "-", "-"])
            continue

        failed: list[str] = []
        nan_ppl: list[str] = []
        by_name: dict[str, dict[str, Any]] = {}
        cpu = ""
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", ""))
            by_name[name] = row
            if not row.get("exists", False):
                failed.append(f"{name}:missing")
            for key in ("bench_returncode", "ppl_returncode", "smoke_returncode"):
                if int(row.get(key, 1)) != 0:
                    failed.append(f"{name}:{key}")
            ppl = row.get("perplexity", {}).get("ppl")
            if not isinstance(ppl, (int, float)) or not math.isfinite(float(ppl)):
                nan_ppl.append(name)
            if not cpu:
                cpu = str(row.get("bench", {}).get("prefill", {}).get("cpu", ""))

        if spec.expected_rows is not None and len(rows) != spec.expected_rows:
            failed.append(f"expected_rows={spec.expected_rows},got={len(rows)}")

        def select_name(*names: str, kind_tokens: tuple[str, ...] = ()) -> str:
            for name in names:
                if name in by_name:
                    return name
            if kind_tokens:
                for candidate in rows:
                    if not isinstance(candidate, dict):
                        continue
                    kind = str(candidate.get("kind", "")).lower()
                    if all(token in kind for token in kind_tokens):
                        return str(candidate.get("name", ""))
            return names[0] if names else ""

        qat_i2s_name = select_name(
            "qwen15b_static_ternary_i2_s",
            "qwen15b_klonly_static_ternary_i2_s",
            kind_tokens=("qat", "i2s"),
        )
        if qat_i2s_name not in by_name:
            for candidate in rows:
                if not isinstance(candidate, dict):
                    continue
                name = str(candidate.get("name", ""))
                kind = str(candidate.get("kind", "")).lower()
                normalized_name = name.lower()
                if (
                    ("static_ternary" in kind and "i2s" in kind)
                    or ("static_ternary" in normalized_name and "i2_s" in normalized_name)
                ):
                    qat_i2s_name = name
                    break

        def ppl_for(name: str) -> str:
            value = by_name.get(name, {}).get("perplexity", {}).get("ppl")
            return f"{float(value):.4g}" if isinstance(value, (int, float)) and math.isfinite(float(value)) else "-"

        def numeric_ppl(name: str) -> float | None:
            value = by_name.get(name, {}).get("perplexity", {}).get("ppl")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
            return None

        def decode_for(name: str) -> str:
            value = by_name.get(name, {}).get("bench", {}).get("decode", {}).get("tok_s")
            return f"{float(value):.2f}" if isinstance(value, (int, float)) else "-"

        qat_i2s_ppl = numeric_ppl(qat_i2s_name)
        reference_ppl: float | None = None
        i2s_reference_ratio: float | None = None
        reference_ppls: list[float] = []
        for candidate in rows:
            if not isinstance(candidate, dict):
                continue
            name = str(candidate.get("name", ""))
            kind = str(candidate.get("kind", "")).lower()
            normalized_name = name.lower()
            if "static_ternary" not in kind and "static_ternary" not in normalized_name:
                continue
            if "i2s" in kind or "i2_s" in normalized_name:
                continue
            value = numeric_ppl(name)
            if value is not None:
                reference_ppls.append(value)
        if qat_i2s_ppl is not None and reference_ppls:
            reference_ppl = min(reference_ppls)
            i2s_reference_ratio = qat_i2s_ppl / reference_ppl
            if i2s_reference_ratio > spec.max_i2s_reference_ratio:
                failed.append(f"{qat_i2s_name}:ppl_ratio={i2s_reference_ratio:.3g}x")

        def fmt_number(value: float | None, digits: int = 4) -> str:
            return f"{value:.{digits}g}" if value is not None and math.isfinite(value) else "-"

        rows_out.append([
            label,
            str(path),
            "PASS" if not failed and not nan_ppl else "FAIL",
            str(len(rows)),
            cpu,
            ",".join(failed) if failed else "-",
            ",".join(nan_ppl) if nan_ppl else "-",
            ppl_for("qwen15b_fp_f16"),
            ppl_for("qwen15b_fp_i2_s"),
            ppl_for(qat_i2s_name),
            fmt_number(reference_ppl),
            fmt_number(i2s_reference_ratio, 6),
            f"{spec.max_i2s_reference_ratio:.6g}",
            decode_for(qat_i2s_name),
        ])
    return md_table(
        [
            "label",
            "path",
            "status",
            "rows",
            "cpu",
            "failed",
            "nan ppl",
            "FP F16 PPL",
            "blind I2_S PPL",
            "QAT I2_S PPL",
            "static ternary ref PPL",
            "I2_S/ref ratio",
            "max ratio",
            "QAT I2_S decode tok/s",
        ],
        rows_out,
    )


def audit_thread_scaling(specs: list[ThreadScalingSpec]) -> str:
    if not specs:
        return "No thread-scaling specs supplied."
    rows_out: list[list[str]] = []
    for spec in specs:
        if not spec.path.exists():
            rows_out.append([spec.label, str(spec.path), "MISSING", "-", "-", "-", "-", "-", "-", "-"])
            continue
        data = load_json(spec.path)
        rows = data.get("rows", [])
        if not isinstance(rows, list):
            rows_out.append([spec.label, str(spec.path), "FAIL", "0", "-", "rows is not a list", "-", "-", "-", "-"])
            continue

        failures: list[str] = []
        by_thread: dict[int, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                failures.append("non-object-row")
                continue
            thread = int(row.get("threads", -1))
            by_thread[thread] = row
            if int(row.get("returncode", 1)) != 0:
                failures.append(f"t{thread}:returncode={row.get('returncode')}")
            for key in ("prefill_tok_s", "decode_tok_s"):
                value = row.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
                    failures.append(f"t{thread}:{key}")

        observed_threads = tuple(sorted(by_thread))
        if spec.expected_threads and observed_threads != spec.expected_threads:
            failures.append(f"threads={','.join(str(v) for v in observed_threads)}")

        prefill_values = {
            thread: float(row["prefill_tok_s"])
            for thread, row in by_thread.items()
            if isinstance(row.get("prefill_tok_s"), (int, float))
        }
        decode_values = {
            thread: float(row["decode_tok_s"])
            for thread, row in by_thread.items()
            if isinstance(row.get("decode_tok_s"), (int, float))
        }
        max_prefill_thread = max(prefill_values, key=lambda thread: prefill_values[thread]) if prefill_values else None
        max_decode_thread = max(decode_values, key=lambda thread: decode_values[thread]) if decode_values else None
        prefill_speedup = None
        if 1 in prefill_values and max_prefill_thread is not None:
            prefill_speedup = prefill_values[max_prefill_thread] / prefill_values[1]

        def fmt(value: float | None, digits: int = 2) -> str:
            return f"{value:.{digits}f}" if value is not None and math.isfinite(value) else "-"

        rows_out.append([
            spec.label,
            str(spec.path),
            "PASS" if not failures else "FAIL",
            str(len(rows)),
            ",".join(str(thread) for thread in observed_threads),
            fmt(prefill_values.get(1)),
            fmt(prefill_values.get(max_prefill_thread) if max_prefill_thread is not None else None),
            str(max_prefill_thread) if max_prefill_thread is not None else "-",
            fmt(prefill_speedup),
            fmt(decode_values.get(max_decode_thread) if max_decode_thread is not None else None),
            str(max_decode_thread) if max_decode_thread is not None else "-",
            ",".join(failures) if failures else "-",
        ])
    return md_table(
        [
            "label",
            "path",
            "status",
            "rows",
            "threads",
            "prefill t1",
            "max prefill",
            "max prefill thread",
            "prefill speedup",
            "max decode",
            "max decode thread",
            "failures",
        ],
        rows_out,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="LABEL=path[:expected_ternary[:expected_scales[:scalar|row[:tie_true|tie_false]]]]",
    )
    parser.add_argument("--lm-eval", action="append", default=[], help="LABEL=path.json")
    parser.add_argument("--perplexity", action="append", default=[], help="LABEL=path.json")
    parser.add_argument("--mc", action="append", default=[], help="LABEL=path.json")
    parser.add_argument("--runtime", action="append", default=[], help="LABEL=path.json")
    parser.add_argument(
        "--gguf-summary",
        action="append",
        default=[],
        help="LABEL=summary.json[:expected_rows[:max_i2s_reference_ratio]]",
    )
    parser.add_argument(
        "--thread-scaling",
        action="append",
        default=[],
        help="LABEL=summary.json[:expected_threads_csv]",
    )
    parser.add_argument("--math", action="append", default=[], help="LABEL=path.json")
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    checkpoints = [parse_checkpoint(spec) for spec in args.checkpoint]
    lm_eval = [parse_label_path(spec) for spec in args.lm_eval]
    perplexity = [parse_label_path(spec) for spec in args.perplexity]
    mc_specs = [parse_label_path(spec) for spec in args.mc]
    runtime = [parse_label_path(spec) for spec in args.runtime]
    gguf_summary = [parse_label_path_count(spec) for spec in args.gguf_summary]
    thread_scaling = [parse_thread_scaling(spec) for spec in args.thread_scaling]
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
        "## GGUF Summary Artifacts",
        audit_gguf_summary(gguf_summary),
        "## Thread-Scaling Artifacts",
        audit_thread_scaling(thread_scaling),
        "## PTQ Math Artifacts",
        audit_math(math_specs),
    ])

    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
