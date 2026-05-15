#!/usr/bin/env python3
"""Audit the queued BitDistill experiment matrix.

The dependency graph gate proves that active downstream jobs point at the
warm-up checkpoint. This audit checks the experiment design itself: which task
families, scale modes, attention weights, teachers, and output roots are queued.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
TASKS = ("mnli", "qnli", "sst2")
ACTIVE_SLURM_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stored_batch_script(job_id: str) -> tuple[str, str]:
    if not job_id:
        return "", "missing job id"
    with tempfile.TemporaryDirectory(prefix="bitdistill-matrix-script-") as tmp:
        path = Path(tmp) / f"job-{job_id}.sh"
        proc = subprocess.run(
            ["scontrol", "write", "batch_script", job_id, str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return "", proc.stderr.strip() or proc.stdout.strip() or f"scontrol exited {proc.returncode}"
        if not path.exists():
            return "", "scontrol did not materialize a batch script"
        return path.read_text(encoding="utf-8", errors="replace"), ""


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_exists(path_text: str) -> bool:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.exists()


def normalize_path_text(value: Any) -> str:
    path = Path(str(value))
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())


def inferred_task_format(row: dict[str, Any]) -> str:
    value = str(row.get("task_format") or "")
    if value:
        return value
    output_dir = str(row.get("output_dir") or "")
    if "causal" in output_dir:
        return "causal_lm"
    if "seqcls" in output_dir:
        return "sequence_classification"
    return ""


def inferred_label_scheme(row: dict[str, Any]) -> str:
    return str(row.get("label_scheme") or "letters")


def inferred_candidate_score(row: dict[str, Any]) -> str:
    return str(row.get("candidate_score") or "mean")


def inferred_exclude_regex(row: dict[str, Any]) -> str:
    value = str(row.get("exclude_linear_regex") or "")
    if value:
        return value
    output_dir = str(row.get("output_dir") or "")
    return "lm_head" if "causal" in output_dir else "score|classifier"


def inferred_field_names(row: dict[str, Any]) -> list[str]:
    fields = []
    for key in ("task_format", "label_scheme", "candidate_score", "exclude_linear_regex"):
        if not row.get(key):
            fields.append(key)
    return fields


def audit_stored_downstream_scripts(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required_snippets = {
        "logit temperature scale default": 'LOGIT_KD_TEMPERATURE_SCALE="${LOGIT_KD_TEMPERATURE_SCALE:-none}"',
        "logit temperature scale arg": '--logit-kd-temperature-scale "$LOGIT_KD_TEMPERATURE_SCALE"',
        "attention KD weight arg": '--attention-kd-weight "$ATTENTION_KD_WEIGHT"',
        "save every arg": '--save-every-steps "$SAVE_EVERY_STEPS"',
    }
    provenance: list[dict[str, Any]] = []
    failing: list[dict[str, Any]] = []
    for row in rows:
        job_id = str(row.get("job_id") or "")
        job_status = row.get("job_status", {}) if isinstance(row.get("job_status"), dict) else {}
        job_state = str(job_status.get("state") or "")
        script_required = job_state in ACTIVE_SLURM_STATES
        script, error = stored_batch_script(job_id)
        missing = [label for label, snippet in required_snippets.items() if script_required and script and snippet not in script]
        passed = (not script_required) or (bool(script) and not error and not missing)
        record = {
            "job_id": job_id,
            "task": row.get("task"),
            "scale": row.get("scale"),
            "task_format": inferred_task_format(row),
            "job_state": job_state,
            "script_required": script_required,
            "script_available": bool(script),
            "script_error": error,
            "sha256": sha256_text(script) if script else "",
            "missing_required_snippets": missing,
            "passed": passed,
        }
        provenance.append(record)
        if script_required and not record["passed"]:
            failing.append(record)
    return provenance, failing


def expected_rows(model_slug: str, warmup_state: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for scale in ("tensor", "row"):
            rows.append(
                {
                    "family": "seqcls_gamma100",
                    "task": task,
                    "task_format": "sequence_classification",
                    "scale": scale,
                    "attention_kd_weight": 100.0,
                    "output_root": f"checkpoints/bitdistill-glue-seqcls-longwarmup/{model_slug}/{task}",
                    "teacher_root": f"checkpoints/bitdistill-glue-seqcls/{model_slug}/{task}",
                    "exclude_linear_regex": "score|classifier",
                    "init_output_head_from_teacher": "0",
                    "warmup_state": warmup_state,
                }
            )
            rows.append(
                {
                    "family": "seqcls_gamma100_headinit",
                    "task": task,
                    "task_format": "sequence_classification",
                    "scale": scale,
                    "attention_kd_weight": 100.0,
                    "output_root": f"checkpoints/bitdistill-glue-seqcls-longwarmup-headinit/{model_slug}/{task}",
                    "teacher_root": f"checkpoints/bitdistill-glue-seqcls/{model_slug}/{task}",
                    "exclude_linear_regex": "score|classifier",
                    "init_output_head_from_teacher": "1",
                    "warmup_state": warmup_state,
                }
            )
    for task in TASKS:
        for scale, root in (
            ("tensor", "checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma"),
            ("row", "checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row"),
        ):
            rows.append(
                {
                    "family": f"seqcls_paper_gamma100000_{scale}",
                    "task": task,
                    "task_format": "sequence_classification",
                    "scale": scale,
                    "attention_kd_weight": 100000.0,
                    "output_root": f"{root}/{model_slug}/{task}",
                    "teacher_root": f"checkpoints/bitdistill-glue-seqcls/{model_slug}/{task}",
                    "exclude_linear_regex": "score|classifier",
                    "init_output_head_from_teacher": "0",
                    "warmup_state": warmup_state,
                }
            )
    for lr, root in (
        (1e-5, "checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr1e-5"),
        (5e-5, "checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr5e-5"),
    ):
        for task in TASKS:
            rows.append(
                {
                    "family": f"seqcls_paper_gamma100000_tensor_lr{lr:g}",
                    "task": task,
                    "task_format": "sequence_classification",
                    "scale": "tensor",
                    "attention_kd_weight": 100000.0,
                    "lr": lr,
                    "output_root": f"{root}/{model_slug}/{task}",
                    "teacher_root": f"checkpoints/bitdistill-glue-seqcls/{model_slug}/{task}",
                    "exclude_linear_regex": "score|classifier",
                    "init_output_head_from_teacher": "0",
                    "warmup_state": warmup_state,
                }
            )
    for task in TASKS:
        rows.append(
            {
                "family": "seqcls_paper_gamma100000_tensor_headinit",
                "task": task,
                "task_format": "sequence_classification",
                "scale": "tensor",
                "attention_kd_weight": 100000.0,
                "lr": 2e-5,
                "output_root": f"checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-headinit/{model_slug}/{task}",
                "teacher_root": f"checkpoints/bitdistill-glue-seqcls/{model_slug}/{task}",
                "exclude_linear_regex": "score|classifier",
                "init_output_head_from_teacher": "1",
                "warmup_state": warmup_state,
            }
        )
    for gamma, root in (
        (1000.0, "checkpoints/bitdistill-glue-seqcls-longwarmup-gamma1k"),
        (10000.0, "checkpoints/bitdistill-glue-seqcls-longwarmup-gamma10k"),
    ):
        rows.append(
            {
                "family": f"mnli_gamma{int(gamma)}",
                "task": "mnli",
                "task_format": "sequence_classification",
                "scale": "tensor",
                "attention_kd_weight": gamma,
                "output_root": f"{root}/{model_slug}/mnli",
                "teacher_root": f"checkpoints/bitdistill-glue-seqcls/{model_slug}/mnli",
                "exclude_linear_regex": "score|classifier",
                "init_output_head_from_teacher": "0",
                "warmup_state": warmup_state,
            }
        )
    for layer in (-1, -2, -4):
        safe_layer = str(abs(layer))
        rows.append(
            {
                "family": f"mnli_layer_sweep_{layer}",
                "task": "mnli",
                "task_format": "sequence_classification",
                "scale": "tensor",
                "attention_kd_weight": 100.0,
                "output_root": (
                    "checkpoints/bitdistill-glue-seqcls-longwarmup-layer-sweep/"
                    f"{model_slug}/mnli/bitdistill-longwarmup-tensor-layer-{safe_layer}"
                ),
                "teacher_root": f"checkpoints/bitdistill-glue-seqcls/{model_slug}/mnli",
                "exclude_linear_regex": "score|classifier",
                "init_output_head_from_teacher": "0",
                "warmup_state": warmup_state,
            }
        )
    for task in TASKS:
        for scale in ("tensor", "row"):
            rows.append(
                {
                    "family": "causal_densehead_gamma100",
                    "task": task,
                    "task_format": "causal_lm",
                    "scale": scale,
                    "attention_kd_weight": 100.0,
                    "output_root": f"checkpoints/bitdistill-glue-causal-longwarmup-densehead/{model_slug}/{task}",
                    "teacher_root": f"checkpoints/bitdistill-glue/{model_slug}/{task}",
                    "exclude_linear_regex": "lm_head",
                    "init_output_head_from_teacher": "0",
                    "warmup_state": warmup_state,
                }
            )
    return rows


def row_matches_expected(row: dict[str, Any], expected: dict[str, Any]) -> bool:
    output_dir = str(row.get("output_dir", ""))
    teacher = str(row.get("teacher", ""))
    actual_gamma = as_float(row.get("attention_kd_weight"))
    actual_lr = as_float(row.get("lr"))
    return (
        row.get("task") == expected["task"]
        and inferred_task_format(row) == expected["task_format"]
        and row.get("scale") == expected["scale"]
        and actual_gamma == expected["attention_kd_weight"]
        and ("lr" not in expected or actual_lr == expected["lr"])
        and output_dir.startswith(str(expected["output_root"]))
        and teacher.startswith(str(expected["teacher_root"]))
        and inferred_exclude_regex(row) == expected["exclude_linear_regex"]
        and str(row.get("init_output_head_from_teacher") or "0") == str(expected["init_output_head_from_teacher"])
        and normalize_path_text(row.get("warmup_state")) == normalize_path_text(expected["warmup_state"])
    )


def find_expected_match(rows: list[dict[str, Any]], expected: dict[str, Any]) -> dict[str, Any] | None:
    matches = [row for row in rows if row_matches_expected(row, expected)]
    if len(matches) == 1:
        return matches[0]
    return None


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, evidence: str, blocker: str = "") -> None:
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence, "blocker": "" if passed else blocker})


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    train_source = Path("train_bitdistill.py")
    train_text = train_source.read_text(encoding="utf-8", errors="replace") if train_source.exists() else ""
    attention_qkv_sum_default = (
        '--attention-qkv-reduction", choices=["sum", "mean"], default="sum"' in train_text
        and 'if qkv_reduction == "sum":' in train_text
    )
    monitor = read_json(args.monitor_json)
    downstream = monitor.get("downstream", []) if isinstance(monitor.get("downstream"), list) else []
    rows = [row for row in downstream if isinstance(row, dict)]
    warmup = monitor.get("warmup", {}) if isinstance(monitor.get("warmup"), dict) else {}
    env = warmup.get("env", {}) if isinstance(warmup.get("env"), dict) else {}
    model = env.get("MODEL", args.model)
    model_slug = str(model).replace("/", "-")
    warmup_state = str(args.warmup_state) if args.warmup_state else str(
        Path(str(warmup.get("output_dir", ""))) / "custom_state_dict.pt"
    )
    expected = expected_rows(model_slug, warmup_state)

    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    output_counts = Counter(str(row.get("output_dir", "")) for row in rows)
    duplicate_outputs = sorted(output for output, count in output_counts.items() if output and count > 1)
    active_job_states = Counter(
        str((row.get("job_status") or {}).get("state", "unknown")) for row in rows if isinstance(row.get("job_status"), dict)
    )
    inferred_field_rows = [
        {"job_id": row.get("job_id"), "fields": inferred_field_names(row)}
        for row in rows
        if inferred_field_names(row)
    ]
    script_provenance, script_failures = audit_stored_downstream_scripts(rows)
    script_required = [record for record in script_provenance if record.get("script_required") is True]

    add_check(checks, "monitor json exists", bool(monitor), str(args.monitor_json), "missing BitDistill monitor JSON")
    add_check(
        checks,
        "train_bitdistill defaults to paper-style Q/K/V attention sum",
        attention_qkv_sum_default,
        f"train_bitdistill.py sha256={sha256_file(train_source)[:12] if train_source.exists() else '-'}",
        "queued jobs rely on train_bitdistill.py defaults, but the Q/K/V reduction is not paper-style sum",
    )
    add_check(checks, "active row count matches design", len(rows) == len(expected), f"rows={len(rows)}, expected={len(expected)}", "wrong active row count")
    add_check(checks, "output directories are unique", not duplicate_outputs, f"duplicates={duplicate_outputs}", "duplicate active output directories")

    expected_results: list[dict[str, Any]] = []
    for item in expected:
        match = find_expected_match(rows, item)
        teacher_metrics = Path(str(match.get("teacher", ""))) / "metrics.json" if match else Path()
        teacher_ok = teacher_metrics.exists() if match else False
        status = (match.get("job_status") or {}).get("state") if match and isinstance(match.get("job_status"), dict) else None
        issues: list[str] = []
        if match is None:
            issues.append("missing expected row")
        if match and str(match.get("dependency", "")).split(":", 1)[-1] not in set(str(job_id) for job_id in monitor.get("warmup_job_ids", [])):
            issues.append("dependency does not point to warm-up job")
        if match and match.get("logit_kd_temperature_scale") != "none":
            issues.append("logit KD temperature scale is not paper-style none")
        if match and inferred_label_scheme(match) != "letters":
            issues.append("label scheme is not letters")
        if match and inferred_candidate_score(match) != "mean":
            issues.append("candidate scoring is not mean")
        if match and as_float(match.get("logit_kd_weight")) != 10.0:
            issues.append("logit KD weight is not 10.0")
        if match and as_float(match.get("logit_temperature")) != 5.0:
            issues.append("logit temperature is not 5.0")
        if match and as_float(match.get("attention_temperature")) != 1.0:
            issues.append("attention temperature is not 1.0")
        if match and as_float(match.get("task_max_steps")) != 1000.0:
            issues.append("task max steps is not 1000")
        if match and not teacher_ok:
            issues.append(f"teacher metrics missing: {teacher_metrics}")
        expected_results.append(
            {
                **item,
                "job_id": match.get("job_id") if match else None,
                "output_dir": match.get("output_dir") if match else None,
                "teacher": match.get("teacher") if match else None,
                "teacher_metrics_exists": teacher_ok,
                "inferred_fields": inferred_field_names(match) if match else [],
                "head_init": str(match.get("init_output_head_from_teacher") or "0") if match else None,
                "job_state": status,
                "issues": issues,
                "passed": not issues,
            }
        )

    missing = [row for row in expected_results if not row["passed"]]
    add_check(
        checks,
        "all expected experiment rows are present and configured",
        not missing,
        f"configured={len(expected_results) - len(missing)}/{len(expected_results)}",
        f"{len(missing)} expected rows missing or misconfigured",
    )
    add_check(
        checks,
        "downstream stored scripts include critical KD/export arguments",
        not script_failures and len(script_required) == sum(active_job_states.get(state, 0) for state in ACTIVE_SLURM_STATES),
        f"active checked={len(script_required)}, total rows={len(script_provenance)}, failures={len(script_failures)}",
        "at least one active downstream job is unavailable or missing a critical paper-style KD/export argument",
    )
    add_check(
        checks,
        "warm-up progress is finite",
        finite_number((warmup.get("latest_step") or {}).get("step")) or isinstance((warmup.get("latest_step") or {}).get("step"), int),
        f"step={(warmup.get('latest_step') or {}).get('step')}/{warmup.get('max_steps')}",
        "warm-up progress missing",
    )

    for check in checks:
        if not check["passed"]:
            blockers.append(check["blocker"])

    return {
        "schema": "bitdistill-job-matrix-audit-v1",
        "date": DATE,
        "monitor_json": str(args.monitor_json),
        "model": model,
        "model_slug": model_slug,
        "warmup_state": warmup_state,
        "train_bitdistill_sha256": sha256_file(train_source) if train_source.exists() else "",
        "attention_qkv_sum_default": attention_qkv_sum_default,
        "warmup_job_ids": monitor.get("warmup_job_ids", []),
        "job_states": dict(active_job_states),
        "inferred_field_rows": inferred_field_rows,
        "expected_rows": len(expected),
        "observed_rows": len(rows),
        "configured_rows": len(expected_results) - len(missing),
        "stored_script_rows": len(script_provenance),
        "stored_script_required_rows": len(script_required),
        "stored_script_failure_count": len(script_failures),
        "stored_script_failures": script_failures,
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "rows": expected_results,
        "blockers": blockers,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    check_rows = [
        [check["name"], "pass" if check["passed"] else "fail", str(check["evidence"]), str(check["blocker"])]
        for check in summary["checks"]
    ]
    row_rows = [
        [
            row["family"],
            row["task"],
            row["task_format"],
            row["scale"],
            fmt(row["attention_kd_weight"]),
            fmt(row["head_init"]),
            fmt(row["job_id"]),
            fmt(row["job_state"]),
            fmt(row["teacher_metrics_exists"]),
            ",".join(row["inferred_fields"]) if row["inferred_fields"] else "none",
            "none" if not row["issues"] else "; ".join(row["issues"]),
        ]
        for row in summary["rows"]
    ]
    sections = [
        f"# BitDistill Job Matrix Audit, {summary['date']}",
        f"Overall status: `{'pass' if summary['passed'] else 'fail'}`.",
        f"Monitor JSON: `{summary['monitor_json']}`.",
        f"Warm-up state: `{summary['warmup_state']}`.",
        f"train_bitdistill.py sha256: `{summary.get('train_bitdistill_sha256', '')[:12]}`. Attention Q/K/V reduction default: `{'sum' if summary.get('attention_qkv_sum_default') else 'not-sum'}`.",
        f"Observed rows: `{summary['observed_rows']}`. Expected rows: `{summary['expected_rows']}`. Configured rows: `{summary['configured_rows']}`.",
        f"Job states: `{summary['job_states']}`.",
        f"Rows with fields inferred from submitter defaults: `{len(summary['inferred_field_rows'])}`.",
        f"Stored downstream scripts checked: `{summary.get('stored_script_required_rows', 0)}` active / `{summary.get('stored_script_rows', 0)}` total rows. Failures: `{summary.get('stored_script_failure_count', 0)}`.",
        "## Checks",
        md_table(["check", "status", "evidence", "blocker"], check_rows),
        "## Expected Matrix",
        md_table(
            [
                "family",
                "task",
                "format",
                "scale",
                "attention gamma",
                "head init",
                "job",
                "state",
                "teacher metrics",
                "inferred fields",
                "issues",
            ],
            row_rows,
        ),
    ]
    if summary["blockers"]:
        sections.extend(["## Blockers", "\n".join(f"- {item}" for item in summary["blockers"])])
    return "\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor-json", type=Path, default=Path(f"benchmark_results/bitdistill_job_monitor_{DATE}.json"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--warmup-state", default="")
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_job_matrix_audit_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_job_matrix_audit_{DATE}.md"))
    args = parser.parse_args()

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(summary)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
