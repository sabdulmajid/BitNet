#!/usr/bin/env python3
"""Audit queued BitDistill producer batch scripts.

Dependency audits prove that finalizers wait for the expected jobs. This audit
checks the other side of that contract: the active producer jobs themselves are
using current batch scripts and the CPU benchmark parser accepts the run
families requested by the stored Slurm script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = datetime.now(timezone.utc).date().isoformat()
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}
CPU_FAMILIES = [
    "short",
    "longwarmup",
    "papergamma",
    "papergamma_row",
    "papergamma_lr1",
    "papergamma_lr5",
    "papergamma_headinit",
]


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_squeue(user: str) -> list[dict[str, str]]:
    proc = run(["squeue", "-h", "-u", user, "-o", "%i\t%j\t%T\t%R"])
    rows: list[dict[str, str]] = []
    if proc.returncode != 0:
        return rows
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        rows.append({"job_id": parts[0], "name": parts[1], "state": parts[2], "reason": parts[3]})
    return rows


def active_jobs_by_name(squeue_rows: list[dict[str, str]], name: str) -> list[dict[str, str]]:
    return [
        row
        for row in squeue_rows
        if row.get("name") == name and row.get("state") in ACTIVE_STATES and row.get("job_id")
    ]


def scontrol_job(job_id: str) -> dict[str, Any]:
    proc = run(["scontrol", "show", "job", job_id])
    dependency_match = re.search(r"\bDependency=(\S+)", proc.stdout)
    dependency_text = dependency_match.group(1) if dependency_match else ""
    return {
        "job_id": job_id,
        "returncode": proc.returncode,
        "dependency_text": dependency_text,
        "dependency_job_ids": sorted(set(re.findall(r":(\d+)", dependency_text)), key=int),
        "raw": proc.stdout,
    }


def stored_script(job_id: str) -> str:
    with tempfile.TemporaryDirectory(prefix="bitdistill-producer-script-") as tmp:
        path = Path(tmp) / f"job-{job_id}.sh"
        proc = run(["scontrol", "write", "batch_script", job_id, str(path)])
        if proc.returncode != 0 or not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")


def active_downstream_ids(monitor: dict[str, Any]) -> list[str]:
    rows = monitor.get("downstream", []) if isinstance(monitor.get("downstream"), list) else []
    ids = {
        str(row.get("job_id"))
        for row in rows
        if isinstance(row, dict) and row.get("job_id")
    }
    return sorted(ids, key=int)


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, evidence: str, blocker: str = "") -> None:
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence, "blocker": "" if passed else blocker})


def parse_cpu_families() -> tuple[bool, str]:
    from benchmark_bitdistill_glue_cpu import parse_runs

    values = [f"{family}:dummy" for family in CPU_FAMILIES]
    try:
        parsed = parse_runs(values)
    except Exception as exc:  # pragma: no cover - report path
        return False, repr(exc)
    parsed_families = [family for family, _ in parsed]
    return parsed_families == CPU_FAMILIES, ",".join(parsed_families)


def py_compile_ok(root: Path, paths: list[Path]) -> tuple[bool, str]:
    proc = run(["python", "-m", "py_compile", *(str(path) for path in paths)])
    evidence = proc.stderr.strip() or proc.stdout.strip() or ",".join(str(path.relative_to(root)) for path in paths)
    return proc.returncode == 0, evidence


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    monitor_path = args.monitor_json if args.monitor_json.is_absolute() else root / args.monitor_json
    monitor = read_json(monitor_path)
    squeue_rows = collect_squeue(args.user or os.environ.get("USER", ""))
    downstream_ids = active_downstream_ids(monitor)

    cpu_jobs = active_jobs_by_name(squeue_rows, "bitdistill-cpu-bench")
    i2sr_jobs = active_jobs_by_name(squeue_rows, "bitdistill-i2sr")
    strict_post = active_jobs_by_name(squeue_rows, "bitdistill-postprocess")
    any_post = active_jobs_by_name(squeue_rows, "bitdistill-postprocess-any")

    current_cpu = (root / "slurm_bitdistill_cpu_benchmark.sh").read_text(encoding="utf-8")
    current_i2sr = (root / "slurm_bitdistill_i2sr_export.sh").read_text(encoding="utf-8")
    current_postprocess = (root / "slurm_bitdistill_postprocess.sh").read_text(encoding="utf-8")
    cpu_helper = root / "benchmarks/benchmark_bitdistill_glue_cpu.py"
    cpu_gate = root / "benchmarks/gate_bitdistill_cpu_benchmark.py"
    i2sr_helper = root / "benchmarks/export_bitdistill_i2sr_suite.py"
    i2sr_converter = root / "benchmarks/convert_static_ternary_to_i2s_gguf.py"
    task_formulation_audit = root / "benchmarks/audit_bitdistill_task_formulation.py"
    cpu_job = sorted(cpu_jobs, key=lambda row: int(row["job_id"]))[-1] if cpu_jobs else {}
    i2sr_job = sorted(i2sr_jobs, key=lambda row: int(row["job_id"]))[-1] if i2sr_jobs else {}
    strict_post_job = sorted(strict_post, key=lambda row: int(row["job_id"]))[-1] if strict_post else {}
    any_post_job = sorted(any_post, key=lambda row: int(row["job_id"]))[-1] if any_post else {}
    cpu_script = stored_script(str(cpu_job.get("job_id", ""))) if cpu_job else ""
    i2sr_script = stored_script(str(i2sr_job.get("job_id", ""))) if i2sr_job else ""
    strict_post_script = stored_script(str(strict_post_job.get("job_id", ""))) if strict_post_job else ""
    any_post_script = stored_script(str(any_post_job.get("job_id", ""))) if any_post_job else ""
    cpu_info = scontrol_job(str(cpu_job.get("job_id", ""))) if cpu_job else {}
    i2sr_info = scontrol_job(str(i2sr_job.get("job_id", ""))) if i2sr_job else {}
    strict_info = scontrol_job(str(strict_post_job.get("job_id", ""))) if strict_post_job else {}
    any_info = scontrol_job(str(any_post_job.get("job_id", ""))) if any_post_job else {}

    cpu_parser_ok, cpu_parser_evidence = parse_cpu_families()
    helpers_compile, helpers_compile_evidence = py_compile_ok(root, [cpu_helper, cpu_gate, i2sr_helper, i2sr_converter, task_formulation_audit])
    stale_cpu_jobs = [row for row in squeue_rows if row.get("job_id") in {"9967", "9997"}]
    checks: list[dict[str, Any]] = []
    add_check(checks, "exactly one active CPU producer", len(cpu_jobs) == 1, f"jobs={cpu_jobs}", "expected one current CPU producer job")
    add_check(
        checks,
        "CPU producer stored script matches current script",
        bool(cpu_script) and sha256_text(cpu_script) == sha256_text(current_cpu),
        f"job={cpu_job.get('job_id')}, stored={sha256_text(cpu_script)[:12] if cpu_script else '-'}, current={sha256_text(current_cpu)[:12]}",
        "queued CPU producer was submitted from a stale Slurm script",
    )
    add_check(
        checks,
        "CPU producer depends on all downstream jobs with afterany",
        set(cpu_info.get("dependency_job_ids", [])) == set(downstream_ids) and str(cpu_info.get("dependency_text", "")).startswith("afterany:"),
        f"job={cpu_job.get('job_id')}, deps={len(cpu_info.get('dependency_job_ids', []))}/{len(downstream_ids)}, dependency={cpu_info.get('dependency_text', '')[:80]}",
        "CPU benchmark should run after all downstream jobs reach terminal state",
    )
    add_check(
        checks,
        "CPU parser accepts queued run families",
        cpu_parser_ok,
        cpu_parser_evidence,
        "benchmark_bitdistill_glue_cpu.py rejects a run family requested by slurm_bitdistill_cpu_benchmark.sh",
    )
    add_check(
        checks,
        "CPU producer script invokes the audited Python benchmark",
        "benchmarks/benchmark_bitdistill_glue_cpu.py" in current_cpu,
        f"helper={cpu_helper.relative_to(root)}, sha256={sha256_file(cpu_helper)[:12]}",
        "CPU Slurm producer does not invoke the audited Python benchmark helper",
    )
    add_check(checks, "stale CPU jobs are not active", not stale_cpu_jobs, f"stale={stale_cpu_jobs}", "old CPU jobs remain active")
    add_check(checks, "exactly one active I2_SR producer", len(i2sr_jobs) == 1, f"jobs={i2sr_jobs}", "expected one I2_SR producer")
    add_check(
        checks,
        "I2_SR producer stored script matches current script",
        bool(i2sr_script) and sha256_text(i2sr_script) == sha256_text(current_i2sr),
        f"job={i2sr_job.get('job_id')}, stored={sha256_text(i2sr_script)[:12] if i2sr_script else '-'}, current={sha256_text(current_i2sr)[:12]}",
        "queued I2_SR producer was submitted from a stale Slurm script",
    )
    add_check(
        checks,
        "I2_SR producer script invokes the audited export helper",
        "benchmarks/export_bitdistill_i2sr_suite.py" in current_i2sr,
        f"helper={i2sr_helper.relative_to(root)}, sha256={sha256_file(i2sr_helper)[:12]}",
        "I2_SR Slurm producer does not invoke the audited export helper",
    )
    add_check(
        checks,
        "producer Python helpers compile",
        helpers_compile,
        helpers_compile_evidence,
        "At least one producer Python helper has a compile error",
    )
    add_check(
        checks,
        "postprocess finalizers depend on current CPU producer",
        cpu_job and str(cpu_job.get("job_id")) in strict_info.get("dependency_job_ids", []) and str(cpu_job.get("job_id")) in any_info.get("dependency_job_ids", []),
        f"cpu={cpu_job.get('job_id')}, strict={strict_info.get('dependency_job_ids', [])[-5:]}, afterany={any_info.get('dependency_job_ids', [])[-5:]}",
        "postprocess jobs do not wait on the current CPU producer",
    )
    add_check(
        checks,
        "current postprocess script includes task-formulation audit",
        "benchmarks/audit_bitdistill_task_formulation.py" in current_postprocess,
        "slurm_bitdistill_postprocess.sh",
        "postprocess would omit the task-formulation claim-control report",
    )
    add_check(
        checks,
        "latest strict postprocess stored script matches current script",
        bool(strict_post_script) and sha256_text(strict_post_script) == sha256_text(current_postprocess),
        f"job={strict_post_job.get('job_id')}, stored={sha256_text(strict_post_script)[:12] if strict_post_script else '-'}, current={sha256_text(current_postprocess)[:12]}",
        "latest strict postprocess job was submitted from a stale Slurm script",
    )
    add_check(
        checks,
        "latest afterany postprocess stored script matches current script",
        bool(any_post_script) and sha256_text(any_post_script) == sha256_text(current_postprocess),
        f"job={any_post_job.get('job_id')}, stored={sha256_text(any_post_script)[:12] if any_post_script else '-'}, current={sha256_text(current_postprocess)[:12]}",
        "latest afterany postprocess job was submitted from a stale Slurm script",
    )

    return {
        "schema": "bitdistill-producer-script-audit-v1",
        "date": DATE,
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "monitor_json": str(monitor_path),
        "downstream_job_ids": downstream_ids,
        "cpu_job": cpu_job,
        "i2sr_job": i2sr_job,
        "producer_helper_hashes": {
            str(cpu_helper.relative_to(root)): sha256_file(cpu_helper),
            str(cpu_gate.relative_to(root)): sha256_file(cpu_gate),
            str(i2sr_helper.relative_to(root)): sha256_file(i2sr_helper),
            str(i2sr_converter.relative_to(root)): sha256_file(i2sr_converter),
            str(task_formulation_audit.relative_to(root)): sha256_file(task_formulation_audit),
        },
        "strict_postprocess_job": strict_post_job,
        "afterany_postprocess_job": any_post_job,
    }


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    rows = [
        [check["name"], "pass" if check["passed"] else "fail", str(check["evidence"]), str(check["blocker"])]
        for check in result["checks"]
    ]
    return "\n\n".join(
        [
            f"# BitDistill Producer Script Audit, {result['date']}",
            f"Overall status: `{'pass' if result['passed'] else 'fail'}`.",
            f"CPU producer: `{result.get('cpu_job', {}).get('job_id', '-')}`.",
            f"I2_SR producer: `{result.get('i2sr_job', {}).get('job_id', '-')}`.",
            f"Downstream rows: `{len(result.get('downstream_job_ids', []))}`.",
            f"Producer helper hashes: `{ {key: value[:12] for key, value in result.get('producer_helper_hashes', {}).items()} }`.",
            "## Checks",
            md_table(["check", "status", "evidence", "blocker"], rows),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--monitor-json", type=Path, default=Path(f"benchmark_results/bitdistill_job_monitor_{DATE}.json"))
    parser.add_argument("--user", default=os.environ.get("USER", ""))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_producer_script_audit_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_producer_script_audit_{DATE}.md"))
    args = parser.parse_args()

    result = build_audit(args)
    output_json = args.output_json if args.output_json.is_absolute() else args.repo_root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else args.repo_root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(result)
    output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
