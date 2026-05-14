#!/usr/bin/env python3
"""Audit that the queued BitDistill postprocess job waits on active producers.

The BitDistill run has a long dependency chain: Stage-2 warm-up releases
downstream task jobs, then export/CPU jobs, then a final postprocess job
materializes reports. This preflight catches a common orchestration failure:
adding a benchmark job without extending the postprocess dependency list.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = datetime.now(timezone.utc).date().isoformat()
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}
DEPENDENCY_RE = re.compile(r":(\d+)")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def collect_squeue(user: str) -> list[dict[str, str]]:
    proc = run(["squeue", "-h", "-u", user, "-o", "%i\t%j\t%T\t%R"])
    if proc.returncode != 0:
        return []
    rows: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        rows.append({"job_id": parts[0], "name": parts[1], "state": parts[2], "reason": parts[3]})
    return rows


def scontrol_job(job_id: str) -> dict[str, Any]:
    proc = run(["scontrol", "show", "job", job_id])
    text = proc.stdout
    dependency_match = re.search(r"\bDependency=(\S+)", text)
    dependency_text = dependency_match.group(1) if dependency_match else ""
    return {
        "job_id": job_id,
        "returncode": proc.returncode,
        "raw": text,
        "dependency_text": dependency_text,
        "dependency_job_ids": sorted(set(DEPENDENCY_RE.findall(dependency_text))),
    }


def active_downstream_ids(monitor: dict[str, Any]) -> list[str]:
    rows = monitor.get("downstream", []) if isinstance(monitor.get("downstream"), list) else []
    job_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = row.get("job_status", {}) if isinstance(row.get("job_status"), dict) else {}
        state = str(status.get("state", ""))
        job_id = str(row.get("job_id", ""))
        if job_id and state in ACTIVE_STATES:
            job_ids.add(job_id)
    return sorted(job_ids)


def active_warmup_ids(monitor: dict[str, Any], squeue_rows: list[dict[str, str]]) -> list[str]:
    job_ids: set[str] = set()
    for job_id in monitor.get("warmup_job_ids", []) if isinstance(monitor.get("warmup_job_ids"), list) else []:
        if job_id:
            job_ids.add(str(job_id))
    warmup = monitor.get("warmup", {}) if isinstance(monitor.get("warmup"), dict) else {}
    env = warmup.get("env", {}) if isinstance(warmup.get("env"), dict) else {}
    if env.get("SLURM_JOB_ID"):
        job_ids.add(str(env["SLURM_JOB_ID"]))

    active_ids = {row["job_id"] for row in squeue_rows if row.get("state") in ACTIVE_STATES}
    return sorted(job_ids & active_ids)


def find_active_jobs_by_name(squeue_rows: list[dict[str, str]], names: set[str]) -> list[dict[str, str]]:
    return [
        row
        for row in squeue_rows
        if row.get("name") in names and row.get("state") in ACTIVE_STATES
    ]


def choose_postprocess(squeue_rows: list[dict[str, str]], name: str, explicit_job_id: str) -> tuple[dict[str, str] | None, list[dict[str, str]]]:
    if explicit_job_id:
        matches = [row for row in squeue_rows if row.get("job_id") == explicit_job_id]
        return (matches[0] if matches else {"job_id": explicit_job_id, "name": name, "state": "unknown", "reason": ""}), matches
    matches = [
        row
        for row in squeue_rows
        if row.get("name") == name and row.get("state") in ACTIVE_STATES
    ]
    if not matches:
        return None, []
    return sorted(matches, key=lambda row: int(row["job_id"]))[-1], matches


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, evidence: str, blocker: str = "") -> None:
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence, "blocker": "" if passed else blocker})


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    monitor_path = args.monitor_json if args.monitor_json.is_absolute() else root / args.monitor_json
    monitor = read_json(monitor_path)
    squeue_rows = collect_squeue(args.user or os.environ.get("USER", ""))
    postprocess, postprocess_matches = choose_postprocess(squeue_rows, args.postprocess_job_name, args.postprocess_job_id)
    producer_names = set(args.extra_job_names)
    extra_jobs = find_active_jobs_by_name(squeue_rows, producer_names)
    warmup_ids = active_warmup_ids(monitor, squeue_rows)
    downstream_ids = active_downstream_ids(monitor)
    expected_ids = sorted(set(warmup_ids) | set(downstream_ids) | {row["job_id"] for row in extra_jobs})

    postprocess_info = scontrol_job(postprocess["job_id"]) if postprocess else {}
    dependency_ids = set(postprocess_info.get("dependency_job_ids", []))
    missing = sorted(set(expected_ids) - dependency_ids)
    stale = sorted(dependency_ids - set(expected_ids))

    checks: list[dict[str, Any]] = []
    add_check(checks, "monitor JSON exists", bool(monitor), str(monitor_path), "missing BitDistill monitor JSON")
    add_check(
        checks,
        "postprocess job is discoverable",
        postprocess is not None,
        f"name={args.postprocess_job_name}, matches={len(postprocess_matches)}",
        "no active postprocess job was found in squeue",
    )
    add_check(
        checks,
        "expected producer jobs are active",
        bool(expected_ids),
        f"warmup={len(warmup_ids)}, downstream={len(downstream_ids)}, extra={len(extra_jobs)}, total={len(expected_ids)}",
        "no active producer jobs were found; this audit should run before postprocess release",
    )
    add_check(
        checks,
        "postprocess depends on every active producer",
        postprocess is not None and not missing,
        f"expected={len(expected_ids)}, dependency_ids={len(dependency_ids)}, missing={missing}",
        "postprocess would run before at least one active benchmark/export/CPU job finishes",
    )

    passed = all(check["passed"] for check in checks)
    return {
        "schema": "bitdistill-postprocess-dependency-audit-v1",
        "date": DATE,
        "passed": passed,
        "checks": checks,
        "monitor_json": str(monitor_path),
        "postprocess": {
            "job": postprocess,
            "matches": postprocess_matches,
            "scontrol_returncode": postprocess_info.get("returncode"),
            "dependency_text": postprocess_info.get("dependency_text"),
            "dependency_job_ids": sorted(dependency_ids),
        },
        "expected_producers": {
            "job_ids": expected_ids,
            "warmup_job_ids": warmup_ids,
            "downstream_job_ids": downstream_ids,
            "extra_jobs": extra_jobs,
        },
        "missing_dependency_job_ids": missing,
        "stale_dependency_job_ids": stale,
        "blockers": [check["blocker"] for check in checks if not check["passed"]],
    }


def render_markdown(result: dict[str, Any]) -> str:
    check_rows = [
        [check["name"], "pass" if check["passed"] else "fail", str(check["evidence"]), str(check["blocker"])]
        for check in result["checks"]
    ]
    producer = result["expected_producers"]
    postprocess = result["postprocess"]
    extra_rows = [
        [row.get("job_id", "-"), row.get("name", "-"), row.get("state", "-"), row.get("reason", "-")]
        for row in producer.get("extra_jobs", [])
    ] or [["none", "-", "-", "-"]]
    return "\n\n".join(
        [
            f"# BitDistill Postprocess Dependency Audit, {result['date']}",
            f"Overall status: `{'pass' if result['passed'] else 'fail'}`.",
            f"Postprocess job: `{(postprocess.get('job') or {}).get('job_id', '-')}`.",
            f"Expected producer jobs: `{len(producer.get('job_ids', []))}`.",
            f"Warmup producer jobs: `{', '.join(producer.get('warmup_job_ids', [])) or '-'}`.",
            f"Missing dependencies: `{result['missing_dependency_job_ids']}`.",
            "## Checks",
            md_table(["check", "status", "evidence", "blocker"], check_rows),
            "## Extra Producer Jobs",
            md_table(["job", "name", "state", "reason"], extra_rows),
            "## Dependency Text",
            f"`{postprocess.get('dependency_text', '')}`",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--monitor-json", type=Path, default=Path(f"benchmark_results/bitdistill_job_monitor_{DATE}.json"))
    parser.add_argument("--postprocess-job-name", default="bitdistill-postprocess")
    parser.add_argument("--postprocess-job-id", default="")
    parser.add_argument("--extra-job-names", nargs="+", default=["bitdistill-i2sr", "bitdistill-cpu-bench", "bitdistill-predtrace"])
    parser.add_argument("--user", default=os.environ.get("USER", ""))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_postprocess_dependency_audit_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_postprocess_dependency_audit_{DATE}.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    result = build_audit(args)
    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
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
