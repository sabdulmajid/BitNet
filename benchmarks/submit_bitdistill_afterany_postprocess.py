#!/usr/bin/env python3
"""Submit a failure-aware BitDistill postprocess finalizer.

The main BitDistill postprocess jobs are intentionally strict: they use
``afterok`` so a successful final report means every producer completed
successfully. For long benchmark waves we also want a diagnostic finalizer that
runs after every producer reaches a terminal state, even if one producer fails.
This script submits that companion job with an ``afterany`` dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = datetime.now(timezone.utc).date().isoformat()
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}


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


def downstream_ids(monitor: dict[str, Any]) -> list[str]:
    ids: set[str] = set()
    for row in monitor.get("downstream", []) if isinstance(monitor.get("downstream"), list) else []:
        if not isinstance(row, dict):
            continue
        status = row.get("job_status", {}) if isinstance(row.get("job_status"), dict) else {}
        if status.get("state") in ACTIVE_STATES and row.get("job_id"):
            ids.add(str(row["job_id"]))
    return sorted(ids, key=int)


def warmup_ids(monitor: dict[str, Any], squeue_rows: list[dict[str, str]]) -> list[str]:
    ids: set[str] = set()
    for job_id in monitor.get("warmup_job_ids", []) if isinstance(monitor.get("warmup_job_ids"), list) else []:
        if job_id:
            ids.add(str(job_id))
    warmup = monitor.get("warmup", {}) if isinstance(monitor.get("warmup"), dict) else {}
    env = warmup.get("env", {}) if isinstance(warmup.get("env"), dict) else {}
    if env.get("SLURM_JOB_ID"):
        ids.add(str(env["SLURM_JOB_ID"]))

    active_ids = {row["job_id"] for row in squeue_rows if row.get("state") in ACTIVE_STATES}
    return sorted(ids & active_ids, key=int)


def active_named_ids(squeue_rows: list[dict[str, str]], names: set[str]) -> list[str]:
    return sorted(
        {
            row["job_id"]
            for row in squeue_rows
            if row.get("name") in names and row.get("state") in ACTIVE_STATES and row.get("job_id")
        },
        key=int,
    )


def active_job_by_name(squeue_rows: list[dict[str, str]], name: str) -> list[dict[str, str]]:
    return [
        row
        for row in squeue_rows
        if row.get("name") == name and row.get("state") in ACTIVE_STATES
    ]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, data: dict[str, Any]) -> None:
    existing_rows = data["existing_afterany_jobs"] or [{"job_id": "none", "state": "-", "reason": "-"}]
    lines = [
        f"# BitDistill Afterany Postprocess Finalizer, {data['date']}",
        "",
        f"Submitted this invocation: `{'true' if data['submitted'] else 'false'}`.",
        f"Job ID: `{data.get('job_id') or '-'}`.",
        f"Dependency type: `{data['dependency_type']}`.",
        f"Producer jobs: `{len(data['producer_job_ids'])}`.",
        "",
        "## Producer Breakdown",
        "",
        "| source | count | job ids |",
        "| --- | ---: | --- |",
        f"| Stage-2 warmup | {len(data['warmup_job_ids'])} | `{', '.join(data['warmup_job_ids'])}` |",
        f"| downstream GLUE/export rows | {len(data['downstream_job_ids'])} | `{', '.join(data['downstream_job_ids'])}` |",
        f"| extra producer jobs | {len(data['extra_job_ids'])} | `{', '.join(data['extra_job_ids'])}` |",
        "",
        "## Existing Afterany Jobs",
        "",
        "| job | state | reason |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {row.get('job_id', '-')} | {row.get('state', '-')} | {str(row.get('reason', '-')).replace('|', '\\|')} |"
        for row in existing_rows
    )
    lines.extend(
        [
            "",
            "## Command",
            "",
            "```bash",
            " ".join(data["command"]),
            "```",
        ]
    )
    if data.get("note"):
        lines.extend(["", "## Note", "", str(data["note"])])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--monitor-json", type=Path, default=Path(f"benchmark_results/bitdistill_job_monitor_{DATE}.json"))
    parser.add_argument("--job-name", default="bitdistill-postprocess-any")
    parser.add_argument("--postprocess-script", type=Path, default=Path("slurm_bitdistill_postprocess.sh"))
    parser.add_argument("--extra-job-names", nargs="+", default=["bitdistill-i2sr", "bitdistill-cpu-bench", "bitdistill-predtrace"])
    parser.add_argument("--user", default=os.environ.get("USER", ""))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_afterany_postprocess_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_afterany_postprocess_{DATE}.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    monitor_path = args.monitor_json if args.monitor_json.is_absolute() else root / args.monitor_json
    monitor = read_json(monitor_path)
    squeue_rows = collect_squeue(args.user)
    existing = active_job_by_name(squeue_rows, args.job_name)
    warm_ids = warmup_ids(monitor, squeue_rows)
    down_ids = downstream_ids(monitor)
    extra_ids = active_named_ids(squeue_rows, set(args.extra_job_names))
    producer_ids = sorted(set(warm_ids) | set(down_ids) | set(extra_ids), key=int)
    if not producer_ids:
        raise SystemExit("no active producer jobs found; refresh the monitor before submitting")

    dependency = "afterany:" + ":".join(producer_ids)
    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        args.job_name,
        "--dependency",
        dependency,
        str(args.postprocess_script),
    ]
    submitted = False
    job_id = ""
    note = ""
    returncode = None
    stdout = ""
    stderr = ""
    if existing and not args.force:
        job_id = sorted(existing, key=lambda row: int(row["job_id"]))[-1]["job_id"]
        note = f"active {args.job_name} job already exists; use --force to submit another"
    elif args.dry_run:
        note = "dry run; no job submitted"
    else:
        proc = run(command)
        returncode = proc.returncode
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if proc.returncode != 0:
            raise SystemExit(f"sbatch failed rc={proc.returncode}: {stderr or stdout}")
        submitted = True
        job_id = stdout.split(";", 1)[0]

    result = {
        "schema": "bitdistill-afterany-postprocess-submission-v1",
        "date": DATE,
        "submitted": submitted,
        "job_id": job_id,
        "job_name": args.job_name,
        "dependency_type": "afterany",
        "producer_job_ids": producer_ids,
        "warmup_job_ids": warm_ids,
        "downstream_job_ids": down_ids,
        "extra_job_ids": extra_ids,
        "existing_afterany_jobs": existing,
        "command": command,
        "monitor_json": str(monitor_path),
        "note": note,
        "sbatch_returncode": returncode,
        "sbatch_stdout": stdout,
        "sbatch_stderr": stderr,
    }
    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    write_json(output_json, result)
    write_markdown(output_md, result)
    print(output_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
