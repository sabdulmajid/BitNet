#!/usr/bin/env python3
"""Write the submission receipt for the 655M Stage-2 afterany audit job."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def build(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "bitnet-stage2-afterany-submission-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quality_claim": "none",
        "status": "dependency_pending",
        "job_id": args.job_id,
        "stage2_job_id": args.stage2_job_id,
        "dependency": args.dependency,
        "script": str(args.script),
        "expected_report_json": str(args.expected_report_json),
        "expected_report_md": str(args.expected_report_md),
        "expected_salvage_json": str(args.expected_salvage_json),
        "expected_ingestion_json": str(args.expected_ingestion_json),
        "expected_watchdog_json": str(args.expected_watchdog_json),
        "caveat": "This afterany job refreshes postmortem/salvage status only. It does not create downstream quality evidence.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "# Stage-2 655.36M Afterany Submission",
            f"Generated: `{report['created_utc']}`",
            f"Status: **{report['status']}**.",
            f"Quality claim: **{report['quality_claim']}**.",
            report["caveat"],
            md_table(
                ["field", "value"],
                [
                    ["job_id", report["job_id"]],
                    ["stage2_job_id", report["stage2_job_id"]],
                    ["dependency", report["dependency"]],
                    ["script", report["script"]],
                    ["expected_report_json", report["expected_report_json"]],
                    ["expected_salvage_json", report["expected_salvage_json"]],
                    ["expected_ingestion_json", report["expected_ingestion_json"]],
                    ["expected_watchdog_json", report["expected_watchdog_json"]],
                ],
            ),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--stage2-job-id", default="10250")
    parser.add_argument("--dependency", default="afterany:10250")
    parser.add_argument("--script", type=Path, default=Path("slurm_stage2_655m_afterany_audit.sh"))
    parser.add_argument(
        "--expected-report-json",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_afterany_audit_2026-05-23.json"),
    )
    parser.add_argument(
        "--expected-report-md",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_afterany_audit_2026-05-23.md"),
    )
    parser.add_argument(
        "--expected-salvage-json",
        type=Path,
        default=Path("benchmarks/results/stage2_snapshot_salvage_2026-05-23.json"),
    )
    parser.add_argument(
        "--expected-ingestion-json",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_ingestion_2026-05-23.json"),
    )
    parser.add_argument(
        "--expected-watchdog-json",
        type=Path,
        default=Path("benchmarks/results/active_gate_watchdog_2026-05-23.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_afterany_submission_2026-05-23.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_afterany_submission_2026-05-23.md"),
    )
    args = parser.parse_args()
    report = build(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
