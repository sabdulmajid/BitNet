#!/usr/bin/env python3
"""Preflight the 655M Stage-2 handoff without creating quality claims.

This checks the exact manifest/handoff path before the dependency-triggered
handoff job runs. Before the final Stage-2 checkpoint exists, it should remain
pending and report the blocking artifact. Once the final checkpoint exists, it
dry-runs manifest creation into a temporary directory and validates the result.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def squeue_state(job_id: str) -> dict[str, str]:
    if not job_id:
        return {"job_id": "", "state": "not_submitted"}
    result = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%i\t%T\t%M\t%l\t%R\t%j"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"job_id": job_id, "state": "not_in_squeue"}
    parts = result.stdout.strip().split("\t", 5)
    return {
        "job_id": parts[0] if len(parts) > 0 else job_id,
        "state": parts[1] if len(parts) > 1 else "unknown",
        "time": parts[2] if len(parts) > 2 else "",
        "time_limit": parts[3] if len(parts) > 3 else "",
        "reason": parts[4] if len(parts) > 4 else "",
        "name": parts[5] if len(parts) > 5 else "",
    }


def check_file(label: str, path: Path, *, required_now: bool = True) -> dict[str, Any]:
    exists = path.exists()
    return {
        "label": label,
        "kind": "file_exists",
        "path": str(path),
        "required_now": required_now,
        "passed": exists if required_now else True,
        "exists": exists,
    }


def check_command(label: str, command: list[str], *, required_now: bool = True) -> dict[str, Any]:
    result = run(command)
    return {
        "label": label,
        "kind": "command",
        "required_now": required_now,
        "passed": result["passed"] if required_now else True,
        "result": result,
    }


def training_save_contract() -> dict[str, Any]:
    source_path = Path("train_bitdistill.py")
    text = source_path.read_text(encoding="utf-8", errors="replace") if source_path.exists() else ""
    checks = [
        {
            "label": "root metrics are written regardless of save_model_artifacts",
            "pattern": '(output_dir / "metrics.json").write_text',
            "passed": '(output_dir / "metrics.json").write_text' in text,
        },
        {
            "label": "root state dict is gated by save_model_artifacts",
            "pattern": 'if args.save_model_artifacts:',
            "passed": 'if args.save_model_artifacts:' in text and 'output_dir / "custom_state_dict.pt"' in text,
        },
        {
            "label": "snapshots write custom_state_dict.pt",
            "pattern": 'snapshot_dir / "custom_state_dict.pt"',
            "passed": 'snapshot_dir / "custom_state_dict.pt"' in text,
        },
        {
            "label": "snapshots write metrics.json",
            "pattern": 'snapshot_dir / "metrics.json"',
            "passed": 'snapshot_dir / "metrics.json"' in text,
        },
        {
            "label": "active producer snapshot.complete flag is legacy false",
            "pattern": '"snapshot"] = {"step": step, "complete": False}',
            "passed": '"snapshot"] = {"step": step, "complete": False}' in text,
        },
    ]
    return {
        "source_path": str(source_path),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "caveat": (
            "The running 655M producer was submitted before any code change here. "
            "For this active run, snapshot usability is audited from actual state/metrics files, "
            "not from the legacy snapshot.complete flag."
        ),
    }


def manifest_command(
    *,
    output_dir: Path,
    parent_manifest: Path,
    run_id: str,
    job_id: str,
    output_json: Path,
    output_md: Path,
) -> list[str]:
    return [
        "python",
        "benchmarks/build_stage2_manifest.py",
        "--output-dir",
        str(output_dir),
        "--parent-manifest",
        str(parent_manifest),
        "--run-id",
        run_id,
        "--job-id",
        job_id,
        "--downstream-status",
        "pending_submission",
        "--downstream-failed-job-id",
        "",
        "--downstream-failure-mode",
        "",
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]


def dry_run_manifest(command: list[str], output_json: Path) -> dict[str, Any]:
    build = run(command)
    validate = (
        run(["python", "benchmarks/validate_stage2_manifest.py", str(output_json)])
        if build["passed"] and output_json.exists()
        else {
            "command": ["python", "benchmarks/validate_stage2_manifest.py", str(output_json)],
            "returncode": None,
            "passed": False,
            "stdout": "",
            "stderr": "manifest build did not produce output json",
        }
    )
    return {
        "build": build,
        "validate": validate,
        "passed": bool(build["passed"] and validate["passed"]),
    }


def classify(
    *,
    preflight_checks: list[dict[str, Any]],
    final_state: dict[str, Any],
    final_metrics: dict[str, Any],
    root_metrics: dict[str, Any],
    slurm_state: str,
    dry_run: dict[str, Any] | None,
) -> str:
    if any(not check["passed"] for check in preflight_checks):
        return "failed_preflight"
    final_ready = bool(final_state["exists"] and final_metrics["exists"] and root_metrics["exists"])
    if dry_run is not None:
        return "ready_for_handoff" if dry_run.get("passed") else "failed_manifest_dry_run"
    if final_ready:
        return "final_artifacts_ready_pending_dry_run"
    if slurm_state.lower() in {"running", "pending"}:
        return "pending_stage2_completion"
    return "failed_missing_final_snapshot"


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != 0.0 and (abs(value) >= 10000.0 or abs(value) < 0.0001):
            return f"{value:.3e}"
        return f"{value:.6f}"
    if isinstance(value, list):
        return " ".join(str(item) for item in value) if value else "none"
    if isinstance(value, dict):
        return ", ".join(f"{key}={fmt(val)}" for key, val in value.items()) if value else "none"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    stage2_submission = read_json(args.stage2_submission)
    handoff_submission = read_json(args.handoff_submission)
    monitor = read_json(args.active_monitor, required=False)
    run_config = stage2_submission["run_config"]
    job_id = str(stage2_submission.get("submitted_job_id", ""))
    output_dir = Path(run_config["output_dir"])
    max_steps = int(run_config["max_steps"])
    parent_manifest = Path(stage2_submission["parent_manifest"]["path"])
    manifest_json = Path(handoff_submission["expected_manifest_json"])
    manifest_md = Path(handoff_submission["expected_manifest_md"])
    run_id = f"qwen25-05b-bitdistill-tensor-stage2-655m-from327m-job{job_id}"
    final_snapshot = output_dir / f"checkpoint-{max_steps}"
    final_state = file_info(final_snapshot / "custom_state_dict.pt")
    final_metrics = file_info(final_snapshot / "metrics.json")
    root_metrics = file_info(output_dir / "metrics.json")
    final_ready = bool(final_state["exists"] and final_metrics["exists"] and root_metrics["exists"])
    slurm = squeue_state(job_id)
    stage2 = monitor.get("stage2", {}) if isinstance(monitor.get("stage2"), dict) else {}
    snapshot_status = stage2.get("snapshot_status", {}) if isinstance(stage2.get("snapshot_status"), dict) else {}

    preflight_checks = [
        check_file("parent manifest exists", parent_manifest),
        check_command("parent manifest validates", ["python", "benchmarks/validate_stage2_manifest.py", str(parent_manifest)]),
        check_file("build_stage2_manifest.py exists", Path("benchmarks/build_stage2_manifest.py")),
        check_file("validate_stage2_manifest.py exists", Path("benchmarks/validate_stage2_manifest.py")),
        check_file("handoff script exists", Path(handoff_submission["script"])),
        check_file("postprocess script exists", Path(handoff_submission["postprocess_script"])),
        check_command("handoff script syntax", ["bash", "-n", handoff_submission["script"]]),
        check_command("postprocess script syntax", ["bash", "-n", handoff_submission["postprocess_script"]]),
        check_file("downstream training script exists", Path("slurm_bitdistill_glue.sh")),
        check_file(
            "FP16 teacher directory exists",
            Path("checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1"),
        ),
    ]
    save_contract = training_save_contract()
    preflight_checks.append(
        {
            "label": "training save contract matches handoff assumptions",
            "kind": "source_contract",
            "path": save_contract["source_path"],
            "required_now": True,
            "passed": save_contract["passed"],
            "exists": Path(save_contract["source_path"]).exists(),
        }
    )

    final_artifact_checks = [
        {**final_state, "label": "final state dict", "required_now": False, "passed": True},
        {**final_metrics, "label": "final snapshot metrics", "required_now": False, "passed": True},
        {**root_metrics, "label": "root metrics", "required_now": False, "passed": True},
    ]

    dry_run: dict[str, Any] | None = None
    dry_run_command: list[str] = []
    expected_command = manifest_command(
        output_dir=output_dir,
        parent_manifest=parent_manifest,
        run_id=run_id,
        job_id=job_id,
        output_json=manifest_json,
        output_md=manifest_md,
    )
    if final_ready and all(check["passed"] for check in preflight_checks):
        with tempfile.TemporaryDirectory(prefix="bitnet-655m-manifest-preflight-") as tmpdir:
            dry_json = Path(tmpdir) / "stage2_manifest_655m_preflight.json"
            dry_md = Path(tmpdir) / "stage2_manifest_655m_preflight.md"
            dry_run_command = manifest_command(
                output_dir=output_dir,
                parent_manifest=parent_manifest,
                run_id=run_id,
                job_id=job_id,
                output_json=dry_json,
                output_md=dry_md,
            )
            dry_run = dry_run_manifest(dry_run_command, dry_json)

    status = classify(
        preflight_checks=preflight_checks,
        final_state=final_state,
        final_metrics=final_metrics,
        root_metrics=root_metrics,
        slurm_state=slurm.get("state", ""),
        dry_run=dry_run,
    )
    return {
        "schema": "bitdistill-655m-handoff-preflight-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quality_claim": "none",
        "status": status,
        "caveat": "This validates the queued handoff path only. It does not run downstream evaluation or update quality claims.",
        "stage2_job_id": job_id,
        "slurm": slurm,
        "latest_step": (stage2.get("latest_step") or {}).get("step") if isinstance(stage2.get("latest_step"), dict) else None,
        "snapshot_status": snapshot_status,
        "output_dir": str(output_dir),
        "final_snapshot": str(final_snapshot),
        "final_artifact_checks": final_artifact_checks,
        "preflight_checks": preflight_checks,
        "training_save_contract": save_contract,
        "expected_manifest_command": expected_command,
        "dry_run_manifest_command": dry_run_command,
        "dry_run": dry_run,
        "expected_manifest_json": str(manifest_json),
        "expected_manifest_md": str(manifest_md),
        "source_paths": {
            "stage2_submission": str(args.stage2_submission),
            "handoff_submission": str(args.handoff_submission),
            "active_monitor": str(args.active_monitor),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    preflight_rows = [
        [
            check["label"],
            check["kind"],
            check.get("path") or " ".join(check.get("result", {}).get("command", [])),
            check["passed"],
            check.get("exists", "-"),
            (check.get("result") or {}).get("returncode", "-"),
        ]
        for check in report["preflight_checks"]
    ]
    final_rows = [
        [check["label"], check["path"], check["exists"], check["size_bytes"]]
        for check in report["final_artifact_checks"]
    ]
    contract_rows = [
        [check["label"], check["pattern"], check["passed"]]
        for check in report["training_save_contract"]["checks"]
    ]
    dry_run = report["dry_run"] or {}
    return "\n\n".join(
        [
            "# Stage-2 655M Handoff Preflight",
            f"Generated: `{report['created_utc']}`",
            f"Status: **{report['status']}**.",
            f"Quality claim: **{report['quality_claim']}**.",
            report["caveat"],
            "## Current State",
            md_table(
                ["field", "value"],
                [
                    ["stage2_job_id", report["stage2_job_id"]],
                    ["slurm_state", report["slurm"].get("state")],
                    ["slurm_time", report["slurm"].get("time")],
                    ["latest_step", report["latest_step"]],
                    ["snapshot_status", report["snapshot_status"].get("status")],
                    ["next_snapshot_step", report["snapshot_status"].get("next_snapshot_step")],
                    ["steps_to_next_snapshot", report["snapshot_status"].get("steps_to_next_snapshot")],
                    ["output_dir", report["output_dir"]],
                    ["final_snapshot", report["final_snapshot"]],
                ],
            ),
            "## Preflight Checks",
            md_table(["check", "kind", "path/command", "passed", "exists", "returncode"], preflight_rows),
            "## Final Artifact Checks",
            md_table(["artifact", "path", "exists", "size_bytes"], final_rows),
            "## Training Save Contract",
            md_table(["check", "source pattern", "passed"], contract_rows),
            report["training_save_contract"]["caveat"],
            "## Manifest Command",
            "`" + " ".join(report["expected_manifest_command"]) + "`",
            "## Dry Run",
            md_table(
                ["field", "value"],
                [
                    ["attempted", bool(report["dry_run_manifest_command"])],
                    ["passed", dry_run.get("passed")],
                    ["build_returncode", (dry_run.get("build") or {}).get("returncode")],
                    ["validate_returncode", (dry_run.get("validate") or {}).get("returncode")],
                ],
            ),
            "## Source Artifacts",
            md_table(["artifact", "path"], [[key, value] for key, value in report["source_paths"].items()]),
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage2-submission",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_submission_2026-05-23.json"),
    )
    parser.add_argument(
        "--handoff-submission",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_handoff_submission_2026-05-23.json"),
    )
    parser.add_argument(
        "--active-monitor",
        type=Path,
        default=Path("benchmarks/results/active_stage2_extension_monitor_2026-05-23.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_handoff_preflight_2026-05-23.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_handoff_preflight_2026-05-23.md"),
    )
    args = parser.parse_args()

    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    return 1 if report["status"].startswith("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
