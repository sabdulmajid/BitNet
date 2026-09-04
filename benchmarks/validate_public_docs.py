#!/usr/bin/env python3
"""Validate public docs against the canonical evidence bundle.

This is intentionally conservative: it checks that the headline README,
CLAIMS, and runtime-contract numbers are still backed by the canonical JSON
bundle and that every artifact referenced by the bundle exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def fmt_ppl(value: float) -> str:
    if abs(value) >= 10000:
        return f"{value:,.3f}"
    return f"{value:.3f}"


def require_contains(label: str, needle: str, haystack: str, errors: list[str]) -> None:
    if needle not in haystack:
        errors.append(f"{label}: missing `{needle}`")


def validate_artifacts(bundle: dict[str, Any], errors: list[str]) -> None:
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("canonical bundle missing artifacts object")
        return
    for label, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            errors.append(f"artifact {label}: not an object")
            continue
        path = artifact.get("path")
        if not isinstance(path, str) or not path:
            errors.append(f"artifact {label}: missing path")
            continue
        if not Path(path).exists():
            errors.append(f"artifact {label}: path does not exist: {path}")
            continue
        expected_sha = artifact.get("sha256")
        if isinstance(expected_sha, str) and expected_sha:
            actual_sha = sha256(Path(path))
            if actual_sha != expected_sha:
                errors.append(f"artifact {label}: sha256 mismatch: {actual_sha} != {expected_sha}")


def validate_reproduction_gap(report: dict[str, Any], errors: list[str]) -> None:
    if report.get("schema") != "bitnet-reproduction-gap-report-v1":
        errors.append(f"reproduction gap: unexpected schema {report.get('schema')}")
    if report.get("status") != "not_reproduced":
        errors.append(f"reproduction gap: unexpected status {report.get('status')}")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("reproduction gap: missing artifacts object")
        return
    for label, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            errors.append(f"reproduction gap artifact {label}: not an object")
            continue
        path = artifact.get("path")
        if not isinstance(path, str) or not path:
            errors.append(f"reproduction gap artifact {label}: missing path")
            continue
        if not Path(path).exists():
            errors.append(f"reproduction gap artifact {label}: path does not exist: {path}")
            continue
        expected_sha = artifact.get("sha256")
        if isinstance(expected_sha, str) and expected_sha:
            actual_sha = sha256(Path(path))
            if actual_sha != expected_sha:
                errors.append(
                    f"reproduction gap artifact {label}: sha256 mismatch: {actual_sha} != {expected_sha}"
                )


def validate_stage2_extension_submission(report: dict[str, Any], errors: list[str]) -> None:
    if report.get("schema") != "bitnet-stage2-extension-submission-v1":
        errors.append(f"stage2 extension: unexpected schema {report.get('schema')}")
    if report.get("status") not in {"pending", "running", "complete", "failed", "cancelled"}:
        errors.append(f"stage2 extension: unexpected status {report.get('status')}")
    parent = report.get("parent_manifest", {})
    if not isinstance(parent, dict):
        errors.append("stage2 extension: missing parent_manifest object")
        return
    parent_path = parent.get("path")
    if not isinstance(parent_path, str) or not parent_path:
        errors.append("stage2 extension: missing parent manifest path")
    elif not Path(parent_path).exists():
        errors.append(f"stage2 extension: parent manifest does not exist: {parent_path}")
    state_dict = parent.get("state_dict_path")
    if not isinstance(state_dict, str) or not state_dict:
        errors.append("stage2 extension: missing parent state_dict_path")
    elif not Path(state_dict).exists():
        errors.append(f"stage2 extension: parent state_dict_path does not exist: {state_dict}")
    config = report.get("run_config", {})
    if not isinstance(config, dict):
        errors.append("stage2 extension: missing run_config object")
        return
    segment = config.get("segment_token_presentations")
    cumulative = config.get("cumulative_token_presentations")
    parent_tokens = parent.get("token_presentations")
    if isinstance(segment, int) and isinstance(cumulative, int) and isinstance(parent_tokens, int):
        if cumulative != parent_tokens + segment:
            errors.append(
                "stage2 extension: cumulative tokens do not equal parent + segment: "
                f"{cumulative} != {parent_tokens} + {segment}"
            )
    else:
        errors.append("stage2 extension: token presentation fields must be integers")


def validate_stage2_handoff_submission(report: dict[str, Any], errors: list[str]) -> None:
    if report.get("schema") != "bitnet-stage2-extension-handoff-submission-v1":
        errors.append(f"stage2 handoff: unexpected schema {report.get('schema')}")
    if report.get("status") not in {"dependency_pending", "running", "submitted_downstream", "failed"}:
        errors.append(f"stage2 handoff: unexpected status {report.get('status')}")
    if report.get("dependency") != "afterok:10250":
        errors.append(f"stage2 handoff: unexpected dependency {report.get('dependency')}")
    script = report.get("script")
    if not isinstance(script, str) or not script:
        errors.append("stage2 handoff: missing script")
    elif not Path(script).exists():
        errors.append(f"stage2 handoff: script does not exist: {script}")
    postprocess_script = report.get("postprocess_script")
    if postprocess_script is not None and not Path(str(postprocess_script)).exists():
        errors.append(f"stage2 handoff: postprocess script does not exist: {postprocess_script}")
    if not report.get("producer_bitnet_commit"):
        errors.append("stage2 handoff: missing producer_bitnet_commit")
    if not report.get("producer_llama_cpp_commit"):
        errors.append("stage2 handoff: missing producer_llama_cpp_commit")


def validate_stage2_afterany_submission(report: dict[str, Any], readme: str, errors: list[str]) -> None:
    if report.get("schema") != "bitnet-stage2-afterany-submission-v1":
        errors.append(f"stage2 afterany: unexpected schema {report.get('schema')}")
    if report.get("quality_claim") != "none":
        errors.append(f"stage2 afterany: unexpected quality_claim {report.get('quality_claim')}")
    if report.get("status") not in {"dependency_pending", "running", "completed", "failed"}:
        errors.append(f"stage2 afterany: unexpected status {report.get('status')}")
    if report.get("dependency") != "afterany:10250":
        errors.append(f"stage2 afterany: unexpected dependency {report.get('dependency')}")
    if report.get("stage2_job_id") != "10250":
        errors.append(f"stage2 afterany: unexpected stage2 job {report.get('stage2_job_id')}")
    script = report.get("script")
    if script != "slurm_stage2_655m_afterany_audit.sh":
        errors.append(f"stage2 afterany: unexpected script {script}")
    elif not Path(script).exists():
        errors.append(f"stage2 afterany: script does not exist: {script}")
    for key in ("expected_report_json", "expected_salvage_json", "expected_ingestion_json", "expected_watchdog_json"):
        value = report.get(key)
        if not isinstance(value, str) or not value:
            errors.append(f"stage2 afterany: missing {key}")
    caveat = report.get("caveat")
    if not isinstance(caveat, str) or "does not create downstream quality evidence" not in caveat:
        errors.append("stage2 afterany: caveat must prohibit quality claims")


def validate_stage2_ingestion(report: dict[str, Any], readme: str, errors: list[str]) -> None:
    if report.get("schema") != "bitdistill-655m-ingestion-audit-v1":
        errors.append(f"stage2 ingestion: unexpected schema {report.get('schema')}")
    if report.get("quality_claim") != "none_until_complete_downstream_trace":
        errors.append(f"stage2 ingestion: unexpected quality_claim {report.get('quality_claim')}")
    allowed = {
        "pending_handoff",
        "handoff_failed",
        "downstream_pending_or_running",
        "downstream_incomplete",
        "downstream_complete_pending_report_ingestion",
        "ingested_reports_rebuilt",
    }
    status = report.get("status")
    if status not in allowed:
        errors.append(f"stage2 ingestion: unexpected status {status}")
    if report.get("target_stage2_tokens") != 655360000:
        errors.append(f"stage2 ingestion: unexpected target tokens {report.get('target_stage2_tokens')}")
    if report.get("consistency_errors") not in ([], None):
        errors.append(f"stage2 ingestion: consistency errors present: {report.get('consistency_errors')}")
    downstream = report.get("downstream")
    if not isinstance(downstream, dict):
        errors.append("stage2 ingestion: missing downstream object")
    else:
        metrics = downstream.get("metrics")
        predictions = downstream.get("predictions")
        if not isinstance(metrics, dict) or not isinstance(metrics.get("exists"), bool):
            errors.append("stage2 ingestion: malformed metrics file info")
        if not isinstance(predictions, dict) or not isinstance(predictions.get("exists"), bool):
            errors.append("stage2 ingestion: malformed predictions file info")
    require_contains(
        "README stage2 ingestion report",
        "stage2_655m_ingestion_2026-05-23.md",
        readme,
        errors,
    )


def validate_stage2_snapshot_salvage(report: dict[str, Any], readme: str, errors: list[str]) -> None:
    if report.get("schema") != "bitdistill-stage2-snapshot-salvage-v1":
        errors.append(f"stage2 snapshot salvage: unexpected schema {report.get('schema')}")
    if report.get("quality_claim") != "none":
        errors.append(f"stage2 snapshot salvage: unexpected quality_claim {report.get('quality_claim')}")
    allowed_statuses = {
        "no_snapshot_expected_yet",
        "waiting_for_snapshot",
        "salvage_available",
        "final_snapshot_available",
        "failed_no_salvage_snapshot",
        "invalid_snapshot_metadata",
    }
    status = report.get("status")
    if status not in allowed_statuses:
        errors.append(f"stage2 snapshot salvage: unexpected status {status}")
    if status == "invalid_snapshot_metadata":
        errors.append("stage2 snapshot salvage: invalid snapshot metadata present")
    if report.get("stage2_job_id") != "10250":
        errors.append(f"stage2 snapshot salvage: unexpected job id {report.get('stage2_job_id')}")
    if report.get("target_cumulative_token_presentations") != 655360000:
        errors.append(
            "stage2 snapshot salvage: unexpected target cumulative tokens "
            f"{report.get('target_cumulative_token_presentations')}"
        )
    snapshots = report.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != 4:
        errors.append("stage2 snapshot salvage: expected four 10k-step snapshot rows")
    else:
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                errors.append("stage2 snapshot salvage: snapshot row is not an object")
                continue
            if snapshot.get("validation_errors") not in ([], None):
                errors.append(
                    "stage2 snapshot salvage: snapshot validation errors present: "
                    f"{snapshot.get('validation_errors')}"
                )
            state = snapshot.get("state")
            metrics = snapshot.get("metrics_file")
            if not isinstance(state, dict) or not isinstance(metrics, dict):
                errors.append("stage2 snapshot salvage: malformed snapshot file info")
    if status == "no_snapshot_expected_yet":
        latest_step = report.get("latest_logged_step")
        save_every_steps = report.get("save_every_steps")
        if isinstance(latest_step, int) and isinstance(save_every_steps, int) and latest_step >= save_every_steps:
            errors.append("stage2 snapshot salvage: no_snapshot_expected_yet after save interval")
    complete_count = report.get("complete_snapshot_count")
    if not isinstance(complete_count, int):
        errors.append("stage2 snapshot salvage: complete_snapshot_count must be an integer")
    best = report.get("best_salvage_snapshot")
    if complete_count and not isinstance(best, dict):
        errors.append("stage2 snapshot salvage: complete snapshots exist but best_salvage_snapshot is missing")
    salvage_command = report.get("salvage_manifest_command")
    if complete_count:
        if not isinstance(salvage_command, list) or "--allow-snapshot-metrics-root" not in salvage_command:
            errors.append("stage2 snapshot salvage: complete snapshot lacks snapshot-metrics manifest command")
    elif salvage_command not in ([], None):
        errors.append("stage2 snapshot salvage: command present before any complete snapshot exists")
    caveat = report.get("caveat")
    if not isinstance(caveat, str) or "does not run downstream evaluation" not in caveat:
        errors.append("stage2 snapshot salvage: caveat must prohibit quality claims")


def validate_stage2_handoff_preflight(report: dict[str, Any], readme: str, errors: list[str]) -> None:
    if report.get("schema") != "bitdistill-655m-handoff-preflight-v1":
        errors.append(f"stage2 handoff preflight: unexpected schema {report.get('schema')}")
    if report.get("quality_claim") != "none":
        errors.append(f"stage2 handoff preflight: unexpected quality_claim {report.get('quality_claim')}")
    allowed_statuses = {
        "pending_stage2_completion",
        "final_artifacts_ready_pending_dry_run",
        "ready_for_handoff",
        "failed_preflight",
        "failed_manifest_dry_run",
        "failed_missing_final_snapshot",
    }
    status = report.get("status")
    if status not in allowed_statuses:
        errors.append(f"stage2 handoff preflight: unexpected status {status}")
    if status and str(status).startswith("failed"):
        errors.append(f"stage2 handoff preflight: failed status {status}")
    if report.get("stage2_job_id") != "10250":
        errors.append(f"stage2 handoff preflight: unexpected job id {report.get('stage2_job_id')}")
    checks = report.get("preflight_checks")
    if not isinstance(checks, list) or not checks:
        errors.append("stage2 handoff preflight: missing preflight checks")
    else:
        for check in checks:
            if not isinstance(check, dict):
                errors.append("stage2 handoff preflight: malformed check row")
                continue
            if check.get("required_now") is True and check.get("passed") is not True:
                errors.append(f"stage2 handoff preflight: required check failed: {check.get('label')}")
    final_checks = report.get("final_artifact_checks")
    if not isinstance(final_checks, list) or len(final_checks) < 3:
        errors.append("stage2 handoff preflight: expected final artifact checks")
    save_contract = report.get("training_save_contract")
    if not isinstance(save_contract, dict):
        errors.append("stage2 handoff preflight: missing training save contract")
    elif save_contract.get("passed") is not True:
        errors.append("stage2 handoff preflight: training save contract failed")
    command = report.get("expected_manifest_command")
    if not isinstance(command, list) or "benchmarks/build_stage2_manifest.py" not in command:
        errors.append("stage2 handoff preflight: missing build_stage2_manifest command")
    if "benchmarks/results/stage2_manifest_655m_2026-05-23.json" not in command:
        errors.append("stage2 handoff preflight: command does not target 655M manifest")
    caveat = report.get("caveat")
    if not isinstance(caveat, str) or "does not run downstream evaluation" not in caveat:
        errors.append("stage2 handoff preflight: caveat must prohibit quality claims")


def validate_gradient_telemetry_submission(report: dict[str, Any], errors: list[str]) -> None:
    if report.get("schema") != "bitdistill-gradient-telemetry-submission-v1":
        errors.append(f"gamma telemetry: unexpected schema {report.get('schema')}")
    if report.get("status") not in {"dependency_pending", "running", "complete", "failed"}:
        errors.append(f"gamma telemetry: unexpected status {report.get('status')}")
    if report.get("dependency") != "afterok:10250":
        errors.append(f"gamma telemetry: unexpected dependency {report.get('dependency')}")
    script = report.get("script")
    if script != "slurm_gamma60_telemetry.sh":
        errors.append(f"gamma telemetry: unexpected script {script}")
    elif not Path(script).exists():
        errors.append(f"gamma telemetry: script does not exist: {script}")
    config = report.get("run_config", {})
    if not isinstance(config, dict):
        errors.append("gamma telemetry: missing run_config object")
        return
    if config.get("attention_kd_weight") != 60:
        errors.append(f"gamma telemetry: expected attention_kd_weight 60, got {config.get('attention_kd_weight')}")
    if config.get("max_steps") != 200:
        errors.append(f"gamma telemetry: expected max_steps 200, got {config.get('max_steps')}")
    if config.get("telemetry_component_grad_norms") is not True:
        errors.append("gamma telemetry: component gradient telemetry is not enabled")
    output_dir = config.get("output_dir")
    if not isinstance(output_dir, str) or "telemetry-gamma60" not in output_dir:
        errors.append(f"gamma telemetry: unexpected output_dir {output_dir}")
    target = report.get("comparison_target", {})
    if not isinstance(target, dict):
        errors.append("gamma telemetry: missing comparison_target object")
    else:
        existing_report = target.get("existing_report")
        if not isinstance(existing_report, str) or not Path(existing_report).exists():
            errors.append(f"gamma telemetry: comparison report does not exist: {existing_report}")
    caveat = report.get("caveat")
    if not isinstance(caveat, str) or "not a quality benchmark" not in caveat:
        errors.append("gamma telemetry: missing non-quality-benchmark caveat")


def validate_active_slurm_batch_scripts(report: dict[str, Any], errors: list[str]) -> None:
    if report.get("schema") != "bitnet-active-slurm-batch-script-audit-v1":
        errors.append(f"slurm batch audit: unexpected schema {report.get('schema')}")
    if report.get("quality_claim") != "none":
        errors.append(f"slurm batch audit: quality_claim must be none, got {report.get('quality_claim')}")
    if report.get("status") != "passed":
        errors.append(f"slurm batch audit: unexpected status {report.get('status')}")
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        errors.append("slurm batch audit: missing checks")
        return
    by_purpose = {
        str(check.get("purpose")): check
        for check in checks
        if isinstance(check, dict) and check.get("purpose") is not None
    }
    for purpose in ("655M Stage-2 handoff", "gamma-60 gradient telemetry", "655M Stage-2 afterany audit"):
        check = by_purpose.get(purpose)
        if not isinstance(check, dict):
            errors.append(f"slurm batch audit: missing purpose {purpose}")
            continue
        if check.get("passed") is not True:
            errors.append(f"slurm batch audit: {purpose} did not pass")
        snippets = check.get("checks")
        if not isinstance(snippets, list) or not snippets:
            errors.append(f"slurm batch audit: {purpose} missing snippet checks")
            continue
        missing = [
            snippet.get("snippet")
            for snippet in snippets
            if isinstance(snippet, dict) and snippet.get("present") is not True
        ]
        if missing:
            errors.append(f"slurm batch audit: {purpose} missing snippets: {missing}")
    dependency_checks = report.get("dependency_checks")
    if not isinstance(dependency_checks, list) or not dependency_checks:
        errors.append("slurm batch audit: missing dependency checks")
    else:
        by_dependency_purpose = {
            str(check.get("purpose")): check
            for check in dependency_checks
            if isinstance(check, dict) and check.get("purpose") is not None
        }
        expected = {
            "655M Stage-2 handoff dependency": ("afterok:10250", "slurm_stage2_655m_handoff.sh"),
            "gamma-60 telemetry dependency": ("afterok:10250", "slurm_gamma60_telemetry.sh"),
            "655M Stage-2 afterany dependency": ("afterany:10250", "slurm_stage2_655m_afterany_audit.sh"),
        }
        for purpose, (dependency, command_suffix) in expected.items():
            check = by_dependency_purpose.get(purpose)
            if not isinstance(check, dict):
                errors.append(f"slurm batch audit: missing dependency purpose {purpose}")
                continue
            if check.get("passed") is not True:
                errors.append(f"slurm batch audit: dependency check failed for {purpose}")
            if check.get("expected_dependency") != dependency:
                errors.append(
                    f"slurm batch audit: {purpose} expected dependency {check.get('expected_dependency')} != {dependency}"
                )
            if check.get("expected_command_suffix") != command_suffix:
                errors.append(
                    f"slurm batch audit: {purpose} expected command suffix "
                    f"{check.get('expected_command_suffix')} != {command_suffix}"
                )
            # Slurm clears dependency metadata once a job starts and may age the
            # full command out of scontrol. Exact scheduler fields are therefore
            # required only while the dependency is still pending; completed jobs
            # remain covered by the audited local script and submission receipt.
            scheduler_state = check.get("slurm", {}).get("state")
            job_state = check.get("job_state")
            if scheduler_state == "PENDING" or job_state == "PENDING":
                if check.get("normalized_dependency") != dependency:
                    errors.append(
                        f"slurm batch audit: {purpose} actual dependency "
                        f"{check.get('normalized_dependency')} != {dependency}"
                    )
                if check.get("command_matches") is not True:
                    errors.append(f"slurm batch audit: {purpose} command suffix did not match")


def validate_active_stage2_monitor(report: dict[str, Any], errors: list[str]) -> None:
    if report.get("schema") != "bitnet-active-stage2-extension-monitor-v1":
        errors.append(f"stage2 monitor: unexpected schema {report.get('schema')}")
    if report.get("quality_claim") != "none":
        errors.append(f"stage2 monitor: quality_claim must be none, got {report.get('quality_claim')}")
    stage2 = report.get("stage2", {})
    if not isinstance(stage2, dict):
        errors.append("stage2 monitor: missing stage2 object")
        return
    if stage2.get("job_id") != "10250":
        errors.append(f"stage2 monitor: unexpected stage2 job {stage2.get('job_id')}")
    if stage2.get("cumulative_token_presentations") != 655360000:
        errors.append(
            "stage2 monitor: unexpected cumulative tokens "
            f"{stage2.get('cumulative_token_presentations')}"
        )
    log_freshness = stage2.get("log_freshness")
    if not isinstance(log_freshness, dict):
        errors.append("stage2 monitor: missing log_freshness object")
    else:
        allowed_log_statuses = {
            "missing_log_running",
            "missing_log_not_running",
            "not_running",
            "stale_running_log",
            "fresh_running_log",
        }
        log_status = log_freshness.get("status")
        if log_status not in allowed_log_statuses:
            errors.append(f"stage2 monitor: unexpected log freshness status {log_status}")
        if log_status in {"missing_log_running", "stale_running_log"}:
            errors.append(f"stage2 monitor: unhealthy running log status {log_status}")
        if log_freshness.get("exists") is not True and log_status == "fresh_running_log":
            errors.append("stage2 monitor: fresh_running_log without existing log")
        if not isinstance(log_freshness.get("stale_after_seconds"), int):
            errors.append("stage2 monitor: log_freshness stale_after_seconds must be an integer")
        age_seconds = log_freshness.get("age_seconds")
        stale_after = log_freshness.get("stale_after_seconds")
        if log_status == "fresh_running_log" and isinstance(age_seconds, (int, float)) and isinstance(stale_after, int):
            if float(age_seconds) > float(stale_after):
                errors.append("stage2 monitor: fresh_running_log has age beyond stale threshold")
        caveat = log_freshness.get("caveat")
        if not isinstance(caveat, str) or "Fresh logs are required" not in caveat:
            errors.append("stage2 monitor: log_freshness caveat must explain freshness requirement")
    log_health = stage2.get("log_health")
    if not isinstance(log_health, dict):
        errors.append("stage2 monitor: missing log_health object")
    else:
        log_health_status = log_health.get("status")
        allowed_log_health_statuses = {"missing_log", "no_steps", "healthy", "unhealthy"}
        if log_health_status not in allowed_log_health_statuses:
            errors.append(f"stage2 monitor: unexpected log_health status {log_health_status}")
        if report.get("status") == "running" and log_health_status != "healthy":
            errors.append(f"stage2 monitor: running producer log health is not healthy: {log_health_status}")
        if not isinstance(log_health.get("parsed_step_rows"), int):
            errors.append("stage2 monitor: log_health parsed_step_rows must be an integer")
        issues = log_health.get("issues")
        if not isinstance(issues, list):
            errors.append("stage2 monitor: log_health issues must be a list")
        elif issues:
            errors.append(f"stage2 monitor: log_health issues present: {issues}")
        fatal_matches = log_health.get("fatal_matches")
        if not isinstance(fatal_matches, list):
            errors.append("stage2 monitor: log_health fatal_matches must be a list")
        elif fatal_matches:
            errors.append(f"stage2 monitor: log_health fatal matches present: {fatal_matches}")
        checks = log_health.get("checks")
        if log_health_status == "healthy":
            if not isinstance(checks, dict):
                errors.append("stage2 monitor: healthy log_health must include checks")
            else:
                for key in (
                    "has_step_rows",
                    "steps_monotonic",
                    "elapsed_monotonic",
                    "finite_numeric_values",
                    "latest_step_within_max_steps",
                ):
                    if checks.get(key) is not True:
                        errors.append(f"stage2 monitor: log_health check failed {key}")
                constant_lr = checks.get("constant_lr_matches_expected")
                if constant_lr not in (True, None):
                    errors.append("stage2 monitor: log_health constant LR check failed")
        caveat = log_health.get("caveat")
        if not isinstance(caveat, str) or "not quality evidence" not in caveat:
            errors.append("stage2 monitor: log_health caveat must prohibit quality claims")
    producer_config = stage2.get("producer_config")
    if not isinstance(producer_config, dict):
        errors.append("stage2 monitor: missing producer_config object")
    else:
        producer_status = producer_config.get("status")
        allowed_producer_statuses = {"missing_log", "missing_header", "mismatched", "matched"}
        if producer_status not in allowed_producer_statuses:
            errors.append(f"stage2 monitor: unexpected producer_config status {producer_status}")
        if report.get("status") == "running" and producer_status != "matched":
            errors.append(f"stage2 monitor: running producer config is not matched: {producer_status}")
        mismatches = producer_config.get("mismatches")
        if not isinstance(mismatches, list):
            errors.append("stage2 monitor: producer_config mismatches must be a list")
        elif mismatches:
            errors.append(f"stage2 monitor: producer_config mismatches present: {mismatches}")
        checks = producer_config.get("checks")
        if producer_status == "matched":
            if not isinstance(checks, list) or not checks:
                errors.append("stage2 monitor: matched producer_config must include checks")
            else:
                by_key = {
                    str(check.get("key")): check
                    for check in checks
                    if isinstance(check, dict) and check.get("key") is not None
                }
                for key in (
                    "SLURM_JOB_ID",
                    "MODEL",
                    "STAGE",
                    "METHOD",
                    "INIT_STATE_MANIFEST",
                    "INIT_STATE_DICT",
                    "SCALE_MODE",
                    "ACTIVATION_QUANTIZATION",
                    "USE_SUBLN",
                    "MAX_SEQ_LEN",
                    "MAX_STEPS",
                    "LR",
                    "LR_SCHEDULER",
                    "SAVE_MODEL_ARTIFACTS",
                    "OUTPUT_DIR",
                ):
                    check = by_key.get(key)
                    if not isinstance(check, dict):
                        errors.append(f"stage2 monitor: producer_config missing check {key}")
                    elif check.get("matched") is not True:
                        errors.append(f"stage2 monitor: producer_config check did not match {key}")
        log_header = producer_config.get("log_header")
        if not isinstance(log_header, dict):
            errors.append("stage2 monitor: producer_config missing log_header")
        else:
            values = log_header.get("values")
            if producer_status == "matched" and not isinstance(values, dict):
                errors.append("stage2 monitor: matched producer_config missing header values")
        caveat = producer_config.get("caveat")
        if not isinstance(caveat, str) or "producer log header" not in caveat:
            errors.append("stage2 monitor: producer_config caveat must explain producer log-header scope")
    time_limit_gate = stage2.get("time_limit_gate")
    if not isinstance(time_limit_gate, dict):
        errors.append("stage2 monitor: missing time_limit_gate object")
    else:
        allowed_time_statuses = {
            "not_running",
            "unknown",
            "likely_walltime_failure",
            "tight_walltime_margin",
            "within_time_limit",
        }
        time_status = time_limit_gate.get("status")
        if time_status not in allowed_time_statuses:
            errors.append(f"stage2 monitor: unexpected time-limit status {time_status}")
        if time_status == "likely_walltime_failure":
            errors.append("stage2 monitor: estimated completion exceeds Slurm time remaining")
        for field in ("elapsed_seconds", "time_limit_seconds", "eta_seconds", "remaining_seconds", "margin_seconds"):
            value = time_limit_gate.get(field)
            if time_status == "within_time_limit" and not isinstance(value, (int, float)):
                errors.append(f"stage2 monitor: time_limit_gate {field} must be numeric")
        if time_status == "within_time_limit" and float(time_limit_gate.get("margin_seconds", -1)) <= 0:
            errors.append("stage2 monitor: within_time_limit has non-positive margin")
        caveat = time_limit_gate.get("caveat")
        if not isinstance(caveat, str) or "runtime-risk signal" not in caveat:
            errors.append("stage2 monitor: time_limit_gate caveat must explain runtime-risk scope")
    latest_step_obj = stage2.get("latest_step")
    latest_step = latest_step_obj.get("step") if isinstance(latest_step_obj, dict) else None
    save_every_steps = stage2.get("save_every_steps")
    snapshot_status = stage2.get("snapshot_status")
    if not isinstance(snapshot_status, dict):
        errors.append("stage2 monitor: missing snapshot_status object")
    else:
        allowed_snapshot_statuses = {
            "log_not_parsed",
            "snapshots_disabled",
            "pre_first_snapshot",
            "snapshots_present",
            "snapshot_due_missing",
            "unknown",
        }
        status = snapshot_status.get("status")
        if status not in allowed_snapshot_statuses:
            errors.append(f"stage2 monitor: unexpected snapshot status {status}")
        if status == "snapshot_due_missing":
            errors.append("stage2 monitor: snapshot is due but missing")
        if snapshot_status.get("save_every_steps") != save_every_steps:
            errors.append(
                "stage2 monitor: snapshot_status save_every_steps mismatch "
                f"{snapshot_status.get('save_every_steps')} != {save_every_steps}"
            )
        if isinstance(latest_step, int) and isinstance(save_every_steps, int) and save_every_steps > 0:
            if latest_step < save_every_steps and status != "pre_first_snapshot":
                errors.append(
                    "stage2 monitor: latest step is before first snapshot but status is "
                    f"{status}"
                )
            if latest_step >= save_every_steps and status == "pre_first_snapshot":
                errors.append("stage2 monitor: pre_first_snapshot status after first snapshot step")
        if not isinstance(snapshot_status.get("missing_output_dir_is_expected"), bool):
            errors.append("stage2 monitor: snapshot_status missing_output_dir_is_expected must be boolean")
        caveat = snapshot_status.get("caveat")
        if not isinstance(caveat, str) or "missing output directory is expected before the first snapshot" not in caveat:
            errors.append("stage2 monitor: snapshot_status caveat must explain pre-first-snapshot behavior")
    downstream = report.get("downstream")
    if not isinstance(downstream, dict):
        errors.append("stage2 monitor: missing downstream object")
        return
    allowed_statuses = {
        "waiting_for_handoff",
        "handoff_failed",
        "pending",
        "running",
        "slurm_completed",
        "slurm_failed",
        "submitted_downstream_not_in_squeue_incomplete",
        "not_submitted_incomplete",
        "complete_artifacts_present",
    }
    if downstream.get("status") not in allowed_statuses:
        errors.append(f"stage2 monitor: unexpected downstream status {downstream.get('status')}")
    output_dir = downstream.get("output_dir")
    if not isinstance(output_dir, str) or "bitdistill-tensor-655mwarmup" not in output_dir:
        errors.append(f"stage2 monitor: unexpected downstream output_dir {output_dir}")
        return
    metrics = downstream.get("metrics")
    predictions = downstream.get("predictions")
    if not isinstance(metrics, dict):
        errors.append("stage2 monitor: downstream metrics is not an object")
    elif metrics.get("path") != f"{output_dir}/metrics.json":
        errors.append(f"stage2 monitor: unexpected downstream metrics path {metrics.get('path')}")
    if not isinstance(predictions, dict):
        errors.append("stage2 monitor: downstream predictions is not an object")
    elif predictions.get("path") != f"{output_dir}/eval_predictions.jsonl":
        errors.append(f"stage2 monitor: unexpected downstream predictions path {predictions.get('path')}")
    complete = downstream.get("complete")
    if complete is True:
        if not isinstance(metrics, dict) or metrics.get("exists") is not True:
            errors.append("stage2 monitor: downstream marked complete without metrics")
        if not isinstance(predictions, dict) or predictions.get("exists") is not True:
            errors.append("stage2 monitor: downstream marked complete without predictions")
    elif complete is not False:
        errors.append(f"stage2 monitor: downstream complete must be boolean, got {complete}")
    caveat = downstream.get("caveat")
    if not isinstance(caveat, str) or "does not compute or claim MNLI accuracy" not in caveat:
        errors.append("stage2 monitor: downstream caveat must prohibit quality claims")
    forbidden_quality_keys = {"accuracy", "mnli_accuracy", "quality", "score"}
    present_forbidden = sorted(forbidden_quality_keys.intersection(downstream.keys()))
    if present_forbidden:
        errors.append(f"stage2 monitor: downstream contains quality-like keys {present_forbidden}")
    postprocess = report.get("postprocess")
    if not isinstance(postprocess, dict):
        errors.append("stage2 monitor: missing postprocess object")
    else:
        caveat = postprocess.get("caveat")
        if not isinstance(caveat, str) or "not quality evidence" not in caveat:
            errors.append("stage2 monitor: postprocess caveat must prohibit quality claims")


def validate_current_goal_status(report: dict[str, Any], errors: list[str]) -> None:
    if report.get("schema") != "bitnet-current-goal-status-v1":
        errors.append(f"current goal status: unexpected schema {report.get('schema')}")
    if report.get("objective_achieved") is not False:
        errors.append("current goal status: objective_achieved must remain false")
    if report.get("completion_status") != "in_progress":
        errors.append(f"current goal status: unexpected completion_status {report.get('completion_status')}")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("current goal status: missing artifacts object")
        return
    for label in ("canonical_bundle", "reproduction_gap", "active_monitor"):
        artifact = artifacts.get(label)
        if not isinstance(artifact, dict):
            errors.append(f"current goal status: missing artifact {label}")
            continue
        path = artifact.get("path")
        if not isinstance(path, str) or not path:
            errors.append(f"current goal status artifact {label}: missing path")
            continue
        if not Path(path).exists():
            errors.append(f"current goal status artifact {label}: path does not exist: {path}")
            continue
        expected_sha = artifact.get("sha256")
        if isinstance(expected_sha, str) and expected_sha:
            actual_sha = sha256(Path(path))
            if actual_sha != expected_sha:
                errors.append(f"current goal status artifact {label}: sha256 mismatch: {actual_sha} != {expected_sha}")
    if report.get("headline_metrics", {}).get("bitdistill_655_36m_status") is None:
        errors.append("current goal status: missing 655.36M status headline")
    active_gate = report.get("active_gate")
    if not isinstance(active_gate, dict):
        errors.append("current goal status: missing active_gate object")
    else:
        required_active_fields = {
            "producer_config_status",
            "log_health_status",
            "snapshot_salvage_status",
            "snapshot_salvage_complete_count",
            "afterany_job_id",
            "afterany_status",
            "afterany_dependency",
            "time_limit_status",
        }
        missing_active = sorted(required_active_fields.difference(active_gate))
        if missing_active:
            errors.append(f"current goal status: missing active gate fields {missing_active}")
        if active_gate.get("producer_config_status") not in {"matched", "missing_log", "missing_header", "mismatched", None}:
            errors.append(
                "current goal status: unexpected producer_config_status "
                f"{active_gate.get('producer_config_status')}"
            )
        if active_gate.get("log_health_status") not in {"healthy", "missing_log", "no_steps", "unhealthy", None}:
            errors.append(f"current goal status: unexpected log_health_status {active_gate.get('log_health_status')}")
        if active_gate.get("afterany_dependency") not in {"afterany:10250", None}:
            errors.append(f"current goal status: unexpected afterany dependency {active_gate.get('afterany_dependency')}")
    requirements = report.get("requirements")
    if not isinstance(requirements, list) or len(requirements) < 5:
        errors.append("current goal status: missing requirement audit rows")
    else:
        names = {str(row.get("requirement")) for row in requirements if isinstance(row, dict)}
        if not {
            "Active 655M evidence-chain guardrails",
            "655M evidence-chain guardrails",
        }.intersection(names):
            errors.append("current goal status: missing 655M evidence-chain guardrail requirement")


def validate_deep_research_handoff(report: dict[str, Any], errors: list[str]) -> None:
    if report.get("schema") != "bitnet-deep-research-handoff-v1":
        errors.append(f"deep research handoff: unexpected schema {report.get('schema')}")
    if report.get("status") != "handoff_not_completion":
        errors.append(f"deep research handoff: unexpected status {report.get('status')}")
    thesis = report.get("thesis")
    if not isinstance(thesis, dict):
        errors.append("deep research handoff: missing thesis object")
    elif "No for the tested dense-Qwen setup" not in str(thesis.get("current_answer", "")):
        errors.append("deep research handoff: thesis does not preserve the negative PTQ answer")
    findings = report.get("completed_findings")
    if not isinstance(findings, list) or len(findings) < 5:
        errors.append("deep research handoff: missing completed findings")
    open_questions = report.get("open_questions")
    if not isinstance(open_questions, list) or len(open_questions) < 4:
        errors.append("deep research handoff: missing open questions")
    next_action = report.get("next_action")
    if not isinstance(next_action, dict):
        errors.append("deep research handoff: missing next_action")
    else:
        if next_action.get("decision_status") != "run_gamma_balanced_downstream":
            errors.append(f"deep research handoff: unexpected next-action status {next_action.get('decision_status')}")
        if next_action.get("blueprint_action") != "run_matched_gamma60_mnli_downstream":
            errors.append(f"deep research handoff: unexpected blueprint action {next_action.get('blueprint_action')}")
        if "single MNLI ablation" not in str(next_action.get("claim_boundary", "")):
            errors.append("deep research handoff: next-action boundary must remain a single MNLI ablation")
    nonclaims = report.get("nonclaims")
    if not isinstance(nonclaims, list) or "universal BitNet converter" not in nonclaims:
        errors.append("deep research handoff: missing universal-converter nonclaim")
    artifacts = report.get("source_artifacts")
    if not isinstance(artifacts, dict):
        errors.append("deep research handoff: missing source artifacts")
        return
    for label in ("current_status", "canonical_bundle", "reproduction_gap"):
        artifact = artifacts.get(label)
        if not isinstance(artifact, dict):
            errors.append(f"deep research handoff: missing artifact {label}")
            continue
        path = artifact.get("path")
        if not isinstance(path, str) or not path:
            errors.append(f"deep research handoff artifact {label}: missing path")
            continue
        if not Path(path).exists():
            errors.append(f"deep research handoff artifact {label}: path does not exist: {path}")
            continue
        expected_sha = artifact.get("sha256")
        if isinstance(expected_sha, str) and expected_sha:
            actual_sha = sha256(Path(path))
            if actual_sha != expected_sha:
                errors.append(f"deep research handoff artifact {label}: sha256 mismatch: {actual_sha} != {expected_sha}")


def validate_benchmark_scoreboard(report: dict[str, Any], readme: str, errors: list[str]) -> None:
    if report.get("schema") != "bitdistill-benchmark-scoreboard-v1":
        errors.append(f"benchmark scoreboard: unexpected schema {report.get('schema')}")
    if report.get("quality_claim") != "scoreboard_from_existing_artifacts_not_new_benchmark":
        errors.append(f"benchmark scoreboard: unexpected quality_claim {report.get('quality_claim')}")
    if report.get("status") != "mixed_supported_and_blocked":
        errors.append(f"benchmark scoreboard: unexpected status {report.get('status')}")
    coverage = report.get("coverage")
    if not isinstance(coverage, dict):
        errors.append("benchmark scoreboard: missing coverage object")
    else:
        if coverage.get("quality_benchmark_count") != 12:
            errors.append(f"benchmark scoreboard: expected 12 quality benchmarks, got {coverage.get('quality_benchmark_count')}")
        if coverage.get("lm_eval_task_count") != 10:
            errors.append(f"benchmark scoreboard: expected 10 lm-eval tasks, got {coverage.get('lm_eval_task_count')}")
        if coverage.get("coverage_gate_passed") is not True:
            errors.append("benchmark scoreboard: coverage gate is not passed")
        if coverage.get("coverage_failed") not in ([], None):
            errors.append(f"benchmark scoreboard: coverage failed checks present: {coverage.get('coverage_failed')}")
    rows = report.get("headline_rows")
    if not isinstance(rows, list) or len(rows) < 10:
        errors.append("benchmark scoreboard: missing headline rows")
    else:
        areas = {str(row.get("area")) for row in rows if isinstance(row, dict)}
        for area in ("Blind ternary PTQ", "BitDistill reproduction", "Packed CPU I2_SR", "MoE / Kimi"):
            if area not in areas:
                errors.append(f"benchmark scoreboard: missing area {area}")
    nonclaims = report.get("nonclaims")
    if not isinstance(nonclaims, list) or len(nonclaims) < 5:
        errors.append("benchmark scoreboard: missing nonclaims")
    require_contains(
        "README benchmark scoreboard report",
        "bitdistill_benchmark_scoreboard_2026-05-23.md",
        readme,
        errors,
    )
    require_contains(
        "README benchmark scoreboard json",
        "bitdistill_benchmark_scoreboard_2026-05-23.json",
        readme,
        errors,
    )


def validate_goal_traceability(report: dict[str, Any], readme: str, errors: list[str]) -> None:
    if report.get("schema") != "bitdistill-goal-traceability-audit-v1":
        errors.append(f"goal traceability: unexpected schema {report.get('schema')}")
    if report.get("quality_claim") != "traceability_from_existing_artifacts_not_new_benchmark":
        errors.append(f"goal traceability: unexpected quality_claim {report.get('quality_claim')}")
    if report.get("objective_achieved") is not False:
        errors.append("goal traceability: objective_achieved must be false")
    if report.get("completion_status") != "in_progress":
        errors.append(f"goal traceability: unexpected completion_status {report.get('completion_status')}")
    requirements = report.get("requirements")
    if not isinstance(requirements, list) or len(requirements) < 10:
        errors.append("goal traceability: missing requirement rows")
    else:
        names = {str(row.get("requirement")) for row in requirements if isinstance(row, dict)}
        for name in (
            "Post-training ternary math audit",
            "Stage-2 continued pretraining",
            "Stage-3 downstream CE + logits KL + attention-relation KD",
            "MoE/Kimi feasibility",
            "Publishable framing",
        ):
            if name not in names:
                errors.append(f"goal traceability: missing requirement {name}")
    source_checks = report.get("source_checks")
    if not isinstance(source_checks, list) or not source_checks:
        errors.append("goal traceability: missing source checks")
    else:
        failed = [row.get("label") for row in source_checks if isinstance(row, dict) and row.get("passed") is not True]
        if failed:
            errors.append(f"goal traceability: failed source checks {failed}")
    require_contains(
        "README goal traceability report",
        "bitdistill_goal_traceability_2026-05-23.md",
        readme,
        errors,
    )
    require_contains(
        "README goal traceability json",
        "bitdistill_goal_traceability_2026-05-23.json",
        readme,
        errors,
    )


def validate_publication_product_plan(report: dict[str, Any], readme: str, errors: list[str]) -> None:
    if report.get("schema") != "bitdistill-publication-product-plan-v1":
        errors.append(f"publication/product plan: unexpected schema {report.get('schema')}")
    if report.get("quality_claim") != "planning_from_existing_artifacts_not_new_benchmark":
        errors.append(f"publication/product plan: unexpected quality_claim {report.get('quality_claim')}")
    if report.get("status") != "research_mvp_with_open_quality_gate":
        errors.append(f"publication/product plan: unexpected status {report.get('status')}")
    product = report.get("product_mvp")
    if not isinstance(product, dict) or product.get("name") != "CPU-first ternary retrofit evaluator":
        errors.append("publication/product plan: missing product MVP framing")
    units = report.get("publishable_units")
    if not isinstance(units, list) or len(units) < 5:
        errors.append("publication/product plan: missing publishable units")
    else:
        names = {str(row.get("unit")) for row in units if isinstance(row, dict)}
        for name in ("Negative PTQ result", "Row-scale runtime contract", "BitDistill reproduction gap", "MoE/Kimi"):
            if name not in names:
                errors.append(f"publication/product plan: missing unit {name}")
    rules = report.get("claim_rules")
    if not isinstance(rules, dict):
        errors.append("publication/product plan: missing claim rules")
    else:
        avoid = rules.get("avoid")
        if not isinstance(avoid, list) or "universal converter" not in avoid:
            errors.append("publication/product plan: missing universal-converter avoid rule")
    require_contains(
        "README publication/product plan report",
        "bitdistill_publication_product_plan_2026-05-23.md",
        readme,
        errors,
    )
    require_contains(
        "README publication/product plan json",
        "bitdistill_publication_product_plan_2026-05-23.json",
        readme,
        errors,
    )


def validate_paper_alignment(report: dict[str, Any], readme: str, errors: list[str]) -> None:
    if report.get("schema") != "bitdistill-paper-alignment-audit-v1":
        errors.append(f"paper alignment: unexpected schema {report.get('schema')}")
    if report.get("quality_claim") != "paper_alignment_not_new_benchmark":
        errors.append(f"paper alignment: unexpected quality_claim {report.get('quality_claim')}")
    if report.get("status") != "not_exact_reproduction":
        errors.append(f"paper alignment: unexpected status {report.get('status')}")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) < 12:
        errors.append("paper alignment: missing alignment rows")
    else:
        axes = {str(row.get("axis")) for row in rows if isinstance(row, dict)}
        for axis in ("Stage-2 token budget", "Stage-2 corpus", "Attention-relation coefficient", "Success criterion"):
            if axis not in axes:
                errors.append(f"paper alignment: missing axis {axis}")
    risks = report.get("highest_risks")
    if not isinstance(risks, list) or len(risks) < 4:
        errors.append("paper alignment: missing highest-risk mismatch list")
    require_contains(
        "README paper alignment report",
        "bitdistill_paper_alignment_2026-05-23.md",
        readme,
        errors,
    )
    require_contains(
        "README paper alignment json",
        "bitdistill_paper_alignment_2026-05-23.json",
        readme,
        errors,
    )


def validate_active_gate_watchdog(report: dict[str, Any], readme: str, errors: list[str]) -> None:
    if report.get("schema") != "bitdistill-active-gate-watchdog-v1":
        errors.append(f"active gate watchdog: unexpected schema {report.get('schema')}")
    if report.get("quality_claim") != "none":
        errors.append(f"active gate watchdog: unexpected quality_claim {report.get('quality_claim')}")
    if report.get("status") != "passed":
        errors.append(f"active gate watchdog: unexpected status {report.get('status')}")
    commands = report.get("commands")
    if not isinstance(commands, list) or len(commands) < 6:
        errors.append("active gate watchdog: missing command rows")
    else:
        failed = [row.get("label") for row in commands if isinstance(row, dict) and row.get("passed") is not True]
        if failed:
            errors.append(f"active gate watchdog: failed commands {failed}")
    summary = report.get("summary")
    if not isinstance(summary, dict) or summary.get("ingestion_status") is None:
        errors.append("active gate watchdog: missing status summary")
    require_contains(
        "README active gate watchdog report",
        "active_gate_watchdog_2026-05-23.md",
        readme,
        errors,
    )
    require_contains(
        "README active gate watchdog json",
        "active_gate_watchdog_2026-05-23.json",
        readme,
        errors,
    )


def validate_next_experiment_blueprint(report: dict[str, Any], readme: str, errors: list[str]) -> None:
    if report.get("schema") != "bitdistill-next-experiment-blueprint-v1":
        errors.append(f"next experiment blueprint: unexpected schema {report.get('schema')}")
    if report.get("quality_claim") != "experiment_blueprint_not_benchmark":
        errors.append(f"next experiment blueprint: unexpected quality_claim {report.get('quality_claim')}")
    if report.get("status") != "run_gamma_balanced_downstream":
        errors.append(f"next experiment blueprint: unexpected status {report.get('status')}")
    current = report.get("current_action")
    if not isinstance(current, dict):
        errors.append("next experiment blueprint: missing current_action")
    else:
        if current.get("action") != "run_matched_gamma60_mnli_downstream":
            errors.append(f"next experiment blueprint: unexpected current action {current.get('action')}")
        commands = current.get("commands")
        if not isinstance(commands, list) or not any(
            "ATTENTION_KD_WEIGHT=60" in command and "sbatch" in command
            for command in commands
            if isinstance(command, str)
        ):
            errors.append("next experiment blueprint: missing gamma-60 downstream command")
        if current.get("runnable_now") is not True:
            errors.append("next experiment blueprint: gamma-60 downstream action must be runnable")
        if "single MNLI ablation" not in str(current.get("claim_boundary", "")):
            errors.append("next experiment blueprint: current action must remain a single MNLI ablation")
    catalog = report.get("action_catalog")
    if not isinstance(catalog, dict):
        errors.append("next experiment blueprint: missing action_catalog")
    else:
        for status in (
            "pending_655m_downstream",
            "run_gamma_balanced_downstream",
            "extend_stage2_curve",
            "replicate_recovery_gate",
            "pause_broad_stage2_audit_recipe",
        ):
            if status not in catalog:
                errors.append(f"next experiment blueprint: missing catalog status {status}")
    require_contains(
        "README next experiment blueprint report",
        "bitdistill_next_experiment_blueprint_2026-05-23.md",
        readme,
        errors,
    )
    require_contains(
        "README next experiment blueprint json",
        "bitdistill_next_experiment_blueprint_2026-05-23.json",
        readme,
        errors,
    )


def validate_readme(bundle: dict[str, Any], readme: str, errors: list[str]) -> None:
    claims = bundle["claims"]
    blind = claims["blind_ptq"]
    qat = claims["qat_distill"]
    bitdistill = claims["bitdistill_reproduction"]
    runtime = claims["row_scale_runtime_contract"]
    i2sr = claims["i2sr_cpu"]
    native = claims["native_classifier"]

    required = {
        "title": "Ternary Retrofit Evaluator and CPU Runtime-Contract Tester",
        "framing": "Extreme ternary quantization is not a file-format conversion problem",
        "not_converter": "not a universal BitNet converter",
        "blind_status": "strong negative result",
        "fp_ppl": fmt_ppl(float(blind["fp_wikitext_ppl"])),
        "ptq_ppl": fmt_ppl(float(blind["ptq_wikitext_ppl"])),
        "fp_mean": fmt(float(blind["fp_ten_task_mean"])),
        "ptq_mean": fmt(float(blind["ptq_ten_task_mean"])),
        "qat_mean": fmt(float(qat["best_row_scale_qat_ten_task_mean"])),
        "qat_recovery": f"{float(qat['recovery_vs_ptq']):+.6f}",
        "qat_gap": f"{float(qat['gap_vs_fp']):+.6f}",
        "bitdistill_status": "No",
        "mnli_40m": fmt(float(bitdistill["controlled_40_96m_mnli"])),
        "mnli_163m": fmt(float(bitdistill["controlled_163_84m_mnli"])),
        "mnli_327m": fmt(float(bitdistill["controlled_327_68m_mnli"])),
        "mnli_delta": f"{float(bitdistill['controlled_327_68m_delta_vs_fp']):+.6f}",
        "stage2_tokens": "655.36M",
        "stage2_final_ce": "3.426713",
        "mnli_655m": "0.729903",
        "mnli_655m_delta": "-0.078248",
        "mnli_655m_ci_low": "-0.086720",
        "mnli_655m_gain": "+0.009883",
        "gamma_paper_grad_ratio": "221.384986",
        "gamma_balanced_grad_ratio": "0.346044",
        "tl2_one_scale": fmt(float(runtime["one_scale_tl2_relative_rms_error"])),
        "row_scale": fmt(float(runtime["exact_fp16_row_scale_relative_rms_error"])),
        "i2sr_file": f"{float(i2sr['row_i2sr']['file_mib']):.1f}",
        "i2sr_ppl": f"{float(i2sr['row_i2sr']['ppl']):.4f}",
        "i2sr_prompt": f"{float(i2sr['row_i2sr']['prompt_tok_s']):.2f}",
        "i2sr_decode": f"{float(i2sr['row_i2sr']['decode_tok_s']):.2f}",
        "native_mnli": fmt(float(native["mnli_accuracy"])),
        "native_agreement": fmt(float(native["pytorch_agreement"])),
        "moe_not_supported": "not supported",
    }
    for label, needle in required.items():
        require_contains(f"README {label}", needle, readme, errors)


def validate_reproduction_gap_docs(
    report: dict[str, Any],
    stage2_handoff: dict[str, Any],
    readme: str,
    claims_doc: str,
    errors: list[str],
) -> None:
    metrics = report["metrics"]
    required = {
        "gap_report_md": "bitdistill_reproduction_gap_2026-05-23.md",
        "gap_report_json": "bitdistill_reproduction_gap_2026-05-23.json",
        "bitnet_default": fmt(float(metrics["bitnet_sft_default_mnli"])),
        "bitnet_best": fmt(float(metrics["bitnet_sft_best_mnli"])),
        "bitnet_vs_paper": f"{float(metrics['bitnet_sft_best_delta_vs_paper_anchor']):+.6f}",
        "bitdistill_latest": fmt(float(metrics["bitdistill_latest_mnli"])),
        "bitdistill_vs_fp": f"{float(metrics['bitdistill_latest_delta_vs_fp16']):+.6f}",
    }
    for label, needle in required.items():
        require_contains(f"README reproduction gap {label}", needle, readme, errors)
        require_contains(f"CLAIMS reproduction gap {label}", needle, claims_doc, errors)
    require_contains("README stage2 extension job", "10250", readme, errors)
    require_contains("README stage2 extension tokens", "655.36M", readme, errors)
    require_contains(
        "README stage2 manifest",
        "stage2_manifest_655m_2026-05-23.md",
        readme,
        errors,
    )
    require_contains("README downstream job", "10260", readme, errors)
    require_contains("README gamma telemetry report", "gamma60_gradient_balance_2026-05-23.md", readme, errors)
    require_contains("README gamma telemetry caveat", "not a task-quality result", readme, errors)
    require_contains(
        "README next decision report",
        "bitdistill_next_decision_2026-05-23.md",
        readme,
        errors,
    )
    require_contains(
        "README next experiment blueprint report",
        "bitdistill_next_experiment_blueprint_2026-05-23.md",
        readme,
        errors,
    )
    require_contains(
        "README current goal status report",
        "current_goal_status_2026-05-23.md",
        readme,
        errors,
    )
    require_contains(
        "README current goal status json",
        "current_goal_status_2026-05-23.json",
        readme,
        errors,
    )
    require_contains(
        "README deep research handoff report",
        "deep_research_handoff_2026-05-23.md",
        readme,
        errors,
    )
    require_contains(
        "README deep research handoff json",
        "deep_research_handoff_2026-05-23.json",
        readme,
        errors,
    )


def validate_claims_doc(bundle: dict[str, Any], claims_doc: str, errors: list[str]) -> None:
    claims = bundle["claims"]
    required = {
        "ptq_rejected": "Rejected",
        "partial": "Supported, partial",
        "bitdistill_not_yet": "Not yet",
        "row_supported": "Supported",
        "i2sr_caveat": "does not beat Q4_K_M",
        "native_not_ready": "Not yet",
        "moe_not_supported": "Not supported",
        "mnli_327": fmt(float(claims["bitdistill_reproduction"]["controlled_327_68m_mnli"])),
        "mnli_655": "0.729903",
        "delta_655": "-0.078248",
    }
    for label, needle in required.items():
        require_contains(f"CLAIMS {label}", needle, claims_doc, errors)


def validate_runtime_doc(bundle: dict[str, Any], runtime_doc: str, errors: list[str]) -> None:
    runtime = bundle["claims"]["row_scale_runtime_contract"]
    i2sr = bundle["claims"]["i2sr_cpu"]
    required = {
        "one_scale": fmt(float(runtime["one_scale_tl2_relative_rms_error"])),
        "row_scale": fmt(float(runtime["exact_fp16_row_scale_relative_rms_error"])),
        "fp_file": f"{float(i2sr['fp_f16']['file_mib']):.1f}",
        "fp_ppl": f"{float(i2sr['fp_f16']['ppl']):.4f}",
        "q4_file": f"{float(i2sr['q4_k_m']['file_mib']):.1f}",
        "q4_ppl": f"{float(i2sr['q4_k_m']['ppl']):.4f}",
        "i2sr_file": f"{float(i2sr['row_i2sr']['file_mib']):.1f}",
        "i2sr_ppl": f"{float(i2sr['row_i2sr']['ppl']):.4f}",
        "not_q4": "not quality/storage competitive with Q4_K_M",
    }
    for label, needle in required.items():
        require_contains(f"RUNTIME_CONTRACT {label}", needle, runtime_doc, errors)


def validate_experiments_doc(experiments_doc: str, readme: str, errors: list[str]) -> None:
    required = {
        "watchdog_command": "python benchmarks/run_active_gate_watchdog.py",
        "ingestion_command": "python benchmarks/audit_stage2_655m_ingestion.py",
        "paper_alignment_command": "python benchmarks/build_bitdistill_paper_alignment_audit.py",
        "watchdog_report": "active_gate_watchdog_2026-05-23.md",
        "ingestion_status": "stage2_655m_ingestion.status == ingested_reports_rebuilt",
        "paper_alignment_status": "bitdistill_paper_alignment.status == not_exact_reproduction",
        "next_blueprint_status": "bitdistill_next_experiment_blueprint.status == run_gamma_balanced_downstream",
        "no_claim_until_ingested": "ingested_reports_rebuilt",
    }
    for label, needle in required.items():
        require_contains(f"EXPERIMENTS {label}", needle, experiments_doc, errors)
    require_contains("README experiments link", "EXPERIMENTS.md", readme, errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("benchmarks/results/canonical_evidence_bundle_2026-05-20.json"),
    )
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--claims", type=Path, default=Path("CLAIMS.md"))
    parser.add_argument("--experiments", type=Path, default=Path("EXPERIMENTS.md"))
    parser.add_argument("--runtime-contract", type=Path, default=Path("RUNTIME_CONTRACT.md"))
    parser.add_argument(
        "--reproduction-gap",
        type=Path,
        default=Path("benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json"),
    )
    parser.add_argument(
        "--stage2-extension",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_submission_2026-05-23.json"),
    )
    parser.add_argument(
        "--stage2-handoff",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_handoff_submission_2026-05-23.json"),
    )
    parser.add_argument(
        "--stage2-afterany",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_afterany_submission_2026-05-23.json"),
    )
    parser.add_argument(
        "--stage2-ingestion",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_ingestion_2026-05-23.json"),
    )
    parser.add_argument(
        "--stage2-snapshot-salvage",
        type=Path,
        default=Path("benchmarks/results/stage2_snapshot_salvage_2026-05-23.json"),
    )
    parser.add_argument(
        "--stage2-handoff-preflight",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_handoff_preflight_2026-05-23.json"),
    )
    parser.add_argument(
        "--gamma-telemetry-submission",
        type=Path,
        default=Path("benchmarks/results/gamma60_telemetry_submission_2026-05-23.json"),
    )
    parser.add_argument(
        "--stage2-monitor",
        type=Path,
        default=Path("benchmarks/results/active_stage2_extension_monitor_2026-05-23.json"),
    )
    parser.add_argument(
        "--current-goal-status",
        type=Path,
        default=Path("benchmarks/results/current_goal_status_2026-05-23.json"),
    )
    parser.add_argument(
        "--deep-research-handoff",
        type=Path,
        default=Path("benchmarks/results/deep_research_handoff_2026-05-23.json"),
    )
    parser.add_argument(
        "--benchmark-scoreboard",
        type=Path,
        default=Path("benchmarks/results/bitdistill_benchmark_scoreboard_2026-05-23.json"),
    )
    parser.add_argument(
        "--goal-traceability",
        type=Path,
        default=Path("benchmarks/results/bitdistill_goal_traceability_2026-05-23.json"),
    )
    parser.add_argument(
        "--publication-product-plan",
        type=Path,
        default=Path("benchmarks/results/bitdistill_publication_product_plan_2026-05-23.json"),
    )
    parser.add_argument(
        "--paper-alignment",
        type=Path,
        default=Path("benchmarks/results/bitdistill_paper_alignment_2026-05-23.json"),
    )
    parser.add_argument(
        "--active-gate-watchdog",
        type=Path,
        default=Path("benchmarks/results/active_gate_watchdog_2026-05-23.json"),
    )
    parser.add_argument(
        "--active-slurm-batch-scripts",
        type=Path,
        default=Path("benchmarks/results/active_slurm_batch_scripts_2026-05-23.json"),
    )
    parser.add_argument(
        "--next-experiment-blueprint",
        type=Path,
        default=Path("benchmarks/results/bitdistill_next_experiment_blueprint_2026-05-23.json"),
    )
    args = parser.parse_args()

    bundle = load_json(args.bundle)
    reproduction_gap = load_json(args.reproduction_gap)
    stage2_extension = load_json(args.stage2_extension)
    stage2_handoff = load_json(args.stage2_handoff)
    stage2_afterany = load_json(args.stage2_afterany)
    stage2_ingestion = load_json(args.stage2_ingestion)
    stage2_snapshot_salvage = load_json(args.stage2_snapshot_salvage)
    stage2_handoff_preflight = load_json(args.stage2_handoff_preflight)
    gamma_telemetry = load_json(args.gamma_telemetry_submission)
    stage2_monitor = load_json(args.stage2_monitor)
    current_goal_status = load_json(args.current_goal_status)
    deep_research_handoff = load_json(args.deep_research_handoff)
    benchmark_scoreboard = load_json(args.benchmark_scoreboard)
    goal_traceability = load_json(args.goal_traceability)
    publication_product_plan = load_json(args.publication_product_plan)
    paper_alignment = load_json(args.paper_alignment)
    active_gate_watchdog = load_json(args.active_gate_watchdog)
    active_slurm_batch_scripts = load_json(args.active_slurm_batch_scripts)
    next_experiment_blueprint = load_json(args.next_experiment_blueprint)
    readme = read_text(args.readme)
    claims_doc = read_text(args.claims)
    experiments_doc = read_text(args.experiments)
    errors: list[str] = []
    validate_artifacts(bundle, errors)
    validate_reproduction_gap(reproduction_gap, errors)
    validate_stage2_extension_submission(stage2_extension, errors)
    validate_stage2_handoff_submission(stage2_handoff, errors)
    validate_stage2_afterany_submission(stage2_afterany, readme, errors)
    validate_stage2_ingestion(stage2_ingestion, readme, errors)
    validate_stage2_snapshot_salvage(stage2_snapshot_salvage, readme, errors)
    validate_stage2_handoff_preflight(stage2_handoff_preflight, readme, errors)
    validate_gradient_telemetry_submission(gamma_telemetry, errors)
    validate_active_stage2_monitor(stage2_monitor, errors)
    validate_current_goal_status(current_goal_status, errors)
    validate_deep_research_handoff(deep_research_handoff, errors)
    validate_benchmark_scoreboard(benchmark_scoreboard, readme, errors)
    validate_goal_traceability(goal_traceability, readme, errors)
    validate_paper_alignment(paper_alignment, readme, errors)
    validate_publication_product_plan(publication_product_plan, readme, errors)
    validate_active_gate_watchdog(active_gate_watchdog, readme, errors)
    validate_next_experiment_blueprint(next_experiment_blueprint, readme, errors)
    validate_active_slurm_batch_scripts(active_slurm_batch_scripts, errors)
    validate_readme(bundle, readme, errors)
    validate_reproduction_gap_docs(reproduction_gap, stage2_handoff, readme, claims_doc, errors)
    validate_claims_doc(bundle, claims_doc, errors)
    validate_experiments_doc(experiments_doc, readme, errors)
    validate_runtime_doc(bundle, read_text(args.runtime_contract), errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"validated public docs against {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
