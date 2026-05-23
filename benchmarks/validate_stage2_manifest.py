#!/usr/bin/env python3
"""Validate Stage-2 checkpoint manifests fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_KEYS = {
    "schema",
    "run_id",
    "job_id",
    "model",
    "stage",
    "method",
    "scale_mode",
    "steps",
    "token_presentations",
    "final_ce",
    "state_dict_path",
    "root_metrics_path",
    "snapshot_metrics_path",
    "state_dict_sha256",
    "git",
}


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    errors: list[str] = []
    missing = sorted(REQUIRED_KEYS.difference(manifest))
    if missing:
        errors.append(f"missing required keys: {', '.join(missing)}")
    if manifest.get("schema") != "bitnet-stage2-checkpoint-manifest-v1":
        errors.append(f"unexpected schema: {manifest.get('schema')}")
    if manifest.get("stage") != "continued_pretrain":
        errors.append(f"stage is not continued_pretrain: {manifest.get('stage')}")
    if manifest.get("method") != "bitdistill":
        errors.append(f"method is not bitdistill: {manifest.get('method')}")
    for key in ["steps", "token_presentations", "final_ce"]:
        if not finite_number(manifest.get(key)):
            errors.append(f"{key} is not finite: {manifest.get(key)!r}")
    existing_paths: dict[str, Path] = {}
    for key in ["state_dict_path", "root_metrics_path", "snapshot_metrics_path"]:
        value = manifest.get(key)
        if not isinstance(value, str) or not value:
            errors.append(f"{key} is empty")
            continue
        path = Path(value)
        existing_paths[key] = path
        if not path.exists():
            errors.append(f"{key} does not exist: {value}")
    state_dict = existing_paths.get("state_dict_path")
    expected_sha = manifest.get("state_dict_sha256")
    if state_dict and state_dict.exists():
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            errors.append(f"state_dict_sha256 is not a sha256 hex digest: {expected_sha!r}")
        else:
            actual_sha = file_sha256(state_dict)
            if actual_sha != expected_sha:
                errors.append(f"state_dict_sha256 mismatch: {actual_sha} != {expected_sha}")
    root_metrics_path = existing_paths.get("root_metrics_path")
    snapshot_metrics_path = existing_paths.get("snapshot_metrics_path")
    root_metrics: dict[str, Any] = {}
    snapshot_metrics: dict[str, Any] = {}
    if root_metrics_path and root_metrics_path.exists():
        root_metrics = read_json(root_metrics_path)
    if snapshot_metrics_path and snapshot_metrics_path.exists():
        snapshot_metrics = read_json(snapshot_metrics_path)
    if root_metrics:
        for manifest_key, metric_key in [
            ("stage", "stage"),
            ("method", "method"),
            ("scale_mode", "scale_mode"),
            ("steps", "steps"),
        ]:
            if manifest.get(manifest_key) != root_metrics.get(metric_key):
                errors.append(
                    f"{manifest_key} does not match root metrics: "
                    f"{manifest.get(manifest_key)!r} != {root_metrics.get(metric_key)!r}"
                )
        metric_tokens = root_metrics.get("effective_train_token_presentations")
        segment_tokens = manifest.get("segment_token_presentations", manifest.get("token_presentations"))
        parent_tokens = manifest.get("parent_token_presentations", 0)
        if isinstance(parent_tokens, int) and isinstance(segment_tokens, int):
            expected_metric_tokens = segment_tokens
            if metric_tokens != expected_metric_tokens:
                errors.append(
                    "root effective_train_token_presentations does not match segment tokens: "
                    f"{metric_tokens!r} != {expected_metric_tokens!r}"
                )
        last = root_metrics.get("last", {}) if isinstance(root_metrics.get("last"), dict) else {}
        if finite_number(manifest.get("final_ce")) and finite_number(last.get("ce")):
            if abs(float(manifest["final_ce"]) - float(last["ce"])) > 1e-9:
                errors.append(f"final_ce does not match root metrics: {manifest['final_ce']!r} != {last['ce']!r}")
        if manifest.get("model") != root_metrics.get("student_model"):
            errors.append(f"model does not match root metrics student_model: {manifest.get('model')!r} != {root_metrics.get('student_model')!r}")
    if snapshot_metrics:
        for manifest_key, metric_key in [
            ("stage", "stage"),
            ("method", "method"),
            ("scale_mode", "scale_mode"),
            ("steps", "steps"),
        ]:
            if manifest.get(manifest_key) != snapshot_metrics.get(metric_key):
                errors.append(
                    f"{manifest_key} does not match snapshot metrics: "
                    f"{manifest.get(manifest_key)!r} != {snapshot_metrics.get(metric_key)!r}"
                )
        snapshot = snapshot_metrics.get("snapshot", {}) if isinstance(snapshot_metrics.get("snapshot"), dict) else {}
        if snapshot.get("step") is not None and snapshot.get("step") != manifest.get("steps"):
            errors.append(f"snapshot step does not match manifest steps: {snapshot.get('step')!r} != {manifest.get('steps')!r}")
    parent_manifest_path = manifest.get("parent_manifest_path")
    if parent_manifest_path:
        if not isinstance(parent_manifest_path, str):
            errors.append(f"parent_manifest_path is not a string: {parent_manifest_path!r}")
        elif not Path(parent_manifest_path).exists():
            errors.append(f"parent_manifest_path does not exist: {parent_manifest_path}")
    parent_state_dict_path = manifest.get("parent_state_dict_path")
    if parent_state_dict_path:
        if not isinstance(parent_state_dict_path, str):
            errors.append(f"parent_state_dict_path is not a string: {parent_state_dict_path!r}")
        elif not Path(parent_state_dict_path).exists():
            errors.append(f"parent_state_dict_path does not exist: {parent_state_dict_path}")
    segment_tokens = manifest.get("segment_token_presentations")
    parent_tokens = manifest.get("parent_token_presentations")
    total_tokens = manifest.get("token_presentations")
    if segment_tokens is not None or parent_tokens is not None:
        if not isinstance(segment_tokens, int):
            errors.append(f"segment_token_presentations is not an int: {segment_tokens!r}")
        if not isinstance(parent_tokens, int):
            errors.append(f"parent_token_presentations is not an int: {parent_tokens!r}")
        if isinstance(segment_tokens, int) and isinstance(parent_tokens, int) and total_tokens != segment_tokens + parent_tokens:
            errors.append(
                "token_presentations does not equal parent + segment: "
                f"{total_tokens!r} != {parent_tokens!r} + {segment_tokens!r}"
            )
    git = manifest.get("git")
    if not isinstance(git, dict) or not git.get("bitnet_commit"):
        errors.append("git.bitnet_commit missing")
    if not isinstance(git, dict) or not git.get("llama_cpp_commit"):
        errors.append("git.llama_cpp_commit missing")
    downstream = manifest.get("downstream")
    if not isinstance(downstream, dict):
        errors.append("downstream object missing")
    elif downstream.get("recommended_init_state_dict") != manifest.get("state_dict_path"):
        errors.append(
            "downstream recommended_init_state_dict does not match state_dict_path: "
            f"{downstream.get('recommended_init_state_dict')!r} != {manifest.get('state_dict_path')!r}"
        )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"validated {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
