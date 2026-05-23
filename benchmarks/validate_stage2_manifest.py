#!/usr/bin/env python3
"""Validate Stage-2 checkpoint manifests fail-closed."""

from __future__ import annotations

import argparse
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
    "git",
}


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
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
    for key in ["state_dict_path", "root_metrics_path", "snapshot_metrics_path"]:
        value = manifest.get(key)
        if not isinstance(value, str) or not value:
            errors.append(f"{key} is empty")
            continue
        if not Path(value).exists():
            errors.append(f"{key} does not exist: {value}")
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

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"validated {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
