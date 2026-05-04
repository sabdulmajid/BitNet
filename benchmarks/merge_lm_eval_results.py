#!/usr/bin/env python3
"""Merge split EleutherAI lm-eval JSON files.

Use this when a benchmark suite is split across Slurm jobs. The script merges
task-keyed sections such as ``results`` and ``samples`` and fails on duplicate
tasks by default so accidental overlap is visible.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


TASK_KEYED_SECTIONS = {
    "configs",
    "group_subtasks",
    "higher_is_better",
    "n-samples",
    "n-shot",
    "results",
    "samples",
    "versions",
}


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("results"), dict):
        raise ValueError(f"{path} does not look like an lm-eval result JSON")
    return data


def merge_task_dict(
    merged: dict[str, Any],
    incoming: dict[str, Any],
    section: str,
    source: Path,
    allow_overwrite: bool,
) -> None:
    incoming_section = incoming.get(section)
    if incoming_section is None:
        return
    if not isinstance(incoming_section, dict):
        raise ValueError(f"{source}: section {section!r} is not a dict")
    merged_section = merged.setdefault(section, {})
    if not isinstance(merged_section, dict):
        raise ValueError(f"merged section {section!r} is not a dict")

    for task, value in incoming_section.items():
        if task in merged_section and not allow_overwrite:
            raise ValueError(
                f"duplicate task {task!r} in section {section!r}; "
                "rerun with --allow-overwrite only if this is intentional"
            )
        merged_section[task] = deepcopy(value)


def merge_results(paths: list[Path], allow_overwrite: bool) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one --input is required")

    first = load_json(paths[0])
    merged = {
        key: deepcopy(value)
        for key, value in first.items()
        if key not in TASK_KEYED_SECTIONS
    }
    merged["merged_from"] = [str(path) for path in paths]

    for path in paths:
        data = load_json(path)
        for section in TASK_KEYED_SECTIONS:
            merge_task_dict(merged, data, section, path, allow_overwrite)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="allow later inputs to overwrite duplicate task entries",
    )
    args = parser.parse_args()

    merged = merge_results(args.input, args.allow_overwrite)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tasks = sorted(merged.get("results", {}))
    print(f"wrote {args.output_json} with {len(tasks)} tasks: {','.join(tasks)}")


if __name__ == "__main__":
    main()
