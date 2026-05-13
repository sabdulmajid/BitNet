#!/usr/bin/env python3
"""Apply selected groups from the generated artifact prune plan.

The default mode is a dry run. Destructive deletion requires --execute and only
works for allow-listed plan groups after path and protected-evidence checks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "bitnet-artifact-prune-application-v1"
PLAN_SCHEMA = "bitnet-artifact-prune-plan-v1"
DEFAULT_PLAN = Path("benchmarks/results/artifact_prune_plan_2026-05-13.json")
DEFAULT_OUTPUT_JSON = Path("benchmark_results/artifact_prune_application_2026-05-13.json")
DEFAULT_OUTPUT_MD = Path("benchmarks/results/artifact_prune_application_2026-05-13.md")
DEFAULT_GROUPS = ("prune_intermediate_checkpoints",)
ALLOW_GROUPS = {
    "prune_intermediate_checkpoints",
    "remove_rebuildable_build_dirs",
    "remove_local_caches",
}


@dataclass
class PruneResult:
    group: str
    path: str
    status: str
    size_bytes: int
    reason: str
    detail: str


def run_text(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def git_is_clean(root: Path) -> bool:
    return run_text(["git", "status", "--porcelain"], root) == ""


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{size} B"


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise SystemExit(f"unexpected prune plan schema in {path}: {plan.get('schema')}")
    return plan


def repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe path: {value}")
    return path


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def protected_hit(path: Path, protected_paths: set[Path]) -> Path | None:
    for protected in protected_paths:
        if path == protected or is_relative_to(path, protected) or is_relative_to(protected, path):
            return protected
    return None


def validate_group_path(group: str, path: Path) -> None:
    if group == "prune_intermediate_checkpoints":
        if len(path.parts) != 3 or path.parts[0] != "checkpoints" or not path.parts[2].startswith("step-"):
            raise ValueError("intermediate checkpoint path must look like checkpoints/<run>/step-*")
    elif group == "remove_rebuildable_build_dirs":
        if path.parts[0] not in {"build", "build-portable-avx2", "build-qwen05b-tl2"} or len(path.parts) != 1:
            raise ValueError("build cleanup path is not an approved rebuildable build directory")
    elif group == "remove_local_caches":
        if path != Path(".hf_cache"):
            raise ValueError("cache cleanup path is not .hf_cache")
    else:
        raise ValueError(f"group is not executable by this script: {group}")


def remove_path(path: Path) -> str:
    if path.is_dir():
        shutil.rmtree(path)
        return "removed directory"
    if path.is_file() or path.is_symlink():
        path.unlink()
        return "removed file"
    return "already absent"


def apply_plan(root: Path, plan: dict[str, Any], groups: list[str], *, execute: bool, require_clean_git: bool) -> dict[str, Any]:
    if require_clean_git and not git_is_clean(root):
        raise SystemExit("git worktree is dirty; rerun after committing/stashing or pass --allow-dirty-git")

    protected_paths = {repo_path(path) for path in plan.get("protected_paths", [])}
    results: list[PruneResult] = []
    total_selected = 0
    total_removed = 0

    for group in groups:
        if group not in ALLOW_GROUPS:
            raise SystemExit(f"group is not allow-listed for deletion: {group}")
        for item in plan.get("groups", {}).get(group, []):
            rel_path = repo_path(item["path"])
            size_bytes = int(item.get("size_bytes") or 0)
            reason = str(item.get("reason") or "")
            total_selected += size_bytes
            try:
                validate_group_path(group, rel_path)
                hit = protected_hit(rel_path, protected_paths)
                if hit is not None:
                    raise ValueError(f"path overlaps protected evidence path: {hit}")
                abs_path = root / rel_path
                if not abs_path.exists():
                    results.append(PruneResult(group, str(rel_path), "absent", 0, reason, "already absent"))
                    continue
                results.append(PruneResult(group, str(rel_path), "would_remove", size_bytes, reason, "validated"))
            except ValueError as exc:
                results.append(PruneResult(group, str(rel_path), "blocked", size_bytes, reason, str(exc)))

    blocked = [result for result in results if result.status == "blocked"]
    if blocked and execute:
        raise SystemExit("refusing to execute because one or more selected paths failed validation")

    if execute:
        for result in results:
            if result.status != "would_remove":
                continue
            detail = remove_path(root / result.path)
            result.status = "removed"
            result.detail = detail
            total_removed += result.size_bytes
    else:
        for result in results:
            if result.status == "would_remove":
                result.detail = "dry run only"

    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plan_schema": plan.get("schema"),
        "plan_git_head": plan.get("git_head"),
        "execute": execute,
        "groups": groups,
        "selected_count": len(results),
        "blocked_count": len(blocked),
        "total_selected_bytes": total_selected,
        "total_removed_bytes": total_removed if execute else 0,
        "results": [result.__dict__ for result in results],
    }


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    rows = [
        [
            item["group"],
            item["path"],
            item["status"],
            format_bytes(int(item["size_bytes"])),
            item["detail"].replace("|", "/"),
        ]
        for item in result["results"]
    ]
    return "\n\n".join(
        [
            "# Artifact Prune Application",
            f"Generated UTC: `{result['generated_utc']}`",
            f"Mode: `{'execute' if result['execute'] else 'dry-run'}`",
            f"Groups: `{', '.join(result['groups'])}`",
            f"Selected paths: `{result['selected_count']}`; blocked: `{result['blocked_count']}`.",
            f"Selected bytes: `{format_bytes(result['total_selected_bytes'])}`.",
            f"Removed bytes: `{format_bytes(result['total_removed_bytes'])}`.",
            md_table(["group", "path", "status", "size", "detail"], rows),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--group", action="append", choices=sorted(ALLOW_GROUPS), help="Plan group to apply. Defaults to intermediate checkpoints.")
    parser.add_argument("--execute", action="store_true", help="Actually delete selected paths. Default is dry-run.")
    parser.add_argument("--allow-dirty-git", action="store_true", help="Do not require a clean Git worktree before pruning.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    plan_path = args.plan_json if args.plan_json.is_absolute() else root / args.plan_json
    groups = args.group or list(DEFAULT_GROUPS)
    plan = load_plan(plan_path)
    result = apply_plan(root, plan, groups, execute=args.execute, require_clean_git=not args.allow_dirty_git)

    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")

    print(render_markdown(result))


if __name__ == "__main__":
    main()
