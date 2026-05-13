#!/usr/bin/env python3
"""Audit whether I2_SR is active in the committed llama.cpp submodule state."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


DATE = "2026-05-13"
SUBMODULE_PATH = Path("3rdparty/llama.cpp")
PATCH_PATH = Path("patches/llama-i2sr-row-scale-qtype.patch")


def run(command: list[str], *, cwd: Path, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=check)


def git_output(command: list[str], *, cwd: Path) -> str:
    return run(command, cwd=cwd, check=True).stdout.strip()


def file_contains(path: Path, needles: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return all(needle in text for needle in needles)


def patch_stat(root: Path, patch: Path) -> dict[str, Any]:
    stat = run(["git", "apply", "--numstat", str(patch)], cwd=root)
    files: list[dict[str, Any]] = []
    for line in stat.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            files.append({"added": parts[0], "deleted": parts[1], "path": parts[2]})
    return {"returncode": stat.returncode, "files": files}


def build_audit(root: Path) -> dict[str, Any]:
    submodule = root / SUBMODULE_PATH
    patch = root / PATCH_PATH
    gitmodules_url = git_output(["git", "config", "-f", ".gitmodules", "--get", "submodule.3rdparty/llama.cpp.url"], cwd=root)
    gitmodules_branch = git_output(["git", "config", "-f", ".gitmodules", "--get", "submodule.3rdparty/llama.cpp.branch"], cwd=root)
    submodule_head = git_output(["git", "rev-parse", "HEAD"], cwd=submodule)
    submodule_short = git_output(["git", "rev-parse", "--short", "HEAD"], cwd=submodule)
    superproject_status = git_output(["git", "status", "--short", "--branch"], cwd=root)
    submodule_status = git_output(["git", "status", "--short", "--branch"], cwd=submodule)
    remote_contains = git_output(["git", "branch", "--remotes", "--contains", "HEAD"], cwd=submodule).splitlines()

    patch_applies = run(["git", "apply", "--check", str(patch)], cwd=root).returncode == 0
    patch_already_applied = run(["git", "apply", "--reverse", "--check", str(patch)], cwd=root).returncode == 0
    stat = patch_stat(root, patch)
    patch_files = [item["path"] for item in stat["files"]]
    patch_touches_submodule = any(path.startswith(str(SUBMODULE_PATH) + "/") for path in patch_files)
    patch_touches_root_runtime = "src/ggml-bitnet-mad.cpp" in patch_files

    active_checks = {
        "ggml_type_i2_sr": file_contains(submodule / "ggml/include/ggml.h", ["GGML_TYPE_I2_SR"]),
        "llama_ftype_i2_sr": file_contains(submodule / "include/llama.h", ["LLAMA_FTYPE_MOSTLY_I2_SR"]),
        "gguf_py_i2_sr": file_contains(submodule / "gguf-py/gguf/constants.py", ["I2_SR", "MOSTLY_I2_SR"]),
        "llama_routes_i2_sr": file_contains(submodule / "src/llama.cpp", ["GGML_TYPE_I2_SR", "LLAMA_FTYPE_MOSTLY_I2_SR"]),
        "root_quantize_i2_sr": file_contains(root / "src/ggml-bitnet-mad.cpp", ["quantize_i2_sr", "GGML_TYPE_I2_SR"]),
    }
    active_runtime_support = all(active_checks.values())
    promotion_ready = (
        active_runtime_support
        and not patch_already_applied
        and bool(remote_contains)
        and not any(line.startswith(" M ") or line.startswith("?? ") for line in submodule_status.splitlines())
    )
    blockers = []
    if not active_runtime_support:
        blockers.append("I2_SR qtype/file-type/runtime support is not present in the active submodule/root runtime files.")
    if patch_applies:
        blockers.append("The I2_SR support still exists as an unapplied patch rather than active committed code.")
    if not remote_contains:
        blockers.append("Submodule HEAD is not reachable from a fetched remote branch; a superproject pointer would not be reproducible.")
    if any(line.startswith(" M ") or line.startswith("?? ") for line in submodule_status.splitlines()):
        blockers.append("Submodule working tree is dirty.")

    return {
        "schema": "bitnet-i2sr-submodule-promotion-audit-v1",
        "date": DATE,
        "promotion_ready": promotion_ready,
        "active_runtime_support": active_runtime_support,
        "patch_applies_cleanly": patch_applies,
        "patch_already_applied": patch_already_applied,
        "patch_touches_submodule": patch_touches_submodule,
        "patch_touches_root_runtime": patch_touches_root_runtime,
        "configured_submodule_url": gitmodules_url,
        "configured_submodule_branch": gitmodules_branch,
        "submodule_head": submodule_head,
        "submodule_short": submodule_short,
        "remote_branches_containing_head": [line.strip() for line in remote_contains],
        "superproject_status": superproject_status,
        "submodule_status": submodule_status,
        "active_checks": active_checks,
        "patch_files": patch_files,
        "blockers": blockers,
        "next_steps": [
            "Create or choose a writable llama.cpp fork/branch for the I2_SR runtime changes.",
            "Apply the submodule portion of patches/llama-i2sr-row-scale-qtype.patch inside that branch and commit it.",
            "Apply the root runtime portion in this superproject or split it into an equivalent top-level commit.",
            "Push the submodule branch, update .gitmodules if the URL changes, then update the superproject submodule pointer.",
            "Run benchmarks/run_i2sr_active_patch_gate.py or the equivalent active-source productization gate after the pointer update.",
        ],
    }


def md_bool(value: bool) -> str:
    return "`true`" if value else "`false`"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    active_rows = [[name, md_bool(value)] for name, value in result["active_checks"].items()]
    patch_rows = [[path] for path in result["patch_files"]]
    blocker_rows = [[blocker] for blocker in result["blockers"]] or [["none"]]
    next_rows = [[step] for step in result["next_steps"]]
    return "\n\n".join(
        [
            f"# I2_SR Submodule Promotion Audit, {DATE}",
            "This audit checks whether the row-scale `I2_SR` runtime is active in the committed source state, not merely available as a patch.",
            f"Promotion ready: {md_bool(result['promotion_ready'])}.",
            f"Active runtime support: {md_bool(result['active_runtime_support'])}.",
            f"Patch applies cleanly: {md_bool(result['patch_applies_cleanly'])}.",
            f"Submodule: `{result['configured_submodule_url']}` branch `{result['configured_submodule_branch']}` at `{result['submodule_short']}`.",
            f"Remote branches containing HEAD: `{', '.join(result['remote_branches_containing_head'])}`.",
            "## Active Source Checks",
            md_table(["check", "present"], active_rows),
            "## Blockers",
            md_table(["blocker"], blocker_rows),
            "## Patch Touches",
            md_table(["path"], patch_rows),
            "## Required Promotion Steps",
            md_table(["step"], next_rows),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_results/i2sr_submodule_promotion_audit_2026-05-13.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/results/i2sr_submodule_promotion_audit_2026-05-13.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    result = build_audit(root)
    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
