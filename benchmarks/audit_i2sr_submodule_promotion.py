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
ROOT_SPLIT_PATCH = Path("patches/bitnet-i2sr-root-runtime.patch")
SUBMODULE_SPLIT_PATCH = Path("patches/llama-i2sr-row-scale-qtype.submodule.patch")
HANDOFF_JSON = Path("benchmark_results/i2sr_promotion_handoff_2026-05-13.json")


def run(command: list[str], *, cwd: Path, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=check)


def git_output(command: list[str], *, cwd: Path) -> str:
    return run(command, cwd=cwd, check=True).stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def remote_write_probe(submodule: Path, branch: str) -> dict[str, Any]:
    completed = run(["git", "push", "--dry-run", "origin", f"HEAD:refs/heads/{branch}"], cwd=submodule)
    stderr = completed.stderr.strip()
    stdout = completed.stdout.strip()
    permission_denied = "Permission to" in stderr and "denied" in stderr
    return {
        "branch": branch,
        "returncode": completed.returncode,
        "writable": completed.returncode == 0,
        "permission_denied": permission_denied,
        "stdout": stdout,
        "stderr": stderr,
    }


def remote_read_probe(url: str) -> dict[str, Any]:
    completed = run(["git", "ls-remote", "--heads", url], cwd=Path.cwd())
    heads = []
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].startswith("refs/heads/"):
            heads.append(parts[1].removeprefix("refs/heads/"))
    return {
        "url": url,
        "returncode": completed.returncode,
        "reachable": completed.returncode == 0,
        "heads": heads,
        "stderr": completed.stderr.strip(),
    }


def build_audit(
    root: Path,
    *,
    check_remote_write: bool = False,
    remote_probe_branch: str = "i2sr-row-scale-runtime",
    candidate_fork_url: str | None = None,
) -> dict[str, Any]:
    submodule = root / SUBMODULE_PATH
    patch = root / PATCH_PATH
    gitmodules_url = git_output(["git", "config", "-f", ".gitmodules", "--get", "submodule.3rdparty/llama.cpp.url"], cwd=root)
    gitmodules_branch = git_output(["git", "config", "-f", ".gitmodules", "--get", "submodule.3rdparty/llama.cpp.branch"], cwd=root)
    submodule_head = git_output(["git", "rev-parse", "HEAD"], cwd=submodule)
    submodule_short = git_output(["git", "rev-parse", "--short", "HEAD"], cwd=submodule)
    superproject_status = git_output(["git", "status", "--short", "--branch"], cwd=root)
    submodule_status = git_output(["git", "status", "--short", "--branch"], cwd=submodule)
    remote_contains = git_output(["git", "branch", "--remotes", "--contains", "HEAD"], cwd=submodule).splitlines()
    remote_write = remote_write_probe(submodule, remote_probe_branch) if check_remote_write else None
    candidate_fork = remote_read_probe(candidate_fork_url) if candidate_fork_url else None
    handoff = read_json(root / HANDOFF_JSON)
    handoff_result = handoff.get("worktree_result") if isinstance(handoff.get("worktree_result"), dict) else {}
    local_handoff = {
        "prepared": bool(handoff_result.get("prepared")),
        "pushed": bool(handoff_result.get("pushed")),
        "commit_sha": handoff_result.get("commit_sha") or "",
        "worktree_dir": handoff_result.get("worktree_dir") or "",
        "branch": handoff_result.get("branch") or remote_probe_branch,
    }

    patch_applies = run(["git", "apply", "--check", str(patch)], cwd=root).returncode == 0
    patch_already_applied = run(["git", "apply", "--reverse", "--check", str(patch)], cwd=root).returncode == 0
    stat = patch_stat(root, patch)
    root_split_patch = root / ROOT_SPLIT_PATCH
    submodule_split_patch = root / SUBMODULE_SPLIT_PATCH
    root_split_applies = root_split_patch.exists() and run(["git", "apply", "--check", str(root_split_patch)], cwd=root).returncode == 0
    submodule_split_applies = submodule_split_patch.exists() and run(["git", "apply", "--check", str(submodule_split_patch)], cwd=submodule).returncode == 0
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
    if remote_write and not remote_write["writable"]:
        blockers.append("Configured llama.cpp submodule remote is not writable from this environment; use a fork or writable branch.")
    if candidate_fork and not candidate_fork["reachable"]:
        blockers.append("Candidate llama.cpp fork URL is not reachable; create the fork or provide the correct writable URL before promotion.")
    if not root_split_applies or not submodule_split_applies:
        blockers.append("Split root/submodule I2_SR promotion patches are missing or do not apply cleanly.")
    if any(line.startswith(" M ") or line.startswith("?? ") for line in submodule_status.splitlines()):
        blockers.append("Submodule working tree is dirty.")

    next_steps = [
        "Create or choose a writable llama.cpp fork/branch for the I2_SR runtime changes.",
    ]
    if local_handoff["prepared"]:
        next_steps.append(
            f"Push prepared local branch `{local_handoff['branch']}` from `{local_handoff['worktree_dir']}` at `{local_handoff['commit_sha']}`."
        )
    else:
        next_steps.append(f"Apply `{SUBMODULE_SPLIT_PATCH}` inside that branch and commit it.")
    next_steps.extend(
        [
            f"Apply `{ROOT_SPLIT_PATCH}` in this superproject or split it into an equivalent top-level commit.",
            "Push the submodule branch, update .gitmodules if the URL changes, then update the superproject submodule pointer.",
            "Run benchmarks/run_i2sr_active_patch_gate.py or the equivalent active-source productization gate after the pointer update.",
        ]
    )

    return {
        "schema": "bitnet-i2sr-submodule-promotion-audit-v1",
        "date": DATE,
        "promotion_ready": promotion_ready,
        "active_runtime_support": active_runtime_support,
        "patch_applies_cleanly": patch_applies,
        "patch_already_applied": patch_already_applied,
        "patch_touches_submodule": patch_touches_submodule,
        "patch_touches_root_runtime": patch_touches_root_runtime,
        "split_patches": {
            "root_patch": str(ROOT_SPLIT_PATCH),
            "root_patch_applies": root_split_applies,
            "submodule_patch": str(SUBMODULE_SPLIT_PATCH),
            "submodule_patch_applies": submodule_split_applies,
        },
        "configured_submodule_url": gitmodules_url,
        "configured_submodule_branch": gitmodules_branch,
        "submodule_head": submodule_head,
        "submodule_short": submodule_short,
        "remote_branches_containing_head": [line.strip() for line in remote_contains],
        "remote_write_probe": remote_write,
        "candidate_fork_probe": candidate_fork,
        "local_handoff": local_handoff,
        "superproject_status": superproject_status,
        "submodule_status": submodule_status,
        "active_checks": active_checks,
        "patch_files": patch_files,
        "blockers": blockers,
        "next_steps": next_steps,
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
    split = result["split_patches"]
    split_rows = [
        [split["root_patch"], md_bool(bool(split["root_patch_applies"]))],
        [split["submodule_patch"], md_bool(bool(split["submodule_patch_applies"]))],
    ]
    blocker_rows = [[blocker] for blocker in result["blockers"]] or [["none"]]
    next_rows = [[step] for step in result["next_steps"]]
    remote_write = result.get("remote_write_probe")
    candidate_fork = result.get("candidate_fork_probe")
    local_handoff = result.get("local_handoff") or {}
    remote_write_lines = []
    if remote_write:
        stderr = remote_write.get("stderr", "").replace("\n", " ")
        remote_write_lines = [
            "## Remote Write Probe",
            md_table(
                ["field", "value"],
                [
                    ["branch", f"`{remote_write.get('branch')}`"],
                    ["returncode", f"`{remote_write.get('returncode')}`"],
                    ["writable", md_bool(bool(remote_write.get("writable")))],
                    ["permission_denied", md_bool(bool(remote_write.get("permission_denied")))],
                    ["stderr", f"`{stderr}`"],
                ],
            ),
        ]
    candidate_fork_lines = []
    if candidate_fork:
        stderr = candidate_fork.get("stderr", "").replace("\n", " ")
        heads = ", ".join(candidate_fork.get("heads") or [])
        candidate_fork_lines = [
            "## Candidate Fork Probe",
            md_table(
                ["field", "value"],
                [
                    ["url", f"`{candidate_fork.get('url')}`"],
                    ["returncode", f"`{candidate_fork.get('returncode')}`"],
                    ["reachable", md_bool(bool(candidate_fork.get("reachable")))],
                    ["heads", f"`{heads}`"],
                    ["stderr", f"`{stderr}`"],
                ],
            ),
        ]
    return "\n\n".join(
        [
            f"# I2_SR Submodule Promotion Audit, {DATE}",
            "This audit checks whether the row-scale `I2_SR` runtime is active in the committed source state, not merely available as a patch.",
            f"Promotion ready: {md_bool(result['promotion_ready'])}.",
            f"Active runtime support: {md_bool(result['active_runtime_support'])}.",
            f"Patch applies cleanly: {md_bool(result['patch_applies_cleanly'])}.",
            f"Submodule: `{result['configured_submodule_url']}` branch `{result['configured_submodule_branch']}` at `{result['submodule_short']}`.",
            f"Remote branches containing HEAD: `{', '.join(result['remote_branches_containing_head'])}`.",
            (
                f"Prepared local handoff: {md_bool(bool(local_handoff.get('prepared')))} "
                f"commit `{local_handoff.get('commit_sha', '')}`."
            ),
            "## Active Source Checks",
            md_table(["check", "present"], active_rows),
            "## Blockers",
            md_table(["blocker"], blocker_rows),
            *remote_write_lines,
            *candidate_fork_lines,
            "## Split Promotion Patches",
            md_table(["path", "applies cleanly"], split_rows),
            "## Patch Touches",
            md_table(["path"], patch_rows),
            "## Required Promotion Steps",
            md_table(["step"], next_rows),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--check-remote-write",
        action="store_true",
        help="Run a non-mutating git push --dry-run probe against the configured submodule origin.",
    )
    parser.add_argument("--remote-probe-branch", default="i2sr-row-scale-runtime")
    parser.add_argument(
        "--candidate-fork-url",
        help="Optional writable llama.cpp fork URL to probe with git ls-remote --heads.",
    )
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_results/i2sr_submodule_promotion_audit_2026-05-13.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/results/i2sr_submodule_promotion_audit_2026-05-13.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    result = build_audit(
        root,
        check_remote_write=args.check_remote_write,
        remote_probe_branch=args.remote_probe_branch,
        candidate_fork_url=args.candidate_fork_url,
    )
    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
