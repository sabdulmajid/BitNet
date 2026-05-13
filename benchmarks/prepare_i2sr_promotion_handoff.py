#!/usr/bin/env python3
"""Prepare an auditable handoff for promoting the I2_SR llama.cpp patch.

The default mode is intentionally non-mutating: it checks local patch
applicability, probes the candidate fork URL, and writes exact commands for the
branch push and superproject pointer update. Use --prepare-worktree to create an
isolated llama.cpp worktree and local commit. Use --push only after the fork URL
is known writable.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


DATE = "2026-05-13"
DEFAULT_BRANCH = "i2sr-row-scale-runtime"
SUBMODULE_PATH = Path("3rdparty/llama.cpp")
ROOT_PATCH = Path("patches/bitnet-i2sr-root-runtime.patch")
SUBMODULE_PATCH = Path("patches/llama-i2sr-row-scale-qtype.submodule.patch")


def run(command: list[str], *, cwd: Path, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=check)


def command_text(command: list[str], *, cwd: Path | None = None) -> str:
    prefix = f"(cd {shlex.quote(str(cwd))} && " if cwd is not None else ""
    suffix = ")" if cwd is not None else ""
    return prefix + " ".join(shlex.quote(str(part)) for part in command) + suffix


def status_clean(path: Path) -> tuple[bool, str]:
    completed = run(["git", "status", "--porcelain"], cwd=path)
    return completed.returncode == 0 and completed.stdout.strip() == "", completed.stdout.strip()


def git_output(command: list[str], *, cwd: Path) -> str:
    return run(command, cwd=cwd, check=True).stdout.strip()


def apply_check(patch: Path, *, cwd: Path) -> dict[str, Any]:
    completed = run(["git", "apply", "--check", str(patch)], cwd=cwd)
    return {
        "patch": str(patch),
        "returncode": completed.returncode,
        "applies": completed.returncode == 0,
        "stderr": completed.stderr.strip(),
    }


def remote_probe(url: str, *, cwd: Path) -> dict[str, Any]:
    completed = run(["git", "ls-remote", "--heads", url], cwd=cwd)
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


def worktree_exists(submodule: Path, worktree_dir: Path) -> bool:
    completed = run(["git", "worktree", "list", "--porcelain"], cwd=submodule)
    needle = f"worktree {worktree_dir}"
    return any(line.strip() == needle for line in completed.stdout.splitlines())


def build_commands(root: Path, fork_url: str, branch: str, worktree_dir: Path) -> dict[str, list[str]]:
    submodule = root / SUBMODULE_PATH
    return {
        "prepare_submodule_branch": [
            command_text(["git", "worktree", "add", "-b", branch, str(worktree_dir), "HEAD"], cwd=submodule),
            command_text(["git", "apply", str(root / SUBMODULE_PATCH)], cwd=worktree_dir),
            command_text(["git", "add", "ggml", "gguf-py", "include", "src"], cwd=worktree_dir),
            command_text(["git", "commit", "-m", "Add I2_SR row-scale runtime"], cwd=worktree_dir),
            command_text(["git", "push", fork_url, f"HEAD:refs/heads/{branch}"], cwd=worktree_dir),
        ],
        "update_superproject_pointer": [
            command_text(["git", "config", "-f", ".gitmodules", "submodule.3rdparty/llama.cpp.url", fork_url], cwd=root),
            command_text(["git", "submodule", "sync", "3rdparty/llama.cpp"], cwd=root),
            command_text(["git", "-C", str(SUBMODULE_PATH), "fetch", fork_url, branch], cwd=root),
            command_text(["git", "-C", str(SUBMODULE_PATH), "checkout", "FETCH_HEAD"], cwd=root),
            command_text(["git", "apply", str(ROOT_PATCH)], cwd=root),
            command_text(["git", "add", ".gitmodules", str(SUBMODULE_PATH), "src/ggml-bitnet-mad.cpp"], cwd=root),
            command_text(["git", "commit", "-m", "Promote I2_SR row-scale runtime"], cwd=root),
        ],
        "post_promotion_gates": [
            command_text(["python", "benchmarks/run_i2sr_active_patch_gate.py"], cwd=root),
            command_text(
                [
                    "python",
                    "benchmarks/audit_i2sr_submodule_promotion.py",
                    "--check-remote-write",
                    "--candidate-fork-url",
                    fork_url,
                ],
                cwd=root,
            ),
            command_text(["python", "benchmarks/audit_product_scope.py"], cwd=root),
            command_text(["python", "benchmarks/audit_objective_completion.py"], cwd=root),
            command_text(["python", "benchmarks/build_evidence_manifest.py"], cwd=root),
        ],
    }


def prepare_worktree(root: Path, worktree_dir: Path, branch: str, fork_url: str, *, push: bool) -> dict[str, Any]:
    submodule = root / SUBMODULE_PATH
    if worktree_dir.exists():
        raise SystemExit(f"worktree path already exists: {worktree_dir}")
    if worktree_exists(submodule, worktree_dir):
        raise SystemExit(f"git worktree already registered: {worktree_dir}")

    steps: list[dict[str, Any]] = []

    def step(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        completed = run(command, cwd=cwd)
        steps.append(
            {
                "command": command_text(command, cwd=cwd),
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )
        if completed.returncode != 0:
            raise SystemExit(f"command failed: {command_text(command, cwd=cwd)}")
        return completed

    step(["git", "worktree", "add", "-b", branch, str(worktree_dir), "HEAD"], submodule)
    step(["git", "apply", str(root / SUBMODULE_PATCH)], worktree_dir)
    step(["git", "add", "ggml", "gguf-py", "include", "src"], worktree_dir)
    step(["git", "commit", "-m", "Add I2_SR row-scale runtime"], worktree_dir)
    commit_sha = git_output(["git", "rev-parse", "HEAD"], cwd=worktree_dir)
    if push:
        step(["git", "push", fork_url, f"HEAD:refs/heads/{branch}"], worktree_dir)
    return {
        "prepared": True,
        "worktree_dir": str(worktree_dir),
        "branch": branch,
        "commit_sha": commit_sha,
        "pushed": push,
        "steps": steps,
    }


def build_handoff(
    root: Path,
    *,
    fork_url: str,
    branch: str,
    worktree_dir: Path,
    prepare: bool,
    push: bool,
    skip_remote_check: bool,
) -> dict[str, Any]:
    submodule = root / SUBMODULE_PATH
    root_clean, root_status = status_clean(root)
    submodule_clean, submodule_status = status_clean(submodule)
    root_patch = root / ROOT_PATCH
    submodule_patch = root / SUBMODULE_PATCH
    root_patch_check = apply_check(root_patch, cwd=root) if root_patch.exists() else {"applies": False, "stderr": "missing"}
    submodule_patch_check = (
        apply_check(submodule_patch, cwd=submodule) if submodule_patch.exists() else {"applies": False, "stderr": "missing"}
    )
    fork = {"url": fork_url, "reachable": None, "returncode": None, "heads": [], "stderr": "skipped"}
    if not skip_remote_check:
        fork = remote_probe(fork_url, cwd=root)

    blockers = []
    prepare_blockers = []
    if not root_clean:
        message = "Root worktree is dirty; commit or stash unrelated changes before promotion."
        blockers.append(message)
        prepare_blockers.append(message)
    if not submodule_clean:
        message = "llama.cpp submodule worktree is dirty; promotion must start from the audited clean state."
        blockers.append(message)
        prepare_blockers.append(message)
    if not root_patch_check["applies"]:
        message = "Root I2_SR patch does not apply cleanly."
        blockers.append(message)
        prepare_blockers.append(message)
    if not submodule_patch_check["applies"]:
        message = "Submodule I2_SR patch does not apply cleanly."
        blockers.append(message)
        prepare_blockers.append(message)
    if not skip_remote_check and not fork.get("reachable"):
        blockers.append("Candidate llama.cpp fork URL is not reachable.")
    if push and not prepare:
        message = "--push requires --prepare-worktree."
        blockers.append(message)
        prepare_blockers.append(message)
    if push and skip_remote_check:
        message = "--push requires a real remote reachability check."
        blockers.append(message)
        prepare_blockers.append(message)
    if push and not fork.get("reachable"):
        prepare_blockers.append("Cannot push because the candidate llama.cpp fork URL is not reachable.")

    commands = build_commands(root, fork_url, branch, worktree_dir)
    worktree_result = None
    can_prepare = root_clean and submodule_clean and root_patch_check["applies"] and submodule_patch_check["applies"]
    if prepare:
        if prepare_blockers:
            raise SystemExit("cannot prepare worktree while blockers remain: " + "; ".join(prepare_blockers))
        if not can_prepare:
            raise SystemExit("cannot prepare worktree; preflight failed")
        worktree_result = prepare_worktree(root, worktree_dir, branch, fork_url, push=push)

    return {
        "schema": "bitnet-i2sr-promotion-handoff-v1",
        "date": DATE,
        "fork_url": fork_url,
        "branch": branch,
        "worktree_dir": str(worktree_dir),
        "root_clean": root_clean,
        "root_status": root_status,
        "submodule_clean": submodule_clean,
        "submodule_status": submodule_status,
        "root_patch_check": root_patch_check,
        "submodule_patch_check": submodule_patch_check,
        "candidate_fork_probe": fork,
        "prepare_requested": prepare,
        "push_requested": push,
        "prepare_blockers": prepare_blockers,
        "worktree_result": worktree_result,
        "ready_for_handoff": can_prepare and (skip_remote_check or bool(fork.get("reachable"))),
        "blockers": blockers,
        "commands": commands,
    }


def md_bool(value: Any) -> str:
    if value is None:
        return "`n/a`"
    return "`true`" if bool(value) else "`false`"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def command_block(commands: list[str]) -> str:
    return "```bash\n" + "\n".join(commands) + "\n```"


def render_markdown(result: dict[str, Any]) -> str:
    fork = result["candidate_fork_probe"]
    rows = [
        ["root clean", md_bool(result["root_clean"])],
        ["submodule clean", md_bool(result["submodule_clean"])],
        ["root patch applies", md_bool(result["root_patch_check"].get("applies"))],
        ["submodule patch applies", md_bool(result["submodule_patch_check"].get("applies"))],
        ["candidate fork reachable", md_bool(fork.get("reachable"))],
        ["ready for handoff", md_bool(result["ready_for_handoff"])],
    ]
    blockers = [[item] for item in result["blockers"]] or [["none"]]
    commands = result["commands"]
    worktree = result.get("worktree_result") or {}
    return "\n\n".join(
        [
            f"# I2_SR Promotion Handoff, {DATE}",
            (
                "This report turns the remaining llama.cpp Git blocker into an executable "
                "handoff. The default script mode is non-mutating; it only prepares a "
                "worktree/commit or pushes when explicitly requested."
            ),
            f"Fork URL: `{result['fork_url']}`.",
            f"Branch: `{result['branch']}`.",
            "## Preflight",
            md_table(["check", "value"], rows),
            "## Blockers",
            md_table(["blocker"], blockers),
            "## Prepare Submodule Branch",
            command_block(commands["prepare_submodule_branch"]),
            "## Update Superproject Pointer",
            command_block(commands["update_superproject_pointer"]),
            "## Post-Promotion Gates",
            command_block(commands["post_promotion_gates"]),
            "## Worktree Result",
            md_table(
                ["field", "value"],
                [
                    ["prepared", md_bool(worktree.get("prepared"))],
                    ["pushed", md_bool(worktree.get("pushed"))],
                    ["commit", f"`{worktree.get('commit_sha', '')}`"],
                    ["worktree", f"`{worktree.get('worktree_dir', '')}`"],
                ],
            ),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--fork-url", default="https://github.com/sabdulmajid/llama.cpp.git")
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument(
        "--worktree-dir",
        type=Path,
        default=Path("benchmark_results/i2sr-promotion-llama-worktree"),
        help="Path for an isolated llama.cpp worktree when --prepare-worktree is used.",
    )
    parser.add_argument("--prepare-worktree", action="store_true", help="Create a local worktree and commit the submodule patch.")
    parser.add_argument("--push", action="store_true", help="Push the prepared branch to --fork-url. Requires --prepare-worktree.")
    parser.add_argument("--skip-remote-check", action="store_true", help="Skip git ls-remote preflight; not allowed with --push.")
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_results/i2sr_promotion_handoff_2026-05-13.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/results/i2sr_promotion_handoff_2026-05-13.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    worktree_dir = args.worktree_dir if args.worktree_dir.is_absolute() else root / args.worktree_dir
    result = build_handoff(
        root,
        fork_url=args.fork_url,
        branch=args.branch,
        worktree_dir=worktree_dir,
        prepare=args.prepare_worktree,
        push=args.push,
        skip_remote_check=args.skip_remote_check,
    )

    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
