#!/usr/bin/env python3
"""Consolidate external inputs required before the remaining benchmarks are honest.

This audit is intentionally not a success gate. It answers a narrower question:
given the current artifact state, what concrete input is still missing before
we can promote I2_SR or run real MoE/Kimi benchmarks?
"""

from __future__ import annotations

import os
import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def latest_artifact(root: Path, pattern: str, fallback: str) -> Path:
    matches = sorted(root.glob(pattern))
    return matches[-1] if matches else root / fallback


def run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True)


def probe_remote(url: str, *, cwd: Path) -> dict[str, Any]:
    completed = run(["git", "ls-remote", "--heads", url], cwd=cwd)
    return {
        "url": url,
        "returncode": completed.returncode,
        "reachable": completed.returncode == 0,
        "stderr": completed.stderr.strip(),
    }


def local_kimi_artifacts(root: Path) -> list[str]:
    search_roots = [root / "models", root / "checkpoints", root / "benchmark_results"]
    artifacts: list[str] = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if "kimi" in path.name.lower() or "qwen2moe" in path.name.lower():
                artifacts.append(str(path.relative_to(root)))
    return sorted(artifacts)


def is_tiny_fixture_artifact(path: str) -> bool:
    lowered = path.lower()
    return "tiny-qwen2moe-fixture" in lowered or "tiny_qwen2moe_fixture" in lowered


def make_requirement(name: str, status: str, evidence: str, unblock_action: str) -> dict[str, str]:
    return {
        "name": name,
        "status": status,
        "evidence": evidence,
        "unblock_action": unblock_action,
    }


def build_audit(root: Path, *, candidate_fork_url: str) -> dict[str, Any]:
    objective = read_json(latest_artifact(root, "benchmark_results/objective_completion_audit_*.json", "benchmark_results/objective_completion_audit_2026-05-13.json"))
    i2sr = read_json(root / "benchmark_results/i2sr_submodule_promotion_audit_2026-05-13.json")
    handoff = read_json(root / "benchmark_results/i2sr_promotion_handoff_2026-05-13.json")
    moe = read_json(latest_artifact(root, "benchmark_results/moe_support_audit_*.json", "benchmark_results/moe_support_audit_2026-05-05.json"))
    moe_contract = read_json(latest_artifact(root, "benchmark_results/moe_packing_contract_*.json", "benchmark_results/moe_packing_contract_2026-05-13.json"))
    scope = read_json(latest_artifact(root, "benchmark_results/product_scope_gate_*.json", "benchmark_results/product_scope_gate_2026-05-13.json"))
    fork_probe = probe_remote(candidate_fork_url, cwd=root)
    gh_path = shutil.which("gh")
    kimi_artifacts = local_kimi_artifacts(root)
    production_moe_artifacts = [artifact for artifact in kimi_artifacts if not is_tiny_fixture_artifact(artifact)]
    tiny_qwen2moe = moe.get("tiny_qwen2moe_fixture", {}) if isinstance(moe.get("tiny_qwen2moe_fixture"), dict) else {}

    i2sr_blocked = not bool(i2sr.get("promotion_ready"))
    handoff_worktree = handoff.get("worktree_result") if isinstance(handoff.get("worktree_result"), dict) else {}
    handoff_prepared = bool(handoff_worktree.get("prepared"))
    handoff_commit = handoff_worktree.get("commit_sha") or ""
    moe_verdict = moe_contract.get("verdict", {}) if isinstance(moe_contract.get("verdict"), dict) else {}
    moe_gates = moe.get("productization_gates", [])
    failed_moe_gates = [gate.get("name") for gate in moe_gates if isinstance(gate, dict) and not gate.get("passed")]

    writable_fork_action = (
        "No action needed; the fork branch is reachable and promotion_ready is true."
        if not i2sr_blocked
        else "Create/provide a reachable writable llama.cpp fork URL, then push the prepared submodule I2_SR patch branch."
    )
    requirements = [
        make_requirement(
            "Writable llama.cpp fork or branch",
            "missing" if i2sr_blocked else "ready",
            (
                f"promotion_ready={i2sr.get('promotion_ready')}; "
                f"candidate_fork_reachable={fork_probe['reachable']}; "
                f"submodule_patch_applies={i2sr.get('patch_applies_cleanly')}; "
                f"local_handoff_prepared={handoff_prepared}; "
                f"local_handoff_commit={handoff_commit or 'n/a'}"
            ),
            writable_fork_action,
        ),
        make_requirement(
            "GitHub automation credential",
            "optional_missing" if gh_path is None else "available",
            f"gh_path={gh_path or 'not_found'}",
            "Install/authenticate GitHub CLI or refresh the GitHub connector token if repository creation/push automation is desired.",
        ),
        make_requirement(
            "Local trained Kimi/Qwen2MoE model artifact",
            "missing" if not production_moe_artifacts else "available",
            f"trained_artifacts={len(production_moe_artifacts)}; tiny_fixture_passed={tiny_qwen2moe.get('passed')}; all_moe_named_artifacts={len(kimi_artifacts)}",
            "Provide a licensed trained Kimi or Qwen2MoE checkpoint/tokenizer artifact plus its FP and quantized baselines. The tiny random Qwen2MoE fixture is not a substitute.",
        ),
        make_requirement(
            "MoE 3D expert tensor packing support",
            "missing" if not moe_verdict.get("moe_packing_ready") else "ready",
            (
                f"tl2_3d={moe_verdict.get('merged_3d_tl2_supported')}; "
                f"i2sr_3d={moe_verdict.get('merged_3d_i2s_i2sr_supported')}; "
                f"2d_control={moe_verdict.get('dense_2d_i2s_control_supported')}"
            ),
            "Implement remaining TL2 3D expert packing and full MoE GGUF/runtime byte tests before any Kimi runtime benchmark.",
        ),
        make_requirement(
            "MoE quality/locality benchmark artifacts",
            "missing" if failed_moe_gates else "ready",
            f"failed_moe_gates={len(failed_moe_gates)}; trained_artifacts={len(production_moe_artifacts)}; tiny_fixture_passed={tiny_qwen2moe.get('passed')}",
            "Run router accuracy, expert locality, quality, throughput, and RSS benchmarks after model and packing support exist.",
        ),
    ]
    missing = [item for item in requirements if item["status"] == "missing"]

    return {
        "schema": "bitnet-unblock-requirements-v1",
        "date": DATE,
        "objective_status": objective.get("completion_status"),
        "objective_complete_count": objective.get("complete_count"),
        "objective_check_count": objective.get("check_count"),
        "scope_status": scope.get("scope_status"),
        "candidate_fork_probe": fork_probe,
        "gh_path": gh_path,
        "local_kimi_artifacts": kimi_artifacts,
        "local_trained_moe_artifacts": production_moe_artifacts,
        "tiny_qwen2moe_fixture_passed": tiny_qwen2moe.get("passed"),
        "requirements": requirements,
        "missing_count": len(missing),
        "can_continue_productively_without_input": len(missing) == 0,
        "next_required_input": missing[0]["unblock_action"] if missing else "",
    }


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    rows = [
        [
            item["name"],
            item["status"],
            item["evidence"].replace("|", "/"),
            item["unblock_action"].replace("|", "/"),
        ]
        for item in result["requirements"]
    ]
    fork = result["candidate_fork_probe"]
    fork_stderr = str(fork["stderr"]).replace("\n", " ")
    next_required_input = (result["next_required_input"] or "none").rstrip(".")
    return "\n\n".join(
        [
            f"# Unblock Requirements Audit, {DATE}",
            (
                "This audit consolidates the external inputs required before the remaining "
                "I2_SR promotion and MoE/Kimi benchmark claims can be completed honestly."
            ),
            f"Objective status: `{result.get('objective_status')}` (`{result.get('objective_complete_count')}/{result.get('objective_check_count')}` complete).",
            f"Product scope: `{result.get('scope_status')}`.",
            f"Can continue productively without new input: `{str(result['can_continue_productively_without_input']).lower()}`.",
            f"Next required input: {next_required_input}.",
            "## Requirements",
            md_table(["requirement", "status", "evidence", "unblock action"], rows),
            "## Candidate Fork Probe",
            md_table(
                ["field", "value"],
                [
                    ["url", f"`{fork['url']}`"],
                    ["returncode", f"`{fork['returncode']}`"],
                    ["reachable", f"`{str(fork['reachable']).lower()}`"],
                    ["stderr", f"`{fork_stderr}`"],
                ],
            ),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--candidate-fork-url", default="https://github.com/sabdulmajid/llama.cpp.git")
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/unblock_requirements_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/unblock_requirements_{DATE}.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    result = build_audit(root, candidate_fork_url=args.candidate_fork_url)
    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
