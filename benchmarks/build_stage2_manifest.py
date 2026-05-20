#!/usr/bin/env python3
"""Materialize a Stage-2 warm-up checkpoint manifest.

The downstream BitDistill jobs should consume an explicit checkpoint manifest
instead of guessing whether a run saved its state dict at the output root or in
the latest snapshot directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path(
    "checkpoints/bitdistill-glue-stage2-curve/"
    "Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-40k"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_sha(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_latest_snapshot(output_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if (path / "custom_state_dict.pt").exists() and (path / "metrics.json").exists():
            candidates.append((step, path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* snapshot with custom_state_dict.pt under {output_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    root_metrics_path = output_dir / "metrics.json"
    if not root_metrics_path.exists():
        raise FileNotFoundError(root_metrics_path)
    snapshot_dir = args.snapshot_dir or discover_latest_snapshot(output_dir)
    state_dict_path = snapshot_dir / "custom_state_dict.pt"
    snapshot_metrics_path = snapshot_dir / "metrics.json"
    if not state_dict_path.exists():
        raise FileNotFoundError(state_dict_path)
    if not snapshot_metrics_path.exists():
        raise FileNotFoundError(snapshot_metrics_path)

    root_metrics = read_json(root_metrics_path)
    snapshot_metrics = read_json(snapshot_metrics_path)
    last = root_metrics.get("last", {}) if isinstance(root_metrics.get("last"), dict) else {}
    repo_root = Path(".").resolve()
    llama_dir = repo_root / "3rdparty/llama.cpp"
    manifest_bitnet_commit = git_sha(repo_root)
    manifest_llama_commit = git_sha(llama_dir) if llama_dir.exists() else ""
    producer_bitnet_commit = args.producer_bitnet_commit or manifest_bitnet_commit
    producer_llama_commit = args.producer_llama_cpp_commit or manifest_llama_commit
    return {
        "schema": "bitnet-stage2-checkpoint-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "job_id": args.job_id,
        "model": args.model,
        "stage": root_metrics.get("stage"),
        "method": root_metrics.get("method"),
        "scale_mode": root_metrics.get("scale_mode"),
        "steps": root_metrics.get("steps"),
        "token_presentations": root_metrics.get("effective_train_token_presentations"),
        "final_ce": last.get("ce"),
        "final_loss": last.get("loss"),
        "final_lr": last.get("lr"),
        "final_grad_norm": last.get("grad_norm"),
        "output_dir": str(output_dir),
        "root_metrics_path": str(root_metrics_path),
        "snapshot_dir": str(snapshot_dir),
        "snapshot_metrics_path": str(snapshot_metrics_path),
        "state_dict_path": str(state_dict_path),
        "state_dict_sha256": file_sha256(state_dict_path),
        "snapshot_complete": bool(snapshot_metrics.get("snapshot", {}).get("complete"))
        if isinstance(snapshot_metrics.get("snapshot"), dict)
        else None,
        "git": {
            "bitnet_commit": producer_bitnet_commit,
            "llama_cpp_commit": producer_llama_commit,
            "producer_bitnet_commit": producer_bitnet_commit,
            "producer_llama_cpp_commit": producer_llama_commit,
            "manifest_bitnet_commit": manifest_bitnet_commit,
            "manifest_llama_cpp_commit": manifest_llama_commit,
        },
        "downstream": {
            "status": args.downstream_status,
            "failed_job_id": "10071",
            "rerun_job_id": args.downstream_rerun_job_id,
            "rerun_output_dir": args.downstream_output_dir,
            "failure_mode": "downstream expected root custom_state_dict.pt, but this Stage-2 run saved snapshot state dicts",
            "recommended_init_state_dict": str(state_dict_path),
        },
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def render_markdown(manifest: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            f"# Stage-2 Checkpoint Manifest: {manifest['run_id']}",
            "This manifest pins the exact warm-up checkpoint consumed by downstream BitDistill jobs.",
            md_table(
                ["field", "value"],
                [
                    ["job_id", manifest["job_id"]],
                    ["model", manifest["model"]],
                    ["method", manifest["method"]],
                    ["scale_mode", manifest["scale_mode"]],
                    ["steps", manifest["steps"]],
                    ["token_presentations", manifest["token_presentations"]],
                    ["final_ce", manifest["final_ce"]],
                    ["state_dict_path", manifest["state_dict_path"]],
                    ["bitnet_commit", manifest["git"]["bitnet_commit"]],
                    ["llama_cpp_commit", manifest["git"]["llama_cpp_commit"]],
                    ["downstream_status", manifest["downstream"]["status"]],
                    ["downstream_rerun_job_id", manifest["downstream"].get("rerun_job_id", "")],
                    ["downstream_rerun_output_dir", manifest["downstream"].get("rerun_output_dir", "")],
                ],
            ),
            "## Downstream Note",
            (
                "Job 10071 failed before training/evaluation because it looked for a root-level "
                "`custom_state_dict.pt`. The valid state dict is the snapshot path recorded above."
            ),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--run-id", default="qwen25-05b-bitdistill-tensor-stage2-40k-job10070")
    parser.add_argument("--job-id", default="10070")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--producer-bitnet-commit", default="")
    parser.add_argument("--producer-llama-cpp-commit", default="")
    parser.add_argument("--downstream-status", default="pending_rerun")
    parser.add_argument("--downstream-rerun-job-id", default="")
    parser.add_argument("--downstream-output-dir", default="")
    parser.add_argument("--output-json", type=Path, default=Path("benchmarks/results/stage2_manifest_2026-05-20.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/results/stage2_manifest_2026-05-20.md"))
    args = parser.parse_args()

    manifest = build_manifest(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(manifest).rstrip() + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
