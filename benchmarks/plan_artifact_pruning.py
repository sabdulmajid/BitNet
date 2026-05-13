#!/usr/bin/env python3
"""Build a dry-run pruning plan for local benchmark artifacts.

The benchmark repo intentionally keeps source, scripts, and public evidence in
Git, while large generated checkpoints and GGUF files stay ignored. This script
separates those two concerns: it protects files cited by the evidence manifest
and benchmark GGUF manifests, then reports which local generated artifacts can
be pruned or archived without invalidating the current public claims.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "bitnet-artifact-prune-plan-v1"


DEFAULT_EVIDENCE_MANIFEST = Path("benchmarks/results/evidence_manifest_2026-05-13.json")
DEFAULT_OUTPUT_JSON = Path("benchmarks/results/artifact_prune_plan_2026-05-13.json")
DEFAULT_OUTPUT_MD = Path("benchmarks/results/artifact_prune_plan_2026-05-13.md")


PROTECTED_CHECKPOINT_DIRS = {
    # Final training checkpoints used by the current reports.
    Path("checkpoints/qwen2.5-1.5b-fineweb-edu/step-5000"),
    Path("checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-5000/step-5000"),
    Path("checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-notiehead-5000/step-5000"),
    Path("checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-row-notiehead-5000/step-5000"),
    Path("checkpoints/qwen2.5-0.5b-fineweb-edu-12/step-1000"),
    Path("checkpoints/qwen2.5-0.5b-fineweb-edu-klonly-1000/step-1000"),
    Path("checkpoints/qwen2.5-0.5b-fineweb-edu-klonly-notiehead-1000/step-1000"),
    Path("checkpoints/qwen2.5-0.5b-fineweb-edu-row-1000/step-1000"),
    # Baseline exports and dense bridge materializations.
    Path("checkpoints/qwen2.5-0.5b-naive-ptq-tensor"),
    Path("checkpoints/qwen2.5-1.5b-naive-ptq-tensor"),
    Path("checkpoints/qwen2.5-1.5b-fineweb-edu/step-5000-static-ternary-dense-f16"),
    Path("checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-5000/step-5000-static-ternary-dense-f16"),
}


REBUILDABLE_DIRS = [
    Path("build"),
    Path("build-portable-avx2"),
    Path("build-qwen05b-tl2"),
]


CACHE_DIRS = [
    Path(".hf_cache"),
]


BENCHMARK_LARGE_FILE_THRESHOLD_BYTES = 256 * 1024 * 1024


@dataclass
class Candidate:
    path: str
    size_bytes: int
    reason: str


def run_text(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def du_bytes(path: Path, root: Path) -> int:
    if not path.exists():
        return 0
    out = run_text(["du", "-sb", str(path)], root)
    if not out:
        return 0
    return int(out.split()[0])


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{size} B"


def rel(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def normalize_repo_path(value: str) -> Path | None:
    if value.startswith(("http://", "https://", "s3://")):
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    if not path.parts:
        return None
    return path


def iter_json_paths(data: Any) -> Iterable[Path]:
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "path" and isinstance(value, str):
                normalized = normalize_repo_path(value)
                if normalized is not None:
                    yield normalized
            else:
                yield from iter_json_paths(value)
    elif isinstance(data, list):
        for value in data:
            yield from iter_json_paths(value)


def collect_protected_paths(root: Path, evidence_manifest: Path) -> tuple[set[Path], dict[str, int]]:
    protected: set[Path] = set()
    counts = defaultdict(int)

    manifest_path = root / evidence_manifest
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in data.get("entries", []):
            if isinstance(entry, dict) and entry.get("exists") and isinstance(entry.get("path"), str):
                normalized = normalize_repo_path(entry["path"])
                if normalized is not None:
                    protected.add(normalized)
                    counts["evidence_manifest_files"] += 1

    for manifest in sorted((root / "benchmarks").glob("*manifest*.json")):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for candidate in iter_json_paths(data):
            if (root / candidate).exists():
                protected.add(candidate)
                counts["benchmark_manifest_paths"] += 1

    for checkpoint_dir in PROTECTED_CHECKPOINT_DIRS:
        if (root / checkpoint_dir).exists():
            protected.add(checkpoint_dir)
            counts["protected_checkpoint_dirs"] += 1

    return protected, dict(counts)


def is_protected(path: Path, protected: set[Path]) -> bool:
    return any(path == item or is_relative_to(path, item) or is_relative_to(item, path) for item in protected)


def add_candidate(groups: dict[str, list[Candidate]], root: Path, group: str, path: Path, reason: str) -> None:
    groups[group].append(Candidate(str(path), du_bytes(root / path, root), reason))


def collect_candidates(root: Path, protected: set[Path]) -> dict[str, list[Candidate]]:
    groups: dict[str, list[Candidate]] = defaultdict(list)

    checkpoint_root = root / "checkpoints"
    if checkpoint_root.exists():
        for path in sorted(checkpoint_root.glob("*/step-*")):
            repo_path = rel(path, root)
            if not path.is_dir() or is_protected(repo_path, protected):
                continue
            add_candidate(
                groups,
                root,
                "prune_intermediate_checkpoints",
                repo_path,
                "Intermediate training checkpoint; final/evidence checkpoints are protected.",
            )

    for path in REBUILDABLE_DIRS:
        if (root / path).exists() and not is_protected(path, protected):
            add_candidate(groups, root, "remove_rebuildable_build_dirs", path, "CMake/runtime build output; can be regenerated.")

    for path in CACHE_DIRS:
        if (root / path).exists() and not is_protected(path, protected):
            add_candidate(groups, root, "remove_local_caches", path, "Download/cache directory; can be repopulated.")

    benchmark_root = root / "benchmark_results"
    if benchmark_root.exists():
        for file_path in sorted(benchmark_root.rglob("*")):
            if not file_path.is_file():
                continue
            repo_path = rel(file_path, root)
            if is_protected(repo_path, protected):
                continue
            size = file_path.stat().st_size
            if size >= BENCHMARK_LARGE_FILE_THRESHOLD_BYTES:
                groups["archive_unreferenced_large_benchmark_files"].append(
                    Candidate(
                        str(repo_path),
                        size,
                        "Large benchmark byproduct not directly cited by the evidence manifest.",
                    )
                )

    model_root = root / "models"
    if model_root.exists():
        for path in sorted(p for p in model_root.iterdir() if p.is_dir()):
            repo_path = rel(path, root)
            add_candidate(
                groups,
                root,
                "keep_by_default_model_artifacts",
                repo_path,
                "Generated model/GGUF directory; useful for reruns and protected if referenced by a benchmark manifest.",
            )

    return dict(groups)


def group_total(candidates: list[Candidate]) -> int:
    return sum(candidate.size_bytes for candidate in candidates)


def shell_quote(path: str) -> str:
    return "'" + path.replace("'", "'\"'\"'") + "'"


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Artifact Prune Plan",
        "",
        f"Generated UTC: `{plan['generated_utc']}`",
        f"Git HEAD at generation time: `{plan.get('git_head') or 'unknown'}`",
        "",
        "This is a dry-run plan. No files were deleted by the generator.",
        "",
        "## Guardrails",
        "",
        f"- Evidence manifest: `{plan['evidence_manifest']}`",
        f"- Protected evidence files: `{plan['protected_counts'].get('evidence_manifest_files', 0)}`",
        f"- Protected benchmark-manifest paths: `{plan['protected_counts'].get('benchmark_manifest_paths', 0)}`",
        f"- Protected checkpoint directories: `{plan['protected_counts'].get('protected_checkpoint_dirs', 0)}`",
        "",
        "## Candidate Storage",
        "",
        "| class | action | items | total |",
        "| --- | --- | ---: | ---: |",
    ]

    action_labels = {
        "prune_intermediate_checkpoints": "safe after review",
        "remove_rebuildable_build_dirs": "safe after review",
        "remove_local_caches": "safe after review",
        "archive_unreferenced_large_benchmark_files": "archive/delete after review",
        "keep_by_default_model_artifacts": "keep by default",
    }
    for group_name, candidates in plan["groups"].items():
        lines.append(
            f"| `{group_name}` | {action_labels.get(group_name, 'review')} | "
            f"{len(candidates)} | {format_bytes(group_total([Candidate(**item) for item in candidates]))} |"
        )

    lines.extend(
        [
            "",
            "## Recommended Sequence",
            "",
            "1. Rebuild the evidence manifest and confirm it still reports zero missing artifacts.",
            "2. Remove only `prune_intermediate_checkpoints` first; this is the main storage win and preserves final checkpoints.",
            "3. Optionally remove CMake build directories and `.hf_cache`; they are rebuildable but may cost time to recreate.",
            "4. Do not remove `keep_by_default_model_artifacts` unless you intentionally accept rerunning GGUF conversion/quantization.",
            "5. Rebuild the evidence manifest again after pruning.",
            "",
            "## Manual Commands",
            "",
            "Review these paths before running any command.",
            "",
        ]
    )

    for group_name in (
        "prune_intermediate_checkpoints",
        "remove_rebuildable_build_dirs",
        "remove_local_caches",
        "archive_unreferenced_large_benchmark_files",
    ):
        candidates = [Candidate(**item) for item in plan["groups"].get(group_name, [])]
        if not candidates:
            continue
        lines.extend([f"### {group_name}", "", "```bash"])
        for candidate in candidates:
            lines.append(f"rm -rf {shell_quote(candidate.path)}")
        lines.extend(["```", ""])

    if plan["groups"].get("keep_by_default_model_artifacts"):
        lines.extend(["## Model Artifacts Kept By Default", ""])
        for item in plan["groups"]["keep_by_default_model_artifacts"]:
            candidate = Candidate(**item)
            lines.append(f"- `{candidate.path}`: {format_bytes(candidate.size_bytes)}")
        lines.append("")

    return "\n".join(lines)


def build_plan(root: Path, evidence_manifest: Path) -> dict[str, Any]:
    protected, protected_counts = collect_protected_paths(root, evidence_manifest)
    groups = collect_candidates(root, protected)
    git_head = run_text(["git", "rev-parse", "--short=12", "HEAD"], root)
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": git_head,
        "evidence_manifest": str(evidence_manifest),
        "protected_counts": protected_counts,
        "protected_paths": sorted(str(path) for path in protected),
        "groups": {name: [asdict(candidate) for candidate in candidates] for name, candidates in sorted(groups.items())},
        "totals": {name: group_total(candidates) for name, candidates in sorted(groups.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--evidence-manifest", type=Path, default=DEFAULT_EVIDENCE_MANIFEST)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    plan = build_plan(root, args.evidence_manifest)

    output_json = root / args.output_json
    output_md = root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(plan) + "\n", encoding="utf-8")

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    for name, total in plan["totals"].items():
        print(f"{name}: {format_bytes(total)}")


if __name__ == "__main__":
    main()
