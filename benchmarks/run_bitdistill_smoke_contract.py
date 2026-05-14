#!/usr/bin/env python3
"""Run fast CPU smoke checks for the BitDistill training entrypoint."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = datetime.now(timezone.utc).date().isoformat()


def run_command(command: list[str], *, cwd: Path, timeout: int = 180) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "returncode": proc.returncode,
        "output_tail": proc.stdout[-6000:],
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, evidence: str, blocker: str = "") -> None:
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence, "blocker": "" if passed else blocker})


def build_report(result: dict[str, Any]) -> str:
    rows = [
        [
            check["name"],
            "pass" if check["passed"] else "fail",
            str(check["evidence"]).replace("\n", " "),
            str(check.get("blocker", "")).replace("\n", " "),
        ]
        for check in result["checks"]
    ]
    lines = [
        f"# BitDistill Smoke Contract, {result['date']}",
        f"Overall status: `{'pass' if result['passed'] else 'fail'}`.",
        f"Work dir: `{result['work_dir']}`.",
        "",
        "| check | status | evidence | blocker |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--work-dir", type=Path, default=Path(f"benchmark_results/bitdistill-smoke-contract-{DATE}"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_smoke_contract_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_smoke_contract_{DATE}.md"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    work_dir = args.work_dir
    if work_dir.exists():
        if "bitdistill-smoke-contract" not in work_dir.name:
            raise SystemExit(f"refusing to remove unexpected work dir: {work_dir}")
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    commands = {
        "help": [py, "train_bitdistill.py", "--help"],
        "py_compile": [py, "-m", "py_compile", "train_bitdistill.py"],
        "continued_pretrain": [
            py,
            "train_bitdistill.py",
            "--smoke-test",
            "--stage",
            "continued_pretrain",
            "--method",
            "bitdistill",
            "--max-steps",
            "2",
            "--device",
            "cpu",
            "--per-device-batch-size",
            "1",
            "--max-seq-len",
            "32",
            "--log-every-steps",
            "1",
            "--output-dir",
            str(work_dir / "continued_pretrain"),
        ],
        "task_sft": [
            py,
            "train_bitdistill.py",
            "--smoke-test",
            "--stage",
            "task_sft",
            "--method",
            "bitdistill",
            "--max-steps",
            "2",
            "--device",
            "cpu",
            "--per-device-batch-size",
            "1",
            "--eval-batch-size",
            "1",
            "--max-seq-len",
            "32",
            "--log-every-steps",
            "1",
            "--output-dir",
            str(work_dir / "task_sft"),
        ],
    }
    runs = {name: run_command(command, cwd=repo_root) for name, command in commands.items()}
    continued = read_json(work_dir / "continued_pretrain" / "metrics.json")
    task = read_json(work_dir / "task_sft" / "metrics.json")

    checks: list[dict[str, Any]] = []
    for name, run in runs.items():
        add_check(checks, f"{name} command exits zero", run["returncode"] == 0, f"returncode={run['returncode']}", "command failed")

    continued_prep = continued.get("preparation", {}) if isinstance(continued.get("preparation"), dict) else {}
    continued_last = continued.get("last", {}) if isinstance(continued.get("last"), dict) else {}
    add_check(checks, "continued-pretrain writes metrics", bool(continued), str(work_dir / "continued_pretrain" / "metrics.json"), "missing metrics")
    add_check(checks, "continued-pretrain takes two steps", continued.get("steps") == 2, f"steps={continued.get('steps')}", "unexpected step count")
    add_check(
        checks,
        "continued-pretrain uses BitLinear and SubLN",
        int(continued_prep.get("bitlinear_replaced", 0)) > 0 and int(continued_prep.get("subln_inserted", 0)) > 0,
        f"bitlinear={continued_prep.get('bitlinear_replaced')}, subln={continued_prep.get('subln_inserted')}",
        "BitNet preparation did not run",
    )
    add_check(checks, "continued-pretrain CE is finite", finite(continued_last.get("ce")), f"ce={continued_last.get('ce')}", "non-finite CE")

    task_prep = task.get("preparation", {}) if isinstance(task.get("preparation"), dict) else {}
    task_last = task.get("last", {}) if isinstance(task.get("last"), dict) else {}
    task_eval = task.get("eval", {}) if isinstance(task.get("eval"), dict) else {}
    add_check(checks, "task-sft writes metrics", bool(task), str(work_dir / "task_sft" / "metrics.json"), "missing metrics")
    add_check(checks, "task-sft takes two steps", task.get("steps") == 2, f"steps={task.get('steps')}", "unexpected step count")
    add_check(
        checks,
        "task-sft uses BitLinear and SubLN",
        int(task_prep.get("bitlinear_replaced", 0)) > 0 and int(task_prep.get("subln_inserted", 0)) > 0,
        f"bitlinear={task_prep.get('bitlinear_replaced')}, subln={task_prep.get('subln_inserted')}",
        "BitNet preparation did not run",
    )
    add_check(checks, "task-sft logits KD is finite", finite(task_last.get("weighted_logit_kd")), f"weighted_logit_kd={task_last.get('weighted_logit_kd')}", "non-finite logits KD")
    add_check(checks, "task-sft attention KD is finite", finite(task_last.get("weighted_attention_kd")), f"weighted_attention_kd={task_last.get('weighted_attention_kd')}", "non-finite attention KD")
    add_check(checks, "task-sft eval accuracy is finite", finite(task_eval.get("accuracy")), f"accuracy={task_eval.get('accuracy')}", "non-finite accuracy")

    result = {
        "schema": "bitdistill-smoke-contract-v1",
        "date": DATE,
        "work_dir": str(work_dir),
        "passed": all(check["passed"] for check in checks),
        "check_count": len(checks),
        "failed": [check["name"] for check in checks if not check["passed"]],
        "checks": checks,
        "runs": runs,
        "continued_pretrain_metrics": continued,
        "task_sft_metrics": task,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_report(result), encoding="utf-8")
    print(build_report(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
