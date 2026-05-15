#!/usr/bin/env python3
"""Audit the local BitDistill loss contract and live loss-balance evidence."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
ATTENTION_CE_RISK_THRESHOLD = 100.0


def line_number(text: str, needle: str) -> int | None:
    for index, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return index
    return None


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, evidence: str, blocker: str = "") -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "evidence": evidence,
            "blocker": "" if passed else blocker,
        }
    )


def static_contract(source: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks_to_run = [
        (
            "SubLN wraps projection inputs before BitLinear replacement",
            "if args.use_subln:",
            "replace_linear_layers(",
            source.find("if args.use_subln:") != -1
            and source.find("replace_linear_layers(") != -1
            and source.find("if args.use_subln:") < source.find("replace_linear_layers("),
            "SubLN should be inserted before nested projections are replaced.",
        ),
        (
            "Attention relation KD uses batchmean KL",
            "F.kl_div(torch.log(student_rows), teacher_rows, reduction=\"batchmean\", log_target=False)",
            None,
            "F.kl_div(torch.log(student_rows), teacher_rows, reduction=\"batchmean\", log_target=False)" in source,
            "attention KL reduction changed or was not found.",
        ),
        (
            "Attention Q/K/V reduction defaults to sum",
            "parser.add_argument(\"--attention-qkv-reduction\", choices=[\"sum\", \"mean\"], default=\"sum\")",
            None,
            "parser.add_argument(\"--attention-qkv-reduction\", choices=[\"sum\", \"mean\"], default=\"sum\")" in source,
            "default Q/K/V reduction is not the audited sum setting.",
        ),
        (
            "Logits KD temperature scaling defaults to none",
            "parser.add_argument(\"--logit-kd-temperature-scale\", choices=[\"none\", \"square\"], default=\"none\")",
            None,
            "parser.add_argument(\"--logit-kd-temperature-scale\", choices=[\"none\", \"square\"], default=\"none\")" in source,
            "default logits KD temperature scaling changed.",
        ),
        (
            "Stage-3 loss is direct weighted sum",
            "loss = ce + weighted_logit_kd + weighted_attention_kd",
            None,
            "loss = ce + weighted_logit_kd + weighted_attention_kd" in source,
            "Stage-3 weighted-sum loss formula was not found.",
        ),
        (
            "Attention weight default is local-safe, not paper gamma",
            "parser.add_argument(\"--attention-kd-weight\", type=float, default=100.0)",
            None,
            "parser.add_argument(\"--attention-kd-weight\", type=float, default=100.0)" in source,
            "default attention weight no longer matches the audited local-safe value.",
        ),
    ]
    for name, first, second, passed, blocker in checks_to_run:
        if second is None:
            evidence = f"line={line_number(source, first)}, needle={first}"
        else:
            evidence = (
                f"first_line={line_number(source, first)}, "
                f"second_line={line_number(source, second)}"
            )
        add_check(checks, name, passed, evidence, blocker)
    return checks


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_live_control(controlled_curve: dict[str, Any]) -> dict[str, Any]:
    rows = controlled_curve.get("rows", []) if isinstance(controlled_curve.get("rows"), list) else []
    live_rows: list[dict[str, Any]] = []
    ratios: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        live = row.get("live_log", {}) if isinstance(row.get("live_log"), dict) else {}
        latest = live.get("latest", {}) if isinstance(live.get("latest"), dict) else {}
        latest_ratio = live.get("latest_weighted_attention_to_ce")
        max_ratio = live.get("max_weighted_attention_to_ce")
        if isinstance(latest_ratio, (int, float)):
            ratios.append(float(latest_ratio))
        if isinstance(max_ratio, (int, float)):
            ratios.append(float(max_ratio))
        if live.get("exists"):
            live_rows.append(
                {
                    "job_id": row.get("job_id"),
                    "label": row.get("label"),
                    "state": (row.get("squeue") or {}).get("state") if isinstance(row.get("squeue"), dict) else None,
                    "latest_step": live.get("latest_step"),
                    "latest_ce": latest.get("ce"),
                    "latest_attention_kd": latest.get("attention_kd"),
                    "latest_weighted_attention_kd": latest.get("weighted_attention_kd"),
                    "latest_weighted_attention_to_ce": latest_ratio,
                    "max_weighted_attention_to_ce": max_ratio,
                    "parsed_steps": live.get("parsed_steps"),
                }
            )
    max_observed = max(ratios) if ratios else None
    return {
        "live_rows": live_rows,
        "max_observed_weighted_attention_to_ce": max_observed,
        "risk": isinstance(max_observed, float) and max_observed >= ATTENTION_CE_RISK_THRESHOLD,
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(item) for item in row) + " |")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|")


def render_markdown(result: dict[str, Any]) -> str:
    check_rows = [
        [check["name"], "pass" if check["passed"] else "fail", check["evidence"], check["blocker"]]
        for check in result["checks"]
    ]
    live_rows = [
        [
            row.get("job_id"),
            row.get("label"),
            row.get("state"),
            row.get("latest_step"),
            row.get("latest_ce"),
            row.get("latest_attention_kd"),
            row.get("latest_weighted_attention_kd"),
            row.get("latest_weighted_attention_to_ce"),
            row.get("max_weighted_attention_to_ce"),
            row.get("parsed_steps"),
        ]
        for row in result["live"]["live_rows"]
    ]
    verdict = "loss-normalization risk" if result["risk"] else "no current live risk"
    return "\n\n".join(
        [
            f"# BitDistill Loss Contract Audit, {result['date']}",
            f"Status: **{result['status']}**. Verdict: **{verdict}**.",
            "This audit is not a quality result. It checks whether the local implementation and live logs make the paper-gamma setting numerically risky under the current loss normalization.",
            "## Static Contract",
            md_table(["check", "status", "evidence", "blocker"], check_rows),
            "## Live Loss Balance",
            md_table(
                [
                    "job",
                    "label",
                    "state",
                    "step",
                    "CE",
                    "attention KD",
                    "weighted attention KD",
                    "weighted attention / CE",
                    "max weighted attention / CE",
                    "parsed steps",
                ],
                live_rows,
            ),
            "## Interpretation",
            (
                f"The risk threshold is weighted-attention/CE >= `{ATTENTION_CE_RISK_THRESHOLD:.1f}`. "
                f"The max observed ratio is `{fmt(result['live']['max_observed_weighted_attention_to_ce'])}`. "
                "If final BitDistill quality remains weak, the first follow-up is loss-normalization and gradient-balance telemetry, not another broad model/task sweep."
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source", type=Path, default=Path("train_bitdistill.py"))
    parser.add_argument("--controlled-json", type=Path, default=Path(f"benchmark_results/bitdistill_controlled_curve_{DATE}.json"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_loss_contract_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_loss_contract_{DATE}.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    source_path = args.source if args.source.is_absolute() else root / args.source
    source = source_path.read_text(encoding="utf-8")
    checks = static_contract(source)
    controlled_path = args.controlled_json if args.controlled_json.is_absolute() else root / args.controlled_json
    live = summarize_live_control(load_json(controlled_path))
    risk = live["risk"]
    status = "loss_normalization_risk" if risk else "pass"
    result = {
        "schema": "bitdistill_loss_contract.v1",
        "date": DATE,
        "status": status,
        "passed": all(check["passed"] for check in checks),
        "risk": risk,
        "risk_threshold_weighted_attention_to_ce": ATTENTION_CE_RISK_THRESHOLD,
        "source": str(source_path.relative_to(root)),
        "controlled_json": str(controlled_path.relative_to(root)),
        "checks": checks,
        "live": live,
    }

    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(result) + "\n", encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
