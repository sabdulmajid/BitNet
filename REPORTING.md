# Reporting Rules

This fork should be presented as an evidence-led research artifact.

## Rules

1. Public claims must cite JSON/manifest artifacts or be explicitly marked as
   interpretation.
2. Missing metrics or predictions mean `pending` or `incomplete`, not failure
   or success.
3. A report with `complete 0/0` is invalid unless it explicitly states why no
   rows were expected.
4. Paper-style tensor-scale BitDistill rows and row-scale retrofit variants
   must be labeled separately.
5. Runtime speed, file size, RSS, task quality, and LM perplexity are separate
   gates.
6. MoE/Kimi claims require trained-model quality and routed runtime evidence.

## Fail-Closed Validator

```bash
python benchmarks/validate_reports_fail_closed.py <json-or-md-report> [...]
```

The validator rejects silent empty reports. For example, the stale 2026-05-17
controlled-curve files are expected to fail because date-based postprocessing
missed the real 2026-05-15/16 artifacts and produced `0/0` rows.

## Public Doc Validator

```bash
python benchmarks/validate_public_docs.py
```

This validator checks that the headline README, claims page, and runtime
contract still contain the canonical evidence-bundle numbers and that every
artifact referenced by the bundle exists.

## Reproduction Gap Reports

Gap reports should distinguish:

- the short/default BitNet-SFT baseline,
- the best tuned BitNet-SFT budget row,
- controlled BitDistill Stage-2 rows,
- FP16 recovery gap,
- loss/gradient-balance telemetry,
- selected-layer pairwise gradient cosines for CE, logits KD, and attention KD
  when objective compatibility is under investigation.

The current report is:

```text
benchmarks/results/bitdistill_reproduction_gap_2026-05-23.md
```

Stage-2 jobs that extend an existing checkpoint must record both segment and
cumulative token presentations. The current pending extension is:

```text
benchmarks/results/stage2_655m_submission_2026-05-23.md
```

The dependent handoff submission is:

```text
benchmarks/results/stage2_655m_handoff_submission_2026-05-23.md
```

Short gradient-balance telemetry jobs must be labeled as diagnostics, not
quality benchmarks. The current equalized-gamma submission is:

```text
benchmarks/results/gamma60_telemetry_submission_2026-05-23.md
```

Gradient magnitudes and gradient directions answer different questions. A
balanced attention/CE norm ratio does not imply compatible updates, so reports
that interpret adaptive loss weighting should include pairwise gradient
cosines when available and state the exact parameter probe used.

`validate_public_docs.py` validates this submission as a diagnostic handoff:
dependency, coefficient, telemetry settings, comparison report, and caveat must
all be present before the public docs pass.

Active long-running gates can be monitored without changing quality claims:

```bash
python benchmarks/monitor_active_stage2_extension.py
```

The generated report must keep `quality_claim` set to `none` until downstream
metrics and prediction traces exist. The active monitor must also track the
expected downstream `metrics.json` and `eval_predictions.jsonl` paths; these
paths are artifact-existence checks only and must not report MNLI accuracy.

Queued Slurm batch scripts should also be audited after resubmission or script
hardening:

```bash
python benchmarks/audit_active_slurm_batch_scripts.py
```

This verifies the stored Slurm script contents for the active handoff and
telemetry jobs. It is job-plumbing evidence only, not quality evidence.

## Preferred Public Labels

| Label | Meaning |
| --- | --- |
| `paper-reproduction` | Same task family, model class, quantization semantics, and recipe target as the paper. |
| `paper-inspired` | Uses similar ingredients but changes budget, backbone, reduction, scale granularity, or task formulation. |
| `retrofit-variant` | Fork-specific extensions such as row-scale ternary and `I2_SR`. |

## Do Not Claim

- arbitrary FP16/BF16 to BitNet conversion,
- paper-level BitDistill reproduction,
- Q4-quality parity for `I2_SR`,
- product-ready native classification,
- Kimi/MoE support.
