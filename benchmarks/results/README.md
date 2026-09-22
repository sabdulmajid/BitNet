# Public Evidence Index

This directory is an append-only archive of sanitized reports. Start with the
current snapshot; do not infer project status from the newest filename alone.

## Current Source of Truth

- [Current evidence summary](current_evidence_2026-09-22.md)
- [Current evidence JSON](current_evidence_2026-09-22.json)
- [Public claim ledger](../../CLAIMS.md)
- [Research status](../../docs/RESEARCH_STATUS.md)
- [Roadmap](../../docs/ROADMAP.md)

## Primary Evidence

| Area | Human report | Machine-readable evidence |
| --- | --- | --- |
| Matched BitDistill control | [audit](bitdistill_adaptive_vs_fixed_matched_audit_2026-09-22.md) | [JSON](bitdistill_adaptive_vs_fixed_matched_audit_2026-09-22.json) |
| Aligned MNLI predictions | [bundle index](bitdistill_matched_prediction_bundle_2026-09-22.md) | [all labels and predictions](bitdistill_matched_prediction_bundle_2026-09-22.json) |
| TL2_SR fidelity/storage/speed | [audit](tl2sr_evidence_audit_2026-09-04.md) | [JSON](tl2sr_evidence_audit_2026-09-04.json) |
| CPU artifact matrix | [matrix](seqcls_native_cpu_matrix_2026-09-04.md) | [JSON](seqcls_native_cpu_matrix_2026-09-04.json) |
| Repeated CPU timing | [report](seqcls_native_cpu_repeated_inplace_2026-09-04.md) | [JSON](seqcls_native_cpu_repeated_inplace_2026-09-04.json) |
| I2 kernel attribution | [profile](i2_kernel_profile_2026-09-04.md) | [JSON](i2_kernel_profile_2026-09-04.json) |
| Historical canonical bundle | [bundle](canonical_evidence_bundle_2026-05-20.md) | [JSON](canonical_evidence_bundle_2026-05-20.json) |

## Evidence Levels

| Label in filename/report | Meaning |
| --- | --- |
| `audit`, `full`, `matched`, `repeated` | Claim-bearing only when status and validation gates pass |
| `sample`, `pilot`, `smoke`, `diagnostic` | Development evidence; never a full benchmark claim |
| `submission`, `monitor`, `watchdog`, `recovery` | Infrastructure/status receipt; no quality claim |
| `plan`, `blueprint`, `handoff` | Intended work; no empirical result |
| `partial`, `pending`, `running`, `blocked` | Incomplete and superseded by a later completed audit |

## Interpretation Rules

1. Prefer completed matched controls over unmatched historical comparisons.
2. Keep paper reproduction, paper-inspired runs, and retrofit variants separate.
3. Keep PyTorch checkpoint quality, packed-runtime fidelity, and CPU speed as
   separate endpoints.
4. Treat a missing artifact or `0/0` denominator as incomplete, never passing.
5. Require source/model hashes for any externally reviewable claim.
6. Use the current snapshot's source receipts to locate the exact supporting
   report.

Older files remain for provenance and postmortem work. They are intentionally
not deleted or silently rewritten when the conclusion changes.
