# BitDistill Next Decision

Generated: `2026-05-23T18:15:10.097106+00:00`

Status: **pending_655m_downstream**.

Quality claim: **decision_support_not_new_benchmark**.

## Recommendation

Wait for the active 655.36M Stage-2 producer, downstream MNLI, and postprocess reports.

## Evidence

| field | value |
| --- | --- |
| latest Stage-2 tokens | 327680000 |
| latest accuracy | 0.720020 |
| latest delta vs FP16 | -0.088130 |
| latest paired CI95 | -0.096749, -0.079511 |
| latest passes FP recovery gate | false |
| previous Stage-2 tokens | 163840000 |
| previous accuracy | 0.691187 |
| marginal Stage-2 gain | 0.028833 |
| gamma status | pending_gamma60_telemetry |
| paper grad attention/CE | 221.384986 |
| gamma60 grad attention/CE | - |
| gamma60 grad reduction factor | - |

## Thresholds

| threshold | value |
| --- | --- |
| target Stage-2 tokens | 655360000 |
| success delta from FP16 | -0.010000 |
| meaningful Stage-2 gain | 0.015000 |
| saturation Stage-2 gain | 0.005000 |
| balanced max grad attention/CE | 10.000000 |
| decision epsilon | 1.000e-12 |

## Evidence Gaps

| gap |
| --- |
| latest controlled row is 327,680,000 tokens, below 655.36M |

## Source Paths

| artifact | path |
| --- | --- |
| reproduction_gap | benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json |
| controlled_curve | benchmarks/results/bitdistill_controlled_curve_2026-05-20.json |
| gamma_balance | benchmarks/results/gamma60_gradient_balance_2026-05-23.json |
| active_monitor | benchmarks/results/active_stage2_extension_monitor_2026-05-23.json |

This report is decision support. It does not create new benchmark evidence and must not be cited as a quality result without the source reports.
