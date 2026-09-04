# BitDistill Next Decision

Generated: `2026-09-04T03:53:37.872608+00:00`

Status: **run_gamma_balanced_downstream**.

Quality claim: **decision_support_not_new_benchmark**.

## Recommendation

The Stage-2 marginal gain is weak and gamma-60 rebalances updates. Run a matched 10k-step downstream MNLI row with the balanced coefficient before spending more Stage-2 tokens.

## Evidence

| field | value |
| --- | --- |
| latest Stage-2 tokens | 655360000 |
| latest accuracy | 0.729903 |
| latest delta vs FP16 | -0.078248 |
| latest paired CI95 | -0.086720, -0.069775 |
| latest passes FP recovery gate | false |
| previous Stage-2 tokens | 327680000 |
| previous accuracy | 0.720020 |
| marginal Stage-2 gain | 0.009883 |
| gamma status | gamma60_rebalanced_attention_updates |
| paper grad attention/CE | 221.384986 |
| gamma60 grad attention/CE | 0.346044 |
| gamma60 grad reduction factor | 639.759089 |

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
| none |

## Source Paths

| artifact | path |
| --- | --- |
| reproduction_gap | benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json |
| controlled_curve | benchmarks/results/bitdistill_controlled_curve_2026-05-23.json |
| gamma_balance | benchmarks/results/gamma60_gradient_balance_2026-05-23.json |
| active_monitor | benchmarks/results/active_stage2_extension_monitor_2026-05-23.json |

This report is decision support. It does not create new benchmark evidence and must not be cited as a quality result without the source reports.
