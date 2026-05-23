# BitDistill Decision Scenarios

Generated: `2026-05-23T17:06:55.191170+00:00`

Quality claim: **decision_policy_not_benchmark**.

This report simulates decision outcomes using existing thresholds. It does not add benchmark evidence or predict the 655M result.

## Thresholds

| threshold | value |
| --- | --- |
| target_stage2_tokens | 655360000 |
| success_delta_from_fp | -0.010000 |
| success_accuracy | 0.798151 |
| meaningful_stage2_gain | 0.015000 |
| saturation_stage2_gain | 0.005000 |
| balanced_max_grad_attention_to_ce | 10.000000 |
| decision_eps | 1.000e-12 |

## Scenario Matrix

| scenario | hypothetical 655M accuracy | delta vs 327M | delta vs FP16 | gamma status | decision | recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| flat 655M | 0.720020 | 0.000000 | -0.088130 | pending | hold_for_gamma_balance | The 655M quality row does not provide enough evidence by itself; wait for gamma-balance telemetry before launching another expensive broad run. |
| saturated 655M, gamma rebalanced | 0.725020 | 0.005000 | -0.083130 | rebalanced | run_gamma_balanced_downstream | The Stage-2 marginal gain is weak and gamma-60 rebalances updates. Run a matched 10k-step downstream MNLI row with the balanced coefficient before spending more Stage-2 tokens. |
| saturated 655M, gamma still dominated | 0.725020 | 0.005000 | -0.083130 | still_dominated | pause_broad_stage2_audit_recipe | The Stage-2 curve appears to saturate and gamma telemetry did not resolve the update imbalance. Stop broad budget scaling and audit recipe alignment. |
| ambiguous mid gain | 0.730020 | 0.010000 | -0.078130 | still_dominated | ambiguous_recovery_continue_with_controls | Evidence is mixed. Run one narrow ablation at a time: either the next Stage-2 point or one gamma-balanced downstream row, but do not expand axes. |
| meaningful gain | 0.735020 | 0.015000 | -0.073130 | pending | extend_stage2_curve | Stage-2 still has meaningful marginal gain. Queue the next controlled point before changing the recipe, while keeping gamma telemetry as a diagnostic. |
| FP recovery gate | 0.798151 | 0.078130 | -0.010000 | pending | replicate_recovery_gate | Do not broaden yet; replicate the recovered row and then run QNLI/SST2 with the same recipe. |

## Interpretation

Use this matrix to audit the policy before the 655M result arrives. When the real downstream prediction trace exists, `bitdistill_next_decision_2026-05-23` is the authoritative report.
