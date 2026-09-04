# BitDistill Attention-Relation Equivalence Audit

Generated: `2026-09-04T04:54:47.801940+00:00`

Status: **published_specification_ambiguous**.

Quality claim: **mathematical_contract_not_task_quality**.

## Result

The relation matrix in Equation 12 is not mathematically equivalent to the L2-normalized relation matrix in Algorithm 1 for general hidden states. The number of relation heads also changes both the loss and its gradient.

## Proof

Equation 12 scales each dot product by the single factor sqrt(d_r), whereas Algorithm 1 scales it by the pair-dependent factor temperature*||a_i||*||a_j||. No global temperature makes those logits equal for arbitrary hidden states unless the relevant norm products are constant (apart from degenerate softmax-equivalent cases). Their KL scales and gradients therefore need not agree, so a fixed attention coefficient is not portable between them.

## Contract Checks

| check | status |
| --- | --- |
| equation_and_pseudocode_losses_differ | pass |
| equation_and_pseudocode_gradients_differ | pass |
| cosine_definition_is_norm_invariant | pass |
| scaled_dot_definition_is_not_norm_invariant | pass |
| split_count_changes_objective | pass |

## Deterministic Probe

| variant | mode | split heads | loss | gradient norm |
| --- | --- | --- | --- | --- |
| algorithm1_cosine_split1 | cosine | 1 | 0.112441957 | 0.0135740032 |
| legacy_cosine_split8 | cosine | 8 | 0.804724634 | 0.0606424734 |
| equation12_scaled_dot_split1 | scaled_dot | 1 | 1.45039296 | 0.253933311 |
| algorithm1_cosine_split1_rescaled | cosine | 1 | 0.112441957 | 0.016631458 |
| equation12_scaled_dot_split1_rescaled | scaled_dot | 1 | 8.11444187 | 0.288834006 |

## Comparisons

| quantity | value |
| --- | --- |
| equation_vs_algorithm_loss_ratio | 12.899037 |
| equation_vs_algorithm_gradient_norm_ratio | 18.7073265 |
| equation_vs_algorithm_gradient_cosine | 0.2437426 |
| split8_vs_split1_loss_ratio | 7.15680032 |
| split8_vs_split1_gradient_norm_ratio | 4.46754523 |
| split8_vs_split1_gradient_cosine | 0.110220827 |
| equation_rescaling_loss_ratio | 5.59465061 |
| algorithm_rescaling_loss_ratio | 1 |

## Decision

Do not interpret gamma sweeps until relation_mode and split_heads are explicit. Run short telemetry pilots for cosine/split1, scaled_dot/split1, and the legacy cosine/split8 control before selecting one full-quality MNLI run.

This synthetic audit proves a mathematical contract difference. It does not establish downstream accuracy or identify which published definition produced the paper's reported scores.
