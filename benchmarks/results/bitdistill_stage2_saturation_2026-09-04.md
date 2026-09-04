# Fixed-Recipe Stage-2 Saturation Audit

This is a conditional extrapolation from three paired MNLI runs, not a claim about all BitDistill recipes.

## Observed Curve

| Stage-2 token presentations | Accuracy | Gain from previous doubling |
| ---: | ---: | ---: |
| `163,840,000` | `0.691187` | `-` |
| `327,680,000` | `0.720020` | `0.028833` |
| `655,360,000` | `0.729903` | `0.009883` |

## Conditional Projection

- Observed gain contraction: `0.342756`.
- Fitted asymptote: `0.735057`.
- Projection at `10,000,000,000` token presentations: `0.734981`.
- Constant-latest-gain sensitivity projection at the target: `0.768758`.
- Required average gain per remaining doubling: `0.017359` (`1.756x` the latest observed gain).
- Paired-bootstrap 95% interval for the asymptote: `[0.723766, 0.760099]`.
- Paired-bootstrap 95% interval at the target budget: `[0.723750, 0.755819]`.
- Paired-bootstrap 95% interval for the constant-latest-gain sensitivity: `[0.741530, 0.795733]`.
- Valid monotone-contraction bootstrap replicates: `19984` / `20000` (`99.920%`).

## Decision

Under the fitted diminishing-returns model, Stage-2 budget alone does not close the local FP16 recovery gap: even the bootstrap upper bound `0.755819` is below the pre-registered recovery target `0.798151`. Even repeating the latest gain without further decay has bootstrap upper bound `0.795733`. Reaching the gate requires the average future gain per doubling to be `1.756x` the latest observed gain, reversing the measured diminishing-return trend. Change the training objective or method contract before scaling this fixed recipe.

The inference is conditional on the current fixed-gamma, tensor-scale, Qwen2.5-0.5B MNLI recipe. It does not rule out adaptive objective balancing, a different backbone, a different Stage-2 corpus, or the paper's unreleased implementation details.
