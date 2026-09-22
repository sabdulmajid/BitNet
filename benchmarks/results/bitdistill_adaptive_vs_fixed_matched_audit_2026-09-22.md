# Matched Adaptive vs Fixed-60 BitDistill Audit

Recovered from completed node-local artifacts on `2026-09-22`. The audit itself completed on `2026-09-04`.

Status: **complete**.

This is a matched, three-seed MNLI comparison across the same Stage-2 checkpoint. It is not a paper-exact reproduction and does not establish generalization beyond this setup.

## Runs

| arm | seed | accuracy | final attention weight | median weighted attention/CE gradient ratio |
| --- | ---: | ---: | ---: | ---: |
| adaptive | 1234 | 0.755782 | 22.8487 | 0.0458014 |
| adaptive | 1235 | 0.756903 | 19.9842 | 0.0494513 |
| adaptive | 1236 | 0.753337 | 25.9193 | 0.0572301 |
| fixed60 | 1234 | 0.758635 | 60 | 0.0828709 |
| fixed60 | 1235 | 0.754457 | 60 | 0.0982454 |
| fixed60 | 1236 | 0.758023 | 60 | 0.104251 |

## Paired Results

| seed | adaptive | fixed60 | adaptive - fixed60 | paired 95% CI | exact McNemar p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1234 | 0.755782 | 0.758635 | -0.002853 | [-0.008585, 0.002879] | 0.346921 |
| 1235 | 0.756903 | 0.754457 | +0.002445 | [-0.003479, 0.008369] | 0.438162 |
| 1236 | 0.753337 | 0.758023 | -0.004687 | [-0.010710, 0.001337] | 0.135725 |

## Aggregate

| adaptive mean | fixed60 mean | mean delta | seed-level paired t 95% CI | conditional example-level 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| 0.755340 | 0.757039 | -0.001698 | [-0.010898, 0.007502] | [-0.005112, 0.001716] |

The preregistered adaptive-superiority test failed. The preregistered fixed-simplicity test also failed because three seeds do not exclude a practically relevant adaptive gain. The honest decision is **inconclusive**: there is no evidence that the controller improves quality, and its observed mean is lower.

Neither arm reached the preregistered paper-recovery floor of `0.798151` (within one accuracy point of the local FP16 baseline).

## Statistical Boundary

The seed-level paired t interval is primary and reflects variation across three training seeds; at `n=3` it has low power. Per-seed intervals and McNemar tests condition on already-trained checkpoints. The example-level interval is secondary and cannot replace training-seed uncertainty.

The machine-readable audit is in [`bitdistill_adaptive_vs_fixed_matched_audit_2026-09-22.json`](bitdistill_adaptive_vs_fixed_matched_audit_2026-09-22.json). It includes run hashes, Stage-2 provenance, telemetry summaries, and preregistered decision rules.
