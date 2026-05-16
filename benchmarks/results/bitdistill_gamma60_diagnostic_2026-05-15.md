# BitDistill Gamma-60 Diagnostic Audit, 2026-05-15

Gamma-60 improves over the matched paper-gamma control but still misses the FP16 recovery gate.

This is a focused loss-normalization diagnostic, not a broad sweep and not a paper-reproduction claim by itself.

## Run State

| field | value |
| --- | --- |
| job id | 10077 |
| squeue state | not_in_squeue |
| squeue elapsed | - |
| candidate metrics | true |
| candidate predictions | true |
| candidate accuracy | 0.738462 |
| matched paper-gamma accuracy | 0.691187 |
| metric delta vs paper-gamma | 0.047275 |
| paired delta vs FP16 | -0.069689 |
| paired delta vs paper-gamma | 0.047275 |
| passes FP recovery gate | false |
| improves over paper-gamma | true |

## Live Loss Balance

| quantity | latest/p50/p95 |
| --- | --- |
| latest step | 10000 |
| latest weighted attention / CE | 2.321450 |
| weighted attention / CE p50 | 1.607406 |
| weighted attention / CE p95 | 6.106116 |
| latest weighted logits / CE | 0.163960 |
| weighted logits / CE p50 | 0.409301 |
| CE/attention equalizing gamma p50 | 37.327967 |
| CE/attention equalizing gamma p95 | 87.906009 |

## Interpretation Gate

Compare this row only against the matched 20k-warmup paper-gamma control. If the final paired delta vs paper-gamma is positive, loss normalization is a likely primary blocker. If it is non-positive, attention-KD dominance is still a measured risk but not sufficient to explain the quality gap.
