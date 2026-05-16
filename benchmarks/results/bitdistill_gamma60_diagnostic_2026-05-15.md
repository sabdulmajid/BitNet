# BitDistill Gamma-60 Diagnostic Audit, 2026-05-15

Pending: the gamma-60 job has not produced full metrics and prediction traces yet.

This is a focused loss-normalization diagnostic, not a broad sweep and not a paper-reproduction claim by itself.

## Run State

| field | value |
| --- | --- |
| job id | 10077 |
| squeue state | RUNNING |
| squeue elapsed | 6:45 |
| candidate metrics | false |
| candidate predictions | false |
| candidate accuracy | - |
| matched paper-gamma accuracy | 0.691187 |
| metric delta vs paper-gamma | - |
| paired delta vs FP16 | - |
| paired delta vs paper-gamma | - |
| passes FP recovery gate | false |
| improves over paper-gamma | false |

## Live Loss Balance

| quantity | latest/p50/p95 |
| --- | --- |
| latest step | 430 |
| latest weighted attention / CE | 2.653340 |
| weighted attention / CE p50 | 1.644974 |
| weighted attention / CE p95 | 3.576358 |
| latest weighted logits / CE | 0.405590 |
| weighted logits / CE p50 | 0.518886 |
| CE/attention equalizing gamma p50 | 36.475351 |
| CE/attention equalizing gamma p95 | 74.772596 |

## Interpretation Gate

Compare this row only against the matched 20k-warmup paper-gamma control. If the final paired delta vs paper-gamma is positive, loss normalization is a likely primary blocker. If it is non-positive, attention-KD dominance is still a measured risk but not sufficient to explain the quality gap.
