# Matched Adaptive vs Fixed-60 Submission

Status: **submitted, results pending**.

This is a pre-registered, matched three-seed MNLI test of whether adaptive
gradient-norm EMA attention-loss balancing improves task quality over a sane
fixed `gamma=60` control. The controls and decision thresholds were recorded
while adaptive seed `1234` was still training and before any adaptive
full-validation accuracy was available. Gamma `60` came from prior historical
and gradient-balance evidence, not from the pending outcomes.

## Jobs

| Arm | Seed | Job | Dependency | Saves model | Status |
| --- | ---: | ---: | --- | --- | --- |
| adaptive | `1234` | `10392` | none | yes | running at registration |
| adaptive | `1235` | `10395` | `afterok:10392` | no | pending |
| adaptive | `1236` | `10396` | `afterok:10395` | no | pending |
| fixed60 | `1234` | `10399` | `afterok:10396` | no | pending |
| fixed60 | `1235` | `10400` | `afterok:10399` | no | pending |
| fixed60 | `1236` | `10401` | `afterok:10400` | no | pending |

Fail-closed audit job `10402` runs with `afterany:10401`. Its staged script
SHA-256 is
`3fd4d5fcc98879a213abe93aafef71f6b687a5d81a6a4ec215558af61db2274e`.
It will run even if a training dependency fails, then mark missing or malformed
artifacts invalid rather than suppressing the comparison.

All six runs use source revision
`526ede7b2c3f33c6a9638de54bdae91e8afe39c6` and Stage-2 state SHA-256
`9fc648a7466adb5f170085cf73d2bf4bd90a500f9de4c2a8f6c68b6cc29fa57d`.
The fixed batch-script hashes are recorded in the JSON manifest.

## Controlled Contract

Held fixed: Qwen2.5-0.5B sequence classification, the same 655.36M-presentation
Stage-2 checkpoint, tensor W1.58A8, SubLN, cosine Q/K/V relations, one relation
head, logits KD weight `10`, 10,000 optimizer steps, cosine learning-rate
schedule, all 392,702 MNLI training examples, all 9,815 matched validation
examples, and seeds `1234`, `1235`, and `1236`.

The intended method difference is:

```text
adaptive: initial gamma=100000, GradNorm EMA target=1, beta=0.9, update every 20 steps
fixed60:  gamma=60 throughout training
```

Only adaptive seed `1234` saves a model payload. The fixed runs omit model
payloads because node-local storage had 31 GB free; metrics, paired predictions,
telemetry, and logs remain mandatory.

## Pre-Registered Decisions

1. Primary endpoint: full MNLI matched-validation accuracy.
2. Primary contrast: adaptive minus fixed60, paired by training seed.
3. Adaptive wins only if all three seed deltas are positive, their mean is at
   least `+0.005`, and the seed-level paired t-interval lower bound exceeds zero.
4. Fixed60 is preferred for simplicity when the seed-level t-interval upper
   bound excludes a `+0.005` adaptive benefit.
5. Paper-level local recovery remains a separate gate: mean accuracy at least
   `0.798151`, within one point of local FP16 (`0.808151`).
6. Per-seed paired intervals and exact McNemar tests are conditional on trained
   checkpoints; they do not replace the primary across-seed uncertainty.

This is a cross-environment method comparison on one model and one task. It is
not a paper-exact reproduction and cannot establish generalization to other
tasks, architectures, or runtimes.
