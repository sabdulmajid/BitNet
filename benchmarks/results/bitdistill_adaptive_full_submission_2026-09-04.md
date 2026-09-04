# BitDistill Adaptive Full-Run Submission

Status: **running**. No quality conclusion is permitted until all three runs and paired audits complete.

| Seed | Job | Dependency | Model artifacts | Status |
| ---: | --- | --- | --- | --- |
| `1234` | `10392` | none | yes | running |
| `1235` | `10395` | `afterok:10392` | no | pending |
| `1236` | `10396` | `afterok:10395` | no | pending |

The jobs run Qwen2.5-0.5B sequence classification from the verified 655M Stage-2 checkpoint. The selected Stage-3 contract is tensor-scale W1.58A8 with SubLN, cosine attention relations, one relation head, logits KD weight `10`, and gradient-norm EMA attention balancing. Each run uses 10,000 optimizer steps, batch size 4, gradient accumulation 4, sequence length 512, the complete MNLI training split, and all 9,815 matched validation examples.

The source is pinned to `526ede7b2c3f33c6a9638de54bdae91e8afe39c6`. The cross-environment stack is RTX A4500, torch `2.6.0+cu118`, Transformers `5.7.0`, Datasets `2.18.0`, and Accelerate `1.13.0`.

## Pre-Registered Decisions

1. Every run must complete 10,000 steps and evaluate exactly 9,815 examples.
2. Adaptive balancing is a quality improvement only if its paired delta over the fixed-gamma 655M reference (`0.729903`) is positive with a 95% confidence interval excluding zero.
3. Paper-level local recovery requires three-seed mean accuracy at least `0.798151`, within one absolute accuracy point of the local FP16 reference (`0.808151`).
4. Stable loss scaling without downstream improvement is not a quality success.
