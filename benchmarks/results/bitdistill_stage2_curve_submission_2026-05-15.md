# BitDistill Stage-2 Curve Submission, 2026-05-15

Queued fixed-recipe Stage-2 budget controls for Qwen2.5-0.5B MNLI. These are intended to compare the pending 20k-warmup recovery run against a matching 5k control and a larger 40k warm-up point.

| job | label | dependency | partition | output |
| --- | --- | --- | --- | --- |
| 10069 | 5k-warmup downstream control | afterany:10067 | midcard | checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-5kwarmup-steps10000-lr2em5-papergamma-headinit |
| 10070 | 40k Stage-2 warm-up | none | dualcard | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-40k |
| 10071 | 40k-warmup downstream control | afterok:10070 | midcard | checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-40kwarmup-steps10000-lr2em5-papergamma-headinit |

## Fixed Recipe

- Qwen2.5-0.5B, MNLI, sequence-classification formulation.
- Tensor-scale paper-style BitDistill path, SubLN enabled, dense classifier head excluded from ternary replacement.
- Distillation layer `-1`, split heads `8`, logits KD weight `10`, attention KD weight `100000`, temperature `5.0`, teacher-head initialization enabled.
- Downstream budget `10000` steps at LR `2e-5`, batch `4`, gradient accumulation `4`.

## Interpretation Gate

No quality claim is made from this submission. The result becomes evidence only after each downstream output has full MNLI predictions and passes the paired FP16 audit.
