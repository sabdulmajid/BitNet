# BitDistill Gamma-60 Diagnostic Submission, 2026-05-16

Submitted Slurm job: `10077`.

Purpose: test the loss-normalization hypothesis with one focused run, not a broad sweep. The completed controlled paper-gamma runs show median CE/attention equalizing gamma near `58-61`, while `ATTENTION_KD_WEIGHT=100000` dominates CE by thousands of times. This job keeps the strongest completed controlled setup fixed and changes only the attention-KD coefficient to `60`.

Configuration:

| field | value |
| --- | --- |
| model | `Qwen/Qwen2.5-0.5B` |
| task | `mnli` |
| formulation | `sequence_classification` |
| method | `bitdistill` |
| scale | `tensor` |
| continued-pretrain init | `checkpoints/bitdistill-glue-longwarmup/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-20k/custom_state_dict.pt` |
| teacher | `checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1` |
| distill layer | `-1` |
| attention split heads | `8` |
| logit KD | `10` |
| attention KD | `60` |
| logit temperature | `5.0` |
| logit temperature scaling | `none` |
| SubLN | `enabled` |
| activation quantization | `enabled` |
| output head init | `teacher score head copied` |
| max steps | `10000` |
| batch / grad accum | `4 / 4` |
| learning rate | `2e-5` |
| output | `checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-steps10000-lr2em5-gamma60-headinit` |

Interpretation rule: compare only against the matched 20k-warmup paper-gamma controlled row (`10068`, MNLI `0.691187`) and the local FP16-SFT reference (`0.807641`). If gamma `60` improves over `10068`, loss normalization is likely a primary reproduction blocker. If it regresses, the paper-gamma dominance may be numerically ugly but not the main quality limiter under this local recipe.
