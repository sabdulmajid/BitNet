# BitDistill Recovery Submission, 2026-05-15

This is a pending, paper-inspired recovery run. It is not a completed
BitDistill reproduction result.

## Submitted Job

| field | value |
| --- | --- |
| job_id | `10068` |
| dependency | `afterany:10067` |
| partition | `midcard` |
| model | `Qwen/Qwen2.5-0.5B` |
| task | `mnli` |
| task_format | `sequence_classification` |
| method | `bitdistill` |
| scale_mode | `tensor` |
| SubLN | `enabled` |
| teacher | `checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1` |
| init_state_dict | `checkpoints/bitdistill-glue-longwarmup/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-20k/custom_state_dict.pt` |
| output_dir | `checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-steps10000-lr2em5-papergamma-headinit` |

## Recipe

| hyperparameter | value |
| --- | --- |
| max_steps | `10000` |
| per_device_batch_size | `4` |
| grad_accum_steps | `4` |
| max_seq_len | `512` |
| learning_rate | `2e-5` |
| logit_kd_weight | `10` |
| logit_temperature | `5.0` |
| attention_kd_weight | `100000` |
| attention_temperature | `1.0` |
| attention_split_heads | `8` |
| distill_layer | `-1` |
| init_output_head_from_teacher | `true` |
| save_model_artifacts | `false` |

## Interpretation Gate

The run is designed to test whether the now-validated tensor-scale BitNet-SFT
baseline can recover toward the local FP16-SFT MNLI accuracy when using:

1. the long existing tensor-scale Stage-2 warmup,
2. a dense FP16 sequence-classification teacher,
3. SubLN-enabled BitDistill,
4. paper-scale attention gamma under the local loss normalization,
5. full MNLI validation predictions.

Success requires full-validation accuracy within about `0.5-1.0` point of the
local FP16-SFT baseline. Until the job completes and paired statistics are
computed, this is only a queued diagnostic.
