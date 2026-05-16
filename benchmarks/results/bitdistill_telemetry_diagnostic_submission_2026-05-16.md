# BitDistill Telemetry Diagnostic Submission, 2026-05-16

This is a narrow diagnostic job, not a new benchmark claim. It waits for the
controlled Stage-2 postprocess job and then records sparse update-balance
telemetry for the same Qwen2.5-0.5B MNLI tensor-scale BitDistill recipe that is
currently the main reproduction blocker.

| field | value |
| --- | --- |
| job id | `10075` |
| dependency | `afterany:10072` |
| partition | `midcard` |
| script | `slurm_bitdistill_glue.sh` |
| stdout | `logs/bitdistill-telemetry-10075.out` |
| stderr | `logs/bitdistill-telemetry-10075.err` |
| output dir | `checkpoints/bitdistill-glue-seqcls-telemetry/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-papergamma-headinit-telemetry-steps200` |

## Recipe

| setting | value |
| --- | --- |
| model | `Qwen/Qwen2.5-0.5B` |
| task | `mnli` |
| formulation | `sequence_classification` |
| method | `bitdistill` |
| scale mode | `tensor` |
| Stage-2 init | `checkpoints/bitdistill-glue-longwarmup/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-20k/custom_state_dict.pt` |
| teacher | `checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1` |
| SubLN | enabled |
| output head init | from teacher |
| attention layer | `-1` |
| logit KD weight | `10` |
| attention KD weight | `100000` |
| logit temperature | `5.0` |
| attention temperature | `1.0` |
| max steps | `200` |
| batch / grad accumulation | `4 / 4` |
| learning rate | `2e-5` |
| warmup steps | `100` |
| max eval samples | `512` |
| save model artifacts | disabled |

## Telemetry

| telemetry option | value |
| --- | --- |
| `TELEMETRY_EVERY_STEPS` | `25` |
| `TELEMETRY_COMPONENT_GRAD_NORMS` | `1` |
| `TELEMETRY_MAX_ELEMENTS_PER_LAYER` | `32768` |

## Purpose

The controlled `163.84M` Stage-2 row improved MNLI to `0.691187` but still sits
`0.116964` below the local FP16-SFT row. Its Slurm log also shows paper-gamma
attention KD dominating CE by thousands under the local loss normalization. This
job is designed to measure whether that dominance also appears in component
gradient norms, ternary-code telemetry, and scale statistics before any broader
model/task sweep is added.
