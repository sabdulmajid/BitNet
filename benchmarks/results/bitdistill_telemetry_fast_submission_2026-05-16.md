# BitDistill Fast Telemetry Submission, 2026-05-16

This is a narrow diagnostic job, not a new benchmark claim. It is scheduled to
start after the active `10069` controlled MNLI row completes, so it can
materialize update-balance telemetry without waiting for the broader controlled
postprocess dependency chain.

| field | value |
| --- | --- |
| job id | `10076` |
| dependency | `afterany:10069` |
| partition | `midcard` |
| script | `slurm_bitdistill_glue.sh` |
| stdout | `logs/bitdistill-telemetry-fast-10076.out` |
| stderr | `logs/bitdistill-telemetry-fast-10076.err` |
| output dir | `checkpoints/bitdistill-glue-seqcls-telemetry-fast/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-papergamma-headinit-after10069-steps200` |

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

The goal is to produce a real, non-smoke `telemetry.jsonl` file containing
component gradient norms, Q/K/V attention split telemetry, activation A8
saturation, sampled ternary flip rate, and scale drift. This should let
`benchmarks/audit_bitdistill_training_dynamics.py` move from `smoke_only` to a
controlled materialized status once the job completes.
