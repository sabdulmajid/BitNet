# BitDistill Row-Warmup Downstream Submission, 2026-05-15

Purpose: queue the first clean row-scale downstream comparisons that depend on the row-scale Stage-2 warm-up branch (`10028`).

These jobs are intentionally tracked outside the strict 38-row tensor-warmup matrix. Their TSV files use the `bitdistill_rowwarmup_downstream_*` prefix, so the existing paper-alignment monitor remains stable while this separate novelty branch matures.

## Dependency

| field | value |
| --- | --- |
| upstream row warm-up job | `10028` |
| dependency mode | `afterok:10028` |
| warm-up state path | `checkpoints/bitdistill-glue-longwarmup-row/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-row-20k/custom_state_dict.pt` |
| partition | `midcard` |

## Submitted Jobs

| family | task | job | scale | layer | attention KD | output directory |
| --- | --- | --- | --- | --- | --- | --- |
| rowwarmup-gamma100 | MNLI | `10029` | row | `-8` | `100.0` | `checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8` |
| rowwarmup-gamma100 | QNLI | `10030` | row | `-8` | `100.0` | `checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8` |
| rowwarmup-gamma100 | SST2 | `10031` | row | `-8` | `100.0` | `checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8` |
| rowwarmup-papergamma | MNLI | `10032` | row | `-8` | `100000` | `checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8` |
| rowwarmup-papergamma | QNLI | `10033` | row | `-8` | `100000` | `checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8` |
| rowwarmup-papergamma | SST2 | `10034` | row | `-8` | `100000` | `checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8` |

## Shared Contract

| field | value |
| --- | --- |
| task format | `sequence_classification` |
| label scheme | `letters` |
| candidate score | `mean` |
| teacher source | `checkpoints/bitdistill-glue-seqcls/.../fp16_sft-tensor-layer-1` |
| task steps | `1000` |
| max train samples | full task train split (`0` cap) |
| max eval samples | full validation split (`0` cap) |
| per-device batch size | `4` |
| gradient accumulation | `4` |
| learning rate | `2e-5` |
| logit KD | `10.0` |
| logit temperature | `5.0` |
| logit temperature scaling | `none` |

## Provenance Checks

| check | status | evidence |
| --- | --- | --- |
| every job is dependency-blocked on row warm-up | pass | jobs `10029`-`10034` show `Dependency=afterok:10028(unfulfilled)` |
| every stored Slurm script matches current launcher | pass | all stored script hashes equal `dd5ea8ef8474` |
| separate TSV tracking | pass | `benchmark_results/bitdistill_rowwarmup_downstream_gamma100_20260515.tsv`, `benchmark_results/bitdistill_rowwarmup_downstream_papergamma_20260515.tsv` |

## Interpretation

- These jobs are the clean row-scale comparison branch needed for the proposed row-scale BitDistill/I2_SR novelty.
- They should be interpreted separately from the currently queued tensor-warmup row Stage-3 jobs.
- If both tensor-warmup row Stage-3 and row-warmup row Stage-3 perform similarly, row-scale is robust to Stage-2 initialization. If row-warmup materially improves task accuracy or export quality, the full row-scale training path becomes the stronger publishable claim.
