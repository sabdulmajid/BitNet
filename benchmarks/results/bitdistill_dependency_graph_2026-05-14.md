# BitDistill Dependency Graph Audit, 2026-05-14

Ready for downstream release: `True`.

## Warm-Up

| log | job | step | max steps | progress | expected state exists | expected state |
| --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | 5850 | 20000 | 0.292500 | false | checkpoints/bitdistill-glue-longwarmup/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-20k/custom_state_dict.pt |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exposes final output directory | pass | log=logs/bitdistill-glue-9894.out, expected_state=checkpoints/bitdistill-glue-longwarmup/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-20k/custom_state_dict.pt |  |
| active downstream jobs point at warm-up final state | pass | active_rows=17, mismatches=0 |  |
| active downstream jobs have FP16 teacher metrics | pass | active_rows=17, missing=0 |  |
| active downstream jobs depend on the running warm-up job | pass | warmup_job=9894, bad=0 |  |

## Warnings

| warning |
| --- |
| 11 output directories appear in multiple historical submission tables; audit uses the latest row per output directory. |
| Warm-up final state is expectedly absent until Stage-2 finishes; downstream jobs are correctly dependency-blocked. |

## Blockers

| blocker |
| --- |
| none |

## Submission Rows

Raw rows: `29`. Deduped rows: `17`. Active rows: `17`.

| job | state | task | format | scale | layer | teacher metrics | warmup match | output dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9925 | PENDING | mnli | - | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| 9926 | PENDING | mnli | - | row | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8 |
| 9927 | PENDING | qnli | - | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| 9928 | PENDING | qnli | - | row | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8 |
| 9929 | PENDING | sst2 | - | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
| 9930 | PENDING | sst2 | - | row | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8 |
| 9931 | PENDING | mnli | - | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| 9932 | PENDING | qnli | - | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| 9933 | PENDING | sst2 | - | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
| 9934 | PENDING | mnli | - | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-gamma1k/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| 9935 | PENDING | mnli | - | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-gamma10k/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| 9943 | PENDING | mnli | causal_lm | tensor | -8 | true | true | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| 9944 | PENDING | mnli | causal_lm | row | -8 | true | true | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8 |
| 9945 | PENDING | qnli | causal_lm | tensor | -8 | true | true | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| 9946 | PENDING | qnli | causal_lm | row | -8 | true | true | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8 |
| 9947 | PENDING | sst2 | causal_lm | tensor | -8 | true | true | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
| 9948 | PENDING | sst2 | causal_lm | row | -8 | true | true | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8 |

## Duplicate Historical Output Dirs

| output dir | superseded job ids |
| --- | --- |
| checkpoints/bitdistill-glue-seqcls-longwarmup-gamma10k/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 | 9912, 9924 |
| checkpoints/bitdistill-glue-seqcls-longwarmup-gamma1k/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 | 9911 |
| checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 | 9906 |
| checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 | 9907 |
| checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 | 9908 |
| checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8 | 9900 |
| checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 | 9899 |
| checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8 | 9902 |
| checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 | 9901 |
| checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8 | 9904 |
| checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 | 9903 |
