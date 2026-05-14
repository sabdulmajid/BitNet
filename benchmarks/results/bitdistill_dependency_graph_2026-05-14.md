# BitDistill Dependency Graph Audit, 2026-05-14

Ready for downstream release: `True`.

## Warm-Up

| log | job | step | max steps | progress | expected state exists | expected state |
| --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | 13280 | 20000 | 0.664000 | false | checkpoints/bitdistill-glue-longwarmup/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-20k/custom_state_dict.pt |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exposes final output directory | pass | log=logs/bitdistill-glue-9894.out, expected_state=checkpoints/bitdistill-glue-longwarmup/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-20k/custom_state_dict.pt |  |
| active downstream jobs point at warm-up final state | pass | active_rows=38, mismatches=0 |  |
| active downstream jobs have FP16 teacher metrics | pass | active_rows=38, missing=0 |  |
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

Raw rows: `61`. Deduped rows: `38`. Active rows: `38`.

| job | state | task | format | scale | layer | teacher metrics | warmup match | output dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9956 | PENDING | mnli | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| 9957 | PENDING | mnli | sequence_classification | row | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8 |
| 9958 | PENDING | qnli | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| 9959 | PENDING | qnli | sequence_classification | row | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8 |
| 9960 | PENDING | sst2 | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
| 9961 | PENDING | sst2 | sequence_classification | row | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8 |
| 9962 | PENDING | mnli | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| 9963 | PENDING | qnli | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| 9964 | PENDING | sst2 | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
| 9965 | PENDING | mnli | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-gamma1k/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| 9966 | PENDING | mnli | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-gamma10k/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| 9943 | PENDING | mnli | causal_lm | tensor | -8 | true | true | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| 9944 | PENDING | mnli | causal_lm | row | -8 | true | true | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8 |
| 9945 | PENDING | qnli | causal_lm | tensor | -8 | true | true | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| 9946 | PENDING | qnli | causal_lm | row | -8 | true | true | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8 |
| 9947 | PENDING | sst2 | causal_lm | tensor | -8 | true | true | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
| 9948 | PENDING | sst2 | causal_lm | row | -8 | true | true | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8 |
| 9971 | PENDING | mnli | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-headinit/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| 9972 | PENDING | mnli | sequence_classification | row | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-headinit/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8 |
| 9973 | PENDING | qnli | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-headinit/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| 9974 | PENDING | qnli | sequence_classification | row | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-headinit/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8 |
| 9975 | PENDING | sst2 | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-headinit/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
| 9976 | PENDING | sst2 | sequence_classification | row | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-headinit/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8 |
| 9978 | PENDING | mnli | sequence_classification | tensor | -1 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-layer-sweep/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-1 |
| 9979 | PENDING | mnli | sequence_classification | tensor | -2 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-layer-sweep/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-2 |
| 9980 | PENDING | mnli | sequence_classification | tensor | -4 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-layer-sweep/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-4 |
| 9981 | PENDING | mnli | sequence_classification | row | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8 |
| 9982 | PENDING | qnli | sequence_classification | row | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8 |
| 9983 | PENDING | sst2 | sequence_classification | row | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8 |
| 9987 | PENDING | mnli | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr1e-5/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| 9988 | PENDING | qnli | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr1e-5/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| 9989 | PENDING | sst2 | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr1e-5/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
| 9990 | PENDING | mnli | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr5e-5/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| 9991 | PENDING | qnli | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr5e-5/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| 9992 | PENDING | sst2 | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr5e-5/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
| 9993 | PENDING | mnli | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-headinit/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| 9994 | PENDING | qnli | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-headinit/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| 9995 | PENDING | sst2 | sequence_classification | tensor | -8 | true | true | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-headinit/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |

## Duplicate Historical Output Dirs

| output dir | superseded job ids |
| --- | --- |
| checkpoints/bitdistill-glue-seqcls-longwarmup-gamma10k/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 | 9912, 9924, 9935 |
| checkpoints/bitdistill-glue-seqcls-longwarmup-gamma1k/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 | 9911, 9934 |
| checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 | 9906, 9931 |
| checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 | 9907, 9932 |
| checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 | 9908, 9933 |
| checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8 | 9900, 9926 |
| checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 | 9899, 9925 |
| checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8 | 9902, 9928 |
| checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 | 9901, 9927 |
| checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8 | 9904, 9930 |
| checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 | 9903, 9929 |
