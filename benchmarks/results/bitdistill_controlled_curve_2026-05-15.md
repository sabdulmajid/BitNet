# BitDistill Controlled Stage-2 Curve Audit, 2026-05-15

Pending: at least one controlled Stage-2 curve row lacks metrics or prediction traces.

| field | value |
| --- | --- |
| reference_predictions | checkpoints/bitdistill-glue-seqcls-predtrace/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1/eval_predictions.jsonl |
| complete | 1/3 |
| passed FP recovery gate | 0/3 |
| success delta from FP16 | -0.010000 |

## Rows

| job | label | state | Stage-2 tokens | paper fraction | metrics | predictions | accuracy | delta vs FP16 | paired CI95 | passes gate | errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10069 | 5k-warmup downstream control | RUNNING | 40960000 | 0.004096 | false | false | - | - | - | false | missing checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-5kwarmup-steps10000-lr2em5-papergamma-headinit/eval_predictions.jsonl |
| 10068 | 20k-warmup downstream control | squeue_error | 163840000 | 0.016384 | true | true | 0.691187 | -0.116964 | [-0.126110, -0.107817] | false |  |
| 10071 | 40k-warmup downstream control | PENDING | 327680000 | 0.032768 | false | false | - | - | - | false | missing checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-40kwarmup-steps10000-lr2em5-papergamma-headinit/eval_predictions.jsonl |

## Loss Components

| job | label | live step | live attn/CE | live max attn/CE | live median attn/CE | live p95 attn/CE | median CE/attn gamma | p95 CE/attn gamma | final CE | final logit KD | final weighted logit KD | final attention KD | final weighted attention KD | live CE | live logit KD | live weighted logit KD | live attention KD | live weighted attention KD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10069 | 5k-warmup downstream control | 1820 | 1130.724918 | 7549.916324 | 1922.517550 | 3749.993306 | 52.014968 | 101.396335 | - | - | - | - | - | 1.484375 | 0.096939 | 0.969391 | 0.016784 | 1678.419800 |
| 10068 | 20k-warmup downstream control | 10000 | 5945.070866 | 37819.641342 | 1729.105844 | 6080.253825 | 57.831811 | 135.086455 | 0.200195 | 0.007469 | 0.074691 | 0.011902 | 1190.173462 | 0.200195 | 0.007469 | 0.074691 | 0.011902 | 1190.173462 |
| 10071 | 40k-warmup downstream control | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

## Interpretation

These rows are the controlled budget test. They should be interpreted only after full MNLI prediction traces exist for each output directory.

