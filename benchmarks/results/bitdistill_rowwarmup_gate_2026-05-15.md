# BitDistill Row-Warmup Gate, 2026-05-15

Model: `Qwen/Qwen2.5-0.5B`.

Threshold: absolute FP16-SFT gap <= `0.01` accuracy.

Any row-warmup family passed: `False`.

## Family Status

| family | complete | passed |
| --- | --- | --- |
| row_warmup_gamma100 | false | false |
| row_warmup_papergamma | false | false |

## Runs

| task | run | family | exists | accuracy | examples | expected | full eval | accuracy 95% CI | FP16 | FP-run | FP-run 95% CI | status | metrics path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | FP16-SFT | baseline | true | 0.807641 | 9815 | 9815 | pass | [0.799724, 0.815318] | 0.807641 | 0.000000 | [-0.011028, 0.011028] | pass | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1/metrics.json |
| mnli | tensor-warmup tensor gamma100 | tensor_warmup_gamma100 | true | 0.641671 | 9815 | 9815 | pass | [0.632131, 0.651100] | 0.807641 | 0.165970 | [0.153691, 0.178250] | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| mnli | tensor-warmup row gamma100 | tensor_warmup_row_gamma100 | true | 0.653591 | 9815 | 9815 | pass | [0.644120, 0.662943] | 0.807641 | 0.154050 | [0.141826, 0.166274] | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| mnli | tensor-warmup tensor paper gamma | tensor_warmup_papergamma | false | - | - | 9815 | fail_or_pending | - | 0.807641 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| mnli | tensor-warmup row paper gamma | tensor_warmup_row_papergamma | false | - | - | 9815 | fail_or_pending | - | 0.807641 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| mnli | row-warmup row gamma100 | row_warmup_gamma100 | false | - | - | 9815 | fail_or_pending | - | 0.807641 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| mnli | row-warmup row paper gamma | row_warmup_papergamma | false | - | - | 9815 | fail_or_pending | - | 0.807641 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| qnli | FP16-SFT | baseline | true | 0.898957 | 5463 | 5463 | pass | [0.890682, 0.906670] | 0.898957 | 0.000000 | [-0.011302, 0.011302] | pass | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/fp16_sft-tensor-layer-1/metrics.json |
| qnli | tensor-warmup tensor gamma100 | tensor_warmup_gamma100 | false | - | - | 5463 | fail_or_pending | - | 0.898957 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| qnli | tensor-warmup row gamma100 | tensor_warmup_row_gamma100 | false | - | - | 5463 | fail_or_pending | - | 0.898957 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| qnli | tensor-warmup tensor paper gamma | tensor_warmup_papergamma | false | - | - | 5463 | fail_or_pending | - | 0.898957 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| qnli | tensor-warmup row paper gamma | tensor_warmup_row_papergamma | false | - | - | 5463 | fail_or_pending | - | 0.898957 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| qnli | row-warmup row gamma100 | row_warmup_gamma100 | false | - | - | 5463 | fail_or_pending | - | 0.898957 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| qnli | row-warmup row paper gamma | row_warmup_papergamma | false | - | - | 5463 | fail_or_pending | - | 0.898957 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| sst2 | FP16-SFT | baseline | true | 0.925459 | 872 | 872 | pass | [0.906098, 0.941087] | 0.925459 | 0.000000 | [-0.024654, 0.024654] | pass | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/fp16_sft-tensor-layer-1/metrics.json |
| sst2 | tensor-warmup tensor gamma100 | tensor_warmup_gamma100 | false | - | - | 872 | fail_or_pending | - | 0.925459 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| sst2 | tensor-warmup row gamma100 | tensor_warmup_row_gamma100 | false | - | - | 872 | fail_or_pending | - | 0.925459 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
| sst2 | tensor-warmup tensor paper gamma | tensor_warmup_papergamma | false | - | - | 872 | fail_or_pending | - | 0.925459 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| sst2 | tensor-warmup row paper gamma | tensor_warmup_row_papergamma | false | - | - | 872 | fail_or_pending | - | 0.925459 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
| sst2 | row-warmup row gamma100 | row_warmup_gamma100 | false | - | - | 872 | fail_or_pending | - | 0.925459 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
| sst2 | row-warmup row paper gamma | row_warmup_papergamma | false | - | - | 872 | fail_or_pending | - | 0.925459 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |

## Comparisons

| task | comparison | left | left n | right | right n | left-right | left-right 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | row_warmup_gamma100_minus_tensor_warmup_tensor_gamma100 | - | - | 0.641671 | 9815 | - | - |
| mnli | row_warmup_gamma100_minus_tensor_warmup_row_gamma100 | - | - | 0.653591 | 9815 | - | - |
| mnli | row_warmup_papergamma_minus_tensor_warmup_tensor_papergamma | - | - | - | - | - | - |
| mnli | row_warmup_papergamma_minus_tensor_warmup_row_papergamma | - | - | - | - | - | - |
| qnli | row_warmup_gamma100_minus_tensor_warmup_tensor_gamma100 | - | - | - | - | - | - |
| qnli | row_warmup_gamma100_minus_tensor_warmup_row_gamma100 | - | - | - | - | - | - |
| qnli | row_warmup_papergamma_minus_tensor_warmup_tensor_papergamma | - | - | - | - | - | - |
| qnli | row_warmup_papergamma_minus_tensor_warmup_row_papergamma | - | - | - | - | - | - |
| sst2 | row_warmup_gamma100_minus_tensor_warmup_tensor_gamma100 | - | - | - | - | - | - |
| sst2 | row_warmup_gamma100_minus_tensor_warmup_row_gamma100 | - | - | - | - | - | - |
| sst2 | row_warmup_papergamma_minus_tensor_warmup_tensor_papergamma | - | - | - | - | - | - |
| sst2 | row_warmup_papergamma_minus_tensor_warmup_row_papergamma | - | - | - | - | - | - |
