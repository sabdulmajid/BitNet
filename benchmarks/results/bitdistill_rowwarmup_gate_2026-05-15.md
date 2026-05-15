# BitDistill Row-Warmup Gate, 2026-05-15

Model: `Qwen/Qwen2.5-0.5B`.

Threshold: absolute FP16-SFT gap <= `0.01` accuracy.

Any row-warmup family passed: `False`.

## Family Status

| family | complete | passed |
| --- | --- | --- |
| row_warmup_gamma100 | true | false |
| row_warmup_papergamma | false | false |

## Runs

| task | run | family | exists | accuracy | examples | expected | full eval | accuracy 95% CI | FP16 | FP-run | FP-run 95% CI | status | metrics path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | FP16-SFT | baseline | true | 0.807641 | 9815 | 9815 | pass | [0.799724, 0.815318] | 0.807641 | 0.000000 | [-0.011028, 0.011028] | pass | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1/metrics.json |
| mnli | tensor-warmup tensor gamma100 | tensor_warmup_gamma100 | true | 0.641671 | 9815 | 9815 | pass | [0.632131, 0.651100] | 0.807641 | 0.165970 | [0.153691, 0.178250] | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| mnli | tensor-warmup row gamma100 | tensor_warmup_row_gamma100 | true | 0.653591 | 9815 | 9815 | pass | [0.644120, 0.662943] | 0.807641 | 0.154050 | [0.141826, 0.166274] | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| mnli | tensor-warmup tensor paper gamma | tensor_warmup_papergamma | true | 0.630260 | 9815 | 9815 | pass | [0.620660, 0.639757] | 0.807641 | 0.177382 | [0.165052, 0.189711] | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| mnli | tensor-warmup row paper gamma | tensor_warmup_row_papergamma | true | 0.617626 | 9815 | 9815 | pass | [0.607968, 0.627192] | 0.807641 | 0.190015 | [0.177636, 0.202394] | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| mnli | row-warmup row gamma100 | row_warmup_gamma100 | true | 0.627713 | 9815 | 9815 | pass | [0.618101, 0.637225] | 0.807641 | 0.179929 | [0.167589, 0.192268] | fail_or_pending | checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| mnli | row-warmup row paper gamma | row_warmup_papergamma | false | - | - | 9815 | fail_or_pending | - | 0.807641 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| qnli | FP16-SFT | baseline | true | 0.898957 | 5463 | 5463 | pass | [0.890682, 0.906670] | 0.898957 | 0.000000 | [-0.011302, 0.011302] | pass | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/fp16_sft-tensor-layer-1/metrics.json |
| qnli | tensor-warmup tensor gamma100 | tensor_warmup_gamma100 | true | 0.787846 | 5463 | 5463 | pass | [0.776804, 0.798483] | 0.898957 | 0.111111 | [0.097642, 0.124580] | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| qnli | tensor-warmup row gamma100 | tensor_warmup_row_gamma100 | true | 0.796998 | 5463 | 5463 | pass | [0.786125, 0.807454] | 0.898957 | 0.101959 | [0.088630, 0.115287] | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| qnli | tensor-warmup tensor paper gamma | tensor_warmup_papergamma | true | 0.759656 | 5463 | 5463 | pass | [0.748145, 0.770802] | 0.898957 | 0.139301 | [0.125435, 0.153166] | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| qnli | tensor-warmup row paper gamma | tensor_warmup_row_papergamma | true | 0.760937 | 5463 | 5463 | pass | [0.749446, 0.772061] | 0.898957 | 0.138019 | [0.124171, 0.151868] | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| qnli | row-warmup row gamma100 | row_warmup_gamma100 | true | 0.779791 | 5463 | 5463 | pass | [0.768608, 0.790581] | 0.898957 | 0.119165 | [0.105578, 0.132753] | fail_or_pending | checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| qnli | row-warmup row paper gamma | row_warmup_papergamma | false | - | - | 5463 | fail_or_pending | - | 0.898957 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| sst2 | FP16-SFT | baseline | true | 0.925459 | 872 | 872 | pass | [0.906098, 0.941087] | 0.925459 | 0.000000 | [-0.024654, 0.024654] | pass | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/fp16_sft-tensor-layer-1/metrics.json |
| sst2 | tensor-warmup tensor gamma100 | tensor_warmup_gamma100 | true | 0.866972 | 872 | 872 | pass | [0.842814, 0.887911] | 0.925459 | 0.058486 | [0.029991, 0.086981] | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| sst2 | tensor-warmup row gamma100 | tensor_warmup_row_gamma100 | true | 0.854358 | 872 | 872 | pass | [0.829391, 0.876217] | 0.925459 | 0.071101 | [0.041911, 0.100291] | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
| sst2 | tensor-warmup tensor paper gamma | tensor_warmup_papergamma | true | 0.841743 | 872 | 872 | pass | [0.816026, 0.864462] | 0.925459 | 0.083716 | [0.053870, 0.113561] | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| sst2 | tensor-warmup row paper gamma | tensor_warmup_row_papergamma | true | 0.837156 | 872 | 872 | pass | [0.811180, 0.860174] | 0.925459 | 0.088303 | [0.058228, 0.118377] | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
| sst2 | row-warmup row gamma100 | row_warmup_gamma100 | true | 0.846330 | 872 | 872 | pass | [0.820879, 0.868743] | 0.925459 | 0.079128 | [0.049517, 0.108740] | fail_or_pending | checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
| sst2 | row-warmup row paper gamma | row_warmup_papergamma | false | - | - | 872 | fail_or_pending | - | 0.925459 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |

## Comparisons

| task | comparison | left | left n | right | right n | left-right | left-right 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | row_warmup_gamma100_minus_tensor_warmup_tensor_gamma100 | 0.627713 | 9815 | 0.641671 | 9815 | -0.013958 | [-0.027429, -0.000488] |
| mnli | row_warmup_gamma100_minus_tensor_warmup_row_gamma100 | 0.627713 | 9815 | 0.653591 | 9815 | -0.025879 | [-0.039298, -0.012459] |
| mnli | row_warmup_papergamma_minus_tensor_warmup_tensor_papergamma | - | - | 0.630260 | 9815 | - | - |
| mnli | row_warmup_papergamma_minus_tensor_warmup_row_papergamma | - | - | 0.617626 | 9815 | - | - |
| qnli | row_warmup_gamma100_minus_tensor_warmup_tensor_gamma100 | 0.779791 | 5463 | 0.787846 | 5463 | -0.008054 | [-0.023491, 0.007382] |
| qnli | row_warmup_gamma100_minus_tensor_warmup_row_gamma100 | 0.779791 | 5463 | 0.796998 | 5463 | -0.017207 | [-0.032521, -0.001893] |
| qnli | row_warmup_papergamma_minus_tensor_warmup_tensor_papergamma | - | - | 0.759656 | 5463 | - | - |
| qnli | row_warmup_papergamma_minus_tensor_warmup_row_papergamma | - | - | 0.760937 | 5463 | - | - |
| sst2 | row_warmup_gamma100_minus_tensor_warmup_tensor_gamma100 | 0.846330 | 872 | 0.866972 | 872 | -0.020642 | [-0.053521, 0.012237] |
| sst2 | row_warmup_gamma100_minus_tensor_warmup_row_gamma100 | 0.846330 | 872 | 0.854358 | 872 | -0.008028 | [-0.041510, 0.025455] |
| sst2 | row_warmup_papergamma_minus_tensor_warmup_tensor_papergamma | - | - | 0.841743 | 872 | - | - |
| sst2 | row_warmup_papergamma_minus_tensor_warmup_row_papergamma | - | - | 0.837156 | 872 | - | - |
