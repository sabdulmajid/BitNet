# BitDistill Reproduction Gate, 2026-05-14

Model: `Qwen/Qwen2.5-0.5B`.

Threshold: absolute FP16-SFT gap <= `0.01` accuracy.

Confidence intervals: accuracy uses Wilson 95% intervals; this aggregate gate uses unpaired normal delta intervals. The paired-prediction audit is the authoritative example-level comparison when `eval_predictions.jsonl` exists.

Strict paper-hyperparameter tensor candidate complete: `False`.

Strict paper-hyperparameter tensor candidate passed: `False`.

Row-scale candidate complete: `False`.

Row-scale candidate passed: `False`.

## Runs

| task | run | family | exists | accuracy | examples | accuracy 95% CI | FP16 | FP-run | FP-run 95% CI | status | metrics path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | FP16-SFT | baseline | true | 0.807641 | 9815 | [0.799724, 0.815318] | 0.807641 | 0.000000 | [-0.011028, 0.011028] | pass | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1/metrics.json |
| mnli | BitNet-SFT | baseline | true | 0.487621 | 9815 | [0.477739, 0.497513] | 0.807641 | 0.320020 | [0.307427, 0.332614] | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-layer-1/metrics.json |
| mnli | BitDistill short tensor layer -1 | short | true | 0.525217 | 9815 | [0.515329, 0.535084] | 0.807641 | 0.282425 | [0.269839, 0.295011] | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-layer-1/metrics.json |
| mnli | BitDistill short row layer -1 | short | true | 0.516556 | 9815 | [0.506665, 0.526434] | 0.807641 | 0.291085 | [0.278494, 0.303676] | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-row-layer-1/metrics.json |
| mnli | BitDistill short tensor layer -8 | short | true | 0.535711 | 9815 | [0.525832, 0.545561] | 0.807641 | 0.271931 | [0.259355, 0.284507] | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-layer-8/metrics.json |
| mnli | BitDistill longwarmup tensor layer -8 gamma100 | longwarmup_gamma100 | false | - | - | - | 0.807641 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| mnli | BitDistill longwarmup row layer -8 | row_scale_candidate | false | - | - | - | 0.807641 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| mnli | BitDistill longwarmup tensor layer -8 paper gamma | paper_hparam_candidate | false | - | - | - | 0.807641 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| mnli | BitDistill longwarmup row layer -8 paper gamma | paper_hparam_row_candidate | false | - | - | - | 0.807641 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| qnli | FP16-SFT | baseline | true | 0.898957 | 5463 | [0.890682, 0.906670] | 0.898957 | 0.000000 | [-0.011302, 0.011302] | pass | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/fp16_sft-tensor-layer-1/metrics.json |
| qnli | BitNet-SFT | baseline | true | 0.596925 | 5463 | [0.583854, 0.609860] | 0.898957 | 0.302032 | [0.286766, 0.317298] | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitnet_sft-tensor-layer-1/metrics.json |
| qnli | BitDistill short tensor layer -1 | short | true | 0.596925 | 5463 | [0.583854, 0.609860] | 0.898957 | 0.302032 | [0.286766, 0.317298] | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-tensor-layer-1/metrics.json |
| qnli | BitDistill short row layer -1 | short | true | 0.618525 | 5463 | [0.605565, 0.631318] | 0.898957 | 0.280432 | [0.265273, 0.295591] | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-row-layer-1/metrics.json |
| qnli | BitDistill short tensor layer -8 | short | false | - | - | - | 0.898957 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-tensor-layer-8/metrics.json |
| qnli | BitDistill longwarmup tensor layer -8 gamma100 | longwarmup_gamma100 | false | - | - | - | 0.898957 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| qnli | BitDistill longwarmup row layer -8 | row_scale_candidate | false | - | - | - | 0.898957 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| qnli | BitDistill longwarmup tensor layer -8 paper gamma | paper_hparam_candidate | false | - | - | - | 0.898957 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| qnli | BitDistill longwarmup row layer -8 paper gamma | paper_hparam_row_candidate | false | - | - | - | 0.898957 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| sst2 | FP16-SFT | baseline | true | 0.925459 | 872 | [0.906098, 0.941087] | 0.925459 | 0.000000 | [-0.024654, 0.024654] | pass | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/fp16_sft-tensor-layer-1/metrics.json |
| sst2 | BitNet-SFT | baseline | true | 0.770642 | 872 | [0.741587, 0.797324] | 0.925459 | 0.154817 | [0.121914, 0.187719] | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitnet_sft-tensor-layer-1/metrics.json |
| sst2 | BitDistill short tensor layer -1 | short | true | 0.815367 | 872 | [0.788251, 0.839717] | 0.925459 | 0.110092 | [0.078994, 0.141190] | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-tensor-layer-1/metrics.json |
| sst2 | BitDistill short row layer -1 | short | true | 0.808486 | 872 | [0.781038, 0.833228] | 0.925459 | 0.116972 | [0.085572, 0.148373] | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-row-layer-1/metrics.json |
| sst2 | BitDistill short tensor layer -8 | short | false | - | - | - | 0.925459 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-tensor-layer-8/metrics.json |
| sst2 | BitDistill longwarmup tensor layer -8 gamma100 | longwarmup_gamma100 | false | - | - | - | 0.925459 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| sst2 | BitDistill longwarmup row layer -8 | row_scale_candidate | false | - | - | - | 0.925459 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
| sst2 | BitDistill longwarmup tensor layer -8 paper gamma | paper_hparam_candidate | false | - | - | - | 0.925459 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| sst2 | BitDistill longwarmup row layer -8 paper gamma | paper_hparam_row_candidate | false | - | - | - | 0.925459 | - | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |

## Row-Scale Comparison

| task | comparison | tensor | tensor n | row | row n | row-tensor | row-tensor 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | gamma100_row_minus_tensor | - | - | - | - | - | - |
| mnli | paper_gamma_row_minus_tensor | - | - | - | - | - | - |
| qnli | gamma100_row_minus_tensor | - | - | - | - | - | - |
| qnli | paper_gamma_row_minus_tensor | - | - | - | - | - | - |
| sst2 | gamma100_row_minus_tensor | - | - | - | - | - | - |
| sst2 | paper_gamma_row_minus_tensor | - | - | - | - | - | - |
