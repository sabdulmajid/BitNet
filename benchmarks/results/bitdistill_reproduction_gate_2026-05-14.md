# BitDistill Reproduction Gate, 2026-05-14

Model: `Qwen/Qwen2.5-0.5B`.

Threshold: absolute FP16-SFT gap <= `0.01` accuracy.

Strict paper-hyperparameter tensor candidate complete: `False`.

Strict paper-hyperparameter tensor candidate passed: `False`.

Row-scale candidate complete: `False`.

Row-scale candidate passed: `False`.

## Runs

| task | run | family | exists | accuracy | FP16 | FP-run | status | metrics path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | FP16-SFT | baseline | true | 0.807641 | 0.807641 | 0.000000 | pass | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1/metrics.json |
| mnli | BitNet-SFT | baseline | true | 0.487621 | 0.807641 | 0.320020 | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-layer-1/metrics.json |
| mnli | BitDistill short tensor layer -1 | short | true | 0.525217 | 0.807641 | 0.282425 | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-layer-1/metrics.json |
| mnli | BitDistill short row layer -1 | short | true | 0.516556 | 0.807641 | 0.291085 | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-row-layer-1/metrics.json |
| mnli | BitDistill short tensor layer -8 | short | true | 0.535711 | 0.807641 | 0.271931 | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-layer-8/metrics.json |
| mnli | BitDistill longwarmup tensor layer -8 gamma100 | longwarmup_gamma100 | false | - | 0.807641 | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| mnli | BitDistill longwarmup row layer -8 | row_scale_candidate | false | - | 0.807641 | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| mnli | BitDistill longwarmup tensor layer -8 paper gamma | paper_hparam_candidate | false | - | 0.807641 | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| qnli | FP16-SFT | baseline | true | 0.898957 | 0.898957 | 0.000000 | pass | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/fp16_sft-tensor-layer-1/metrics.json |
| qnli | BitNet-SFT | baseline | true | 0.596925 | 0.898957 | 0.302032 | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitnet_sft-tensor-layer-1/metrics.json |
| qnli | BitDistill short tensor layer -1 | short | true | 0.596925 | 0.898957 | 0.302032 | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-tensor-layer-1/metrics.json |
| qnli | BitDistill short row layer -1 | short | true | 0.618525 | 0.898957 | 0.280432 | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-row-layer-1/metrics.json |
| qnli | BitDistill short tensor layer -8 | short | false | - | 0.898957 | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-tensor-layer-8/metrics.json |
| qnli | BitDistill longwarmup tensor layer -8 gamma100 | longwarmup_gamma100 | false | - | 0.898957 | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| qnli | BitDistill longwarmup row layer -8 | row_scale_candidate | false | - | 0.898957 | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| qnli | BitDistill longwarmup tensor layer -8 paper gamma | paper_hparam_candidate | false | - | 0.898957 | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| sst2 | FP16-SFT | baseline | true | 0.925459 | 0.925459 | 0.000000 | pass | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/fp16_sft-tensor-layer-1/metrics.json |
| sst2 | BitNet-SFT | baseline | true | 0.770642 | 0.925459 | 0.154817 | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitnet_sft-tensor-layer-1/metrics.json |
| sst2 | BitDistill short tensor layer -1 | short | true | 0.815367 | 0.925459 | 0.110092 | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-tensor-layer-1/metrics.json |
| sst2 | BitDistill short row layer -1 | short | true | 0.808486 | 0.925459 | 0.116972 | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-row-layer-1/metrics.json |
| sst2 | BitDistill short tensor layer -8 | short | false | - | 0.925459 | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-tensor-layer-8/metrics.json |
| sst2 | BitDistill longwarmup tensor layer -8 gamma100 | longwarmup_gamma100 | false | - | 0.925459 | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| sst2 | BitDistill longwarmup row layer -8 | row_scale_candidate | false | - | 0.925459 | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
| sst2 | BitDistill longwarmup tensor layer -8 paper gamma | paper_hparam_candidate | false | - | 0.925459 | - | fail_or_pending | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |

## Row-Scale Comparison

| task | tensor | row | row-tensor |
| --- | --- | --- | --- |
| mnli | - | - | - |
| qnli | - | - | - |
| sst2 | - | - | - |
