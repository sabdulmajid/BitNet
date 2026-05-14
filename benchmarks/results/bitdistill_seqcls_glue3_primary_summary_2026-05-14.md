# BitDistill GLUE Summary, 2026-05-14

Model: `Qwen/Qwen2.5-0.5B`.

Overall threshold pass: `False` with max FP gap `0.01`.

## Metrics

| task | run | exists | accuracy | examples | steps | format | labels | score | eval mode | excluded linears | head init | last CE | last wLogitKD | last wAttnKD | attn weight | metrics path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | fp16_sft | yes | 0.807641 | 9815.000000 | 1000.000000 | sequence_classification | letters | mean | - | score\|classifier | - | 0.085449 | 0.000000 | 0.000000 | 100.000000 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1/metrics.json |
| mnli | bitnet_sft | yes | 0.487621 | 9815.000000 | 1000.000000 | sequence_classification | letters | mean | - | score\|classifier | - | 0.644531 | 0.000000 | 0.000000 | 100.000000 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-layer-1/metrics.json |
| mnli | bitdistill_tensor | yes | 0.525217 | 9815.000000 | 1000.000000 | sequence_classification | letters | mean | - | score\|classifier | - | 0.365234 | 4.512222 | 1.396076 | 100.000000 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-layer-1/metrics.json |
| mnli | bitdistill_row | yes | 0.516556 | 9815.000000 | 1000.000000 | sequence_classification | letters | mean | - | score\|classifier | false | 0.296875 | 5.578142 | 1.561546 | 100.000000 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-row-layer-1/metrics.json |
| qnli | fp16_sft | yes | 0.898957 | 5463.000000 | 1000.000000 | sequence_classification | letters | mean | - | score\|classifier | false | 0.028931 | 0.000000 | 0.000000 | 100.000000 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/fp16_sft-tensor-layer-1/metrics.json |
| qnli | bitnet_sft | yes | 0.596925 | 5463.000000 | 1000.000000 | sequence_classification | letters | mean | - | score\|classifier | false | 0.714844 | 0.000000 | 0.000000 | 100.000000 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitnet_sft-tensor-layer-1/metrics.json |
| qnli | bitdistill_tensor | yes | 0.596925 | 5463.000000 | 1000.000000 | sequence_classification | letters | mean | - | score\|classifier | false | 0.738281 | 12.720361 | 1.245865 | 100.000000 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-tensor-layer-1/metrics.json |
| qnli | bitdistill_row | yes | 0.618525 | 5463.000000 | 1000.000000 | sequence_classification | letters | mean | - | score\|classifier | false | 0.531250 | 9.721390 | 1.294893 | 100.000000 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-row-layer-1/metrics.json |
| sst2 | fp16_sft | yes | 0.925459 | 872.000000 | 1000.000000 | sequence_classification | letters | mean | - | score\|classifier | false | 0.113770 | 0.000000 | 0.000000 | 100.000000 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/fp16_sft-tensor-layer-1/metrics.json |
| sst2 | bitnet_sft | yes | 0.770642 | 872.000000 | 1000.000000 | sequence_classification | letters | mean | - | score\|classifier | false | 0.953125 | 0.000000 | 0.000000 | 100.000000 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitnet_sft-tensor-layer-1/metrics.json |
| sst2 | bitdistill_tensor | yes | 0.815367 | 872.000000 | 1000.000000 | sequence_classification | letters | mean | - | score\|classifier | false | 1.148438 | 4.108821 | 0.961861 | 100.000000 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-tensor-layer-1/metrics.json |
| sst2 | bitdistill_row | yes | 0.808486 | 872.000000 | 1000.000000 | sequence_classification | letters | mean | - | score\|classifier | false | 1.039062 | 3.922959 | 1.142420 | 100.000000 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-row-layer-1/metrics.json |

## Verdicts

| task | FP16 | BitNet-SFT | BitDistill | FP-BitNet | FP-BitDistill | row-tensor | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | 0.807641 | 0.487621 | 0.525217 | 0.320020 | 0.282425 | -0.008660 | fail |
| qnli | 0.898957 | 0.596925 | 0.596925 | 0.302032 | 0.302032 | 0.021600 | fail |
| sst2 | 0.925459 | 0.770642 | 0.815367 | 0.154817 | 0.110092 | -0.006881 | fail |
