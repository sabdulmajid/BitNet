# BitDistill GLUE Summary, 2026-05-14

Model: `Qwen/Qwen2.5-0.5B`.

Root: `checkpoints/bitdistill-glue-causal-longwarmup-densehead`. FP root: `checkpoints/bitdistill-glue`. BitNet root: `checkpoints/bitdistill-glue`.

Overall threshold pass: `False` with max FP gap `0.01`.

## Metrics

| task | run | exists | accuracy | examples | steps | format | labels | score | eval mode | excluded linears | head init | last CE | last wLogitKD | logit temp scale | last wAttnKD | attn weight | metrics path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | fp16_sft | yes | 0.829852 | 9815.000000 | 1000.000000 | causal_lm | - | - | - | - | - | 0.088660 | - | - | - | - | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1/metrics.json |
| mnli | bitnet_sft | yes | 0.517983 | 9815.000000 | 1000.000000 | causal_lm | - | - | - | - | - | 0.649247 | - | - | - | - | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-layer-1/metrics.json |
| mnli | bitdistill_tensor | no | - | - | - | - | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| mnli | bitdistill_row | no | - | - | - | - | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| qnli | fp16_sft | yes | 0.900970 | 5463.000000 | 1000.000000 | causal_lm | - | - | - | - | - | 0.309719 | - | - | - | - | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/qnli/fp16_sft-tensor-layer-1/metrics.json |
| qnli | bitnet_sft | yes | 0.614681 | 5463.000000 | 1000.000000 | causal_lm | - | - | - | - | - | 0.126270 | - | - | - | - | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/qnli/bitnet_sft-tensor-layer-1/metrics.json |
| qnli | bitdistill_tensor | no | - | - | - | - | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| qnli | bitdistill_row | no | - | - | - | - | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| sst2 | fp16_sft | yes | 0.939220 | 872.000000 | 1000.000000 | causal_lm | - | - | - | - | - | 0.073379 | 0.000000 | - | 0.000000 | 0.000010 | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/sst2/fp16_sft-tensor-layer-1/metrics.json |
| sst2 | bitnet_sft | yes | 0.831422 | 872.000000 | 1000.000000 | causal_lm | - | - | - | - | - | 0.604714 | 0.000000 | - | 0.000000 | 0.000010 | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/sst2/bitnet_sft-tensor-layer-1/metrics.json |
| sst2 | bitdistill_tensor | no | - | - | - | - | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| sst2 | bitdistill_row | no | - | - | - | - | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |

## Verdicts

| task | FP16 | BitNet-SFT | BitDistill | FP-BitNet | FP-BitDistill | row-tensor | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | 0.829852 | 0.517983 | - | 0.311870 | - | - | fail |
| qnli | 0.900970 | 0.614681 | - | 0.286290 | - | - | fail |
| sst2 | 0.939220 | 0.831422 | - | 0.107798 | - | - | fail |
