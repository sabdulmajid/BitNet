# BitDistill GLUE Summary, 2026-05-15

Model: `Qwen/Qwen2.5-0.5B`.

Root: `checkpoints/bitdistill-glue-causal-longwarmup-densehead`. FP root: `checkpoints/bitdistill-glue`. BitNet root: `checkpoints/bitdistill-glue`.

Overall threshold pass: `False` with max FP gap `0.01`.

## Metrics

| task | run | exists | accuracy | examples | steps | format | labels | score | eval mode | excluded linears | head init | last CE | last wLogitKD | logit temp scale | last wAttnKD | attn weight | metrics path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | fp16_sft | yes | 0.829852 | 9815.000000 | 1000.000000 | causal_lm | - | - | - | - | - | 0.088660 | - | - | - | - | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1/metrics.json |
| mnli | bitnet_sft | yes | 0.517983 | 9815.000000 | 1000.000000 | causal_lm | - | - | - | - | - | 0.649247 | - | - | - | - | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-layer-1/metrics.json |
| mnli | bitdistill_tensor | yes | 0.615181 | 9815.000000 | 1000.000000 | causal_lm | letters | mean | single_forward_single_token_labels | lm_head | false | 0.706981 | 0.061914 | none | 1.106632 | 100.000000 | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| mnli | bitdistill_row | yes | 0.608355 | 9815.000000 | 1000.000000 | causal_lm | letters | mean | single_forward_single_token_labels | lm_head | false | 0.654137 | 0.060663 | none | 1.106820 | 100.000000 | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| qnli | fp16_sft | yes | 0.900970 | 5463.000000 | 1000.000000 | causal_lm | - | - | - | - | - | 0.309719 | - | - | - | - | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/qnli/fp16_sft-tensor-layer-1/metrics.json |
| qnli | bitnet_sft | yes | 0.614681 | 5463.000000 | 1000.000000 | causal_lm | - | - | - | - | - | 0.126270 | - | - | - | - | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/qnli/bitnet_sft-tensor-layer-1/metrics.json |
| qnli | bitdistill_tensor | yes | 0.765697 | 5463.000000 | 1000.000000 | causal_lm | letters | mean | single_forward_single_token_labels | lm_head | false | 0.358421 | 0.032653 | none | 1.226281 | 100.000000 | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| qnli | bitdistill_row | yes | 0.770822 | 5463.000000 | 1000.000000 | causal_lm | letters | mean | single_forward_single_token_labels | lm_head | false | 0.600795 | 0.030709 | none | 1.247302 | 100.000000 | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| sst2 | fp16_sft | yes | 0.939220 | 872.000000 | 1000.000000 | causal_lm | - | - | - | - | - | 0.073379 | 0.000000 | - | 0.000000 | 0.000010 | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/sst2/fp16_sft-tensor-layer-1/metrics.json |
| sst2 | bitnet_sft | yes | 0.831422 | 872.000000 | 1000.000000 | causal_lm | - | - | - | - | - | 0.604714 | 0.000000 | - | 0.000000 | 0.000010 | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/sst2/bitnet_sft-tensor-layer-1/metrics.json |
| sst2 | bitdistill_tensor | yes | 0.833716 | 872.000000 | 1000.000000 | causal_lm | letters | mean | single_forward_single_token_labels | lm_head | false | 0.076595 | 0.032288 | none | 0.551736 | 100.000000 | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| sst2 | bitdistill_row | yes | 0.840596 | 872.000000 | 1000.000000 | causal_lm | letters | mean | single_forward_single_token_labels | lm_head | false | 0.108806 | 0.032606 | none | 0.601614 | 100.000000 | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |

## Verdicts

| task | FP16 | BitNet-SFT | BitDistill | FP-BitNet | FP-BitDistill | row-tensor | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | 0.829852 | 0.517983 | 0.615181 | 0.311870 | 0.214671 | -0.006826 | fail |
| qnli | 0.900970 | 0.614681 | 0.765697 | 0.286290 | 0.135274 | 0.005125 | fail |
| sst2 | 0.939220 | 0.831422 | 0.833716 | 0.107798 | 0.105505 | 0.006881 | fail |
