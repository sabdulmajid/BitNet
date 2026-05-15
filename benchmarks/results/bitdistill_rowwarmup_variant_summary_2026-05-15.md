# BitDistill Row-Warmup Variant Summary, 2026-05-15

Model: `Qwen/Qwen2.5-0.5B`.

FP reference root: `checkpoints/bitdistill-glue-seqcls`.

## Runs

| root | task | run | accuracy | FP16 | FP-run | examples | steps | method | format | scale | layer | head init | state loaded | shape mismatches | last CE | last wLogitKD | logit temp scale | last wAttnKD | attn QKV reduction | metrics path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100 | mnli | bitdistill-longwarmup-row-layer-8 | 0.627713 | 0.807641 | 0.179929 | 9815.000000 | 1000.000000 | bitdistill | sequence_classification | row | -8.000000 | false | true | 0.000000 | 0.236328 | 0.191733 | none | 1.978707 | sum | checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100 | qnli | bitdistill-longwarmup-row-layer-8 | 0.779791 | 0.898957 | 0.119165 | 5463.000000 | 1000.000000 | bitdistill | sequence_classification | row | -8.000000 | false | true | 0.000000 | 0.255859 | 0.248060 | none | 2.004323 | sum | checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100 | sst2 | bitdistill-longwarmup-row-layer-8 | 0.846330 | 0.925459 | 0.079128 | 872.000000 | 1000.000000 | bitdistill | sequence_classification | row | -8.000000 | false | true | 0.000000 | 1.062500 | 0.349534 | none | 1.277080 | sum | checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
| checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma | mnli | bitdistill-longwarmup-row-layer-8 | 0.617830 | 0.807641 | 0.189812 | 9815.000000 | 1000.000000 | bitdistill | sequence_classification | row | -8.000000 | false | true | 0.000000 | 0.361328 | 0.221521 | none | 1638.004028 | sum | checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma | qnli | bitdistill-longwarmup-row-layer-8 | 0.777046 | 0.898957 | 0.121911 | 5463.000000 | 1000.000000 | bitdistill | sequence_classification | row | -8.000000 | false | true | 0.000000 | 0.453125 | 0.283975 | none | 1548.720581 | sum | checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma | sst2 | bitdistill-longwarmup-row-layer-8 | 0.830275 | 0.925459 | 0.095183 | 872.000000 | 1000.000000 | bitdistill | sequence_classification | row | -8.000000 | false | true | 0.000000 | 0.843750 | 0.220247 | none | 974.834961 | sum | checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
