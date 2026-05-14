# BitDistill Loss-Scale Audit, 2026-05-14

Paper classification attention gamma: `100000.0`.

gamma=1e5 is finite in the local smoke test, but it can dominate CE by orders of magnitude under this implementation's relation-loss normalization. Treat paper-gamma jobs as strict paper-hyperparameter stress tests and compare them to gamma=100 diagnostics.

Materialized rows with attention KD: `11`.

Projected paper-gamma attention/CE range: `837.539005` to `1.304e+04`.

## Runs

| run | exists | steps | scale | layer | CE | attention KD | actual gamma | actual weighted AD | actual AD/CE | projected paper weighted AD | projected paper AD/CE | metrics path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli:short-tensor-layer-1 | true | 1000 | tensor | -1 | 0.365234 | 0.013961 | 100.000000 | 1.396076 | 3.822410 | 1.396e+03 | 3.822e+03 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-layer-1/metrics.json |
| qnli:short-tensor-layer-1 | true | 1000 | tensor | -1 | 0.738281 | 0.012459 | 100.000000 | 1.245865 | 1.687521 | 1.246e+03 | 1.688e+03 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-tensor-layer-1/metrics.json |
| sst2:short-tensor-layer-1 | true | 1000 | tensor | -1 | 1.148438 | 0.009619 | 100.000000 | 0.961861 | 0.837539 | 961.861201 | 837.539005 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-tensor-layer-1/metrics.json |
| mnli:short-row-layer-1 | true | 1000 | row | -1 | 0.296875 | 0.015615 | 100.000000 | 1.561546 | 5.259944 | 1.562e+03 | 5.260e+03 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-row-layer-1/metrics.json |
| qnli:short-row-layer-1 | true | 1000 | row | -1 | 0.531250 | 0.012949 | 100.000000 | 1.294893 | 2.437445 | 1.295e+03 | 2.437e+03 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-row-layer-1/metrics.json |
| sst2:short-row-layer-1 | true | 1000 | row | -1 | 1.039062 | 0.011424 | 100.000000 | 1.142420 | 1.099472 | 1.142e+03 | 1.099e+03 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-row-layer-1/metrics.json |
| mnli:short-tensor-layer-8 | true | 1000 | tensor | -8 | 0.589844 | 0.008727 | 100.000000 | 0.872714 | 1.479569 | 872.714352 | 1.480e+03 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-layer-8/metrics.json |
| qnli:short-tensor-layer-8 | false | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-tensor-layer-8/metrics.json |
| sst2:short-tensor-layer-8 | false | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-tensor-layer-8/metrics.json |
| mnli:paperlogit-tensor-layer-1 | true | 1000 | tensor | -1 | 0.566406 | 0.010731 | 100.000000 | 1.073127 | 1.894624 | 1.073e+03 | 1.895e+03 | checkpoints/bitdistill-glue-seqcls-paperlogit/Qwen-Qwen2.5-0.5B/mnli/bitdistill-paperlogit-tensor-layer-1/metrics.json |
| qnli:paperlogit-tensor-layer-1 | false | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-paperlogit/Qwen-Qwen2.5-0.5B/qnli/bitdistill-paperlogit-tensor-layer-1/metrics.json |
| sst2:paperlogit-tensor-layer-1 | false | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-paperlogit/Qwen-Qwen2.5-0.5B/sst2/bitdistill-paperlogit-tensor-layer-1/metrics.json |
| mnli:paperlogit-row-layer-1 | true | 1000 | row | -1 | 0.523438 | 0.011150 | 100.000000 | 1.114958 | 2.130070 | 1.115e+03 | 2.130e+03 | checkpoints/bitdistill-glue-seqcls-paperlogit/Qwen-Qwen2.5-0.5B/mnli/bitdistill-paperlogit-row-layer-1/metrics.json |
| qnli:paperlogit-row-layer-1 | false | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-paperlogit/Qwen-Qwen2.5-0.5B/qnli/bitdistill-paperlogit-row-layer-1/metrics.json |
| sst2:paperlogit-row-layer-1 | false | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-paperlogit/Qwen-Qwen2.5-0.5B/sst2/bitdistill-paperlogit-row-layer-1/metrics.json |
| mnli:paperlogit-headinit-tensor-layer-1 | true | 1000 | tensor | -1 | 0.554688 | 0.010777 | 100.000000 | 1.077731 | 1.942951 | 1.078e+03 | 1.943e+03 | checkpoints/bitdistill-glue-seqcls-paperlogit/Qwen-Qwen2.5-0.5B/mnli/bitdistill-paperlogit-headinit-tensor-layer-1/metrics.json |
| qnli:paperlogit-headinit-tensor-layer-1 | false | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-paperlogit/Qwen-Qwen2.5-0.5B/qnli/bitdistill-paperlogit-headinit-tensor-layer-1/metrics.json |
| sst2:paperlogit-headinit-tensor-layer-1 | false | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-paperlogit/Qwen-Qwen2.5-0.5B/sst2/bitdistill-paperlogit-headinit-tensor-layer-1/metrics.json |
| mnli:longwarmup-tensor-gamma100 | false | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| qnli:longwarmup-tensor-gamma100 | false | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| sst2:longwarmup-tensor-gamma100 | false | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| mnli:longwarmup-row-gamma100 | false | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| qnli:longwarmup-row-gamma100 | false | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| sst2:longwarmup-row-gamma100 | false | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
| mnli:longwarmup-tensor-papergamma | false | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| qnli:longwarmup-tensor-papergamma | false | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| sst2:longwarmup-tensor-papergamma | false | - | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| smoke:papergamma | true | 2 | tensor | -1 | 0.657691 | 0.085731 | 1.000e+05 | 8.573e+03 | 1.304e+04 | 8.573e+03 | 1.304e+04 | benchmark_results/tmp_bitdistill_papergamma_smoke/metrics.json |
