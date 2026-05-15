# BitDistill Loss-Scale Audit, 2026-05-15

Paper classification attention gamma: `100000.0`.

gamma=1e5 is finite in the local smoke test, but it can dominate CE by orders of magnitude under this implementation's relation-loss normalization. Legacy rows used a Q/K/V mean; the projected paper-gamma column converts those rows to the paper-style Q/K/V sum before estimating scale. Treat paper-gamma jobs as strict paper-hyperparameter stress tests and compare them to gamma=100 diagnostics.

Materialized rows with attention KD: `11`.

Projected paper-gamma attention/CE range: `2.118e+03` to `1.578e+04`.

## Runs

| run | exists | steps | scale | layer | CE | attention KD | QKV reduction | paper-equiv attention KD | actual gamma | actual weighted AD | actual AD/CE | projected paper weighted AD | projected paper AD/CE | metrics path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli:short-tensor-layer-1 | true | 1000 | tensor | -1 | 0.365234 | 0.013961 | legacy_mean | 0.041882 | 100.000000 | 1.396076 | 3.822410 | 4.188e+03 | 1.147e+04 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-layer-1/metrics.json |
| qnli:short-tensor-layer-1 | true | 1000 | tensor | -1 | 0.738281 | 0.012459 | legacy_mean | 0.037376 | 100.000000 | 1.245865 | 1.687521 | 3.738e+03 | 5.063e+03 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-tensor-layer-1/metrics.json |
| sst2:short-tensor-layer-1 | true | 1000 | tensor | -1 | 1.148438 | 0.009619 | legacy_mean | 0.028856 | 100.000000 | 0.961861 | 0.837539 | 2.886e+03 | 2.513e+03 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-tensor-layer-1/metrics.json |
| mnli:short-row-layer-1 | true | 1000 | row | -1 | 0.296875 | 0.015615 | legacy_mean | 0.046846 | 100.000000 | 1.561546 | 5.259944 | 4.685e+03 | 1.578e+04 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-row-layer-1/metrics.json |
| qnli:short-row-layer-1 | true | 1000 | row | -1 | 0.531250 | 0.012949 | legacy_mean | 0.038847 | 100.000000 | 1.294893 | 2.437445 | 3.885e+03 | 7.312e+03 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-row-layer-1/metrics.json |
| sst2:short-row-layer-1 | true | 1000 | row | -1 | 1.039062 | 0.011424 | legacy_mean | 0.034273 | 100.000000 | 1.142420 | 1.099472 | 3.427e+03 | 3.298e+03 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-row-layer-1/metrics.json |
| mnli:short-tensor-layer-8 | true | 1000 | tensor | -8 | 0.589844 | 0.008727 | legacy_mean | 0.026181 | 100.000000 | 0.872714 | 1.479569 | 2.618e+03 | 4.439e+03 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-layer-8/metrics.json |
| qnli:short-tensor-layer-8 | false | - | - | - | - | - | legacy_mean | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-tensor-layer-8/metrics.json |
| sst2:short-tensor-layer-8 | false | - | - | - | - | - | legacy_mean | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-tensor-layer-8/metrics.json |
| mnli:paperlogit-tensor-layer-1 | true | 1000 | tensor | -1 | 0.566406 | 0.010731 | legacy_mean | 0.032194 | 100.000000 | 1.073127 | 1.894624 | 3.219e+03 | 5.684e+03 | checkpoints/bitdistill-glue-seqcls-paperlogit/Qwen-Qwen2.5-0.5B/mnli/bitdistill-paperlogit-tensor-layer-1/metrics.json |
| qnli:paperlogit-tensor-layer-1 | false | - | - | - | - | - | legacy_mean | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-paperlogit/Qwen-Qwen2.5-0.5B/qnli/bitdistill-paperlogit-tensor-layer-1/metrics.json |
| sst2:paperlogit-tensor-layer-1 | false | - | - | - | - | - | legacy_mean | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-paperlogit/Qwen-Qwen2.5-0.5B/sst2/bitdistill-paperlogit-tensor-layer-1/metrics.json |
| mnli:paperlogit-row-layer-1 | true | 1000 | row | -1 | 0.523438 | 0.011150 | legacy_mean | 0.033449 | 100.000000 | 1.114958 | 2.130070 | 3.345e+03 | 6.390e+03 | checkpoints/bitdistill-glue-seqcls-paperlogit/Qwen-Qwen2.5-0.5B/mnli/bitdistill-paperlogit-row-layer-1/metrics.json |
| qnli:paperlogit-row-layer-1 | false | - | - | - | - | - | legacy_mean | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-paperlogit/Qwen-Qwen2.5-0.5B/qnli/bitdistill-paperlogit-row-layer-1/metrics.json |
| sst2:paperlogit-row-layer-1 | false | - | - | - | - | - | legacy_mean | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-paperlogit/Qwen-Qwen2.5-0.5B/sst2/bitdistill-paperlogit-row-layer-1/metrics.json |
| mnli:paperlogit-headinit-tensor-layer-1 | true | 1000 | tensor | -1 | 0.554688 | 0.010777 | legacy_mean | 0.032332 | 100.000000 | 1.077731 | 1.942951 | 3.233e+03 | 5.829e+03 | checkpoints/bitdistill-glue-seqcls-paperlogit/Qwen-Qwen2.5-0.5B/mnli/bitdistill-paperlogit-headinit-tensor-layer-1/metrics.json |
| qnli:paperlogit-headinit-tensor-layer-1 | false | - | - | - | - | - | legacy_mean | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-paperlogit/Qwen-Qwen2.5-0.5B/qnli/bitdistill-paperlogit-headinit-tensor-layer-1/metrics.json |
| sst2:paperlogit-headinit-tensor-layer-1 | false | - | - | - | - | - | legacy_mean | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-paperlogit/Qwen-Qwen2.5-0.5B/sst2/bitdistill-paperlogit-headinit-tensor-layer-1/metrics.json |
| mnli:longwarmup-tensor-gamma100 | false | - | - | - | - | - | legacy_mean | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| qnli:longwarmup-tensor-gamma100 | false | - | - | - | - | - | legacy_mean | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| sst2:longwarmup-tensor-gamma100 | false | - | - | - | - | - | legacy_mean | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| mnli:longwarmup-row-gamma100 | false | - | - | - | - | - | legacy_mean | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| qnli:longwarmup-row-gamma100 | false | - | - | - | - | - | legacy_mean | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| sst2:longwarmup-row-gamma100 | false | - | - | - | - | - | legacy_mean | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
| mnli:longwarmup-tensor-papergamma | false | - | - | - | - | - | legacy_mean | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| qnli:longwarmup-tensor-papergamma | false | - | - | - | - | - | legacy_mean | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| sst2:longwarmup-tensor-papergamma | false | - | - | - | - | - | legacy_mean | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| smoke:papergamma | true | 2 | tensor | -1 | 5.606061 | 0.118751 | sum | 0.118751 | 1.000e+05 | 1.188e+04 | 2.118e+03 | 1.188e+04 | 2.118e+03 | benchmark_results/tmp_bitdistill_papergamma_smoke/metrics.json |
