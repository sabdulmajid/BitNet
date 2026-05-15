# BitDistill GLUE CPU Benchmark Gate, 2026-05-15

Input: `benchmark_results/bitdistill_glue_cpu_xeon_2026-05-15.json`.

Passed: `true`.

Critical run set: `['short:fp16_sft-tensor-layer-1', 'short:bitnet_sft-tensor-layer-1', 'short:bitdistill-tensor-layer-1', 'short:bitdistill-row-layer-1', 'longwarmup:bitdistill-longwarmup-tensor-layer-8', 'longwarmup:bitdistill-longwarmup-row-layer-8', 'papergamma:bitdistill-longwarmup-tensor-layer-8', 'papergamma_row:bitdistill-longwarmup-row-layer-8', 'papergamma_lr1:bitdistill-longwarmup-tensor-layer-8', 'papergamma_lr5:bitdistill-longwarmup-tensor-layer-8', 'papergamma_headinit:bitdistill-longwarmup-tensor-layer-8']`.

Threads: `12`. Batch size: `8`. Max eval samples: `512`. Child timeout seconds: `900`.

Full-quality contract: `{'mnli': 9815, 'qnli': 5463, 'sst2': 872}` examples from each checkpoint's stored full validation metric.

This gate validates PyTorch CPU sampled task-runtime rows and stored full task-quality metrics; it is not a packed llama.cpp/I2_SR runtime gate.

## Hardware

| field | value |
| --- | --- |
| CPU model | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz |
| OS logical CPUs | 24 |
| cpuinfo logical CPUs | 24 |
| cpuinfo physical cores | 12 |
| requested threads | 12 |
| ISA flags | avx2=true, avx512bw=true, avx512dq=true, avx512f=true, avx512vl=true, bmi2=true, fma=true |
| platform | Linux-5.15.0-161-generic-x86_64-with-glibc2.35 |
| python | 3.13.5 |

## Critical Rows

| task | family | run | present | complete | status | sampled accuracy | sampled n | expected sampled n | stored full accuracy | stored full n | full quality | examples/s | RSS load MiB | max RSS MiB | blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | short | fp16_sft-tensor-layer-1 | true | true | complete | 0.791016 | 512 | 512 | 0.807641 | 9815.000000 | true | 6.319930 | 4254.089844 | 4423.277344 |  |
| mnli | short | bitnet_sft-tensor-layer-1 | true | true | complete | 0.496094 | 512 | 512 | 0.487621 | 9815.000000 | true | 3.456502 | 4421.023438 | 4694.792969 |  |
| mnli | short | bitdistill-tensor-layer-1 | true | true | complete | 0.537109 | 512 | 512 | 0.525217 | 9815.000000 | true | 3.116637 | 4419.269531 | 4750.101562 |  |
| mnli | short | bitdistill-row-layer-1 | true | true | complete | 0.500000 | 512 | 512 | 0.516556 | 9815.000000 | true | 3.142816 | 4419.722656 | 4710.214844 |  |
| mnli | longwarmup | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.654297 | 512 | 512 | 0.641671 | 9815.000000 | true | 3.093551 | 4416.050781 | 4748.136719 |  |
| mnli | longwarmup | bitdistill-longwarmup-row-layer-8 | true | true | complete | 0.664062 | 512 | 512 | 0.653591 | 9815.000000 | true | 3.361551 | 4418.792969 | 4713.136719 |  |
| mnli | papergamma | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.615234 | 512 | 512 | 0.630260 | 9815.000000 | true | 2.979899 | 4418.144531 | 5100.261719 |  |
| mnli | papergamma_row | bitdistill-longwarmup-row-layer-8 | true | true | complete | 0.585938 | 512 | 512 | 0.617626 | 9815.000000 | true | 3.301501 | 4418.593750 | 4891.460938 |  |
| mnli | papergamma_lr1 | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.578125 | 512 | 512 | 0.604381 | 9815.000000 | true | 3.028186 | 4419.171875 | 4954.957031 |  |
| mnli | papergamma_lr5 | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.599609 | 512 | 512 | 0.642384 | 9815.000000 | true | 2.740737 | 5058.027344 | 5439.882812 |  |
| mnli | papergamma_headinit | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.595703 | 512 | 512 | 0.627815 | 9815.000000 | true | 2.756553 | 5007.300781 | 5322.949219 |  |
| qnli | short | fp16_sft-tensor-layer-1 | true | true | complete | 0.898438 | 512 | 512 | 0.898957 | 5463.000000 | true | 4.946787 | 4254.019531 | 4512.812500 |  |
| qnli | short | bitnet_sft-tensor-layer-1 | true | true | complete | 0.570312 | 512 | 512 | 0.596925 | 5463.000000 | true | 2.648767 | 4418.746094 | 5076.402344 |  |
| qnli | short | bitdistill-tensor-layer-1 | true | true | complete | 0.609375 | 512 | 512 | 0.596925 | 5463.000000 | true | 2.531845 | 4419.777344 | 5033.382812 |  |
| qnli | short | bitdistill-row-layer-1 | true | true | complete | 0.603516 | 512 | 512 | 0.618525 | 5463.000000 | true | 2.428779 | 5008.726562 | 5449.882812 |  |
| qnli | longwarmup | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.761719 | 512 | 512 | 0.787846 | 5463.000000 | true | 2.562941 | 4416.613281 | 4992.882812 |  |
| qnli | longwarmup | bitdistill-longwarmup-row-layer-8 | true | true | complete | 0.767578 | 512 | 512 | 0.796998 | 5463.000000 | true | 2.567045 | 4418.636719 | 4984.898438 |  |
| qnli | papergamma | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.728516 | 512 | 512 | 0.759656 | 5463.000000 | true | 2.514052 | 4419.656250 | 5181.031250 |  |
| qnli | papergamma_row | bitdistill-longwarmup-row-layer-8 | true | true | complete | 0.728516 | 512 | 512 | 0.760937 | 5463.000000 | true | 2.573962 | 4418.855469 | 5099.183594 |  |
| qnli | papergamma_lr1 | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.746094 | 512 | 512 | 0.757459 | 5463.000000 | true | 2.620683 | 4419.101562 | 5121.574219 |  |
| qnli | papergamma_lr5 | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.763672 | 512 | 512 | 0.790957 | 5463.000000 | true | 2.485811 | 4417.316406 | 5107.421875 |  |
| qnli | papergamma_headinit | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.750000 | 512 | 512 | 0.762951 | 5463.000000 | true | 2.555828 | 4418.062500 | 5128.800781 |  |
| sst2 | short | fp16_sft-tensor-layer-1 | true | true | complete | 0.941406 | 512 | 512 | 0.925459 | 872.000000 | true | 10.355743 | 3824.789062 | 3957.402344 |  |
| sst2 | short | bitnet_sft-tensor-layer-1 | true | true | complete | 0.792969 | 512 | 512 | 0.770642 | 872.000000 | true | 4.079304 | 4418.769531 | 4865.867188 |  |
| sst2 | short | bitdistill-tensor-layer-1 | true | true | complete | 0.835938 | 512 | 512 | 0.815367 | 872.000000 | true | 4.092037 | 4420.160156 | 4755.609375 |  |
| sst2 | short | bitdistill-row-layer-1 | true | true | complete | 0.814453 | 512 | 512 | 0.808486 | 872.000000 | true | 4.128891 | 4419.671875 | 4763.390625 |  |
| sst2 | longwarmup | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.869141 | 512 | 512 | 0.866972 | 872.000000 | true | 3.934726 | 4416.746094 | 4920.132812 |  |
| sst2 | longwarmup | bitdistill-longwarmup-row-layer-8 | true | true | complete | 0.865234 | 512 | 512 | 0.854358 | 872.000000 | true | 4.001375 | 5059.042969 | 5318.375000 |  |
| sst2 | papergamma | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.853516 | 512 | 512 | 0.841743 | 872.000000 | true | 4.178354 | 4417.691406 | 4687.871094 |  |
| sst2 | papergamma_row | bitdistill-longwarmup-row-layer-8 | true | true | complete | 0.847656 | 512 | 512 | 0.837156 | 872.000000 | true | 3.950266 | 4418.007812 | 4808.511719 |  |
| sst2 | papergamma_lr1 | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.857422 | 512 | 512 | 0.846330 | 872.000000 | true | 4.140466 | 4418.386719 | 4778.820312 |  |
| sst2 | papergamma_lr5 | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.857422 | 512 | 512 | 0.836009 | 872.000000 | true | 4.001179 | 4417.222656 | 4732.531250 |  |
| sst2 | papergamma_headinit | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.851562 | 512 | 512 | 0.834862 | 872.000000 | true | 3.881937 | 4418.070312 | 4918.707031 |  |
