# BitDistill GLUE CPU Benchmark Gate, 2026-05-15

Input: `benchmark_results/bitdistill_glue_cpu_fast_2026-05-15.json`.

Passed: `true`.

Critical run set: `['short:fp16_sft-tensor-layer-1', 'short:bitnet_sft-tensor-layer-1', 'longwarmup:bitdistill-longwarmup-tensor-layer-8', 'longwarmup:bitdistill-longwarmup-row-layer-8', 'papergamma:bitdistill-longwarmup-tensor-layer-8']`.

Threads: `12`. Batch size: `8`. Max eval samples: `16`. Child timeout seconds: `420`.

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
| mnli | short | fp16_sft-tensor-layer-1 | true | true | complete | 0.812500 | 16 | 16 | 0.807641 | 9815.000000 | true | 4.933843 | 4255.410156 | 4379.222656 |  |
| mnli | short | bitnet_sft-tensor-layer-1 | true | true | complete | 0.375000 | 16 | 16 | 0.487621 | 9815.000000 | true | 2.532144 | 4992.964844 | 5265.449219 |  |
| mnli | longwarmup | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.500000 | 16 | 16 | 0.641671 | 9815.000000 | true | 2.378624 | 5008.121094 | 5300.070312 |  |
| mnli | longwarmup | bitdistill-longwarmup-row-layer-8 | true | true | complete | 0.625000 | 16 | 16 | 0.653591 | 9815.000000 | true | 2.445707 | 4419.503906 | 4829.078125 |  |
| mnli | papergamma | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.562500 | 16 | 16 | 0.630260 | 9815.000000 | true | 2.394263 | 5026.562500 | 5299.042969 |  |
| qnli | short | fp16_sft-tensor-layer-1 | true | true | complete | 0.875000 | 16 | 16 | 0.898957 | 5463.000000 | true | 5.440132 | 3856.445312 | 4037.324219 |  |
| qnli | short | bitnet_sft-tensor-layer-1 | true | true | complete | 0.562500 | 16 | 16 | 0.596925 | 5463.000000 | true | 3.087550 | 4419.484375 | 4678.917969 |  |
| qnli | longwarmup | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.812500 | 16 | 16 | 0.787846 | 5463.000000 | true | 2.685601 | 4419.105469 | 4723.539062 |  |
| qnli | longwarmup | bitdistill-longwarmup-row-layer-8 | true | true | complete | 0.687500 | 16 | 16 | 0.796998 | 5463.000000 | true | 2.476174 | 5059.210938 | 5318.832031 |  |
| qnli | papergamma | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.750000 | 16 | 16 | 0.759656 | 5463.000000 | true | 2.374129 | 4419.617188 | 4728.832031 |  |
| sst2 | short | fp16_sft-tensor-layer-1 | true | true | complete | 1.000000 | 16 | 16 | 0.925459 | 872.000000 | true | 10.951954 | 3775.738281 | 3899.277344 |  |
| sst2 | short | bitnet_sft-tensor-layer-1 | true | true | complete | 0.937500 | 16 | 16 | 0.770642 | 872.000000 | true | 4.361395 | 4420.214844 | 4679.617188 |  |
| sst2 | longwarmup | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 1.000000 | 16 | 16 | 0.866972 | 872.000000 | true | 4.240927 | 4420.644531 | 4679.980469 |  |
| sst2 | longwarmup | bitdistill-longwarmup-row-layer-8 | true | true | complete | 1.000000 | 16 | 16 | 0.854358 | 872.000000 | true | 4.275414 | 4418.714844 | 4678.265625 |  |
| sst2 | papergamma | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 1.000000 | 16 | 16 | 0.841743 | 872.000000 | true | 4.445281 | 4418.636719 | 4677.968750 |  |
