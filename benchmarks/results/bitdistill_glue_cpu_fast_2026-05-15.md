# BitDistill GLUE CPU Benchmark, 2026-05-15

Model: `Qwen/Qwen2.5-0.5B`.

Threads: `12`. Batch size: `8`. Max eval samples: `16`. Dtype: `fp32`. Child timeout: `420` seconds.

This is PyTorch CPU sequence-classification runtime, not packed `I2_SR`/llama.cpp inference. The `accuracy` column is measured on the sampled CPU subset when `max_eval_samples > 0`; full task quality is the stored full validation metric.

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

## Runs

| task | run | family | status | sampled accuracy | sampled examples | stored full accuracy | stored full examples | examples/s | mean batch s | p95 batch s | RSS load MiB | max RSS MiB | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | fp16_sft-tensor-layer-1 | short | complete | 0.812500 | 16 | 0.807641 | 9815.000000 | 4.933843 | 1.617783 | 1.967161 | 4255.410156 | 4379.222656 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1 |
| mnli | bitnet_sft-tensor-layer-1 | short | complete | 0.375000 | 16 | 0.487621 | 9815.000000 | 2.532144 | 3.155685 | 3.484980 | 4992.964844 | 5265.449219 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-layer-1 |
| mnli | bitdistill-longwarmup-tensor-layer-8 | longwarmup | complete | 0.500000 | 16 | 0.641671 | 9815.000000 | 2.378624 | 3.359610 | 3.718457 | 5008.121094 | 5300.070312 | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| mnli | bitdistill-longwarmup-row-layer-8 | longwarmup | complete | 0.625000 | 16 | 0.653591 | 9815.000000 | 2.445707 | 3.267343 | 3.800718 | 4419.503906 | 4829.078125 | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8 |
| mnli | bitdistill-longwarmup-tensor-layer-8 | papergamma | complete | 0.562500 | 16 | 0.630260 | 9815.000000 | 2.394263 | 3.337649 | 3.701820 | 5026.562500 | 5299.042969 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| qnli | fp16_sft-tensor-layer-1 | short | complete | 0.875000 | 16 | 0.898957 | 5463.000000 | 5.440132 | 1.466905 | 1.579594 | 3856.445312 | 4037.324219 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/fp16_sft-tensor-layer-1 |
| qnli | bitnet_sft-tensor-layer-1 | short | complete | 0.562500 | 16 | 0.596925 | 5463.000000 | 3.087550 | 2.587322 | 2.757273 | 4419.484375 | 4678.917969 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitnet_sft-tensor-layer-1 |
| qnli | bitdistill-longwarmup-tensor-layer-8 | longwarmup | complete | 0.812500 | 16 | 0.787846 | 5463.000000 | 2.685601 | 2.975174 | 3.169986 | 4419.105469 | 4723.539062 | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| qnli | bitdistill-longwarmup-row-layer-8 | longwarmup | complete | 0.687500 | 16 | 0.796998 | 5463.000000 | 2.476174 | 3.227060 | 3.411340 | 5059.210938 | 5318.832031 | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8 |
| qnli | bitdistill-longwarmup-tensor-layer-8 | papergamma | complete | 0.750000 | 16 | 0.759656 | 5463.000000 | 2.374129 | 3.365915 | 3.559397 | 4419.617188 | 4728.832031 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| sst2 | fp16_sft-tensor-layer-1 | short | complete | 1.000000 | 16 | 0.925459 | 872.000000 | 10.951954 | 0.728186 | 0.857392 | 3775.738281 | 3899.277344 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/fp16_sft-tensor-layer-1 |
| sst2 | bitnet_sft-tensor-layer-1 | short | complete | 0.937500 | 16 | 0.770642 | 872.000000 | 4.361395 | 1.832005 | 1.924775 | 4420.214844 | 4679.617188 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitnet_sft-tensor-layer-1 |
| sst2 | bitdistill-longwarmup-tensor-layer-8 | longwarmup | complete | 1.000000 | 16 | 0.866972 | 872.000000 | 4.240927 | 1.884123 | 2.015654 | 4420.644531 | 4679.980469 | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
| sst2 | bitdistill-longwarmup-row-layer-8 | longwarmup | complete | 1.000000 | 16 | 0.854358 | 872.000000 | 4.275414 | 1.867023 | 2.057677 | 4418.714844 | 4678.265625 | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8 |
| sst2 | bitdistill-longwarmup-tensor-layer-8 | papergamma | complete | 1.000000 | 16 | 0.841743 | 872.000000 | 4.445281 | 1.797404 | 2.016726 | 4418.636719 | 4677.968750 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
