# BitDistill GLUE CPU Benchmark Gate, 2026-05-15

Input: `benchmark_results/bitdistill_glue_cpu_2026-05-15.json`.

Passed: `true`.

Critical run set: `['short:fp16_sft-tensor-layer-1', 'short:bitnet_sft-tensor-layer-1', 'short:bitdistill-tensor-layer-1', 'short:bitdistill-row-layer-1', 'longwarmup:bitdistill-longwarmup-tensor-layer-8', 'longwarmup:bitdistill-longwarmup-row-layer-8', 'papergamma:bitdistill-longwarmup-tensor-layer-8', 'papergamma_row:bitdistill-longwarmup-row-layer-8', 'papergamma_lr1:bitdistill-longwarmup-tensor-layer-8', 'papergamma_lr5:bitdistill-longwarmup-tensor-layer-8', 'papergamma_headinit:bitdistill-longwarmup-tensor-layer-8']`.

Threads: `12`. Batch size: `8`. Max eval samples: `512`. Child timeout seconds: `900`.

Full-quality contract: `{'mnli': 9815, 'qnli': 5463, 'sst2': 872}` examples from each checkpoint's stored full validation metric.

This gate validates PyTorch CPU sampled task-runtime rows and stored full task-quality metrics; it is not a packed llama.cpp/I2_SR runtime gate.

## Hardware

| field | value |
| --- | --- |
| CPU model | AMD Ryzen Threadripper PRO 5945WX 12-Cores |
| OS logical CPUs | 24 |
| cpuinfo logical CPUs | 24 |
| cpuinfo physical cores | 12 |
| requested threads | 12 |
| ISA flags | avx2=true, avx512bw=false, avx512dq=false, avx512f=false, avx512vl=false, bmi2=true, fma=true |
| platform | Linux-5.15.0-161-generic-x86_64-with-glibc2.35 |
| python | 3.13.5 |

## Critical Rows

| task | family | run | present | complete | status | sampled accuracy | sampled n | expected sampled n | stored full accuracy | stored full n | full quality | examples/s | RSS load MiB | max RSS MiB | blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | short | fp16_sft-tensor-layer-1 | true | true | complete | 0.791016 | 512 | 512 | 0.807641 | 9815.000000 | true | 9.379139 | 4264.566406 | 4431.117188 |  |
| mnli | short | bitnet_sft-tensor-layer-1 | true | true | complete | 0.486328 | 512 | 512 | 0.487621 | 9815.000000 | true | 6.606628 | 4429.316406 | 4821.886719 |  |
| mnli | short | bitdistill-tensor-layer-1 | true | true | complete | 0.541016 | 512 | 512 | 0.525217 | 9815.000000 | true | 6.896125 | 4427.777344 | 4941.496094 |  |
| mnli | short | bitdistill-row-layer-1 | true | true | complete | 0.498047 | 512 | 512 | 0.516556 | 9815.000000 | true | 7.026077 | 4426.496094 | 5083.160156 |  |
| mnli | longwarmup | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.646484 | 512 | 512 | 0.641671 | 9815.000000 | true | 5.899504 | 4968.421875 | 5288.964844 |  |
| mnli | longwarmup | bitdistill-longwarmup-row-layer-8 | true | true | complete | 0.675781 | 512 | 512 | 0.653591 | 9815.000000 | true | 7.782687 | 4429.277344 | 4744.800781 |  |
| mnli | papergamma | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.613281 | 512 | 512 | 0.630260 | 9815.000000 | true | 6.971964 | 4427.562500 | 4981.089844 |  |
| mnli | papergamma_row | bitdistill-longwarmup-row-layer-8 | true | true | complete | 0.582031 | 512 | 512 | 0.617626 | 9815.000000 | true | 7.511386 | 4428.929688 | 5018.781250 |  |
| mnli | papergamma_lr1 | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.574219 | 512 | 512 | 0.604381 | 9815.000000 | true | 5.629624 | 5000.792969 | 5328.632812 |  |
| mnli | papergamma_lr5 | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.613281 | 512 | 512 | 0.642384 | 9815.000000 | true | 6.196852 | 5018.000000 | 5340.128906 |  |
| mnli | papergamma_headinit | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.589844 | 512 | 512 | 0.627815 | 9815.000000 | true | 7.349255 | 4430.078125 | 4964.121094 |  |
| qnli | short | fp16_sft-tensor-layer-1 | true | true | complete | 0.898438 | 512 | 512 | 0.898957 | 5463.000000 | true | 8.244979 | 4264.402344 | 4575.859375 |  |
| qnli | short | bitnet_sft-tensor-layer-1 | true | true | complete | 0.572266 | 512 | 512 | 0.596925 | 5463.000000 | true | 5.785050 | 4428.250000 | 5014.558594 |  |
| qnli | short | bitdistill-tensor-layer-1 | true | true | complete | 0.615234 | 512 | 512 | 0.596925 | 5463.000000 | true | 5.617994 | 4429.828125 | 5060.894531 |  |
| qnli | short | bitdistill-row-layer-1 | true | true | complete | 0.601562 | 512 | 512 | 0.618525 | 5463.000000 | true | 4.926247 | 5018.417969 | 5387.800781 |  |
| qnli | longwarmup | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.761719 | 512 | 512 | 0.787846 | 5463.000000 | true | 5.829508 | 4430.644531 | 4834.027344 |  |
| qnli | longwarmup | bitdistill-longwarmup-row-layer-8 | true | true | complete | 0.767578 | 512 | 512 | 0.796998 | 5463.000000 | true | 4.807509 | 5015.273438 | 5367.457031 |  |
| qnli | papergamma | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.730469 | 512 | 512 | 0.759656 | 5463.000000 | true | 5.809975 | 4430.738281 | 4847.859375 |  |
| qnli | papergamma_row | bitdistill-longwarmup-row-layer-8 | true | true | complete | 0.728516 | 512 | 512 | 0.760937 | 5463.000000 | true | 5.694707 | 4429.183594 | 5075.695312 |  |
| qnli | papergamma_lr1 | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.751953 | 512 | 512 | 0.757459 | 5463.000000 | true | 5.591501 | 4429.710938 | 5149.730469 |  |
| qnli | papergamma_lr5 | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.759766 | 512 | 512 | 0.790957 | 5463.000000 | true | 5.666964 | 4429.910156 | 4991.394531 |  |
| qnli | papergamma_headinit | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.750000 | 512 | 512 | 0.762951 | 5463.000000 | true | 4.974457 | 5007.660156 | 5355.578125 |  |
| sst2 | short | fp16_sft-tensor-layer-1 | true | true | complete | 0.941406 | 512 | 512 | 0.925459 | 872.000000 | true | 18.225072 | 4265.140625 | 4365.234375 |  |
| sst2 | short | bitnet_sft-tensor-layer-1 | true | true | complete | 0.794922 | 512 | 512 | 0.770642 | 872.000000 | true | 9.747764 | 4430.175781 | 4758.144531 |  |
| sst2 | short | bitdistill-tensor-layer-1 | true | true | complete | 0.835938 | 512 | 512 | 0.815367 | 872.000000 | true | 8.645011 | 5050.449219 | 5323.476562 |  |
| sst2 | short | bitdistill-row-layer-1 | true | true | complete | 0.812500 | 512 | 512 | 0.808486 | 872.000000 | true | 9.010471 | 4428.738281 | 4840.207031 |  |
| sst2 | longwarmup | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.867188 | 512 | 512 | 0.866972 | 872.000000 | true | 10.049761 | 4429.019531 | 4719.945312 |  |
| sst2 | longwarmup | bitdistill-longwarmup-row-layer-8 | true | true | complete | 0.863281 | 512 | 512 | 0.854358 | 872.000000 | true | 9.402918 | 4430.531250 | 4849.746094 |  |
| sst2 | papergamma | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.849609 | 512 | 512 | 0.841743 | 872.000000 | true | 7.675149 | 5025.292969 | 5290.199219 |  |
| sst2 | papergamma_row | bitdistill-longwarmup-row-layer-8 | true | true | complete | 0.845703 | 512 | 512 | 0.837156 | 872.000000 | true | 9.815497 | 4429.835938 | 4792.093750 |  |
| sst2 | papergamma_lr1 | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.851562 | 512 | 512 | 0.846330 | 872.000000 | true | 7.519133 | 4974.480469 | 5233.996094 |  |
| sst2 | papergamma_lr5 | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.851562 | 512 | 512 | 0.836009 | 872.000000 | true | 9.211185 | 4429.933594 | 4866.687500 |  |
| sst2 | papergamma_headinit | bitdistill-longwarmup-tensor-layer-8 | true | true | complete | 0.845703 | 512 | 512 | 0.834862 | 872.000000 | true | 10.114614 | 4429.558594 | 4800.460938 |  |
