# BitDistill GLUE CPU Benchmark, 2026-05-15

Model: `Qwen/Qwen2.5-0.5B`.

Threads: `12`. Batch size: `8`. Max eval samples: `512`. Dtype: `fp32`. Child timeout: `900` seconds.

This is PyTorch CPU sequence-classification runtime, not packed `I2_SR`/llama.cpp inference. The `accuracy` column is measured on the sampled CPU subset when `max_eval_samples > 0`; full task quality is the stored full validation metric.

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

## Runs

| task | run | family | status | sampled accuracy | sampled examples | stored full accuracy | stored full examples | examples/s | mean batch s | p95 batch s | RSS load MiB | max RSS MiB | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | fp16_sft-tensor-layer-1 | short | complete | 0.791016 | 512 | 0.807641 | 9815.000000 | 9.379139 | 0.851905 | 1.696914 | 4264.566406 | 4431.117188 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1 |
| mnli | bitnet_sft-tensor-layer-1 | short | complete | 0.486328 | 512 | 0.487621 | 9815.000000 | 6.606628 | 1.209920 | 2.180761 | 4429.316406 | 4821.886719 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-layer-1 |
| mnli | bitdistill-tensor-layer-1 | short | complete | 0.541016 | 512 | 0.525217 | 9815.000000 | 6.896125 | 1.159126 | 1.533789 | 4427.777344 | 4941.496094 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-layer-1 |
| mnli | bitdistill-row-layer-1 | short | complete | 0.498047 | 512 | 0.516556 | 9815.000000 | 7.026077 | 1.137648 | 1.527969 | 4426.496094 | 5083.160156 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-row-layer-1 |
| mnli | bitdistill-tensor-layer-8 | short | complete | 0.525391 | 512 | 0.535711 | 9815.000000 | 7.302692 | 1.094520 | 1.432612 | 4428.066406 | 4929.210938 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-layer-8 |
| mnli | bitdistill-longwarmup-tensor-layer-8 | longwarmup | complete | 0.646484 | 512 | 0.641671 | 9815.000000 | 5.899504 | 1.355103 | 1.698671 | 4968.421875 | 5288.964844 | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| mnli | bitdistill-longwarmup-row-layer-8 | longwarmup | complete | 0.675781 | 512 | 0.653591 | 9815.000000 | 7.782687 | 1.026960 | 1.399410 | 4429.277344 | 4744.800781 | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8 |
| mnli | bitdistill-longwarmup-tensor-layer-8 | papergamma | complete | 0.613281 | 512 | 0.630260 | 9815.000000 | 6.971964 | 1.146490 | 1.642094 | 4427.562500 | 4981.089844 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| mnli | bitdistill-longwarmup-row-layer-8 | papergamma_row | complete | 0.582031 | 512 | 0.617626 | 9815.000000 | 7.511386 | 1.064097 | 1.386306 | 4428.929688 | 5018.781250 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8 |
| mnli | bitdistill-longwarmup-tensor-layer-8 | papergamma_lr1 | complete | 0.574219 | 512 | 0.604381 | 9815.000000 | 5.629624 | 1.420087 | 1.877787 | 5000.792969 | 5328.632812 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr1e-5/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| mnli | bitdistill-longwarmup-tensor-layer-8 | papergamma_lr5 | complete | 0.613281 | 512 | 0.642384 | 9815.000000 | 6.196852 | 1.290004 | 1.746093 | 5018.000000 | 5340.128906 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr5e-5/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| mnli | bitdistill-longwarmup-tensor-layer-8 | papergamma_headinit | complete | 0.589844 | 512 | 0.627815 | 9815.000000 | 7.349255 | 1.087578 | 1.442036 | 4430.078125 | 4964.121094 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-headinit/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| qnli | fp16_sft-tensor-layer-1 | short | complete | 0.898438 | 512 | 0.898957 | 5463.000000 | 8.244979 | 0.969173 | 1.381214 | 4264.402344 | 4575.859375 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/fp16_sft-tensor-layer-1 |
| qnli | bitnet_sft-tensor-layer-1 | short | complete | 0.572266 | 512 | 0.596925 | 5463.000000 | 5.785050 | 1.381739 | 1.843895 | 4428.250000 | 5014.558594 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitnet_sft-tensor-layer-1 |
| qnli | bitdistill-tensor-layer-1 | short | complete | 0.615234 | 512 | 0.596925 | 5463.000000 | 5.617994 | 1.422889 | 1.863247 | 4429.828125 | 5060.894531 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-tensor-layer-1 |
| qnli | bitdistill-row-layer-1 | short | complete | 0.601562 | 512 | 0.618525 | 5463.000000 | 4.926247 | 1.622858 | 2.135512 | 5018.417969 | 5387.800781 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-row-layer-1 |
| qnli | bitdistill-tensor-layer-8 | short | missing | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-tensor-layer-8 |
| qnli | bitdistill-longwarmup-tensor-layer-8 | longwarmup | complete | 0.761719 | 512 | 0.787846 | 5463.000000 | 5.829508 | 1.371207 | 1.868387 | 4430.644531 | 4834.027344 | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| qnli | bitdistill-longwarmup-row-layer-8 | longwarmup | complete | 0.767578 | 512 | 0.796998 | 5463.000000 | 4.807509 | 1.662928 | 2.159831 | 5015.273438 | 5367.457031 | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8 |
| qnli | bitdistill-longwarmup-tensor-layer-8 | papergamma | complete | 0.730469 | 512 | 0.759656 | 5463.000000 | 5.809975 | 1.375840 | 1.869450 | 4430.738281 | 4847.859375 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| qnli | bitdistill-longwarmup-row-layer-8 | papergamma_row | complete | 0.728516 | 512 | 0.760937 | 5463.000000 | 5.694707 | 1.403687 | 1.909705 | 4429.183594 | 5075.695312 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8 |
| qnli | bitdistill-longwarmup-tensor-layer-8 | papergamma_lr1 | complete | 0.751953 | 512 | 0.757459 | 5463.000000 | 5.591501 | 1.429636 | 1.912923 | 4429.710938 | 5149.730469 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr1e-5/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| qnli | bitdistill-longwarmup-tensor-layer-8 | papergamma_lr5 | complete | 0.759766 | 512 | 0.790957 | 5463.000000 | 5.666964 | 1.410554 | 1.841192 | 4429.910156 | 4991.394531 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr5e-5/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| qnli | bitdistill-longwarmup-tensor-layer-8 | papergamma_headinit | complete | 0.750000 | 512 | 0.762951 | 5463.000000 | 4.974457 | 1.607098 | 2.140247 | 5007.660156 | 5355.578125 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-headinit/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| sst2 | fp16_sft-tensor-layer-1 | short | complete | 0.941406 | 512 | 0.925459 | 872.000000 | 18.225072 | 0.438175 | 0.561786 | 4265.140625 | 4365.234375 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/fp16_sft-tensor-layer-1 |
| sst2 | bitnet_sft-tensor-layer-1 | short | complete | 0.794922 | 512 | 0.770642 | 872.000000 | 9.747764 | 0.819909 | 0.943194 | 4430.175781 | 4758.144531 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitnet_sft-tensor-layer-1 |
| sst2 | bitdistill-tensor-layer-1 | short | complete | 0.835938 | 512 | 0.815367 | 872.000000 | 8.645011 | 0.924605 | 1.048107 | 5050.449219 | 5323.476562 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-tensor-layer-1 |
| sst2 | bitdistill-row-layer-1 | short | complete | 0.812500 | 512 | 0.808486 | 872.000000 | 9.010471 | 0.887061 | 1.030149 | 4428.738281 | 4840.207031 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-row-layer-1 |
| sst2 | bitdistill-tensor-layer-8 | short | missing | - | - | - | - | - | - | - | - | - | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-tensor-layer-8 |
| sst2 | bitdistill-longwarmup-tensor-layer-8 | longwarmup | complete | 0.867188 | 512 | 0.866972 | 872.000000 | 10.049761 | 0.795251 | 0.928962 | 4429.019531 | 4719.945312 | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
| sst2 | bitdistill-longwarmup-row-layer-8 | longwarmup | complete | 0.863281 | 512 | 0.854358 | 872.000000 | 9.402918 | 0.850008 | 0.985140 | 4430.531250 | 4849.746094 | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8 |
| sst2 | bitdistill-longwarmup-tensor-layer-8 | papergamma | complete | 0.849609 | 512 | 0.841743 | 872.000000 | 7.675149 | 1.041541 | 1.219214 | 5025.292969 | 5290.199219 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
| sst2 | bitdistill-longwarmup-row-layer-8 | papergamma_row | complete | 0.845703 | 512 | 0.837156 | 872.000000 | 9.815497 | 0.814229 | 0.948062 | 4429.835938 | 4792.093750 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8 |
| sst2 | bitdistill-longwarmup-tensor-layer-8 | papergamma_lr1 | complete | 0.851562 | 512 | 0.846330 | 872.000000 | 7.519133 | 1.063157 | 1.245402 | 4974.480469 | 5233.996094 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr1e-5/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
| sst2 | bitdistill-longwarmup-tensor-layer-8 | papergamma_lr5 | complete | 0.851562 | 512 | 0.836009 | 872.000000 | 9.211185 | 0.867710 | 0.987687 | 4429.933594 | 4866.687500 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr5e-5/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
| sst2 | bitdistill-longwarmup-tensor-layer-8 | papergamma_headinit | complete | 0.845703 | 512 | 0.834862 | 872.000000 | 10.114614 | 0.790145 | 0.956798 | 4429.558594 | 4800.460938 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-headinit/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
