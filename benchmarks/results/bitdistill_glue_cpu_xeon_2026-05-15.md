# BitDistill GLUE CPU Benchmark, 2026-05-15

Model: `Qwen/Qwen2.5-0.5B`.

Threads: `12`. Batch size: `8`. Max eval samples: `512`. Dtype: `fp32`. Child timeout: `900` seconds.

Progress: `33` / `33` terminal rows; complete rows `33`; failed/timeout/missing rows `0`.

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
| mnli | fp16_sft-tensor-layer-1 | short | complete | 0.791016 | 512 | 0.807641 | 9815.000000 | 6.319930 | 1.262717 | 1.957522 | 4254.089844 | 4423.277344 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1 |
| mnli | bitnet_sft-tensor-layer-1 | short | complete | 0.496094 | 512 | 0.487621 | 9815.000000 | 3.456502 | 2.311458 | 2.958272 | 4421.023438 | 4694.792969 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-layer-1 |
| mnli | bitdistill-tensor-layer-1 | short | complete | 0.537109 | 512 | 0.525217 | 9815.000000 | 3.116637 | 2.563814 | 3.259592 | 4419.269531 | 4750.101562 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-layer-1 |
| mnli | bitdistill-row-layer-1 | short | complete | 0.500000 | 512 | 0.516556 | 9815.000000 | 3.142816 | 2.542372 | 3.121351 | 4419.722656 | 4710.214844 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-row-layer-1 |
| mnli | bitdistill-longwarmup-tensor-layer-8 | longwarmup | complete | 0.654297 | 512 | 0.641671 | 9815.000000 | 3.093551 | 2.582987 | 3.265595 | 4416.050781 | 4748.136719 | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| mnli | bitdistill-longwarmup-row-layer-8 | longwarmup | complete | 0.664062 | 512 | 0.653591 | 9815.000000 | 3.361551 | 2.376817 | 3.036956 | 4418.792969 | 4713.136719 | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8 |
| mnli | bitdistill-longwarmup-tensor-layer-8 | papergamma | complete | 0.615234 | 512 | 0.630260 | 9815.000000 | 2.979899 | 2.681628 | 3.478403 | 4418.144531 | 5100.261719 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| mnli | bitdistill-longwarmup-row-layer-8 | papergamma_row | complete | 0.585938 | 512 | 0.617626 | 9815.000000 | 3.301501 | 2.420116 | 3.209252 | 4418.593750 | 4891.460938 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8 |
| mnli | bitdistill-longwarmup-tensor-layer-8 | papergamma_lr1 | complete | 0.578125 | 512 | 0.604381 | 9815.000000 | 3.028186 | 2.638810 | 3.369222 | 4419.171875 | 4954.957031 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr1e-5/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| mnli | bitdistill-longwarmup-tensor-layer-8 | papergamma_lr5 | complete | 0.599609 | 512 | 0.642384 | 9815.000000 | 2.740737 | 2.915880 | 3.728574 | 5058.027344 | 5439.882812 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr5e-5/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| mnli | bitdistill-longwarmup-tensor-layer-8 | papergamma_headinit | complete | 0.595703 | 512 | 0.627815 | 9815.000000 | 2.756553 | 2.899154 | 3.725460 | 5007.300781 | 5322.949219 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-headinit/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8 |
| qnli | fp16_sft-tensor-layer-1 | short | complete | 0.898438 | 512 | 0.898957 | 5463.000000 | 4.946787 | 1.613493 | 2.339844 | 4254.019531 | 4512.812500 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/fp16_sft-tensor-layer-1 |
| qnli | bitnet_sft-tensor-layer-1 | short | complete | 0.570312 | 512 | 0.596925 | 5463.000000 | 2.648767 | 3.016481 | 3.875019 | 4418.746094 | 5076.402344 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitnet_sft-tensor-layer-1 |
| qnli | bitdistill-tensor-layer-1 | short | complete | 0.609375 | 512 | 0.596925 | 5463.000000 | 2.531845 | 3.156019 | 4.052669 | 4419.777344 | 5033.382812 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-tensor-layer-1 |
| qnli | bitdistill-row-layer-1 | short | complete | 0.603516 | 512 | 0.618525 | 5463.000000 | 2.428779 | 3.290095 | 4.204681 | 5008.726562 | 5449.882812 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-row-layer-1 |
| qnli | bitdistill-longwarmup-tensor-layer-8 | longwarmup | complete | 0.761719 | 512 | 0.787846 | 5463.000000 | 2.562941 | 3.117701 | 4.056064 | 4416.613281 | 4992.882812 | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| qnli | bitdistill-longwarmup-row-layer-8 | longwarmup | complete | 0.767578 | 512 | 0.796998 | 5463.000000 | 2.567045 | 3.112706 | 4.000247 | 4418.636719 | 4984.898438 | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8 |
| qnli | bitdistill-longwarmup-tensor-layer-8 | papergamma | complete | 0.728516 | 512 | 0.759656 | 5463.000000 | 2.514052 | 3.178412 | 4.174162 | 4419.656250 | 5181.031250 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| qnli | bitdistill-longwarmup-row-layer-8 | papergamma_row | complete | 0.728516 | 512 | 0.760937 | 5463.000000 | 2.573962 | 3.104298 | 4.087913 | 4418.855469 | 5099.183594 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8 |
| qnli | bitdistill-longwarmup-tensor-layer-8 | papergamma_lr1 | complete | 0.746094 | 512 | 0.757459 | 5463.000000 | 2.620683 | 3.048926 | 3.945909 | 4419.101562 | 5121.574219 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr1e-5/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| qnli | bitdistill-longwarmup-tensor-layer-8 | papergamma_lr5 | complete | 0.763672 | 512 | 0.790957 | 5463.000000 | 2.485811 | 3.214519 | 4.255240 | 4417.316406 | 5107.421875 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr5e-5/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| qnli | bitdistill-longwarmup-tensor-layer-8 | papergamma_headinit | complete | 0.750000 | 512 | 0.762951 | 5463.000000 | 2.555828 | 3.126368 | 4.000951 | 4418.062500 | 5128.800781 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-headinit/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8 |
| sst2 | fp16_sft-tensor-layer-1 | short | complete | 0.941406 | 512 | 0.925459 | 872.000000 | 10.355743 | 0.770224 | 1.019044 | 3824.789062 | 3957.402344 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/fp16_sft-tensor-layer-1 |
| sst2 | bitnet_sft-tensor-layer-1 | short | complete | 0.792969 | 512 | 0.770642 | 872.000000 | 4.079304 | 1.958825 | 2.276183 | 4418.769531 | 4865.867188 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitnet_sft-tensor-layer-1 |
| sst2 | bitdistill-tensor-layer-1 | short | complete | 0.835938 | 512 | 0.815367 | 872.000000 | 4.092037 | 1.952699 | 2.263557 | 4420.160156 | 4755.609375 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-tensor-layer-1 |
| sst2 | bitdistill-row-layer-1 | short | complete | 0.814453 | 512 | 0.808486 | 872.000000 | 4.128891 | 1.935261 | 2.262875 | 4419.671875 | 4763.390625 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-row-layer-1 |
| sst2 | bitdistill-longwarmup-tensor-layer-8 | longwarmup | complete | 0.869141 | 512 | 0.866972 | 872.000000 | 3.934726 | 2.030866 | 2.358377 | 4416.746094 | 4920.132812 | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
| sst2 | bitdistill-longwarmup-row-layer-8 | longwarmup | complete | 0.865234 | 512 | 0.854358 | 872.000000 | 4.001375 | 1.997015 | 2.322985 | 5059.042969 | 5318.375000 | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8 |
| sst2 | bitdistill-longwarmup-tensor-layer-8 | papergamma | complete | 0.853516 | 512 | 0.841743 | 872.000000 | 4.178354 | 1.912323 | 2.215614 | 4417.691406 | 4687.871094 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
| sst2 | bitdistill-longwarmup-row-layer-8 | papergamma_row | complete | 0.847656 | 512 | 0.837156 | 872.000000 | 3.950266 | 2.022869 | 2.318092 | 4418.007812 | 4808.511719 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8 |
| sst2 | bitdistill-longwarmup-tensor-layer-8 | papergamma_lr1 | complete | 0.857422 | 512 | 0.846330 | 872.000000 | 4.140466 | 1.929846 | 2.212217 | 4418.386719 | 4778.820312 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr1e-5/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
| sst2 | bitdistill-longwarmup-tensor-layer-8 | papergamma_lr5 | complete | 0.857422 | 512 | 0.836009 | 872.000000 | 4.001179 | 1.997100 | 2.333841 | 4417.222656 | 4732.531250 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr5e-5/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
| sst2 | bitdistill-longwarmup-tensor-layer-8 | papergamma_headinit | complete | 0.851562 | 512 | 0.834862 | 872.000000 | 3.881937 | 2.058525 | 2.368929 | 4418.070312 | 4918.707031 | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-headinit/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8 |
