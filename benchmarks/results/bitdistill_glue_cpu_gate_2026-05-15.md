# BitDistill GLUE CPU Benchmark Gate, 2026-05-15

Input: `benchmark_results/bitdistill_glue_cpu_2026-05-15.json`.

Passed: `false`.

Threads: `-`. Batch size: `-`. Max eval samples: `-`. Child timeout seconds: `-`.

Full-quality contract: `{'mnli': 9815, 'qnli': 5463, 'sst2': 872}` examples from each checkpoint's stored full validation metric.

This gate validates PyTorch CPU sampled task-runtime rows and stored full task-quality metrics; it is not a packed llama.cpp/I2_SR runtime gate.

## Hardware

| field | value |
| --- | --- |
| CPU model | - |
| OS logical CPUs | - |
| cpuinfo logical CPUs | - |
| cpuinfo physical cores | - |
| requested threads | - |
| ISA flags | - |
| platform | - |
| python | - |

## Critical Rows

| task | family | run | present | complete | status | sampled accuracy | sampled n | expected sampled n | stored full accuracy | stored full n | full quality | examples/s | RSS load MiB | max RSS MiB | blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | short | fp16_sft-tensor-layer-1 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| mnli | short | bitnet_sft-tensor-layer-1 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| mnli | short | bitdistill-tensor-layer-1 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| mnli | short | bitdistill-row-layer-1 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| mnli | longwarmup | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| mnli | longwarmup | bitdistill-longwarmup-row-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| mnli | papergamma | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| mnli | papergamma_row | bitdistill-longwarmup-row-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| mnli | papergamma_lr1 | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| mnli | papergamma_lr5 | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| mnli | papergamma_headinit | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| qnli | short | fp16_sft-tensor-layer-1 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| qnli | short | bitnet_sft-tensor-layer-1 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| qnli | short | bitdistill-tensor-layer-1 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| qnli | short | bitdistill-row-layer-1 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| qnli | longwarmup | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| qnli | longwarmup | bitdistill-longwarmup-row-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| qnli | papergamma | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| qnli | papergamma_row | bitdistill-longwarmup-row-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| qnli | papergamma_lr1 | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| qnli | papergamma_lr5 | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| qnli | papergamma_headinit | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| sst2 | short | fp16_sft-tensor-layer-1 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| sst2 | short | bitnet_sft-tensor-layer-1 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| sst2 | short | bitdistill-tensor-layer-1 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| sst2 | short | bitdistill-row-layer-1 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| sst2 | longwarmup | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| sst2 | longwarmup | bitdistill-longwarmup-row-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| sst2 | papergamma | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| sst2 | papergamma_row | bitdistill-longwarmup-row-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| sst2 | papergamma_lr1 | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| sst2 | papergamma_lr5 | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |
| sst2 | papergamma_headinit | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json |

## Blockers

- missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-15.json
