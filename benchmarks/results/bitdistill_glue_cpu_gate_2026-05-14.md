# BitDistill GLUE CPU Benchmark Gate, 2026-05-14

Input: `benchmark_results/bitdistill_glue_cpu_2026-05-14.json`.

Passed: `false`.

Threads: `-`. Batch size: `-`. Max eval samples: `-`. Child timeout seconds: `-`.

This gate validates PyTorch CPU task-runtime rows only; it is not a packed llama.cpp/I2_SR runtime gate.

## Critical Rows

| task | family | run | present | complete | status | accuracy | examples/s | RSS load MiB | max RSS MiB | blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | short | fp16_sft-tensor-layer-1 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| mnli | short | bitnet_sft-tensor-layer-1 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| mnli | short | bitdistill-tensor-layer-1 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| mnli | short | bitdistill-row-layer-1 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| mnli | longwarmup | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| mnli | longwarmup | bitdistill-longwarmup-row-layer-8 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| mnli | papergamma | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| qnli | short | fp16_sft-tensor-layer-1 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| qnli | short | bitnet_sft-tensor-layer-1 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| qnli | short | bitdistill-tensor-layer-1 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| qnli | short | bitdistill-row-layer-1 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| qnli | longwarmup | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| qnli | longwarmup | bitdistill-longwarmup-row-layer-8 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| qnli | papergamma | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| sst2 | short | fp16_sft-tensor-layer-1 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| sst2 | short | bitnet_sft-tensor-layer-1 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| sst2 | short | bitdistill-tensor-layer-1 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| sst2 | short | bitdistill-row-layer-1 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| sst2 | longwarmup | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| sst2 | longwarmup | bitdistill-longwarmup-row-layer-8 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |
| sst2 | papergamma | bitdistill-longwarmup-tensor-layer-8 | false | false | - | - | - | - | - | missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json |

## Blockers

- missing input artifact benchmark_results/bitdistill_glue_cpu_2026-05-14.json
