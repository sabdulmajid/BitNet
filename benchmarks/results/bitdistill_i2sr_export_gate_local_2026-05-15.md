# BitDistill Causal Packed-Ternary Export Gate, 2026-05-15

Results dir: `benchmark_results/bitdistill-causal-longwarmup-i2sr-local-2026-05-15`.

Passed: `true`.

This gate validates packed causal-LM ternary artifacts only; it does not validate sequence-classification heads.

Expected format split: tensor-scale paper baselines use scalar `I2_S`; row-scale novelty runs use stable `I2_SR`.

| task | scale | qtype | expected kind | complete | exported | ternary keys | ftype | manifest | file MiB | PPL | prefill tok/s | decode tok/s | max RSS GiB | blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | tensor | i2_s | bitdistill_tensor_bitnet25_i2_s | true | true | 168 | MOSTLY_I2_S | benchmark_results/bitdistill-causal-longwarmup-i2sr-local-2026-05-15/manifest.json | 611.102844 | 278656.166400 | 519.533058 | 38.951548 | 0.697208 |  |
| mnli | row | i2_sr | bitdistill_row_bitnet25_i2_sr | true | true | 168 | MOSTLY_I2_SR | benchmark_results/bitdistill-causal-longwarmup-i2sr-local-2026-05-15/manifest.json | 612.263000 | 234478.130600 | 519.610950 | 38.559266 | 0.698128 |  |
| qnli | tensor | i2_s | bitdistill_tensor_bitnet25_i2_s | true | true | 168 | MOSTLY_I2_S | benchmark_results/bitdistill-causal-longwarmup-i2sr-local-2026-05-15/manifest.json | 611.102844 | 155846.123800 | 520.030218 | 38.911870 | 0.697086 |  |
| qnli | row | i2_sr | bitdistill_row_bitnet25_i2_sr | true | true | 168 | MOSTLY_I2_SR | benchmark_results/bitdistill-causal-longwarmup-i2sr-local-2026-05-15/manifest.json | 612.263000 | 241942.003800 | 518.670571 | 38.625700 | 0.698238 |  |
| sst2 | tensor | i2_s | bitdistill_tensor_bitnet25_i2_s | true | true | 168 | MOSTLY_I2_S | benchmark_results/bitdistill-causal-longwarmup-i2sr-local-2026-05-15/manifest.json | 611.102844 | 192532.032100 | 516.897098 | 38.951125 | 0.697227 |  |
| sst2 | row | i2_sr | bitdistill_row_bitnet25_i2_sr | true | true | 168 | MOSTLY_I2_SR | benchmark_results/bitdistill-causal-longwarmup-i2sr-local-2026-05-15/manifest.json | 612.263000 | 347660.334700 | 520.372546 | 38.818632 | 0.698181 |  |
