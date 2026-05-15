# BitDistill Causal Packed-Ternary Export Gate, 2026-05-15

Results dir: `benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-15`.

Passed: `true`.

This gate validates packed causal-LM ternary artifacts only; it does not validate sequence-classification heads.

Expected format split: tensor-scale paper baselines use scalar `I2_S`; row-scale novelty runs use stable `I2_SR`.

| task | scale | qtype | expected kind | complete | exported | ternary keys | ftype | manifest | file MiB | PPL | prefill tok/s | decode tok/s | max RSS GiB | blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | tensor | i2_s | bitdistill_tensor_bitnet25_i2_s | true | true | 168 | MOSTLY_I2_S | benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-15/manifest.json | 611.102844 | 278656.166400 | 1175.744678 | 103.179564 | 0.697216 |  |
| mnli | row | i2_sr | bitdistill_row_bitnet25_i2_sr | true | true | 168 | MOSTLY_I2_SR | benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-15/manifest.json | 612.263000 | 234478.130600 | 1182.971653 | 93.780233 | 0.698441 |  |
| qnli | tensor | i2_s | bitdistill_tensor_bitnet25_i2_s | true | true | 168 | MOSTLY_I2_S | benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-15/manifest.json | 611.102844 | 155846.123800 | 1256.320416 | 102.849593 | 0.697124 |  |
| qnli | row | i2_sr | bitdistill_row_bitnet25_i2_sr | true | true | 168 | MOSTLY_I2_SR | benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-15/manifest.json | 612.263000 | 241942.003800 | 1259.647628 | 99.083992 | 0.698208 |  |
| sst2 | tensor | i2_s | bitdistill_tensor_bitnet25_i2_s | true | true | 168 | MOSTLY_I2_S | benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-15/manifest.json | 611.102844 | 192532.032100 | 1258.661773 | 105.411067 | 0.697029 |  |
| sst2 | row | i2_sr | bitdistill_row_bitnet25_i2_sr | true | true | 168 | MOSTLY_I2_SR | benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-15/manifest.json | 612.263000 | 347660.334700 | 1189.249793 | 96.629554 | 0.698242 |  |
