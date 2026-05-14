# BitDistill Causal Packed-Ternary Export Gate, 2026-05-14

Results dir: `benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14`.

Passed: `false`.

This gate validates packed causal-LM ternary artifacts only; it does not validate sequence-classification heads.

Expected format split: tensor-scale paper baselines use scalar `I2_S`; row-scale novelty runs use stable `I2_SR`.

| task | scale | qtype | complete | exported | ternary keys | ftype | file MiB | PPL | prefill tok/s | decode tok/s | max RSS GiB | blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | tensor | i2_s | false | false | - | - | - | - | - | - | - | missing export summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/export_summary.json; missing export row; missing suite summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/gguf_suite/summary.json; missing memory summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/memory/summary.json |
| mnli | row | i2_sr | false | false | - | - | - | - | - | - | - | missing export summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/export_summary.json; missing export row; missing suite summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/gguf_suite/summary.json; missing memory summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/memory/summary.json |
| qnli | tensor | i2_s | false | false | - | - | - | - | - | - | - | missing export summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/export_summary.json; missing export row; missing suite summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/gguf_suite/summary.json; missing memory summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/memory/summary.json |
| qnli | row | i2_sr | false | false | - | - | - | - | - | - | - | missing export summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/export_summary.json; missing export row; missing suite summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/gguf_suite/summary.json; missing memory summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/memory/summary.json |
| sst2 | tensor | i2_s | false | false | - | - | - | - | - | - | - | missing export summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/export_summary.json; missing export row; missing suite summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/gguf_suite/summary.json; missing memory summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/memory/summary.json |
| sst2 | row | i2_sr | false | false | - | - | - | - | - | - | - | missing export summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/export_summary.json; missing export row; missing suite summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/gguf_suite/summary.json; missing memory summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/memory/summary.json |

## Blockers

- missing export row
- missing export summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/export_summary.json
- missing memory summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/memory/summary.json
- missing suite summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/gguf_suite/summary.json
