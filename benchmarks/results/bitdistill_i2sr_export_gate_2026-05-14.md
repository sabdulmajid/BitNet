# BitDistill Causal I2_SR Export Gate, 2026-05-14

Results dir: `benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14`.

Passed: `false`.

This gate validates packed causal-LM I2_SR artifacts only; it does not validate sequence-classification heads.

| task | scale | complete | exported | ternary keys | file MiB | PPL | prefill tok/s | decode tok/s | max RSS GiB | blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | tensor | false | false | - | - | - | - | - | - | missing export summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/export_summary.json; missing export row; missing suite summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/gguf_suite/summary.json; missing memory summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/memory/summary.json |
| mnli | row | false | false | - | - | - | - | - | - | missing export summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/export_summary.json; missing export row; missing suite summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/gguf_suite/summary.json; missing memory summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/memory/summary.json |
| qnli | tensor | false | false | - | - | - | - | - | - | missing export summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/export_summary.json; missing export row; missing suite summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/gguf_suite/summary.json; missing memory summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/memory/summary.json |
| qnli | row | false | false | - | - | - | - | - | - | missing export summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/export_summary.json; missing export row; missing suite summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/gguf_suite/summary.json; missing memory summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/memory/summary.json |
| sst2 | tensor | false | false | - | - | - | - | - | - | missing export summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/export_summary.json; missing export row; missing suite summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/gguf_suite/summary.json; missing memory summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/memory/summary.json |
| sst2 | row | false | false | - | - | - | - | - | - | missing export summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/export_summary.json; missing export row; missing suite summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/gguf_suite/summary.json; missing memory summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/memory/summary.json |

## Blockers

- missing export row
- missing export summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/export_summary.json
- missing memory summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/memory/summary.json
- missing suite summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/gguf_suite/summary.json
