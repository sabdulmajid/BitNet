# Sequence-Classification Native I2_SR Batching Audit, 2026-05-15

This audit checks whether the native token-ID classifier path is invariant to embedding prompt batching. It intentionally targets low-margin MNLI rows that changed between batch-1 and batch-4 samples.

| field | value |
| --- | --- |
| status | batching_parity_mismatch |
| targets | [15, 35] |
| all predictions invariant | false |
| max relative RMS vs alone | 0.305153 |
| changed cases | 4 |
| ready for batched product benchmark | false |

## Cases

| target | case | target pos | pred | margin | rel RMS vs alone | indices |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | alone | 0 | 1 | 0.019177 | 0.000000 | [15] |
| 15 | pos0 | 0 | 1 | 0.019177 | 0.000000 | [15, 16, 17, 18] |
| 15 | pos1 | 1 | 2 | 0.010487 | 0.110423 | [14, 15, 16, 17] |
| 15 | pos2 | 2 | 1 | 0.019177 | 0.000000 | [13, 14, 15, 16] |
| 15 | pos3 | 3 | 2 | 0.001524 | 0.114326 | [12, 13, 14, 15] |
| 35 | alone | 0 | 0 | 0.124395 | 0.000000 | [35] |
| 35 | pos0 | 0 | 0 | 0.124395 | 0.000000 | [35, 36, 37, 38] |
| 35 | pos1 | 1 | 0 | 0.124395 | 0.000000 | [34, 35, 36, 37] |
| 35 | pos2 | 2 | 1 | 0.156405 | 0.305153 | [33, 34, 35, 36] |
| 35 | pos3 | 3 | 1 | 0.084456 | 0.221412 | [32, 33, 34, 35] |

## Interpretation

Native batched logits are not invariant for the audited rows. Do not promote batched throughput or full-validation numbers until the llama.cpp sequence embedding path has a stable batching contract.
