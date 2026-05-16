# TL2 Group-Scale Viability Audit, 2026-05-15

Do not implement group/tile-scale TL2 as the quality-preserving row-scale path; the best available fp16 group-scale row still misses the strict output-fidelity gate.

## Decision Summary

| field | value |
| --- | --- |
| strict threshold | 0.010000 |
| loose threshold | 0.100000 |
| current one-scale TL2 error | 1.904230 |
| best tensor L2 one-scale error | 0.161173 |
| best fp16 group-scale error | 0.098692 |
| best fp16 group-scale MiB | 0.615234 |
| exact row fp16 error | 0.000197 |
| exact row fp16 MiB | 1.230469 |
| strict group-scale viable | false |
| loose group-scale viable | true |
| exact row required for strict fidelity | true |
| best group / exact row error ratio | 499.843218 |

## Row-Scale Checkpoint Group Sweep

| group size | relative output RMS error | scale MiB |
| --- | --- | --- |
| 2 | 0.098692 | 0.615234 |
| 4 | 0.120769 | 0.307617 |
| 8 | 0.130677 | 0.153809 |
| 16 | 0.136894 | 0.076904 |
| 32 | 0.142844 | 0.038452 |
| 64 | 0.153960 | 0.019226 |
| 128 | 0.154392 | 0.009613 |
| 256 | 0.158026 | 0.004807 |
| 512 | 0.159537 | 0.002510 |
| 1024 | 0.160028 | 0.001389 |

## Tensor-Scale Control

| field | value |
| --- | --- |
| tensor-scale checkpoint current TL2 error | 0.000000 |
| tensor-scale checkpoint row-fp16 error | 0.000175 |

## Interpretation

The grouped-scale rows answer whether TL2 can use one scale per output-row group instead of one scale per row. For the audited Qwen2.5-1.5B row-scale checkpoint, even the best available fp16 group-scale setting misses the strict `0.01` relative-output-RMS fidelity gate by roughly two orders of magnitude compared with exact row fp16 scales. Group-scale TL2 may be useful as a future speed/quality experiment, but it should not close the current row-scale TL2 objective blocker. The quality-preserving path is exact row-scale metadata or a different proven scale model.
