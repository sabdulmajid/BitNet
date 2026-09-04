# TL2_SR Kernel Contract, 2026-09-04

Generated lookup-table kernels are compared with the explicit reference
`Y[b, i] = row_scale[i] * sum_j(T[i, j] * Q8(X[b, j])) / activation_scale[b]`.
The test uses nonuniform row scales, so a scalar-scale implementation cannot pass.

Status: **pass**

| M | K | batch | relative RMS error | max abs error | status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 896 | 896 | 1 | 5.0649799e-08 | 2.38418579e-07 | pass |
| 896 | 896 | 8 | 5.13104865e-08 | 2.38418579e-07 | pass |
| 896 | 896 | 32 | 5.11629459e-08 | 2.38418579e-07 | pass |
| 128 | 896 | 1 | 4.0823064e-08 | 1.1920929e-07 | pass |
| 4864 | 896 | 1 | 5.11720977e-08 | 2.38418579e-07 | pass |
| 896 | 4864 | 1 | 5.49851045e-08 | 4.76837158e-07 | pass |

This proves the generated matrix kernels for these shapes and batches. It does not,
by itself, prove end-to-end model quality or throughput.
