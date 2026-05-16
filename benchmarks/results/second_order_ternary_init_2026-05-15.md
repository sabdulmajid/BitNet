# Second-Order Ternary Initialization Audit, 2026-05-15

Diagonal-Hessian weighted ternary least-squares reduces synthetic output reconstruction error versus absmean row-scale initialization, so it is a credible next training initializer. The current training hook exposes the unweighted least-squares fixed-point initializer; activation-calibrated diagonal Hessian collection remains a separate implementation step. This is not a GLUE or language-model quality claim until a full BitNet-SFT/BitDistill run uses it.

## Setup

| field | value |
| --- | --- |
| rows | 512 |
| cols | 512 |
| calibration samples | 2048 |
| trials per activation profile | 12 |
| coordinate-descent iterations | 8 |
| quality proven | false |
| unweighted LS training integrated | true |
| diagonal-Hessian training integrated | false |

## Reconstruction Results

| activation profile | method | mean rel RMS | std | mean zero fraction |
| --- | --- | --- | --- | --- |
| isotropic | row_absmean_retrofit | 0.512512 | 0.000753 | 0.309852 |
| isotropic | row_diag_hessian_ls | 0.434913 | 0.000741 | 0.457930 |
| isotropic | tensor_absmean_paper | 0.513070 | 0.000738 | 0.310128 |
| isotropic | tensor_diag_hessian_ls | 0.435859 | 0.000704 | 0.459408 |
| lognormal_diag | row_absmean_retrofit | 0.512531 | 0.003629 | 0.309607 |
| lognormal_diag | row_diag_hessian_ls | 0.417728 | 0.013596 | 0.450852 |
| lognormal_diag | tensor_absmean_paper | 0.513203 | 0.003815 | 0.309928 |
| lognormal_diag | tensor_diag_hessian_ls | 0.435891 | 0.003218 | 0.459536 |

## Paired Candidate Delta

Negative values mean the diagonal-Hessian row-scale initializer had lower reconstruction error than row absmean.

| activation profile | mean delta | std | wins | trials |
| --- | --- | --- | --- | --- |
| isotropic | -0.077599 | 0.000625 | 12 | 12 |
| lognormal_diag | -0.094803 | 0.014438 | 12 | 12 |

## Source Integration

| check | status |
| --- | --- |
| least-squares initializer helper exists | pass |
| BitDistill CLI exposes opt-in init mode | pass |
| trained checkpoint loads are not reinitialized | pass |

## Math

For diagonal activation covariance `H=diag(h)`, each output row minimizes `sum_j h_j (w_j - s t_j)^2` with `t_j in {-1,0,+1}`. For fixed `s`, the optimal code is nonzero when `|w_j| > s/2`. For fixed codes, the optimal row scale is `s = sum_j h_j |w_j| 1(|w_j|>s/2) / sum_j h_j 1(|w_j|>s/2)`. The script iterates these two closed-form steps.

## Next Gate

Run MNLI BitNet-SFT with --ternary-init-mode ls against the absmean baseline, then add calibration activation collection for the diagonal-Hessian variant and repeat the paired full-validation comparison.
