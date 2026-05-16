# Ternary Threshold Dynamics Audit, 2026-05-15

Status: **measured_increase**.

Snapshot root: `checkpoints/bitdistill-glue-longwarmup-row/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-row-20k`.

Scale mode: `row`. Primary boundary band: `±0.05` around `|W|/alpha=0.5`.

This is a mechanism audit for saved Stage-2 weights. It supports or rejects the narrow claim that continued pretraining moves latent FP weights near ternary transition boundaries; it does not by itself prove task quality or paper reproduction.

## Snapshot Trend

| step | projection tensors | elements | threshold ±0.05 | zero-region | code 0 | code -1 | code +1 | scale mean | scale std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 168 | 357826560 | 0.060531 | 0.320747 | 0.320747 | 0.339458 | 0.339795 | 0.015763 | 0.004429 |
| 5000 | 168 | 357826560 | 0.065942 | 0.321348 | 0.321348 | 0.339162 | 0.339489 | 0.015811 | 0.004435 |
| 10000 | 168 | 357826560 | 0.071502 | 0.321886 | 0.321886 | 0.338897 | 0.339217 | 0.015860 | 0.004436 |
| 15000 | 168 | 357826560 | 0.074593 | 0.322118 | 0.322118 | 0.338778 | 0.339104 | 0.015882 | 0.004435 |
| 20000 | 168 | 357826560 | 0.075766 | 0.322202 | 0.322202 | 0.338733 | 0.339065 | 0.015888 | 0.004434 |

## Final-Step Family Breakdown

| family | elements | threshold ±0.05 | zero-region | code 0 | scale mean | scale std |
| --- | --- | --- | --- | --- | --- | --- |
| down_proj | 104595456 | 0.078409 | 0.322752 | 0.322752 | 0.014373 | 0.002642 |
| gate_proj | 104595456 | 0.074954 | 0.319163 | 0.319163 | 0.016790 | 0.003633 |
| k_proj | 2752512 | 0.072763 | 0.327698 | 0.327698 | 0.020621 | 0.012748 |
| o_proj | 19267584 | 0.080107 | 0.334224 | 0.334224 | 0.013782 | 0.004184 |
| q_proj | 19267584 | 0.066170 | 0.328196 | 0.328196 | 0.018022 | 0.010669 |
| up_proj | 104595456 | 0.074514 | 0.320615 | 0.320615 | 0.015153 | 0.001970 |
| v_proj | 2752512 | 0.093492 | 0.345508 | 0.345508 | 0.015255 | 0.003665 |

## Interpretation

The primary threshold-band fraction changed by 0.015235 from step 1000 to 20000. Monotonic non-decreasing: true. This supports the mechanism that continued pretraining keeps more latent weights close to ternary transition boundaries, at least for this saved row-scale Stage-2 run.
