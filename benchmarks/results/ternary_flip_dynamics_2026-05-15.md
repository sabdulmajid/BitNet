# Ternary Flip Dynamics Audit, 2026-05-15

Status: **pass**.

Snapshot root: `checkpoints/bitdistill-glue-longwarmup-row/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-row-20k`.

This is offline telemetry over saved Stage-2 checkpoints. It does not replace live per-step gradient or activation telemetry, but it measures whether ternary codes are still changing during continued pretraining.

## Snapshot Code Fractions

| step | tensors | elements | -1 fraction | 0 fraction | +1 fraction |
| --- | --- | --- | --- | --- | --- |
| 1000 | 169 | 493961216 | 0.339050 | 0.320018 | 0.340932 |
| 10000 | 169 | 493961216 | 0.326694 | 0.345881 | 0.327424 |
| 20000 | 169 | 493961216 | 0.321804 | 0.356683 | 0.321512 |

## Pairwise Flip Rates

| pair | shared tensors | elements | flip rate | 0->nonzero | nonzero->0 | -1<->+1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1000->10000 | 169 | 493961216 | 0.165956 | 0.062889 | 0.088752 | 0.014315 |
| 10000->20000 | 169 | 493961216 | 0.064547 | 0.026814 | 0.037616 | 0.000117 |

## Family Flip Rates

| pair | family | elements | flip rate | 0->nonzero | nonzero->0 | -1<->+1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1000->10000 | down_proj | 104595456 | 0.067529 | 0.033065 | 0.034453 | 0.000011 |
| 1000->10000 | gate_proj | 104595456 | 0.067999 | 0.033342 | 0.034655 | 0.000001 |
| 1000->10000 | k_proj | 2752512 | 0.086731 | 0.043343 | 0.043259 | 0.000129 |
| 1000->10000 | lm_head | 136134656 | 0.421141 | 0.139217 | 0.230066 | 0.051858 |
| 1000->10000 | o_proj | 19267584 | 0.077583 | 0.037896 | 0.039662 | 0.000025 |
| 1000->10000 | q_proj | 19267584 | 0.103194 | 0.052120 | 0.050595 | 0.000479 |
| 1000->10000 | up_proj | 104595456 | 0.062574 | 0.030745 | 0.031829 | 0.000000 |
| 1000->10000 | v_proj | 2752512 | 0.073260 | 0.035253 | 0.037932 | 0.000074 |
| 10000->20000 | down_proj | 104595456 | 0.026958 | 0.013275 | 0.013684 | 0.000000 |
| 10000->20000 | gate_proj | 104595456 | 0.026031 | 0.012836 | 0.013194 | 0.000000 |
| 10000->20000 | k_proj | 2752512 | 0.025505 | 0.012813 | 0.012692 | 0.000000 |
| 10000->20000 | lm_head | 136134656 | 0.165069 | 0.063139 | 0.101504 | 0.000426 |
| 10000->20000 | o_proj | 19267584 | 0.029439 | 0.014507 | 0.014932 | 0.000000 |
| 10000->20000 | q_proj | 19267584 | 0.032545 | 0.016440 | 0.016105 | 0.000000 |
| 10000->20000 | up_proj | 104595456 | 0.024206 | 0.011954 | 0.012252 | 0.000000 |
| 10000->20000 | v_proj | 2752512 | 0.026567 | 0.013237 | 0.013330 | 0.000000 |

## Interpretation

A nonzero flip rate means the Stage-2 student is not merely storing a fixed ternary projection; the discrete codes continue to move under the warm-up objective. This supports the framing that ternary retrofit is a training-dynamics problem, not only a packing problem.
