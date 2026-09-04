# BitDistill Adaptive Full-Run Audit

Generated: `2026-09-04T14:06:25.437022+00:00`

Status: **complete**.

Three-seed cross-environment MNLI quality gate; not a paper-exact reproduction.

| seed | status | accuracy | delta vs fixed | paired CI vs fixed | McNemar vs fixed | delta vs gamma60 | paired CI vs gamma60 | delta vs FP16 | paired CI vs FP16 | final gamma | median grad A/CE | median probe A/CE | median global/last probe | max grad A/CE | max A8 clipped | mean ternary flips | blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1234 | complete | 0.755782 | 0.0258788 | [0.0182132, 0.0335443] | 4.33257e-11 | 0.0173204 | [0.0111231, 0.0235178] | -0.0523688 | [-0.0608675, -0.0438702] | 22.8487 | 0.0458014 | 0.956879 | 0.0401069 | 0.430439 | 0 | 0.00906075 | [] |
| 1235 | complete | 0.756903 | 0.0269995 | [0.0192141, 0.0347849] | 1.26525e-11 | 0.0184412 | [0.0115013, 0.0253811] | -0.0512481 | [-0.0596574, -0.0428388] | 19.9842 | 0.0494513 | 0.930571 | 0.0475532 | 0.249511 | 0 | 0.00902864 | [] |
| 1236 | complete | 0.753337 | 0.0234335 | [0.0156308, 0.0312362] | 4.59151e-09 | 0.0148752 | [0.00793482, 0.0218156] | -0.0548141 | [-0.0632973, -0.0463309] | 25.9193 | 0.0572301 | 1.03161 | 0.0694326 | 0.12837 | 0 | 0.00902156 | [] |

## Aggregate

| completed seeds | mean accuracy | sample SD | seed-mean t CI |
| --- | --- | --- | --- |
| 3 | 0.75534 | 0.00182352 | [0.750811, 0.75987] |

## Decisions

| gate | result |
| --- | --- |
| all runs complete | True |
| all seeds improve over fixed gamma with paired CI > 0 | pass |
| three-seed mean within one point of FP16 | fail |

## Telemetry Boundary

The global attention/CE norm and controller probe are not same-support, same-microbatch measurements. Their reported ratio is descriptive only: the global norm covers all trainable parameters on the telemetry microbatch, while the probe is the most recent controller update on selected-layer Q/K/V parameters.
