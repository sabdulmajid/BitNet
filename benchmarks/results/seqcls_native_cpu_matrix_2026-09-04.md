# Native MNLI CPU Deployment Matrix

Generated: `2026-09-04T09:34:58.689755+00:00`. Status: **valid_sample_matrix**.

## Artifacts

| artifact | function | accuracy | MiB | prompt tok/s | examples/s | peak RSS MiB |
| --- | --- | --- | --- | --- | --- | --- |
| fp16_teacher | fp16_teacher | 0.789062 | 948.109589 | 357.468452 | 9.614434 | 1016.878906 |
| q4_0_teacher | fp16_teacher | 0.675781 | 335.840057 | 230.174991 | 6.227864 | 955.890625 |
| i2_sr_student | qat_student | 0.669922 | 352.617432 | 173.101075 | 4.697959 | 956.980469 |
| i2_sr_q8_embedding_student | qat_student | 0.667969 | 230.903564 | 147.526154 | 4.005960 | 957.042969 |

## Paired Comparisons

| comparison | estimand | accuracy delta | paired 95% CI | wins / losses | prediction agreement | McNemar p | size factor | speed factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q4_0_teacher_vs_fp16_teacher | same-model format effect | -0.113281 | [-0.154296875, -0.07421875] | 28 / 86 | 0.761719 | 4.842e-08 | 2.823099 | 0.643903 |
| i2_sr_student_vs_fp16_teacher | deployed-model effect | -0.119141 | [-0.162109375, -0.076171875] | 37 / 98 | 0.705078 | 1.502e-07 | 2.688777 | 0.484242 |
| i2_sr_q8_embedding_student_vs_fp16_teacher | deployed-model effect | -0.121094 | [-0.166015625, -0.078125] | 38 / 100 | 0.701172 | 1.288e-07 | 4.106085 | 0.412697 |
| i2_sr_q8_embedding_student_vs_i2_sr_student | same-model format effect | -0.001953 | [-0.01171875, 0.0078125] | 3 / 4 | 0.982422 | 1.000000 | 1.527120 | 0.852254 |

## Interpretation

The matrix is contract-valid. Q4_0 versus F16 estimates a format-only effect on the same teacher. I2_SR versus F16 estimates the end-to-end deployed-student tradeoff and must not be interpreted as a pure quantization-format effect. Statistical and systems conclusions are limited to this fixed MNLI sample and hardware.

## Claim Boundary

- The fixed sample is the first N MNLI validation_matched rows; it is not a randomized benchmark sample.
- Accuracy intervals are paired over examples; throughput is a single-run measurement and has no timing confidence interval.
- I2_SR is a separately trained QAT student, so its comparison with the FP16 teacher includes training and format effects.
- Peak RSS includes runtime overhead and shared libraries; GGUF bytes are the cleaner storage measurement.
- General language-model quality, other tasks, other CPUs, and energy use are outside this matrix.

## Validation

No contract violations.
