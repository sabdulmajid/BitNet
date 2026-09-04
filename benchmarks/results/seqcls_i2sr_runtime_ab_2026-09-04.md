# I2_SR Runtime A/B Benchmark

Generated: `2026-09-04T10:18:06.770341+00:00`. Status: **valid**.

Protocol: `4` rotated repetitions over the first `128` MNLI validation examples, `12` threads pinned to `0-11`.

| artifact | baseline tok/s | candidate tok/s | candidate / baseline | paired 95% CI | max abs logit delta | predictions identical |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| i2_sr_student | 178.938 | 261.355 | 1.4619 | [1.3686, 1.5616] | 0.000e+00 | True |
| i2_sr_q8_embedding_student | 199.048 | 285.270 | 1.4358 | [1.2857, 1.6035] | 0.000e+00 | True |

## Runtime Revisions

- Baseline BitNet: `a6cf8361422be440d742eeca05b6379a8d7b9caa`
- Baseline llama.cpp: `7fe586546fef1aff17cddabc2ca262d3da4fba15`
- Candidate BitNet: `f56d63c7abd40335be10360b17c9e96410ca0802`
- Candidate llama.cpp: `d0223354b78e96656bcbe46d86b7808e90706df3`
- Fingerprinted source differences: `3rdparty/llama.cpp/ggml/src/ggml.c`

## Interpretation

The candidate preserves predictions and produces a statistically positive local throughput ratio for both I2_SR artifacts under this protocol.

## Claim Boundary

- The estimand is a runtime-implementation effect: model bytes, prompts, thread count, and affinity are identical.
- Ratios are paired by repetition and summarized on the log scale with a Student-t interval.
- Four repetitions characterize local run variability; they do not establish portability to other CPUs or workloads.
- This benchmark does not change or re-evaluate model quality relative to FP16.
