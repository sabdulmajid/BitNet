# Repeated Native MNLI CPU Throughput

Generated: `2026-09-04T09:58:10.443412+00:00`. Status: **valid**.

Protocol: `4` interleaved repetitions over the first `128` MNLI validation examples, `12` threads pinned to `0-11`.

| artifact | mean tok/s | mean 95% CI | range | speed / FP16 | ratio 95% CI | predictions stable |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fp16_teacher | 440.733 | [439.165, 442.300] | 439.450-441.610 | 1.000 | [1.000, 1.000] | True |
| q4_0_teacher | 389.748 | [387.730, 391.765] | 388.400-390.960 | 0.884 | [0.878, 0.890] | True |
| i2_sr_student | 286.300 | [285.035, 287.565] | 285.600-287.400 | 0.650 | [0.646, 0.653] | True |
| i2_sr_q8_embedding_student | 266.500 | [265.939, 267.061] | 266.010-266.840 | 0.605 | [0.603, 0.607] | True |

## Interpretation

All prediction and token-count contracts are stable across repetitions. Throughput comparisons may be reported with paired run-level intervals.

## Claim Boundary

- Intervals use a two-sided Student-t interval over 4 execution repetitions; they quantify run variability, not model-quality uncertainty.
- Ratios are paired by repetition and summarized on the log scale.
- The I2_SR artifacts are trained students; speed comparisons are valid deployed-artifact comparisons, not isolated kernel microbenchmarks.
- Results apply to this CPU, affinity, executable, shared libraries, prompt set, and sequence-isolated classifier path.
