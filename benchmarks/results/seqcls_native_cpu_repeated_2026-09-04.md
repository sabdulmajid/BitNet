# Repeated Native MNLI CPU Throughput

Generated: `2026-09-04T09:43:00.946488+00:00`. Status: **valid**.

Protocol: `4` interleaved repetitions over the first `128` MNLI validation examples, `12` threads pinned to `0-11`.

| artifact | mean tok/s | mean 95% CI | range | speed / FP16 | ratio 95% CI | predictions stable |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fp16_teacher | 275.385 | [252.117, 298.653] | 260.880-294.260 | 1.000 | [1.000, 1.000] | True |
| q4_0_teacher | 201.035 | [183.174, 218.896] | 185.250-211.740 | 0.730 | [0.666, 0.799] | True |
| i2_sr_student | 175.285 | [171.409, 179.161] | 173.460-178.720 | 0.637 | [0.575, 0.706] | True |
| i2_sr_q8_embedding_student | 145.642 | [126.959, 164.326] | 128.990-153.950 | 0.528 | [0.475, 0.588] | True |

## Interpretation

All prediction and token-count contracts are stable across repetitions. Throughput comparisons may be reported with paired run-level intervals.

## Claim Boundary

- Intervals use a two-sided Student-t interval over 4 execution repetitions; they quantify run variability, not model-quality uncertainty.
- Ratios are paired by repetition and summarized on the log scale.
- The I2_SR artifacts are trained students; speed comparisons are valid deployed-artifact comparisons, not isolated kernel microbenchmarks.
- Results apply to this CPU, affinity, executable, shared libraries, prompt set, and sequence-isolated classifier path.
