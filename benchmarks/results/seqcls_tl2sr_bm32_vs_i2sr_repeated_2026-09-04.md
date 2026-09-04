# Repeated Native MNLI CPU Throughput

Generated: `2026-09-04T12:57:49.201928+00:00`. Status: **valid**.

Protocol: `5` interleaved repetitions over the first `128` MNLI validation examples, `12` threads pinned to `0-11`.

| artifact | mean tok/s | mean 95% CI | range | speed / i2_sr | ratio 95% CI | predictions stable |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| i2_sr | 248.838 | [248.348, 249.328] | 248.470-249.380 | 1.000 | [1.000, 1.000] | True |
| tl2_sr_bm32 | 228.638 | [228.307, 228.969] | 228.400-228.980 | 0.919 | [0.917, 0.921] | True |

## Interpretation

All prediction and token-count contracts are stable across repetitions. Throughput comparisons may be reported with paired run-level intervals.

## Claim Boundary

- Intervals use a two-sided Student-t interval over 5 execution repetitions; they quantify run variability, not model-quality uncertainty.
- Ratios are paired by repetition and summarized on the log scale.
- Trained-student speed comparisons are valid deployed-artifact comparisons, not isolated kernel microbenchmarks.
- Results apply to this CPU, affinity, executable, shared libraries, prompt set, and sequence-isolated classifier path.
