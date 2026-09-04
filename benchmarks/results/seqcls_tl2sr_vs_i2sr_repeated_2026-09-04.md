# Repeated Native MNLI CPU Throughput

Generated: `2026-09-04T12:52:43.810932+00:00`. Status: **valid**.

Protocol: `5` interleaved repetitions over the first `128` MNLI validation examples, `12` threads pinned to `0-11`.

| artifact | mean tok/s | mean 95% CI | range | speed / i2_sr | ratio 95% CI | predictions stable |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| i2_sr | 248.516 | [247.361, 249.671] | 247.230-249.300 | 1.000 | [1.000, 1.000] | True |
| tl2_sr | 211.988 | [211.517, 212.459] | 211.340-212.270 | 0.853 | [0.848, 0.858] | True |

## Interpretation

All prediction and token-count contracts are stable across repetitions. Throughput comparisons may be reported with paired run-level intervals.

## Claim Boundary

- Intervals use a two-sided Student-t interval over 5 execution repetitions; they quantify run variability, not model-quality uncertainty.
- Ratios are paired by repetition and summarized on the log scale.
- Trained-student speed comparisons are valid deployed-artifact comparisons, not isolated kernel microbenchmarks.
- Results apply to this CPU, affinity, executable, shared libraries, prompt set, and sequence-isolated classifier path.
