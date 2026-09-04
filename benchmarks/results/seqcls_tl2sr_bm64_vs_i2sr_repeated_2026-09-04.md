# Repeated Native MNLI CPU Throughput

Generated: `2026-09-04T12:47:27.040592+00:00`. Status: **valid**.

Protocol: `5` interleaved repetitions over the first `128` MNLI validation examples, `12` threads pinned to `0-11`.

| artifact | mean tok/s | mean 95% CI | range | speed / i2_sr | ratio 95% CI | predictions stable |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| i2_sr | 242.970 | [233.811, 252.129] | 231.110-249.340 | 1.000 | [1.000, 1.000] | True |
| tl2_sr_bm64 | 210.406 | [198.068, 222.744] | 197.980-218.030 | 0.866 | [0.835, 0.897] | True |

## Interpretation

All prediction and token-count contracts are stable across repetitions. Throughput comparisons may be reported with paired run-level intervals.

## Claim Boundary

- Intervals use a two-sided Student-t interval over 5 execution repetitions; they quantify run variability, not model-quality uncertainty.
- Ratios are paired by repetition and summarized on the log scale.
- Trained-student speed comparisons are valid deployed-artifact comparisons, not isolated kernel microbenchmarks.
- Results apply to this CPU, affinity, executable, shared libraries, prompt set, and sequence-isolated classifier path.
