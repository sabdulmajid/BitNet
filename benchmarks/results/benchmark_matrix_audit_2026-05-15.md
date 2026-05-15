# Benchmark Matrix Audit, 2026-05-15

This audit verifies that the public side-by-side comparison is backed by complete benchmark artifacts rather than one-off samples.

Overall status: PASS.

Complete quality benchmarks: `12`.

## Quality Matrix

| benchmark | metric | status | model families covered |
| --- | --- | --- | --- |
| WikiText perplexity | perplexity | pass | 6 |
| FineWeb heldout perplexity | perplexity | pass | 6 |
| arc_challenge | acc_norm | pass | 6 |
| arc_easy | acc_norm | pass | 6 |
| hellaswag | acc_norm | pass | 6 |
| piqa | acc_norm | pass | 6 |
| winogrande | acc | pass | 6 |
| boolq | acc | pass | 6 |
| copa | acc | pass | 6 |
| openbookqa | acc_norm | pass | 6 |
| sciq | acc_norm | pass | 6 |
| truthfulqa_mc1 | acc | pass | 6 |

## Runtime Matrix

| field | value |
| --- | --- |
| finite Xeon rows | 5 |
| headline labels | FP F16, FP Q4_K_M, FP Q8_0, row I2_SR, row TQ2_0 |
| RSS contexts | 512, 2048, 8192, 32768 |
| Q4 vs I2_SR ratios | {"decode_speedup": 1.19061682213809, "file_ratio": 1.2881333039180543, "ppl_ratio": 3.0323232796303237, "prefill_speedup": 2.298817760610607, "rss512_ratio": 1.2685742155747033} |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| At least ten distinct side-by-side quality benchmarks are complete | pass | complete=12/12; benchmarks=['WikiText perplexity', 'FineWeb heldout perplexity', 'arc_challenge', 'arc_easy', 'hellaswag', 'piqa', 'winogrande', 'boolq', 'copa', 'openbookqa', 'sciq', 'truthfulqa_mc1'] |  |
| Both perplexity benchmarks cover all six model families | pass | complete=['WikiText perplexity', 'FineWeb heldout perplexity'] |  |
| All ten lm-eval tasks cover all six model families | pass | complete_tasks=10/10 |  |
| Each lm-eval model family has full logged sample count | pass | sample_counts={'FP': 22382, 'naive PTQ': 22382, 'QAT hidden-MSE': 22382, 'QAT KL-only': 22382, 'QAT KL-only dense lm_head': 22382, 'QAT KL-only row dense lm_head': 22382} |  |
| Paired statistical reports exist for the key quality comparisons | pass | present=4/4 |  |
| Xeon CPU matrix has finite PPL, size, RSS, prefill, and decode rows | pass | rows=5; labels=['FP F16', 'FP Q4_K_M', 'FP Q8_0', 'row I2_SR', 'row TQ2_0'] |  |
| I2_SR RSS is measured at four context lengths | pass | contexts=[512, 2048, 8192, 32768] |  |
| TL2 row-scale is explicitly excluded from success claims | pass | ready=False; failed=9; path=benchmark_results/tl2_row_scale_runtime_contract_2026-05-15.json |  |

## Verdict

The side-by-side dense-Qwen comparison has at least ten complete quality benchmarks plus finite Xeon runtime/RSS evidence. The full original objective remains partial only because TL2 row-scale support is explicitly blocked.
