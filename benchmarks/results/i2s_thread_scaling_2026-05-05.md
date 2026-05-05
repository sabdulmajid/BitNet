# Row-Scale I2_S Thread Scaling, 2026-05-05

Hardware: Intel Xeon Silver 4116, portable AVX2 patched row-scale `I2_S`
build, no BLAS.

Artifact:
`models/qwen2.5-1.5b-klonly-row-notie-static-ternary-dense/qwen15b_klonly_row_notie_static_ternary_i2_s_rowscale.gguf`

Command for passing rows:

```bash
llama-bench -p 512 -n 128 -ngl 0 -r 3
```

## Results

| threads | status | prefill tok/s | decode tok/s |
| ---: | --- | ---: | ---: |
| 1 | `llama-bench` segfault | - | - |
| 2 | `llama-bench` segfault | - | - |
| 4 | pass | 83.17 | 19.63 |
| 8 | pass | 154.42 | 19.35 |
| 12 | pass | 211.63 | 18.82 |
| 16 | pass | 197.88 | 19.32 |
| 24 | pass | 247.75 | 17.81 |

Control checks:

- `llama-cli -t 1` on the same row-scale `I2_S` artifact succeeds and produces
  coherent output, so single-thread inference is not categorically broken.
- `llama-bench -t 1` on the FP `Q4_K_M` control succeeds, so the low-thread
  benchmark crash is specific to the patched row-scale `I2_S` path.

## Interpretation

Prefill scales meaningfully with threads: 24 threads is about `2.98x` faster
than 4 threads on the `-p 512` prompt benchmark. Decode does not scale; it stays
near `18-20 tok/s` and drops at 24 threads. Product claims should separate
prompt ingestion from autoregressive decode, because decode latency is the
limiting path for interactive generation on this CPU.

The `llama-bench` segfault at 1 and 2 threads is a prototype stability issue.
It should be fixed before describing the row-scale `I2_S` layout as production
ready, even though `llama-cli -t 1` succeeds.
