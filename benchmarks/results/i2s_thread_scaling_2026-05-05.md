# Row-Scale I2_S Thread Scaling, 2026-05-05

Hardware: Intel Xeon Silver 4116, portable AVX2 patched row-scale `I2_S`
build, no BLAS. The patch includes heap temporary buffers in the I2_S prompt
GEMM/GEMV path; this removes the earlier `llama-bench` crash at 1 and 2
threads.

Artifact:
`models/qwen2.5-1.5b-klonly-row-notie-static-ternary-dense/qwen15b_klonly_row_notie_static_ternary_i2_s_rowscale.gguf`

Command:

```bash
llama-bench -p 512 -n 128 -ngl 0 -r 3
```

## Results

| threads | status | prefill tok/s | decode tok/s |
| ---: | --- | ---: | ---: |
| 1 | pass | 22.02 | 8.57 |
| 2 | pass | 43.42 | 14.74 |
| 4 | pass | 84.41 | 19.49 |
| 8 | pass | 154.94 | 19.38 |
| 12 | pass | 217.60 | 18.95 |
| 16 | pass | 197.98 | 19.19 |
| 24 | pass | 245.31 | 18.34 |

Control checks:

- Before the heap-buffer fix, `llama-bench -t 1/-t 2` segfaulted at prompt
  length 256 or larger while `llama-cli -t 1` on the same artifact succeeded.
- After the heap-buffer fix, `llama-bench -t 1/-t 2` pass on the same `-p 512
  -n 128` probe.

## Interpretation

Prefill scales meaningfully with threads: 24 threads is about `11.14x` faster
than 1 thread and about `2.91x` faster than 4 threads on the `-p 512` prompt
benchmark. Decode improves from 1 to 4 threads, then stays near `18-20 tok/s`.
Product claims should separate
prompt ingestion from autoregressive decode, because decode latency is the
limiting path for interactive generation on this CPU.

The previous low-thread `llama-bench` segfault was caused by large stack
temporary buffers in the I2_S prompt path. The tracked row-scale patch now uses
heap temporary buffers for that path.
