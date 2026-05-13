# Direct Scalar I2_S GGUF Export, 2026-05-13

## Verdict

Direct scalar-scale static-ternary `I2_S` GGUF export is now mechanically
possible in this fork without modifying the vendored `llama.cpp` submodule.
It is **not** a quality-preserving product path for the tested Qwen2.5-0.5B
checkpoint.

This result narrows the engineering question:

- scalar/tensor-scale ternary checkpoints can be packed directly as `I2_S`;
- row-scale checkpoints are intentionally rejected by this writer;
- the current x86 `ACT_PARALLEL` scalar `I2_S` Qwen2.5-0.5B artifact is
  smaller and faster than FP16 on CPU, but quality is still weak: PPL
  `423.4528` versus FP16 `18.0986`;
- the earlier row-group direct scalar artifact is retained as a negative
  control because it produced `NaN` PPL and punctuation spam;
- publishable quality still requires the row-scale/distillation path plus a
  stable row-scale packed format, not scalar direct `I2_S`.

## Implementation

Script:

- `benchmarks/convert_static_ternary_to_i2s_gguf.py`

The writer imports the existing `3rdparty/llama.cpp/convert_hf_to_gguf.py`
model metadata/tokenizer path, overrides tensor preparation, and packs
`*.ternary_weight` tensors directly into the existing tensor-scale `I2_S`
layout:

- four ternary codes per byte, encoded as `{-1,0,1} -> {0,1,2}`;
- the byte order matches the active x86 `ACT_PARALLEL` layout in this fork:
  each flat row-major group of 128 ternary codes is packed into 32 bytes;
- one float32 tensor scale appended after the packed code bytes;
- `output.weight` kept F16 by default, matching `llama-quantize` policy;
- non-ternary tensors copied as F32 for 1D/norm tensors and F16 for dense 2D
  tensors;
- row/non-scalar `weight_scale` tensors rejected.

The vendored Python GGUF constants do not expose `I2_S` or `MOSTLY_I2_S`.
Instead of committing a detached submodule edit, the script uses integer-backed
fallbacks:

- `GGML_TYPE_I2_S = 36`;
- `LLAMA_FTYPE_MOSTLY_I2_S = 40`.

The C++ runtime already understands these values.

## Conversion Evidence

Tiny sanity checkpoint:

- checkpoint: `checkpoints/eval-ternary-tiny`;
- output: `models/eval-ternary-tiny-direct/tiny_static_ternary_direct_i2_s_selfcontained.gguf`;
- `has_native_gguf_python_constants=false`;
- packed tensors: `7`;
- F16 output tensors: `1`;
- copied tensors: `4`;
- output tensors: `12`;
- output size: `25,339,392` bytes;
- `llama-cli` loaded the file as `I2_S - 2 bpw ternary` with `7` `i2_s`
  tensors.

Qwen2.5-0.5B scalar-scale KL-only checkpoint, current x86 `ACT_PARALLEL`
writer:

- checkpoint: `checkpoints/qwen2.5-0.5b-fineweb-edu-klonly-1000/step-1000`;
- output: `models/qwen2.5-0.5b-direct-static-ternary/qwen05b_klonly_direct_i2_s_x86act.gguf`;
- `has_native_gguf_python_constants=false`;
- packed tensors: `168`;
- F16 output tensors: `1`;
- copied tensors: `122`;
- output tensors: `291`;
- packed `I2_S` payload bytes: `89,462,016`;
- output size: `640,231,936` bytes;
- `llama-cli` loaded the file as `I2_S - 2 bpw ternary` with `168` `i2_s`
  tensors and a `604.90 MiB` CPU weight buffer.

Row-scale rejection test:

- checkpoint: `checkpoints/qwen2.5-0.5b-fineweb-edu-row-1000/step-1000`;
- first rejected scale:
  `model.layers.0.self_attn.q_proj.weight_scale` with shape `(896, 1)`;
- outcome: expected failure before writing a misleading scalar `I2_S` file.

The generic `llama-gguf r n` reader still rejects the raw tensor-data layout
with `failed to read tensor data`, while `llama-cli` and `llama-bench` load and
execute it through the model runtime. Treat this as a tooling-compatibility
warning.

## CPU Benchmark

Command:

```bash
python benchmarks/run_gguf_suite.py \
  --models-json benchmarks/gguf_qwen05b_direct_i2s_x86act_manifest.json \
  --out-dir benchmark_results/direct-i2s-qwen05b-klonly-x86act-2026-05-13 \
  --llama-bin-dir build-portable-avx2/bin \
  --perplexity-file benchmark_results/gguf-ppl/wikitext2_test_excerpt.txt \
  --ppl-chunks 16 \
  --ctx-size 512 \
  --threads 12 \
  --prompt-tokens 512 \
  --gen-tokens 64 \
  --smoke-tokens 16 \
  --repetitions 3
```

Result:

| artifact | file MiB | PPL | PPL tok/s | prefill tok/s | decode tok/s | smoke |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-0.5B FP16 | 948.1 | 18.0986 | 239.73 | 367.95 | 16.68 | `Paris. It is the capital of France. It is the capital of France.` |
| Qwen2.5-0.5B scalar direct `I2_S`, x86 ACT pack | 610.6 | 423.4528 | 307.75 | 540.97 | 40.04 | `the most important and important part of the country. The most important part of the` |

The current scalar direct `I2_S` artifact is approximately `35.6%` smaller than
FP16, `1.47x` faster on prefill, and `2.40x` faster on decode in this run. It
is finite after the x86 packing fix, but it is not product quality: PPL is
`23.4x` worse than the FP16 baseline on the same fixed excerpt.

Historical row-group negative control:

| artifact | file MiB | PPL | PPL tok/s | prefill tok/s | decode tok/s | smoke |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-0.5B scalar direct `I2_S`, old row-group pack | 610.6 | NaN | 291.05 | 523.69 | 37.30 | `!!!!!!!!!!!!!!!!` |

That failure showed the same packing bug later isolated in the first `I2_SR`
candidate: the writer emitted row-group-of-four bytes, while this fork's active
x86 runtime expects the `ACT_PARALLEL` 128-code layout.

## Interpretation

This is a useful engineering milestone but still a negative product-quality
result.

The runtime can ingest a directly packed scalar-scale ternary checkpoint, so
the remaining blocker is not simply "GGUF cannot carry ternary tensors." The
blocker is partly mathematical/architectural: scalar ternary constraints are
not enough for this Qwen checkpoint to approach FP quality, and the best quality
ablation we have measured uses row-wise scales plus distillation. Existing
tensor-scale `I2_S` cannot represent those row-wise scales without a new stable
layout or new GGUF quantization type.

The current product-safe path remains:

1. train/distill under the exact ternary forward constraint;
2. keep Qwen's tied output head dense unless an output-head policy is trained;
3. use row-wise scales for quality;
4. export through dense/TQ2_0 for reproducible quality today;
5. promote the row-scale `I2_S` prototype into a stable format before claiming
   row-scale `I2_S` deployment.
