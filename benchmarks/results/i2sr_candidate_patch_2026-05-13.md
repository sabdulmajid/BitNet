# I2_SR Candidate Patch Note, 2026-05-13

This note records the next row-scale packed-runtime step after the
`I2_S`-overloading prototype. It is not a product-complete benchmark result.

## What Changed

- Added candidate runtime patch: `patches/llama-i2sr-row-scale-qtype.patch`.
- Extended `benchmarks/convert_static_ternary_to_i2s_gguf.py` with
  `--row-scale-qtype i2_sr`.
- Kept default behavior unchanged:
  - scalar checkpoints still emit existing tensor-scale `I2_S`,
  - row-scale checkpoints are still rejected unless explicitly allowed,
  - `--row-scale-prototype` still emits the older overloaded `I2_S` prototype.

## Patch Contract

The candidate patch defines a separate row-scale ternary type instead of
changing the existing `I2_S` binary contract:

| item | candidate value |
| --- | ---: |
| `GGML_TYPE_I2_SR` | `40` |
| `LLAMA_FTYPE_MOSTLY_I2_SR` | `41` |
| packed codes | same BitNet 1x4 row-interleaved 2-bit codes |
| scales | one FP32 scale per output row |
| scalar `I2_S` compatibility | preserved |

## Validation Performed

The patch was applied temporarily, built, converted back into a patch artifact,
and then reversed so the vendored llama.cpp tree remained clean.

| check | result |
| --- | --- |
| `git apply --check patches/llama-i2sr-row-scale-qtype.patch` | pass |
| `cmake --build build-portable-avx2 -j 4` with the patch applied | pass |
| direct writer help/compile check | pass |
| Qwen2.5-0.5B row-scale writer smoke with `--row-scale-qtype i2_sr` | pass |

Writer-smoke facts:

| field | value |
| --- | ---: |
| source checkpoint | `checkpoints/qwen2.5-0.5b-fineweb-edu-row-1000/step-1000` |
| packed row-scale tensors | `168` |
| dense F16 output tensor | `1` |
| copied tensors | `122` |
| output tensors | `291` |
| emitted qtype | `I2_SR` / numeric dtype `40` |
| emitted file type | `MOSTLY_I2_SR` / numeric ftype `41` |
| temporary GGUF size | `641,448,448` bytes |

The temporary GGUF was removed after the smoke test; the small summary is kept
under `benchmark_results/i2sr-writer-smoke-2026-05-13/summary.json`.

## Full 1.5B Follow-Up

After this smoke test, the same patch was applied to the portable AVX2 build
and the strong Qwen2.5-1.5B KL-only row-scale dense-`lm_head` checkpoint was
converted with `--row-scale-qtype i2_sr`.

| field | value |
| --- | ---: |
| packed row-scale tensors | `196` |
| output tensors | `339` |
| GGUF size | `1,270,157,888` bytes |
| fixed-excerpt PPL | `20,074,699.9423` |
| prompt throughput | `212.10` tok/s |
| decode throughput | `19.01` tok/s |

The artifact loads and runs, but quality is catastrophic. The known-good
row-scale packed prototype on the same checkpoint reaches PPL `38.8832`, so
this result points to a remaining semantic/layout mismatch in the direct
`I2_SR` path. The detailed negative result is recorded in
`benchmarks/results/i2sr_qwen15b_candidate_2026-05-13.md`.

## Current Status

This is a real engineering step, but it is not a publishable quality claim.
The productization gate still fails because the stable qtype is a candidate
patch rather than the active vendored runtime, and the first full applied
`I2_SR` artifact quality-fails. Required next work is layout-equivalence
debugging against the known-good patched `llama-quantize` row-scale path.
