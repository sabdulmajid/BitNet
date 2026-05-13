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
| packed codes | same active x86 `I2_S` `ACT_PARALLEL` 128-code layout |
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

The first artifact loaded and ran, but quality was catastrophic. The known-good
row-scale packed prototype on the same checkpoint reaches PPL `38.8832`, so
that result pointed to a semantic/layout mismatch in the direct `I2_SR` path.
The detailed negative result is recorded in
`benchmarks/results/i2sr_qwen15b_candidate_2026-05-13.md`.

The mismatch was then traced to the direct writer's row-group-of-four pack
order. This fork's x86 runtime defines `ACT_PARALLEL`, so the active layout is
the chunked 128-code layout emitted by `quantize_i2_s`. After changing the
direct writer to that layout, the fixed `I2_SR` artifact reached PPL `38.8477`,
prompt throughput `211.67` tok/s, and decode throughput `19.07` tok/s. The fix
is recorded in `benchmarks/results/i2sr_x86act_fix_2026-05-13.md`.

## Current Status

This is now a quality-preserving engineering step, but it is still not a
product-ready upstream claim. The productization gate fails because the stable
qtype is a candidate patch rather than the active vendored runtime. Required
next work is promoting the qtype/runtime contract and adding regression tests
for both scalar `I2_S` and row-scale `I2_SR` packing.
