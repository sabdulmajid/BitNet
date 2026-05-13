# I2_S Row-Scale Format Compatibility Audit, 2026-05-13

This audit separates two claims that must not be conflated: row-scale packed ternary execution is physically possible, but the current prototype is not a stable product format because it reuses the existing `I2_S` type while changing the tensor payload layout.

## Measured Evidence

| artifact | fixed PPL | ratio vs TQ2_0 | file MiB | interpretation |
| --- | --- | --- | --- | --- |
| row-scale TQ2_0 reference | 38.8224 | 1.0000 |  |  |
| default row-scale I2_S | 1197135.5848 | 30836.21 | 1208.9 | fails row-scale scales |
| patched row-scale I2_S prototype | 38.8832 | 1.0016 | 1211.3 | preserves row-scale quality |

## Code-Level Format Evidence

| check | value |
| --- | --- |
| current source stores one I2_S scale | True |
| patch changes I2_S nbytes | True |
| patch indexes per-row scales | True |
| patch defines a new row-scale qtype | False |
| patch overloads existing I2_S type | True |

## Verdict

- Row-scale packed `I2_S` is physically possible: `True`.
- The default `I2_S` layout fails row-scale checkpoints: `True`.
- The current patch is product-format safe: `False`.
- A compatibility-safe new GGUF quantization type or explicit versioned layout is required: `True`.
- Direct `ternary_state_dict.pt` GGUF writing remains required: `True`.

## Production Gate

Do not market the row-scale `I2_S` result as default/upstream `I2_S`. The benchmark proves the missing scale semantics and CPU kernel are feasible. Productization requires a new row-scale-aware GGUF type, writer, reader, `ggml_nbytes` accounting, matmul/get-rows kernels, backward-compatibility tests for tensor-scale `I2_S`, and direct export from `ternary_state_dict.pt` without materializing dense F16 weights.
