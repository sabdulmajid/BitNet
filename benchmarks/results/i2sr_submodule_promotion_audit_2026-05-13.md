# I2_SR Submodule Promotion Audit, 2026-05-13

This audit checks whether the row-scale `I2_SR` runtime is active in the committed source state, not merely available as a patch.

Promotion ready: `false`.

Active runtime support: `false`.

Patch applies cleanly: `true`.

Submodule: `https://github.com/Eddie-Wang1120/llama.cpp.git` branch `merge-dev` at `1f86f058`.

Remote branches containing HEAD: `origin/merge-dev`.

## Active Source Checks

| check | present |
| --- | --- |
| ggml_type_i2_sr | `false` |
| llama_ftype_i2_sr | `false` |
| gguf_py_i2_sr | `false` |
| llama_routes_i2_sr | `false` |
| root_quantize_i2_sr | `false` |

## Blockers

| blocker |
| --- |
| I2_SR qtype/file-type/runtime support is not present in the active submodule/root runtime files. |
| The I2_SR support still exists as an unapplied patch rather than active committed code. |

## Patch Touches

| path |
| --- |
| src/ggml-bitnet-mad.cpp |
| 3rdparty/llama.cpp/ggml/include/ggml.h |
| 3rdparty/llama.cpp/ggml/src/ggml-quants.c |
| 3rdparty/llama.cpp/ggml/src/ggml-quants.h |
| 3rdparty/llama.cpp/ggml/src/ggml.c |
| 3rdparty/llama.cpp/gguf-py/gguf/constants.py |
| 3rdparty/llama.cpp/include/llama.h |
| 3rdparty/llama.cpp/src/llama.cpp |

## Required Promotion Steps

| step |
| --- |
| Create or choose a writable llama.cpp fork/branch for the I2_SR runtime changes. |
| Apply the submodule portion of patches/llama-i2sr-row-scale-qtype.patch inside that branch and commit it. |
| Apply the root runtime portion in this superproject or split it into an equivalent top-level commit. |
| Push the submodule branch, update .gitmodules if the URL changes, then update the superproject submodule pointer. |
| Run benchmarks/run_i2sr_active_patch_gate.py or the equivalent active-source productization gate after the pointer update. |
