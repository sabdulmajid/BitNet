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
| Configured llama.cpp submodule remote is not writable from this environment; use a fork or writable branch. |

## Remote Write Probe

| field | value |
| --- | --- |
| branch | `i2sr-row-scale-runtime` |
| returncode | `128` |
| writable | `false` |
| permission_denied | `true` |
| stderr | `remote: Permission to Eddie-Wang1120/llama.cpp.git denied to sabdulmajid. fatal: unable to access 'https://github.com/Eddie-Wang1120/llama.cpp.git/': The requested URL returned error: 403` |

## Split Promotion Patches

| path | applies cleanly |
| --- | --- |
| patches/bitnet-i2sr-root-runtime.patch | `true` |
| patches/llama-i2sr-row-scale-qtype.submodule.patch | `true` |

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
| Apply `patches/llama-i2sr-row-scale-qtype.submodule.patch` inside that branch and commit it. |
| Apply `patches/bitnet-i2sr-root-runtime.patch` in this superproject or split it into an equivalent top-level commit. |
| Push the submodule branch, update .gitmodules if the URL changes, then update the superproject submodule pointer. |
| Run benchmarks/run_i2sr_active_patch_gate.py or the equivalent active-source productization gate after the pointer update. |
