# I2_SR Submodule Promotion Audit, 2026-05-13

This audit checks whether the row-scale `I2_SR` runtime is active in the committed source state, not merely available as a patch.

Promotion ready: `true`.

Active runtime support: `true`.

Patch applies cleanly: `false`.

Submodule: `https://github.com/sabdulmajid/llama.cpp.git` branch `i2sr-row-scale-runtime` at `106eac0c8`.

Remote branches containing HEAD: `origin/i2sr-row-scale-runtime`.

Prepared local handoff: `true` commit `cb38942486a05e3a1212e6eba1462e178478f459`.

## Active Source Checks

| check | present |
| --- | --- |
| ggml_type_i2_sr | `true` |
| llama_ftype_i2_sr | `true` |
| gguf_py_i2_sr | `true` |
| llama_routes_i2_sr | `true` |
| quantize_cli_i2_sr | `true` |
| root_quantize_i2_sr | `true` |

## Blockers

| blocker |
| --- |
| none |

## Warnings

| warning |
| --- |
| A split promotion patch no longer applies cleanly, but the active source is already promoted and the submodule HEAD is reachable from the configured fork branch. |

## Remote Write Probe

| field | value |
| --- | --- |
| branch | `i2sr-row-scale-runtime` |
| returncode | `0` |
| writable | `true` |
| permission_denied | `false` |
| stderr | `Everything up-to-date` |

## Candidate Fork Probe

| field | value |
| --- | --- |
| url | `https://github.com/sabdulmajid/llama.cpp` |
| returncode | `0` |
| reachable | `true` |
| heads | `i2sr-row-scale-runtime, master` |
| stderr | `` |

## Split Promotion Patches

| path | applies cleanly | already applied | covered |
| --- | --- | --- | --- |
| patches/bitnet-i2sr-root-runtime.patch | `false` | `true` | `true` |
| patches/llama-i2sr-row-scale-qtype.submodule.patch | `false` | `false` | `false` |

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
| No external I2_SR promotion input remains; keep the fork branch, submodule pointer, and active-source productization gate in sync. |
