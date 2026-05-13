# Row-Scale Qtype Productization Gate, 2026-05-13

This gate checks whether the row-scale packed ternary path has moved from a local `I2_S`-overloading prototype to a compatibility-safe deployable qtype.

Scenario: `I2_SR candidate patch applied`.

Note: Generated after applying patches/llama-i2sr-row-scale-qtype.patch to the clean active source tree; the patch was reversed after this audit so the repository remains patch-distribution based.

Overall status: `pass`.

## Gates

| gate | status | evidence | blocker |
| --- | --- | --- | --- |
| row-scale semantics are physically feasible | pass | prototype/TQ2_0 PPL ratio=1.001566 |  |
| default tensor-scale I2_S failure is proven | pass | default/TQ2_0 PPL ratio=30836.21 |  |
| stable GGML row-scale qtype is defined | pass | 3rdparty/llama.cpp/ggml/include/ggml.h |  |
| stable llama file type is defined | pass | 3rdparty/llama.cpp/include/llama.h |  |
| runtime routes stable qtype without changing I2_S | pass | ggml.c/llama.cpp source scan plus row-scale patch scan |  |
| direct writer emits stable row-scale qtype | pass | benchmarks/convert_static_ternary_to_i2s_gguf.py |  |
| stable qtype benchmark evidence exists | pass | benchmarks/results/evidence_manifest_2026-05-13.json |  |
| stable qtype benchmark preserves quality | pass | benchmark_results/i2sr-row-scale-qwen15b-x86act-suite-2026-05-13/summary.json |  |
| direct I2_SR packing matches known-good x86 layout | pass | benchmark_results/i2s-packing-layout-verify-2026-05-13/summary.json |  |

## Observations

- Prototype row-scale `I2_S` / `TQ2_0` PPL ratio: `1.001566`.
- Default row-scale `I2_S` / `TQ2_0` PPL ratio: `30836.21`.
- Current row-scale patch overloads existing `I2_S`: `True`.
- Stable GGML qtype present: `True`.
- Stable llama file type present: `True`.
- Direct writer emits stable row-scale qtype: `True`.
- Stable qtype benchmark present in manifest: `True`.
- Stable qtype benchmark quality acceptable: `True`.
- Stable qtype benchmark max finite PPL: `38.8477`.
- Direct `I2_SR` packing byte-layout verification passed: `True` (5/5 tensors).

## Interpretation

The productization claim is positive for this audited source state: the stable row-scale qtype is defined, routed, benchmarked, and byte-layout checked. This does not by itself prove the patch is merged upstream; it proves the runtime contract represented by this source state.
