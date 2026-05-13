# Row-Scale Qtype Productization Gate, 2026-05-13

This gate checks whether the row-scale packed ternary path has moved from a local `I2_S`-overloading prototype to a compatibility-safe deployable qtype.

Overall status: `fail`.

## Gates

| gate | status | evidence | blocker |
| --- | --- | --- | --- |
| row-scale semantics are physically feasible | pass | prototype/TQ2_0 PPL ratio=1.001566 |  |
| default tensor-scale I2_S failure is proven | pass | default/TQ2_0 PPL ratio=30836.21 |  |
| stable GGML row-scale qtype is defined | fail | 3rdparty/llama.cpp/ggml/include/ggml.h | No separate GGML_TYPE_I2_SR/I2_RS enum is present; current patch overloads GGML_TYPE_I2_S. |
| stable llama file type is defined | fail | 3rdparty/llama.cpp/include/llama.h | No separate LLAMA_FTYPE_MOSTLY_I2_SR/I2_RS value is present. |
| runtime routes stable qtype without changing I2_S | fail | ggml.c/llama.cpp source scan plus row-scale patch scan | Runtime evidence still indicates an overloaded I2_S patch rather than a separate row-scale qtype path. |
| direct writer emits stable row-scale qtype | pass | benchmarks/convert_static_ternary_to_i2s_gguf.py |  |
| stable qtype benchmark evidence exists | fail | benchmarks/results/evidence_manifest_2026-05-13.json | Evidence manifest has no stable I2_SR/I2_RS row-scale benchmark suite. |

## Observations

- Prototype row-scale `I2_S` / `TQ2_0` PPL ratio: `1.001566`.
- Default row-scale `I2_S` / `TQ2_0` PPL ratio: `30836.21`.
- Current row-scale patch overloads existing `I2_S`: `True`.
- Stable GGML qtype present: `False`.
- Stable llama file type present: `False`.
- Direct writer emits stable row-scale qtype: `True`.
- Stable qtype benchmark present in manifest: `False`.

## Interpretation

The feasibility claim is positive: the patched prototype preserves row-scale quality. The productization claim is negative: the source tree does not yet define a separate row-scale qtype, the direct writer now has an `I2_SR` emission mode, and the manifest has no stable-qtype benchmark. This keeps row-scale packed deployment in research/prototype status.
