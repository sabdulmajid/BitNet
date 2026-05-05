# TL2 Shape Support Audit, 2026-05-05

## Coverage

| model | kernel config | config exists | eligible tensors | unique shapes | kernel shapes | supported tensors | missing tensors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5-0.5b | active include | no | 168 | 4 | 0 | 0 | 168 |
| qwen2.5-0.5b | BitNet 3B preset | yes | 168 | 4 | 3 | 0 | 168 |
| qwen2.5-0.5b | BitNet large preset | yes | 168 | 4 | 3 | 0 | 168 |
| qwen2.5-0.5b | Llama3 8B preset | yes | 168 | 4 | 4 | 0 | 168 |
| qwen2.5-1.5b | active include | no | 196 | 4 | 0 | 0 | 196 |
| qwen2.5-1.5b | BitNet 3B preset | yes | 196 | 4 | 3 | 0 | 196 |
| qwen2.5-1.5b | BitNet large preset | yes | 196 | 4 | 3 | 56 | 140 |
| qwen2.5-1.5b | Llama3 8B preset | yes | 196 | 4 | 4 | 0 | 196 |

## Build Flags

| CMake cache | BITNET_X86_TL2 |
| --- | --- |
| build/CMakeCache.txt | OFF |
| build-portable-avx2/CMakeCache.txt | OFF |
| build-qwen05b-tl2/CMakeCache.txt | ON |

## Verdict

The checked Qwen checkpoints have dense matrix shapes that are not covered by the active TL2 config or by the bundled BitNet/Llama preset configs. At least one checked build cache has `BITNET_X86_TL2=ON`, but the default checked build caches remain `OFF`; TL2 measurements must name the exact model-specific binary used. Custom TL2 code generation appears possible for the audited dense shapes, but it requires generating model-specific LUT kernels, rebuilding the runtime with `-DBITNET_X86_TL2=ON`, converting with the matching kernel config, and then running the same PPL/throughput/RSS audits.

## qwen2.5-0.5b Shapes

| shape (out x in) | tensor count | example tensor |
| --- | --- | --- |
| 128 x 896 | 48 | model.layers.0.self_attn.k_proj.weight |
| 896 x 896 | 48 | model.layers.0.self_attn.o_proj.weight |
| 896 x 4864 | 24 | model.layers.0.mlp.down_proj.weight |
| 4864 x 896 | 48 | model.layers.0.mlp.gate_proj.weight |

## qwen2.5-0.5b Custom TL2 Codegen

A syntactically valid TL2 codegen parameterization exists for these dense matrix shapes:

```bash
python utils/codegen_tl2.py --shape 128,896 --shape 896,896 --shape 896,4864 --shape 4864,896 --BM 128,224,224,256 --BK 192,192,192,192 --bm 32,32,32,32
```

## qwen2.5-1.5b Shapes

| shape (out x in) | tensor count | example tensor |
| --- | --- | --- |
| 256 x 1536 | 56 | model.layers.0.self_attn.k_proj.weight |
| 1536 x 1536 | 56 | model.layers.0.self_attn.o_proj.weight |
| 1536 x 8960 | 28 | model.layers.0.mlp.down_proj.weight |
| 8960 x 1536 | 56 | model.layers.0.mlp.gate_proj.weight |

## qwen2.5-1.5b Custom TL2 Codegen

A syntactically valid TL2 codegen parameterization exists for these dense matrix shapes:

```bash
python utils/codegen_tl2.py --shape 256,1536 --shape 1536,1536 --shape 1536,8960 --shape 8960,1536 --BM 256,256,256,256 --BK 192,192,192,192 --bm 32,32,32,32
```
