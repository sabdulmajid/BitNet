# Qwen2.5-0.5B TL2 Probe, 2026-05-05

## What Was Tested

This probe tested whether the BitNet TL2 path can be made Qwen-aware for a
dense Qwen checkpoint. It does not test Kimi/MoE, and it does not validate the
strong Qwen2.5-1.5B row-scale checkpoint.

Inputs:

- Checkpoint:
  `checkpoints/qwen2.5-0.5b-fineweb-edu-klonly-notiehead-1000/step-1000`
- Converter:
  `utils/convert-hf-to-gguf-bitnet.py --outtype tl2`
- Custom TL2 shapes:
  `128x896`, `896x896`, `896x4864`, `4864x896`
- Generated artifact:
  `benchmark_results/tl2_probe/qwen05b-bitnet-tl2.gguf`

The generated GGUF is `599.5 MiB`; llama.cpp reports `168` TL2 tensors,
`122` F32 tensors, `593.78 MiB` model size, and `10.08 BPW`.
The mechanical evidence audit is
`benchmark_results/evidence_audit/qwen05b_tl2_probe.md`.

## Engineering Result

Qwen2 TL2 conversion is now technically possible for this dense checkpoint only
after model-specific code generation:

```bash
python utils/codegen_tl2.py \
  --shape 128,896 --shape 896,896 --shape 896,4864 --shape 4864,896 \
  --BM 128,224,224,256 \
  --BK 192,192,192,192 \
  --bm 32,32,32,32
```

The converter also now accepts `--kernel-config` or `BITNET_KERNEL_CONFIG`, so
it no longer depends only on a hard-coded `include/kernel_config.ini` path.

## Runtime Result

Generic AVX2 build, `build-portable-avx2/bin`:

| model | smoke rc | bench rc | PPL rc | prefill tok/s | decode tok/s | PPL |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| qwen05b_qat_tl2 | -11 | 0 | -11 | 265.78 | 21.63 | n/a |

The generic build can load and benchmark the TL2 artifact, but standard smoke
generation and perplexity segfault. A manual `--no-warmup` smoke run avoids the
crash but produces repeated punctuation, so this is not a usable deployment
path.

Qwen-specific TL2/AVX-512 build, `build-qwen05b-tl2/bin`:

| model | format | file MiB | prefill tok/s | decode tok/s | PPL | smoke |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| qwen05b_fp_f16 | F16 | 948.1 | 329.30 | 16.40 | 18.0984 | Paris... |
| qwen05b_fp_q4_k_m | Q4_K_M | 379.4 | 213.60 | 35.61 | 18.8239 | Paris... |
| qwen05b_qat_i2_s | I2_S | 489.5 | 533.90 | 49.38 | NaN | !!!!!!!! |
| qwen05b_qat_tl2 | TL2 | 599.5 | 229.52 | 22.95 | NaN | nonsensical mixed text |

## Verdict

This is a partial engineering success and a quality failure.

The repo can be extended to emit a dense Qwen TL2 GGUF if exact TL2 kernel
shapes are generated first and the runtime is rebuilt with matching LUT code.
That is not a generic arbitrary-model path: it is model-shape-specific codegen
plus a model-specific binary.

The measured Qwen2.5-0.5B ternary checkpoint is not useful. Both I2_S and TL2
return NaN perplexity, and generated text is degenerate. This result supports
the broader thesis that execution kernels are not the main blocker; the blocker
is producing a ternary student whose weights and scales preserve model quality.

## Next Step

The only TL2 result worth pursuing next is a strong checkpoint:

1. Generate TL2 kernels for the Qwen2.5-1.5B row-scale dense-head shapes.
2. Decide how TL2 should represent row scales; the current TL1/TL2 converter
   uses one tensor scale from `max(abs(W))`, which will not preserve row-scale
   QAT behavior. The scale-semantics audit at
   `benchmarks/results/tl2_scale_semantics_2026-05-05.md` measures the induced
   row-scale 1.5B error at `1.904230` relative Frobenius/output RMS.
3. Convert and benchmark only after the scale semantics are fixed; otherwise a
   TL2 run on the strong row-scale checkpoint would likely reproduce the same
   row-scale-loss failure seen in default I2_S.
