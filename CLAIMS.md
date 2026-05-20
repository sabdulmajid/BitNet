# Claims

This fork is a research artifact, not a product claim that arbitrary models can
be converted to BitNet.

## Current Position

Extreme ternary quantization is not a file-format conversion problem. It is a
representation-learning problem plus a runtime-contract problem.

## Evidence-Led Claims

| Claim | Status | Evidence |
| --- | --- | --- |
| Blind FP/BF16 to ternary PTQ is viable for tested Qwen2.5-1.5B | Rejected | FP PPL `13.901`; naive PTQ PPL `3,813,121.803`. FP ten-task mean `0.644169`; PTQ mean `0.348671`. |
| QAT/distillation helps | Supported, partial | Best row-scale QAT mean `0.499459`, `+0.150788` over naive PTQ, still `-0.144710` below FP. |
| BitDistill is reproduced | Not yet | 40.96M and 163.84M controlled MNLI rows remain below FP; 327.68M downstream rerun is pending. |
| Row-scale runtime semantics matter | Supported | TL2 one-scale RMS error `1.904230`; exact row-scale RMS error `0.000197`. |
| `I2_SR` is a working row-scale CPU path | Supported with caveat | Runs on Xeon and is faster than FP16 decode, but does not beat Q4_K_M on quality or file size. |
| Native classifier runtime is product-ready | Not yet | Full MNLI native path runs, but accuracy `0.652165` and PyTorch agreement `0.976668` are below product gates. |
| Kimi/MoE support is proven | Not supported | Tiny Qwen2MoE fixtures only. |

## Language To Avoid

- Do not say "universal BitNet converter."
- Do not say "reproduced BitDistill" until the controlled reproduction closes.
- Do not say "`I2_SR` beats Q4" because current evidence says it does not on
  quality or file size.
- Do not say "Kimi support" without trained MoE quality and runtime evidence.

## Source Of Truth

Use the canonical evidence bundle, not hand-copied report snippets:

- `benchmarks/results/canonical_evidence_bundle_2026-05-20.json`
- `benchmarks/results/canonical_evidence_bundle_2026-05-20.md`
