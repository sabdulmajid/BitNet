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
| BitDistill is reproduced | Not yet | Fixed-gamma MNLI rises from `0.720020` at `327.68M` to `0.729903` at `655.36M` (delta vs FP `-0.078248`); the best completed loss-balanced tensor run reaches `0.738462` but remains `-0.069689` below FP (paired 95% CI `[-0.078431, -0.060947]`). |
| Local loss-scale alignment improves task quality | Strong historical control; replication pending | At the matched `163.84M` checkpoint, the gamma `60` run beats gamma `100,000` by `+0.047275`, paired CI `[0.039256, 0.055293]`, exact McNemar `p=9.07e-31`. All available serialized controls match and the step-1 loss fingerprint is exact, but the historical artifacts lack serialized seed/source revision. |
| Scaling the unchanged fixed recipe to 10B presentations is justified | Rejected conditionally | A geometric fit projects `0.734981` at 10B (95% interval `[0.723750, 0.755819]`). Even repeating the latest gain without further decay projects `0.768758` (95% interval `[0.741530, 0.795733]`), below the `0.798151` gate. This is not a general BitDistill limit. |
| Row-scale runtime semantics matter | Supported | TL2 one-scale RMS error `1.904230`; exact row-scale RMS error `0.000197`. |
| `I2_SR` is a working row-scale CPU path | Supported with caveat | Runs on Xeon and is faster than FP16 decode, but does not beat Q4_K_M on quality or file size. |
| Native classifier runtime preserves task quality | Supported for one artifact; not product-ready | On 9,815 MNLI examples, native `I2_SR` accuracy is `0.652165` versus PyTorch `0.653591`: paired delta `-0.001426`, CI `[-0.004193, 0.001341]`, exact McNemar `p=0.348`. Exact prediction agreement remains `0.976668`; the non-inferiority margin was retrospective and the model itself is weak. |
| Mixed `I2_SR` plus Q8 embedding reduces classifier storage | Supported for one artifact | `230.90 MiB`, `4.106x` smaller than FP16 and `1.527x` smaller than base I2_SR. Against base I2_SR on 512 fixed MNLI examples: delta `-0.001953`, paired CI `[-0.011719, 0.007812]`, prediction agreement `0.982422`. |
| Removing I2 output staging improves classifier throughput | Supported for tested workload | Old/new binary A/B gives base I2_SR ratio `1.4619`, CI `[1.3686, 1.5616]`, and mixed I2_SR+Q8 ratio `1.4358`, CI `[1.2857, 1.6035]`, with bit-identical logits and only `ggml.c` differing in the runtime source fingerprint. |
| I2_SR accelerates sequence-isolated classification on Xeon 4116 | Rejected for tested workload | After the runtime optimization, four interleaved pinned runs give I2_SR/FP16 geometric throughput ratio `0.650`, CI `[0.646, 0.653]`; mixed I2_SR+Q8 is `0.605`, CI `[0.603, 0.607]`. This does not reject a causal-decode speedup. |
| Kimi/MoE support is proven | Not supported | Tiny Qwen2MoE fixtures only. |

## Reproduction Gap Update

The current gap report is:

- `benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json`
- `benchmarks/results/bitdistill_reproduction_gap_2026-05-23.md`

The earlier short BitNet-SFT default was undertrained: it measured `0.487621`.
A 10k-step BitNet-SFT budget row reaches MNLI `0.628935`, which is `+0.020935`
above the paper BitNet-SFT anchor for Qwen2.5-0.5B MNLI. The remaining failure
is therefore not only the BitNet-SFT baseline. The completed `655.36M` Stage-2
BitDistill row reaches `0.729903`, still `-0.078248` below the local FP16-SFT
reference. Doubling continuation from `327.68M` to `655.36M` presentations adds
only `+0.009883`; a paired-bootstrap saturation audit rejects scaling that
unchanged fixed-gamma recipe as the next move. The active controlled test uses
adaptive gradient balancing while holding the checkpoint, task, and 10k-step
budget fixed, then replicates across three seeds.

A second, matched three-seed arm now tests the adaptive controller directly
against fixed `gamma=60`. This control was submitted before adaptive
full-validation quality was available and holds source, checkpoint, objective,
schedule, data, and seeds fixed. Until all six runs and the fail-closed paired
audit complete, it supports no quality claim. The pre-registered method gate is
documented in
`benchmarks/results/bitdistill_adaptive_vs_fixed_submission_2026-09-04.json`.

## Language To Avoid

- Do not say "universal BitNet converter."
- Do not say "reproduced BitDistill" until the controlled reproduction closes.
- Do not say "`I2_SR` beats Q4" because current evidence says it does not on
  quality or file size.
- Do not claim a general CPU speedup: the causal decode path improves, while
  the controlled sequence-classification path is slower than FP16.
- Do not say "Kimi support" without trained MoE quality and runtime evidence.

## Source Of Truth

Use the canonical evidence bundle, not hand-copied report snippets:

- `benchmarks/results/canonical_evidence_bundle_2026-05-20.json`
- `benchmarks/results/canonical_evidence_bundle_2026-05-20.md`
