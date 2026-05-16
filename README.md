# BitNet Retrofit Research Fork

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This fork tests whether pretrained FP16/BF16 language models can be adapted to
BitNet-style W1.58A8 ternary CPU inference.

**Current verdict: blind post-training FP/BF16 to ternary conversion is not a
viable general retrofit path for arbitrary pretrained models.** The credible
path is narrower: train or distill task-specific ternary students, preserve the
scale semantics they learned, and report quality, speed, and memory as separate
gates.

This is a research and evaluation fork. It does not claim that arbitrary Qwen,
Kimi, or other open-weight models can be converted to 1.58-bit form with no
quality loss.

## Research Thesis

Extreme ternary quantization is not just a file format. It changes the model
family:

```text
FP model:       W in R^(m x n)
ternary model:  W ~= scale * T, where T in {-1, 0, +1}^(m x n)
```

Blind PTQ asks whether an already-trained FP solution can be projected into the
ternary family. Our tested answer is no. QAT and distillation ask whether the
solution can be moved into that family while preserving task behavior. Our
tested answer is partial recovery, not paper-level FP recovery yet.

The strongest fork-specific systems result is that row-scale ternary students
need a matching row-scale runtime contract. Collapsing learned row scales to one
tensor scale breaks outputs; preserving row scales through `I2_SR` keeps the
packed CPU path faithful to the trained checkpoint.

## Claim Ledger

| Claim | Status | Evidence |
| --- | --- | --- |
| One-click arbitrary FP/BF16 to ternary conversion works | **No** | Qwen2.5-1.5B naive PTQ drops ten-task mean from `0.644169` to `0.348671`; WikiText PPL jumps from `13.901` to `3,813,121.803`. |
| QAT/distillation can recover useful signal | **Partially** | Best dense row-scale QAT reaches ten-task mean `0.499459`, improving over naive PTQ by `+0.150788` paired mean accuracy but remaining `-0.144710` behind FP. |
| BitDistill paper-level GLUE reproduction is complete | **No** | Local Qwen2.5 FP16-SFT MNLI is close to the paper anchor (`0.807641` vs `0.799100`), and a tensor-scale CE-only BitNet-SFT budget sweep can clear the weaker paper BitNet-SFT anchor (`0.628935` vs `0.608000`). That is not BitDistill recovery: controlled BitDistill and Qwen3 paper-alignment rows remain below FP quality. Qwen3 tensor BitDistill recovers strongly over BitNet-SFT on QNLI (`0.861065` vs `0.587040`) and SST2 (`0.871560` vs `0.799312`), but still trails FP by paired deltas `-0.060040` and `-0.058486`. Qwen3 row-scale SST2 reaches `0.877294`, but still trails FP by `-0.052752`. MNLI layer selection matters: layer `-8` beats layer `-1` by paired `+0.028528`, but remains `-0.077738` behind FP. |
| Row-scale semantics matter | **Yes** | TL2 one-scale relative output RMS error is `1.904230`; exact FP16 row scales reduce it to `0.000197`. |
| Packed row-scale CPU inference works for compatible causal artifacts | **Yes, audited path only** | `I2_SR` Qwen2.5-1.5B row-scale run on Xeon Silver 4116: PPL `38.8477`, prompt `211.67 tok/s`, decode `19.07 tok/s`, file `1211.3 MiB`. |
| Packed sequence-classification deployment is solved | **No, native plumbing only** | Native single-artifact `bitnet-qwen` GGUF classifier-head execution matches the sidecar path. A repaired 64-example MNLI CPU sample using direct token IDs reaches saved-PyTorch agreement `0.96875`, accuracy `0.59375`, and RSS `950.64 MiB`, but it is still sample-only and not product-ready. |
| Kimi/MoE retrofit is proven | **No** | Tiny Qwen2MoE fixtures prove converter/runtime plumbing only. No trained MoE quality, Kimi mapping, expert-locality, or CPU product result is proven. |

## Evidence Snapshot

The detailed matrices live in the reports under `benchmarks/results/`. The
top-level evidence is intentionally short:

- **PTQ negative result:** Qwen2.5-1.5B naive ternary PTQ collapses from FP
  ten-task mean `0.644169` to `0.348671`, with WikiText PPL rising from
  `13.901` to `3,813,121.803`.
- **QAT/distillation partial recovery:** the best dense row-scale QAT run
  reaches ten-task mean `0.499459`, a real improvement over naive PTQ but still
  `-0.144710` behind FP.
- **BitDistill status:** Qwen2.5 MNLI FP16-SFT is close to the paper anchor
  (`0.807641` local vs `0.799100` paper), but controlled BitDistill remains
  below FP (`0.691187` at 163.84M Stage-2 tokens; `0.738462` in the gamma-60
  diagnostic). Qwen3 tensor BitDistill recovers from `0.587040` BitNet-SFT to
  `0.861065` on QNLI, and from `0.799312` BitNet-SFT to `0.871560` on SST2.
  Row-scale SST2 reaches `0.877294`. These are real task recoveries, but they
  still trail their FP references by paired deltas `-0.060040`, `-0.058486`,
  and `-0.052752`, respectively.
- **Layer-selection signal:** the Qwen3 MNLI attention-layer sweep confirms
  that single-layer attention distillation is sensitive to layer choice. Layer
  `-8` is best at `0.752012`, paired `+0.028528` over layer `-1`; layer `-4`
  is a smaller lift at `+0.009883`; layer `-2` is slightly lower than `-1`
  with CI crossing zero. The best layer still trails FP `0.829750` by
  `-0.077738`.
- **Row-scale caution:** row-scale is a runtime-contract contribution, not a
  universal accuracy win. On completed Qwen3 paper-gamma rows it is lower than
  tensor-scale on MNLI (`-0.027305` paired delta) and QNLI (`-0.012630`). On
  SST2 it is slightly higher (`+0.005734`), but the CI crosses zero:
  `[-0.011536, 0.023004]`.
- **Initializer caution:** least-squares ternary initializers are negative
  transfer results on Qwen2.5 MNLI under the current BitNet-SFT recipe.
  Unweighted LS reaches `0.361895`; calibrated diag-LS reaches `0.350993`;
  the matched absmean baseline is `0.628935`. The diag-LS paired delta is
  `-0.277942` with CI `[-0.290856, -0.265028]`.
- **CPU runtime:** row-scale `I2_SR` runs on the Xeon Silver 4116 with PPL
  `38.8477`, prompt `211.67 tok/s`, decode `19.07 tok/s`, and file size
  `1211.3 MiB`. This proves a compatible packed path, not a Q4_K_M quality or
  storage win.

## What This Fork Adds

- PTQ math and empirical audits showing why blind FP/BF16 to ternary projection
  collapses tested dense-Qwen checkpoints.
- BitLinear/QAT/BitDistill-style training components for Qwen-style models:
  SubLN insertion, Stage-2 continued pretraining, Stage-3 CE + logits KL +
  Q/K/V attention-relation distillation, attention-layer sweeps, and telemetry.
- Row-scale ternary training/export experiments and paired statistical audits.
- A llama.cpp fork with a stable `I2_SR` row-scale packed runtime path for
  compatible causal-LM artifacts.
- A Qwen-compatible `bitnet-qwen` packed graph prototype for preserving Qwen
  SiLU/SwiGLU semantics and Q/K/V bias slots.
- Native GGUF metadata and loader support for a dense Qwen sequence-classifier
  head, currently proven only as plumbing. The evaluator now supports direct
  token-ID prompts because decoding Qwen sentence-pair token IDs back to text
  is not lossless at some BPE boundaries.
- A native batching audit that prevents overclaiming batched seqcls throughput:
  audited low-margin rows change logits and predictions with sequence position
  inside a multi-prompt embedding batch, so batched numbers are blocked until
  the runtime contract is fixed.
- Explicit product gates that prevent fast but unusable artifacts from being
  reported as successful LLMs.

The llama.cpp submodule points at:

```text
https://github.com/sabdulmajid/llama.cpp
```

with active work on `i2sr-row-scale-runtime`.

## Evidence Labels

| Label | Meaning |
| --- | --- |
| `paper-reproduction` | Same task family, model class, quantization semantics, and recipe target as the BitDistill paper. |
| `paper-inspired` | Uses BitDistill-like ingredients but changes budget, backbone, loss normalization, scale granularity, or task formulation. |
| `retrofit-variant` | Fork-specific extensions such as row-scale ternary and `I2_SR`; these are not standard BitNet claims. |

## Not Yet Proven

- One-click universal FP/BF16-to-ternary conversion.
- Paper-level BitDistill reproduction.
- FP-quality 1.58-bit Qwen from this retrofit recipe.
- Full native packed `Qwen2ForSequenceClassification` product validation:
  full GLUE quality, batching parity, RSS, and throughput from the same GGUF.
- General-LM quality for task-distilled causal prompt-scoring exports.
- Kimi or trained MoE quality, speed, memory, routing locality, or CPU product
  viability.
- TL2 support for row-scale Qwen. Current TL2 uses one scalar scale; generated
  x86 TL2 qgemm multiplies by `Scales[0]`. Use `I2_SR` for row-scale
  checkpoints until a new TL2 row/group-scale contract is implemented.
  A grouped-scale audit also shows that the best available fp16 group-scale
  row (`group2`) still has `0.098692` relative output RMS error, while exact
  fp16 row scales reach `0.000197`; group/tile scales should not close the
  strict TL2 blocker without new quality evidence.

## Current Plan

The next work is deliberately narrow:

1. Finish and postprocess the controlled Qwen2.5-0.5B MNLI Stage-2 token-budget
   curve.
2. Audit BitDistill loss/update balance, especially attention-KD normalization
   and gamma scaling.
3. Keep paper-style tensor-scale BitDistill separate from row-scale
   `retrofit-variant` results.
4. Close the product gap by upgrading the native sequence-classification token-ID
   sample into a faithful full-validation evaluator. The current 64-example
   sample has high agreement with saved PyTorch predictions but still shows
   residual packed-runtime drift, and batching parity currently fails, so full
   product claims remain premature.
5. Keep MoE/Kimi as future work until the dense case is solved.

Detailed current status and next steps are in
[Current Research Plan](benchmarks/results/current_research_plan_2026-05-16.md).

## Repository Map

| Path | Purpose |
| --- | --- |
| `benchmarks/` | benchmark, audit, conversion, and report-generation scripts |
| `benchmarks/results/` | public Markdown reports with parsed results and verdicts |
| `benchmark_results/` | raw JSON summaries and benchmark artifacts |
| `experiments/` | small mathematical audits and viability probes |
| `utils/` | Hugging Face conversion and preprocessing utilities |
| `3rdparty/llama.cpp` | llama.cpp fork with active `I2_SR` support |
| `src/ggml-bitnet-mad.cpp` | BitNet CPU quantization/runtime integration |

## Reproduce The Evidence Snapshot

The current public reports use `BITNET_REPORT_DATE=2026-05-15`.

```bash
export BITNET_REPORT_DATE=2026-05-15

python benchmarks/audit_bitnet_sft_budget_sweep.py
python benchmarks/audit_bitdistill_stage2_curve.py
python benchmarks/audit_bitdistill_controlled_curve.py
python benchmarks/audit_bitdistill_gamma60_diagnostic.py
python benchmarks/audit_bitdistill_root_cause.py
python benchmarks/audit_research_redirect_claims.py
python benchmarks/audit_seqcls_runtime_gap.py
python benchmarks/build_seqcls_runtime_implementation_plan.py
python benchmarks/audit_seqcls_native_i2sr_smoke.py
python benchmarks/audit_seqcls_native_mismatch.py --prompt-input token_ids
python benchmarks/audit_seqcls_native_batching.py
python benchmarks/benchmark_seqcls_native_i2sr_cpu.py --max-samples 64 --prompt-input token_ids \
  --output-json benchmark_results/seqcls_native_i2sr_cpu_mnli_64_token_ids_2026-05-15.json \
  --output-md benchmarks/results/seqcls_native_i2sr_cpu_mnli_64_token_ids_2026-05-15.md
python benchmarks/audit_tl2_negative_result.py
python benchmarks/audit_benchmark_coverage.py
python benchmarks/build_evidence_manifest.py \
  --output-json benchmarks/results/evidence_manifest_2026-05-15.json \
  --output-md benchmarks/results/evidence_manifest_2026-05-15.md
```

Runtime smoke:

```bash
cmake --build build-portable-avx2 --target llama-cli llama-bench llama-perplexity llama-quantize -j 12
./build-portable-avx2/bin/llama-quantize --help | rg "I2_SR|I2_S"
```

## Primary Reports

- [Current Research Plan](benchmarks/results/current_research_plan_2026-05-16.md)
- [Research redirect](benchmarks/results/research_redirect_2026-05-15.md)
- [Research redirect claim gate](benchmarks/results/research_redirect_claims_2026-05-15.md)
- [Qwen side-by-side summary](benchmarks/results/qwen_side_by_side_2026-05-15.md)
- [BitDistill root-cause audit](benchmarks/results/bitdistill_root_cause_audit_2026-05-15.md)
- [BitDistill controlled curve audit](benchmarks/results/bitdistill_controlled_curve_2026-05-15.md)
- [BitNet-SFT budget sweep audit](benchmarks/results/bitnet_sft_budget_sweep_2026-05-15.md)
- [Sequence-classification runtime gap audit](benchmarks/results/seqcls_runtime_gap_2026-05-15.md)
- [Sequence-classification runtime implementation plan](benchmarks/results/seqcls_runtime_implementation_plan_2026-05-15.md)
- [Sequence-classification native I2_SR smoke](benchmarks/results/seqcls_native_i2sr_smoke_2026-05-15.md)
- [Sequence-classification native I2_SR mismatch audit](benchmarks/results/seqcls_native_mismatch_audit_2026-05-15.md)
- [Sequence-classification native I2_SR batching audit](benchmarks/results/seqcls_native_batching_audit_2026-05-15.md)
- [Sequence-classification native I2_SR CPU token-ID sample](benchmarks/results/seqcls_native_i2sr_cpu_mnli_64_token_ids_2026-05-15.md)
- [CPU tradeoff frontier audit](benchmarks/results/cpu_tradeoff_frontier_2026-05-15.md)
- [TL2 group-scale viability audit](benchmarks/results/tl2_group_scale_viability_2026-05-15.md)
- [TL2 row-scale implementation plan](benchmarks/results/tl2_row_scale_implementation_plan_2026-05-15.md)
- [TL2 negative-result audit](benchmarks/results/tl2_negative_result_2026-05-15.md)
- [Product scope gate](benchmarks/results/product_scope_gate_2026-05-15.md)
- [Evidence manifest](benchmarks/results/evidence_manifest_2026-05-15.md)

## References

- [BitNet Distillation](https://arxiv.org/html/2510.13998v1), Microsoft Research.
- [BitNet b1.58 2B4T Technical Report](https://arxiv.org/html/2504.12285v1), Microsoft Research.
- [bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/html/2502.11880v1), Microsoft Research.
- [llama.cpp](https://github.com/ggml-org/llama.cpp), ggml-org.

## Upstream Attribution

This fork is based on Microsoft's BitNet / bitnet.cpp work and uses llama.cpp
as the CPU inference substrate. Upstream BitNet release claims apply to native
BitNet models, not to the arbitrary-retrofit experiments reported here.
