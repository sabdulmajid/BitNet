# Claim Ledger

Last updated: 2026-09-22

This file defines what may be said publicly about the fork. The adjacent
[current evidence snapshot](benchmarks/results/current_evidence_2026-09-22.md)
is the compact source of truth.

## Supported

| Claim | Evidence | Boundary |
| --- | --- | --- |
| Naive tensor-absmean ternary PTQ fails for the tested Qwen2.5-1.5B setup. | FP/PTQ WikiText PPL `13.901`/`3,813,121.803`; ten-task mean `0.644169`/`0.348671`. | This rejects one quantizer and setup, not optimized PTQ generally. |
| Training under ternary constraints recovers useful signal. | Best row-scale QAT ten-task mean `0.499459`, `+0.150788` over naive PTQ. | It remains `-0.144710` below FP and is not acceptable general-LM recovery. |
| The local BitDistill recipe has not reproduced FP16 MNLI. | FP16 `0.808151`; adaptive mean `0.755340`; fixed-60 mean `0.757039`. | Independent, non-paper-exact Qwen2.5-0.5B experiment. |
| Row-scale semantics must be preserved by the runtime. | One-scale relative RMS error `1.904230`; exact row scales `0.000197`. | Tested projections and formats only. |
| `I2_SR` is a working row-scale packed path. | Causal artifact: `1211.3 MiB`, PPL `38.8477`, prompt `211.67 tok/s`, decode `19.07 tok/s`. | It does not beat Q4_K_M on file size or quality. |
| Native packed classification can preserve a source student's task behavior. | Full MNLI native/PyTorch `0.652165`/`0.653591`; delta `-0.001426`, CI `[-0.004193, 0.001341]`. | The student itself is weak and exact agreement is `0.976668`. |
| Mixed I2/Q8 reduces the tested classifier file. | `230.90 MiB`, `4.106x` smaller than FP16 and `1.527x` smaller than base I2_SR. | Fixed 512-example same-student format comparison for quality. |
| Removing I2 output staging improved the local implementation. | Base/mixed speed ratios `1.4619x`/`1.4358x`, with bit-identical logits. | Local implementation A/B, not a model-quality gain. |
| `TL2_SR` preserves the tested row-scale representation and saves projection bytes. | `18/18` kernel cases; full MNLI delta `+0.001426`, CI `[-0.000917, 0.003872]`; projection bytes `-12.862%`. | Whole-file reduction is only `3.154%`. |

## Rejected

| Claim | Evidence |
| --- | --- |
| The adaptive attention-loss controller improves matched fixed-60 training. | Adaptive minus fixed mean `-0.001698`; seed 95% CI `[-0.010898, 0.007502]`; recommendation `inconclusive`. |
| Scaling the unchanged fixed-gamma recipe alone is the justified next experiment. | MNLI gains contract from `+0.074580` to `+0.028833` to `+0.009883` across measured Stage-2 doublings. |
| Local LS or diagonal-Hessian LS initialization improves downstream quality. | Baseline MNLI `0.628935`; LS `0.361895`; diagonal LS `0.350993`. |
| `TL2_SR` accelerates the tested classifier over `I2_SR`. | BM128 `0.853x`, BM64 `0.866x`, BM32 `0.919x`; every paired interval is below `1.0`. |
| Packed ternary accelerates sequence-isolated classification on Xeon 4116. | I2_SR/FP16 `0.650x`, CI `[0.646, 0.653]`; mixed I2/Q8 `0.605x`, CI `[0.603, 0.607]`. |
| A8 activation quantization is the main remaining projection bottleneck. | A8 is `5.49%`, CI `[5.29%, 5.69%]`; I2 arithmetic is `94.51%`; ideal free-A8 bound `1.0581x`. |

## Not Proven

- A universal FP16/BF16-to-ternary converter.
- Paper-exact BitDistill reproduction.
- General-language quality preservation by the task-trained causal exports.
- Superiority over Q4_K_M at a common quality, storage, and speed point.
- Independent reproduction of PT2-LLM, CAT-Q, ScaleQ-1.58, or TWLA.
- Kimi support, real MoE quality, expert paging, or routed CPU speed.
- Energy reduction or generalization beyond the measured hardware.

## Required Language

- Say **"naive absmean PTQ fails in the tested setup"**, not "ternary PTQ is
  impossible."
- Say **"independent BitDistill-style implementation"**, not "BitDistill
  reproduced."
- Say **"row-scale retrofit format"** for `I2_SR`; it is not standard BitNet.
- Report causal decode and sequence classification as different workloads.
- Call ScaleQ-1.58 and BITCOS recent preprints until their status changes.
- Call MoE support plumbing-only until a real checkpoint passes quality and
  runtime gates.

## Evidence Files

- [Current evidence](benchmarks/results/current_evidence_2026-09-22.json)
- [Matched BitDistill audit](benchmarks/results/bitdistill_adaptive_vs_fixed_matched_audit_2026-09-22.json)
- [Matched prediction bundle](benchmarks/results/bitdistill_matched_prediction_bundle_2026-09-22.json)
- [TL2_SR audit](benchmarks/results/tl2sr_evidence_audit_2026-09-04.json)
- [CPU matrix](benchmarks/results/seqcls_native_cpu_matrix_2026-09-04.json)
- [Repeated CPU timings](benchmarks/results/seqcls_native_cpu_repeated_inplace_2026-09-04.json)
