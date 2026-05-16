# Current Research Plan - 2026-05-16

## Verdict

The original product hypothesis is rejected by the current evidence:

> Blind post-training FP/BF16 to ternary conversion is not a viable general
> retrofit path for arbitrary pretrained models.

The project should now be framed as a CPU-first ternary retrofit evaluation
stack:

> Can a pretrained teacher train a task-specific ternary student, can the
> trained student's scale semantics be preserved in a packed CPU format, and
> can quality/speed/memory tradeoffs be reported honestly enough to decide
> whether the artifact is useful?

This separates the negative algorithmic result from the positive systems
result.

## What Has Been Proven

| Result | Status | Evidence |
| --- | --- | --- |
| Blind PTQ collapse | Proven for tested dense Qwen setup | Qwen2.5-1.5B ten-task mean falls from `0.644169` FP to `0.348671` naive PTQ; WikiText PPL rises from `13.901` to `3,813,121.803`. |
| Training under ternary constraints helps | Proven partially | Best row-scale QAT improves ten-task mean by `+0.150788` over naive PTQ, but remains `-0.144710` behind FP. |
| BitNet-SFT baseline can clear the paper anchor | Proven for one local MNLI row | CE-only Qwen2.5-0.5B BitNet-SFT reaches `0.628935` vs paper BitNet-SFT anchor `0.608000`. |
| BitDistill FP recovery is not reproduced | Proven not yet complete | Controlled BitDistill rows reach `0.616607` and `0.691187`, still far from local FP16-SFT `0.807641`. |
| Row-scale runtime contract matters | Proven by output audit | One-scale TL2 relative output RMS error `1.904230`; exact row scales `0.000197`. |
| TL2 group/tile-scale compromise is enough | Rejected for strict fidelity | Best available fp16 group-scale row is `0.098692` relative output RMS; exact fp16 row scales are `0.000197`. |
| `I2_SR` can preserve row-scale ternary semantics in packed CPU inference | Proven for compatible causal-LM artifacts | Qwen2.5-1.5B `I2_SR` runs on Xeon Silver 4116 with PPL `38.8477`, prompt `211.67 tok/s`, decode `19.07 tok/s`. |
| Native packed sequence-classification product is solved | Not proven | Current classifier path is a Python sidecar over an `I2_SR` backbone, not native GGUF inference. |
| Kimi/MoE product viability is proven | Not proven | Current MoE work is tiny-fixture plumbing only. |

## What Changed In The Research Question

Original question:

```text
Can arbitrary pretrained FP16/BF16 models be post-hoc converted to BitNet-style
W1.58A8 CPU inference?
```

Current answer:

```text
No, not with blind PTQ or lossless conversion.
```

Current research question:

```text
Can task-specific ternary students be trained from pretrained teachers, and can
row-scale CPU formats preserve the semantics that make those students useful?
```

That is a stronger and more defensible direction. It is no longer a universal
converter story. It is a training, evaluation, and runtime-contract story.

## Current Active Runs

At the last local check:

| Job | Partition / node | Purpose | Status |
| ---: | --- | --- | --- |
| `10040` | `dualcard / ece-nebula10` | Qwen3 paper-alignment run | running near `19310/20000` downstream steps |
| `10077` | `midcard / ece-nebula12` | gamma-60 BitDistill diagnostic | running near `4470/10000` downstream steps |
| `10070` | `dualcard` | controlled 327.68M Stage-2 row | pending |
| `10079` | `midcard` | unweighted LS BitNet-SFT initializer benchmark | pending |
| `10080` | `midcard` | calibrated diag-LS BitNet-SFT initializer benchmark | pending |

No quality claim should be made from these rows until their postprocess audits
materialize JSON and Markdown reports.

## Critical Interpretation

### 1. The weak early BitNet-SFT result was not final

The original local BitNet-SFT MNLI result was `0.487621`, far below the paper
anchor `0.608000`. Longer CE-only budget changed that:

```text
best completed CE-only BitNet-SFT: 0.628935
paper BitNet-SFT anchor:          0.608000
```

This means the low early baseline was mostly undertraining/schedule, not proof
that the BitLinear replacement was fundamentally broken.

### 2. The current blocker is BitDistill recovery, not BitNet-SFT viability

The local FP16-SFT baseline is strong:

```text
local FP16-SFT MNLI:  0.807641
paper FP16 anchor:   0.799100
```

The controlled BitDistill rows are improving with Stage-2 budget:

```text
40.96M tokens:   0.616607
163.84M tokens:  0.691187
```

But they remain outside the success gate of recovering to within about `0.5` to
`1.0` accuracy point of FP16-SFT. The remaining question is whether the curve
continues upward with more Stage-2 tokens or saturates because loss/update
balance is misaligned.

### 3. Paper gamma is not portable without matching loss normalization

Completed paper-gamma rows show weighted-attention/CE ratios in the thousands:

```text
40.96M row:   4718.947626
163.84M row:  5945.070866
```

The median equalizing gamma estimated from raw losses is around `58-61`, not
`100000`, under this implementation. That does not prove the paper coefficient
is bad. It proves the local reduction/normalization contract is different
enough that the coefficient cannot be interpreted without telemetry.

### 4. Row-scale is a fork contribution, not a BitDistill reproduction

Row-scale improves several retrofit results and has a clear runtime contract
through `I2_SR`. It should be labeled `retrofit-variant`, not
`paper-reproduction`, because the BitDistill paper describes per-tensor
absmean quantization in its baseline equations.

### 5. The product artifact gap remains

The strongest quality path is sequence classification. The strongest packed
runtime path is causal-LM `I2_SR`. The sidecar prototype narrows the gap, but it
is not a native deployed task model yet.

A credible product needs one artifact that carries both:

```text
quality evidence + CPU runtime evidence
```

## Next Decision Gates

### Gate A: Controlled Stage-2 Curve

Finish the fixed-recipe tensor-scale rows:

| Stage-2 token presentations | Status |
| ---: | --- |
| `40.96M` | complete, MNLI `0.616607` |
| `163.84M` | complete, MNLI `0.691187` |
| `327.68M` | pending |

Decision:

- If accuracy continues rising, scale Stage-2 before changing the method.
- If accuracy saturates far below FP16, prioritize loss normalization, update
  balance, SubLN initialization/timing, and optimizer schedule.

### Gate B: Gamma-60 Diagnostic

The gamma-60 run tests whether the empirically equalized attention-KD scale is
healthier than paper-gamma under the local loss normalization.

Decision:

- If gamma-60 improves accuracy or stability, run a small normalized-gamma
  sweep with fixed Stage-2 checkpoints.
- If it does not, look beyond scalar gamma toward attention-layer choice,
  teacher quality, and Stage-2 budget.

### Gate C: Initializer Diagnostics

Synthetic LS and diagonal-Hessian ternary initialization reduce row-wise output
error in controlled tests:

```text
row absmean rel RMS:       ~0.5125
row diag-Hessian LS RMS:   ~0.4177 to 0.4349
```

Pending jobs test whether this initialization helps actual MNLI BitNet-SFT.

Decision:

- If initializer quality improves, use it as the standard retrofit
  initialization before QAT.
- If it does not, keep it as a negative result and do not add complexity to the
  main pipeline.

### Gate D: Runtime Product Path

Pick one deployable task formulation:

1. Native packed sequence classification in llama.cpp / GGUF, or
2. Causal prompt scoring as the primary task format.

Until that choice is made, quality and runtime are proven on related but not
identical artifacts.

### Gate E: TL2 Row-Scale Contract

The new group-scale viability audit prevents a premature kernel detour:

```text
current one-scale TL2 error: 1.904230
best fp16 group-scale error: 0.098692
exact fp16 row-scale error:  0.000197
```

Decision:

- Do not treat group/tile-scale TL2 as the quality-preserving row-scale fix.
- Exact row-scale metadata, or a different scale model with audited fidelity,
  is required before TL2 can close the objective blocker.

## Immediate Work Plan

1. Postprocess jobs `10040`, `10077`, `10079`, `10080`, and `10070` when they
   finish. Commit only audited JSON/Markdown, not raw log assumptions.
2. Keep the top-level README as a claim ledger, not a dense experiment dump.
3. Update reports with a strict label on every row:
   `paper-reproduction`, `paper-inspired`, or `retrofit-variant`.
4. Do not run new broad sweeps until the controlled Stage-2 curve and gamma-60
   diagnostic are materialized.
5. Keep MoE/Kimi in future work until dense Qwen quality/runtime is one
   deployed artifact.

## Publishable Framing

Do not claim:

- universal BitNet conversion,
- full BitDistill reproduction,
- row-scale as standard BitNet,
- Kimi/MoE support,
- useful general-LM quality from task-distilled causal exports.

Defensible framing:

> This fork provides an independent CPU-first ternary retrofit evaluation stack.
> It shows that blind PTQ fails for tested dense-Qwen checkpoints, that
> QAT/distillation partially recovers quality, and that row-scale ternary
> students require a matching packed runtime contract such as `I2_SR`.

This is publishable only if the next reports keep the distinction between
negative PTQ, BitDistill reproduction, row-scale runtime contribution, and
product readiness strict.
