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
| BitNet-SFT baseline can clear the paper anchor | Proven for one Qwen2.5 MNLI tensor-scale schedule/budget row | CE-only Qwen2.5-0.5B BitNet-SFT reaches `0.628935` vs paper BitNet-SFT anchor `0.608000`. This is a baseline sanity check, not BitDistill recovery and not a general Qwen3 result. |
| BitDistill FP recovery is not reproduced | Proven not yet complete | Controlled BitDistill rows reach `0.616607` and `0.691187`, still far from local FP16-SFT `0.807641`. |
| Loss-normalized attention KD helps | Proven for one matched diagnostic | Gamma-60 reaches MNLI `0.738462`, improving over the matched paper-gamma row by `+0.047275`, but still remains `-0.069689` behind FP16. |
| LS or diag-LS ternary initialization improves task quality | Rejected for current Qwen2.5 MNLI recipe | Unweighted LS reaches `0.361895`; calibrated diag-LS reaches `0.350993`; both trail the matched absmean baseline `0.628935`. Diag-LS paired delta is `-0.277942`, CI `[-0.290856, -0.265028]`. |
| Qwen3 paper-alignment is not reproduced | Proven for completed MNLI/QNLI/SST2 rows | Qwen3-0.6B-Base MNLI FP16-SFT is `0.829750`; BitNet-SFT is `0.477127`; tensor BitDistill layer `-1` improves to `0.723484` but remains `-0.106266` paired delta behind FP. The MNLI layer sweep confirms sensitivity: layer `-8` reaches `0.752012` and is best, paired `+0.028528` over layer `-1`, but still remains `-0.077738` behind FP; layer `-4` is `+0.009883` over `-1`; layer `-2` is not useful. QNLI FP16-SFT is `0.921106`; BitNet-SFT is `0.587040`; tensor BitDistill improves to `0.861065` but remains `-0.060040` paired delta behind FP. SST2 FP16-SFT is `0.930046`; BitNet-SFT is `0.799312`; tensor BitDistill improves to `0.871560` and row BitDistill to `0.877294`, but both remain behind FP by paired deltas `-0.058486` and `-0.052752`. |
| Row-scale runtime contract matters | Proven by output audit | One-scale TL2 relative output RMS error `1.904230`; exact row scales `0.000197`. |
| TL2 group/tile-scale compromise is enough | Rejected for strict fidelity | Best available fp16 group-scale row is `0.098692` relative output RMS; exact fp16 row scales are `0.000197`. |
| `I2_SR` can preserve row-scale ternary semantics in packed CPU inference | Proven for compatible causal-LM artifacts | Qwen2.5-1.5B `I2_SR` runs on Xeon Silver 4116 with PPL `38.8477`, prompt `211.67 tok/s`, decode `19.07 tok/s`. |
| Native packed sequence-classification product is solved | No, native plumbing only | A single-artifact `bitnet-qwen` GGUF carries the classifier head and matches the sidecar path. Direct token-ID evaluation fixes a Qwen BPE pair-boundary artifact; the repaired 64-example MNLI CPU sample reaches saved-PyTorch agreement `0.96875`, accuracy `0.59375`, and RSS `950.64 MiB`, but remains sample-only. |
| Native seqcls batching is product-ready | Rejected for now | Low-margin rows `15` and `35` change logits/predictions depending on sequence position inside a batch; max relative RMS vs alone is `0.305153`. |
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

At the latest local check:

| Job | Partition / node | Purpose | Status |
| ---: | --- | --- | --- |
| `10049` | `dualcard / ece-nebula10` | Qwen3 SST2 FP16-SFT reference | complete; accuracy `0.930046` |
| `10050` | `dualcard / ece-nebula10` | Qwen3 SST2 BitNet-SFT | complete; accuracy `0.799312` |
| `10079` | `midcard / ece-nebula12` | unweighted LS BitNet-SFT initializer benchmark | complete; negative transfer vs absmean |
| `10051` | `dualcard / ece-nebula10` | Qwen3 SST2 tensor BitDistill | complete; accuracy `0.871560`; paired delta vs FP `-0.058486` |
| `10052` | `dualcard / ece-nebula10` | Qwen3 SST2 row BitDistill | complete; accuracy `0.877294`; paired delta vs FP `-0.052752` |
| `10053` | `dualcard / ece-nebula10` | Qwen3 MNLI attention-layer sweep, layer `-8` | complete; accuracy `0.752012`; paired delta vs layer `-1` `+0.028528` |
| `10054` | `dualcard / ece-nebula10` | Qwen3 MNLI attention-layer sweep, layer `-2` | complete; accuracy `0.717779`; paired delta vs layer `-1` `-0.005706` |
| `10055` | `dualcard / ece-nebula10` | Qwen3 MNLI attention-layer sweep, layer `-4` | complete; accuracy `0.733367`; paired delta vs layer `-1` `+0.009883` |
| `10070` | `dualcard / ece-nebula10` | controlled 327.68M Stage-2 row | running |
| `10080` | `midcard / ece-nebula12` | calibrated diag-LS BitNet-SFT initializer benchmark | complete; negative transfer vs absmean, accuracy `0.350993` |
| `10081` | `midcard / ece-nebula12` | native single-artifact MNLI full validation in safe prompt-batch-size-1 mode | running |

No ternary quality claim should be made from the running or pending rows until
their postprocess audits materialize JSON and Markdown reports.

## Critical Interpretation

### 1. The weak early BitNet-SFT result was not final

The original local Qwen2.5 BitNet-SFT MNLI result was `0.487621`, far below
the paper anchor `0.608000`. A tensor-scale CE-only schedule/budget sweep
changed that:

```text
best completed CE-only BitNet-SFT: 0.628935
paper BitNet-SFT anchor:          0.608000
```

This means the low early Qwen2.5 baseline was mostly undertraining/schedule,
not proof that the BitLinear replacement was fundamentally broken. It does not
explain the Qwen3 paper-alignment failure, where the completed MNLI BitNet-SFT
row is still only `0.477127` against local FP16-SFT `0.829750`, and the
completed QNLI BitNet-SFT row is `0.587040` against local FP16-SFT `0.921106`.

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
`100000`, under this implementation. The completed gamma-60 diagnostic reaches
MNLI `0.738462`, a paired `+0.047275` over the matched paper-gamma control.
That does not prove the paper coefficient is bad. It proves the local
reduction/normalization contract is different enough that coefficient values
cannot be transferred without telemetry.

### 4. Row-scale is a fork contribution, not a BitDistill reproduction

Row-scale improves several retrofit results and has a clear runtime contract
through `I2_SR`. It should be labeled `retrofit-variant`, not
`paper-reproduction`, because the BitDistill paper describes per-tensor
absmean quantization in its baseline equations.

The completed Qwen3 MNLI/QNLI/SST2 branch is a useful caution: row-scale is not
monotonically better. On Qwen3-0.6B-Base MNLI, row minus tensor is `-0.027305`
with CI `[-0.034073, -0.020537]`; on QNLI it is `-0.012630` with CI
`[-0.019826, -0.005435]`. SST2 is slightly positive at `+0.005734`, but the CI
`[-0.011536, 0.023004]` crosses zero. Row-scale remains a systems/runtime
contribution and a retrofit variant, not a general accuracy guarantee.

The Qwen3 MNLI layer-selection sweep also confirms the paper's warning that
single-layer attention distillation is sensitive to which layer is selected.
Layer `-8` improves over the layer `-1` baseline by `+0.028528` paired accuracy
with CI `[0.022008, 0.035048]`, layer `-4` improves by `+0.009883` with CI
`[0.003959, 0.015807]`, and layer `-2` is slightly lower with CI crossing zero.
The best accuracy, `0.752012`, is still `-0.077738` behind FP16-SFT. Layer
selection matters, but it is not sufficient under the current
paper-gamma/loss-normalization contract.

### 5. The product artifact gap has narrowed, but remains

The strongest quality path is sequence classification. The strongest packed
runtime path was causal-LM `I2_SR`; the new `bitnet-qwen` native classifier
smoke proves that a single GGUF can carry the dense classifier head and emit
matching logits. A direct-token input path was added to `llama-embedding`
because Qwen sentence-pair token IDs cannot always be decoded to text and
re-tokenized losslessly. The repaired native MNLI CPU sample is measurable but
still sample-only: 64 examples, accuracy `0.59375`, saved-prediction agreement
`0.96875`, child peak RSS `950.64 MiB`.

A faster batch-4 sample reaches `2.639` examples/sec and happens to agree with
the saved PyTorch predictions on the first 64 examples, but it is not a valid
product benchmark yet. The batching audit shows that the same low-margin rows
can change logits and predictions with sequence position inside the batch.
The nearest-single-logit probe shows every drifted target row remains closest
to its own single-prompt logits, so the failure is not explained by a simple
output-row swap.

A credible product needs one artifact that carries both:

```text
quality evidence + CPU runtime evidence
```

The next source-owned work is not a broad full split run yet. It is to explain
the remaining native-vs-PyTorch mismatch, then rerun full MNLI only after the
sample gate reaches high agreement.

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

The gamma-60 run tested whether the empirically equalized attention-KD scale is
healthier than paper-gamma under the local loss normalization.

Decision:

- It did improve accuracy: `0.738462` vs matched paper-gamma `0.691187`.
- It did not recover FP: paired delta vs FP16 is `-0.069689`.
- Next step is a small normalized-gamma sweep with fixed Stage-2 checkpoints,
  not a broad search over unrelated axes.

### Gate C: Initializer Diagnostics

Synthetic LS and diagonal-Hessian ternary initialization reduce row-wise output
error in controlled tests:

```text
row absmean rel RMS:       ~0.5125
row diag-Hessian LS RMS:   ~0.4177 to 0.4349
```

Completed and pending jobs test whether this initialization helps actual MNLI
BitNet-SFT.

Decision:

- Unweighted LS initialization does not transfer to task quality under the
  current Qwen2.5 MNLI BitNet-SFT recipe: `0.361895` vs absmean `0.628935`.
- Calibrated diag-LS is worse under the same recipe: `0.350993` vs absmean
  `0.628935`, paired delta `-0.277942` with CI
  `[-0.290856, -0.265028]`.
- Do not promote LS-style ternary initialization into the main recipe. The
  synthetic reconstruction gains are not translating into task quality.

### Gate D: Runtime Product Path

The deployable task formulation is now native packed sequence classification,
but only at plumbing maturity.

Current proof:

```text
single-prompt native GGUF logits match sidecar logits
relative RMS delta: 0.000000103
prompt eval:        finite single-prompt smoke timing only; not a benchmark
64-example MNLI:    accuracy 0.59375, saved-prediction agreement 0.96875
token IDs:          direct-token path required; text round-trip is not lossless
batching parity:    failed; max relative RMS vs alone 0.305153 on audited rows
batching diagnosis: position-dependent drift, not simple output-row swap
```

Missing before product claims:

```text
full MNLI validation
batched inference parity
RSS measurement
throughput distribution
comparison against FP16 and Q4 task artifacts
residual packed-runtime drift analysis for the remaining disagreements
batch-invariant native embedding/classifier logits
```

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

1. Postprocess jobs `10070` and `10080` when they
   finish. Commit only audited JSON/Markdown, not raw log assumptions.
2. Keep the top-level README as a claim ledger, not a dense experiment dump.
3. Update reports with a strict label on every row:
   `paper-reproduction`, `paper-inspired`, or `retrofit-variant`.
4. Do not run new broad sweeps until the controlled Stage-2 curve and gamma-60
   diagnostic are materialized.
5. Debug the native seqcls sample mismatch before running full same-artifact
   validation, then keep MoE/Kimi in future work until dense Qwen has one
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
