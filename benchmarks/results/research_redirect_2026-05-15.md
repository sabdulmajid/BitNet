# Research Redirect, 2026-05-15

## Verdict

The research question has changed.

The original question was whether arbitrary pretrained FP16/BF16 models could
be post-hoc projected into BitNet-style W1.58A8 ternary CPU inference. Current
evidence rejects that path for the tested dense-Qwen checkpoints.

The defensible current question is narrower:

> Can a pretrained teacher train a task-specific ternary student, and can the
> packed CPU runtime preserve the scale semantics learned by that student?

That separates the negative algorithmic result from the positive systems
result.

## Proven So Far

| result | status | evidence |
| --- | --- | --- |
| Blind PTQ-to-ternary fails for tested Qwen | proven for tested setup | Qwen2.5-1.5B ten-task mean `0.644169` FP to `0.348671` naive PTQ; WikiText PPL `13.901` to `3,813,121.803`. |
| QAT/distillation recovers signal | proven partially | Best row-scale dense-Qwen ten-task mean `0.499459`; paired recovery over naive PTQ `+0.150788` with 95% CI `[+0.053427, +0.248149]`. |
| Row-scale semantics matter | proven for current best row-scale checkpoint | TL2 one-scale relative output RMS error `1.904230`; exact FP16 row scales `0.000197` with only `1.230469 MiB` scale overhead. |
| Packed row-scale CPU runtime is feasible | proven for dense causal artifact | `I2_SR` Xeon result: PPL `38.8477`, prompt `211.67 tok/s`, decode `19.07 tok/s`, file `1211.3 MiB`. |
| Paper-level BitDistill is reproduced | not proven | FP16-SFT MNLI is close to paper (`0.807641` vs `0.799100`), but BitNet-SFT remains below anchor (`0.487621` default; best completed budget row `0.574733` vs paper `0.608000`). |
| Kimi/MoE works | not proven | Tiny Qwen2MoE fixtures prove routing/packing smoke only; no Kimi mapping or trained MoE quality exists. |

## The Narrow Blocker

The key reproduction failure is now specific:

```text
FP16-SFT learns MNLI locally.
BitNet-SFT does not match the paper's BitNet-SFT anchor.
```

| MNLI row | local | paper anchor |
| --- | ---: | ---: |
| FP16-SFT | `0.807641` | `0.799100` |
| BitNet-SFT default | `0.487621` | `0.608000` |
| BitNet-SFT best completed budget row | `0.574733` | `0.608000` |

This means the next work should not be broad ablation expansion. It should
explain why BitNet-SFT is low before claiming anything about BitDistill.

Completed controls rule out several simple causes:

- Decoder projection replacement count is correct: `168/168`.
- The sequence-classification head remains dense.
- Removing activation quantization only moves MNLI from `0.487621` to
  `0.493734`, so A8 is not the dominant cause.
- Current SubLN-only insertion worsens MNLI to `0.350280`, so SubLN placement,
  initialization, or timing is not aligned enough to explain the paper result.
- A 1000-step LR sweep improved the best short row to `0.523892`.
- The completed 3000-step sweep improves further to `0.574733`, with `48,000`
  optimizer examples, or `0.122230` MNLI epochs, but the best row still misses
  the paper anchor by `0.033267`.

## Canonical Next Matrix

Keep the next wave narrow until the BitNet-SFT baseline is explained.

| axis | selected value |
| --- | --- |
| Model | Qwen2.5-0.5B first |
| Task | MNLI first |
| Formulation | `Qwen2ForSequenceClassification` |
| Scale mode | tensor-scale paper-style first |
| Baselines | FP16-SFT, BitNet-SFT, BitDistill |
| Current downstream budgets | `1000`, `3000`, `10000` steps |
| Metrics | full validation accuracy, paired predictions, last CE, replacement counts, A8/SubLN settings |
| Success criterion | within about `0.5-1.0` accuracy point of FP16-SFT |

Decision rule:

- If longer BitNet-SFT rows move monotonically toward `0.608000`, the short
  rows were mostly undertrained.
- If the curve saturates below the paper anchor, investigate equation-level
  recipe mismatch before running larger BitDistill jobs.
- Row-scale should remain a separate `retrofit-variant`, not a
  paper-reproduction row.

## Product Framing

Do not market this as a universal BitNet converter.

The credible product is a CPU-first retrofit evaluator:

1. Ingest a dense Hugging Face checkpoint.
2. Train or distill a task-specific ternary student.
3. Export tensor-scale or row-scale packed GGUF.
4. Benchmark quality, PPL, file size, RSS, prompt/decode throughput, and paired
   statistical deltas.
5. Return a safe claim label: unsupported, paper-inspired, retrofit-variant, or
   paper-reproduction.

The product is still useful when it says "no", because it prevents deploying a
model whose ternary artifact is fast but not accurate.

## Publishable Angle

The publishable work is not discovering BitDistill. Microsoft did.

The plausible contribution is:

- independent reproduction attempt with strict claim gates,
- negative boundary study for blind PTQ and short-budget ternary retrofit,
- row-scale ternary as a retrofit relaxation,
- `I2_SR` packed CPU runtime contract for preserving row-scale semantics,
- task-specific success versus general-LM failure boundary,
- explicit MoE/Kimi feasibility limits.

## Immediate Actions

1. Finish and ingest the BitNet-SFT `3000`/`10000` step budget curve.
2. If the curve remains weak, audit BitLinear equations, SubLN placement/timing,
   dense-head treatment, optimizer schedule, and ternary code/scale dynamics
   before broader BitDistill sweeps.
3. Keep Qwen3/Qwen2.5 paper-alignment jobs labeled as partial until full
   validation rows close the FP16 gap.
4. Keep row-scale `I2_SR` as a separate systems contribution.
5. Move real Kimi/MoE claims to future work until trained quality and routing
   locality are measured.
