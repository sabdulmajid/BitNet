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
| TL2 row-scale packed runtime is ready | not proven; explicitly blocked | The current TL2 runtime contract has `11` checks and remains `false`: converter scale collapse, missing row-scale TL2 storage, one-scale transform metadata, generated `Scales[0]` qgemm, unoffset x86 dispatch, missing loader sidecars, and no passing row-scale TL2 benchmark. |
| Sequence-classification packed path is possible | prototype only; native head blocked | MNLI long-warmup row-scale checkpoint (`0.653591` PyTorch accuracy) exports as a `352.6 MiB` `I2_SR` backbone plus `10.8 KiB` dense head sidecar. A Qwen-compatible `bitnet-qwen` graph repairs the dominant runtime mismatch: hidden cosine is `0.994091`, hidden relative RMS is `0.108662`, and the 64-example sidecar CPU probe reaches `0.578125` accuracy with `0.921875` agreement against saved PyTorch predictions. Native GGUF classifier inference and full-split CPU validation are not implemented. |
| Paper-level BitDistill is reproduced | not proven | FP16-SFT MNLI is close to paper (`0.807641` vs `0.799100`), and BitNet-SFT now clears its paper anchor (`0.628935` vs `0.608000`) after more budget. This is not yet BitDistill or FP16-level recovery. |
| Kimi/MoE works | not proven | Tiny Qwen2MoE fixtures prove routing/packing smoke only; no Kimi mapping or trained MoE quality exists. |

## Baseline Status

The earlier reproduction failure was specific:

```text
FP16-SFT learns MNLI locally.
Short-budget BitNet-SFT does not match the paper's BitNet-SFT anchor.
```

| MNLI row | local | paper anchor |
| --- | ---: | ---: |
| FP16-SFT | `0.807641` | `0.799100` |
| BitNet-SFT default | `0.487621` | `0.608000` |
| BitNet-SFT best completed budget row | `0.628935` | `0.608000` |

The first `10000`-step BitNet-SFT row changes the interpretation: the weak
baseline was substantially undertrained at the shorter budgets. The current
blocker is now BitDistill recovery toward the FP16 task model, not whether the
CE-only BitNet-SFT baseline can reach the paper's weaker BitNet-SFT anchor.

Completed controls rule out several simple causes:

- Decoder projection replacement count is correct: `168/168`.
- The sequence-classification head remains dense.
- Removing activation quantization only moves MNLI from `0.487621` to
  `0.493734`, so A8 is not the dominant cause.
- Current SubLN-only insertion worsens MNLI to `0.350280`, so SubLN placement,
  initialization, or timing is not aligned enough to explain the paper result.
- A direct SubLN activation audit on a local Qwen2.5-0.5B checkpoint shows why
  this is plausible: adding the local SubLN wrappers inserts `48` RMSNorms,
  normalizes audited projection inputs to near unit RMS, and causes last-token
  logit relative RMS drift `0.768044` before any warmup.
- A 1000-step LR sweep improved the best short row to `0.523892`.
- The completed 3000-step sweep improves further to `0.574733`, with `48,000`
  optimizer examples, or `0.122230` MNLI epochs.
- The first 10000-step row reaches `0.628935`, with `160,000` optimizer
  examples, or `0.407434` MNLI epochs. It exceeds the paper BitNet-SFT anchor
  by `0.020935`, but remains `0.178706` below the local FP16-SFT row.
- The second 10000-step row at `lr=5e-5` reaches `0.607845`, effectively the
  paper BitNet-SFT anchor but weaker than the `lr=2e-5` row. This confirms
  that the BitNet-SFT baseline is budget-viable but schedule-sensitive.
- Against the saved FP16 prediction trace, the paired candidate-minus-FP16
  delta is `-0.179215`, 95% CI `[-0.189580, -0.168851]`, McNemar
  `p=3.438389e-240`.

Telemetry coverage is deliberately scoped. The current script and saved metrics
record CE, logits KD, attention KD, weighted KD terms, final ternary code
fractions, and offline SubLN perturbation. They do not yet record
per-component gradient norms, ternary flip rates during active training,
per-layer scale trajectories, activation int8 saturation, or Q/K/V-split
attention losses. Therefore the current evidence supports loss-scale and
static-mechanics triage, but it does not yet prove which objective term
dominates the update direction.

An offline Stage-2 snapshot audit now measures ternary-code movement for the
existing Qwen2.5-0.5B row-scale warm-up. Across `493,961,216` ternary elements,
the code flip rate is `0.165956` from step `1000` to `10000` and `0.064547`
from step `10000` to `20000`; the zero-code fraction rises from `0.320018` to
`0.356683`. This is not live gradient telemetry, but it does show that warm-up
continues to move the discrete ternary representation.

The controlled Stage-2 recovery audit now parses Slurm logs and full prediction
traces. The `163.84M`-token paper-gamma row completed with MNLI accuracy
`0.691187`, paired delta versus local FP16 `-0.116964`, and 95% CI
`[-0.126110, -0.107817]`. This is materially better than the earlier
short-budget BitDistill rows, but it is still outside the paper-style FP
recovery gate. Its final log has weighted-attention/CE `5945.070866`, median
weighted-attention/CE `1729.105844`, p95 weighted-attention/CE `6080.253825`,
and max observed weighted-attention/CE `37819.641342`; the median raw
CE/attention equalizing gamma is `57.831811`, not `100000`. That keeps loss
normalization and update-balance as first-class reproduction risks.

## Canonical Next Matrix

Keep the next wave narrow now that the BitNet-SFT anchor is cleared.

| axis | selected value |
| --- | --- |
| Model | Qwen2.5-0.5B first |
| Task | MNLI first |
| Formulation | `Qwen2ForSequenceClassification` |
| Scale mode | tensor-scale paper-style first |
| Baselines | FP16-SFT, BitNet-SFT, BitDistill |
| Current downstream budgets | `1000`, `3000`, `10000` steps; controlled Stage-2 rows at `40.96M`, `163.84M`, and `327.68M` token presentations |
| Metrics | full validation accuracy, paired predictions, last CE, replacement counts, A8/SubLN settings |
| Success criterion | within about `0.5-1.0` accuracy point of FP16-SFT |

Decision rule:

- Finish the controlled Stage-2 token-budget curve before adding new sweep axes.
- Treat the completed `163.84M` row as positive movement, not a reproduction.
- Treat row-scale as a separate `retrofit-variant`, not a paper-reproduction row.

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

The current product blocker is now explicitly audited and partially narrowed.
The strict GLUE quality branch has `15` `Qwen2ForSequenceClassification`
checkpoint configs and `0` native causal-export-compatible configs. One MNLI
checkpoint now exports as a packed `I2_SR` decoder backbone plus dense score-head
sidecar. The original `bitnet-25` graph mismatch is repaired by a dedicated
`bitnet-qwen` graph that keeps the BitNet/SubLN/I2_SR loader contract while
using Qwen SiLU/SwiGLU FFN semantics and Q/K/V projection-bias tensors. On the
audited MNLI sample, token IDs match, hidden cosine is `0.994091`, hidden
relative RMS is `0.108662`, and score-logit relative RMS is `0.091918`. On a
64-example CPU sidecar probe, accuracy is `0.578125` and agreement with saved
PyTorch predictions is `0.921875`. This is a useful prototype, but not a
deployable classifier: the classifier head is still not native GGUF metadata or
runtime code, the hidden contract is not bit-exact, and full-split CPU
quality/RSS/throughput have not been measured on one native artifact.

For row-scale causal artifacts, the supported packed runtime remains `I2_SR`.
TL2 is not a shortcut: the converter recomputes one scalar scale, ggml byte
sizing has no row-scale TL2 sidecar, the transform stores one scale, generated
x86 TL2 qgemm multiplies by `Scales[0]`, and the x86 dispatch passes the same
scale pointer to each qgemm call. A quality-preserving TL2 variant needs an
explicit new row/group-scale metadata contract plus fresh PPL, task, throughput,
and RSS benchmarks.

## Publishable Angle

The publishable work is not discovering BitDistill. Microsoft did.

The plausible contribution is:

- independent reproduction attempt with strict claim gates,
- negative boundary study for blind PTQ and short-budget ternary retrofit,
- row-scale ternary as a retrofit relaxation,
- `I2_SR` packed CPU runtime contract for preserving row-scale semantics,
- sidecar-to-native sequence-classification runtime bridge for the same task
  checkpoints,
- task-specific success versus general-LM failure boundary,
- explicit MoE/Kimi feasibility limits.

## Immediate Actions

1. Finish and ingest the controlled Stage-2 rows at `40.96M` and `327.68M`
   token presentations.
2. Use the cleared CE-only BitNet-SFT anchor as the controlled baseline for the
   next BitDistill interpretation.
3. If the controlled curve remains weak, audit loss normalization, SubLN placement/timing,
   dense-head treatment, optimizer schedule, and ternary code/scale dynamics.
4. Add opt-in gradient-component, flip-rate, scale-trajectory, activation
   saturation, and Q/K/V-split telemetry after the active queued jobs finish.
5. Keep Qwen3/Qwen2.5 paper-alignment jobs labeled as partial until full
   validation rows close the FP16 gap.
6. Promote the `bitnet-qwen` sidecar sequence-classification smoke into native GGUF classifier
   metadata/head execution, then run full MNLI/QNLI/SST2 on CPU.
7. Keep row-scale `I2_SR` as a separate systems contribution.
8. Move real Kimi/MoE claims to future work until trained quality and routing
   locality are measured.
