# BitNet Retrofit Research Fork

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This fork investigates whether pretrained FP16/BF16 language models can be
retrofitted into BitNet-style W1.58A8 / ternary CPU inference without training
from scratch.

The short answer from the current evidence is:

**No, blind post-training ternarization is not a lossless or acceptable retrofit
path for arbitrary pretrained models.** The useful path is **QAT/distillation
under ternary forward constraints**, plus a runtime format that preserves the
scale semantics the distilled model learned.

This repository is therefore a research fork, not a claim that arbitrary Qwen,
Kimi, or other open-weight models can be converted to 1.58-bit form with no
quality loss.

## What This Fork Adds

The upstream Microsoft BitNet project provides optimized inference machinery
for native BitNet-style models. This fork uses that codebase to test a harder
question: can we take standard pretrained models and make them CPU-native after
the fact?

Added work in this fork includes:

- Qwen2.5 dense-model retrofit experiments with BitLinear replacement.
- FSDP-compatible ternary checkpoint export repair.
- Post-training ternarization, QAT, KL-only distillation, dense-head, and
  row-scale ablations.
- Ten-task lm-eval, WikiText, FineWeb-heldout, prompt sanity, paired-delta,
  CPU throughput, file-size, and RSS/context scaling reports.
- Direct static-ternary GGUF export for dense Qwen checkpoints.
- A stable row-scale `I2_SR` llama.cpp qtype/file type for preserving row-wise
  ternary scales in packed CPU inference.
- Routed `ggml_mul_mat_id` support for packed `I2_S`/`I2_SR` expert tensors,
  allowing tiny Qwen2MoE merged experts to execute through the CPU MoE path.
- A BitDistill smoke contract that now validates PyTorch QAT, tensor-scale
  `I2_S` GGUF export, row-scale `I2_SR` GGUF export, and SubLN key remapping.
- MoE/Kimi feasibility audits that separate generic routing support from real
  Kimi/MoE benchmark evidence.
- Tiny random Qwen2MoE FP16 and ternary `I2_SR` GGUF runtime fixtures proving
  generic converter, merged-expert packing, and CPU routed execution plumbing,
  without claiming Kimi or trained MoE quality.

The llama.cpp submodule now points at the writable fork:

`https://github.com/sabdulmajid/llama.cpp`

with the active `i2sr-row-scale-runtime` branch.

## Current Verdict

| claim | status | evidence |
| --- | --- | --- |
| Arbitrary FP16/BF16 to ternary conversion is lossless | **No** | Qwen2.5-1.5B naive PTQ collapses from ten-task mean `0.644169` to `0.348671`; WikiText PPL jumps from `13.901` to `3,813,121.803`. |
| Distillation/QAT can recover useful signal | **Yes, partially** | Best row-scale dense-Qwen run reaches ten-task mean `0.499459`, well above naive PTQ but below FP. |
| Stable CPU row-scale packed inference exists for dense Qwen | **Yes, for the audited path** | `I2_SR` productization gate passes `9/9`; Xeon I2_SR PPL `38.8477`, prompt `211.67 tok/s`, decode `19.07 tok/s`. |
| BitDistill paper-level GLUE reproduction is achieved here | **No, not yet** | Qwen2.5-0.5B gamma=100, strict paper-gamma tensor, strict paper-gamma row, LR=`1e-5`/`5e-5`, and strict paper-gamma head-init GLUE3 sequence-classification runs are complete with full paired prediction traces. They improve over BitNet-SFT but still miss FP16-SFT by `0.058486-0.203260` absolute accuracy depending on task/run. Clean row-warmup, full-budget/backbone-scale search, and full CPU-quality gates remain open. |
| TL2 is ready for the best row-scale checkpoint | **No** | Runtime contract gate fails: current TL2 one-scale error is `1.904230` relative output RMS; exact fp16 row scales would be `0.000197` with only `1.230469` MiB of scales, but converter/runtime/kernel metadata do not carry them. |
| Kimi/MoE retrofit is proven | **No** | Tiny random Qwen2MoE FP16 and ternary `I2_SR` fixtures now pass converter/runtime smoke; the ternary fixture packs 3 merged row-scale expert tensors, runs routed CPU inference, and records `419.29` decode tok/s at `142.48` MiB RSS. A Kimi-K2 config audit still shows the real target needs Kimi/DeepSeekV3 loading, MLA metadata conversion, shared-expert mapping, block-FP8 import, and trained MoE quality/locality benchmarks before product claims are defensible. |

## BitDistill Reproduction Status

Microsoft's BitDistill paper changes the answer from "PTQ is enough" to
"task-specific QAT/distillation may work if the training recipe is strong
enough." That does not contradict the negative PTQ result in this fork.

This fork now implements the key BitDistill components for Qwen-style models:
SubLN insertion, Stage-2 continued pretraining, Stage-3 CE + logits KL +
Q/K/V attention-relation distillation, attention-layer sweep support, and both
paper-style tensor-scale and experimental row-scale ternary students. New
BitDistill jobs default to paper-style logits KL scaling and paper-style
summation over Q/K/V relation losses.

The implementation smoke gate currently passes `40/40` checks, including
tensor-scale GGUF export and row-scale `I2_SR` GGUF export for a tiny
causal-LM BitDistill checkpoint. Those export checks use a smoke-only synthetic
tokenizer stub, so they prove tensor packing and metadata wiring, not text
generation quality.

The BitDistill claim-control reports now enforce full GLUE validation counts:
MNLI `9,815`, QNLI `5,463`, and SST2 `872` examples. Aggregate reproduction
rows, paired prediction traces, and CPU runtime rows cannot pass as quality
evidence unless their stored full-validation metrics match those counts. CPU
runtime sampling remains allowed for speed measurement, but the gate labels it
as sampled runtime and separately requires the full task-quality metric.
Packed causal-LM exports are also gated for provenance: the `I2_S`/`I2_SR`
GGUF file, converter summary, benchmark suite, RSS probe, qtype, SubLN mapping,
and manifest path must all agree.

A separate task-formulation audit prevents overclaiming across incompatible
GLUE setups. The strict local reproduction branch is
`Qwen2ForSequenceClassification`; causal-LM prompt-scoring rows are useful
deployment diagnostics and export candidates, but they are not mixed into the
headline sequence-classification table. Against the BitDistill excerpt's
Qwen2.5-0.5B MNLI anchor, the local FP16-SFT baseline is close
(`0.807641` vs `0.799100`), while BitNet-SFT (`0.487621` vs `0.608000`) and
short-budget BitDistill (`0.525217` vs `0.799800`) remain far below the paper
target. That makes the current gap concrete: the baseline task is learnable,
but the ternary recovery recipe is not yet reproduced.

Current completed Qwen2.5-0.5B GLUE sequence-classification short-budget
diagnostics:

| task | FP16-SFT | BitNet-SFT | BitDistill tensor | BitDistill row |
| --- | ---: | ---: | ---: | ---: |
| MNLI | `0.807641` | `0.487621` | `0.525217` | `0.516556` |
| QNLI | `0.898957` | `0.596925` | `0.596925` | `0.618525` |
| SST2 | `0.925459` | `0.770642` | `0.815367` | `0.808486` |

Current gamma=100 long-warmup sequence-classification diagnostics:

| task | FP16-SFT | BitNet-SFT | long-warmup tensor | tensor FP gap | long-warmup row | row FP gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MNLI | `0.807641` | `0.487621` | `0.641671` | `0.165970` | `0.653591` | `0.154050` |
| QNLI | `0.898957` | `0.596925` | `0.787846` | `0.111111` | `0.796998` | `0.101959` |
| SST2 | `0.925459` | `0.770642` | `0.866972` | `0.058486` | `0.854358` | `0.071101` |

The strict paper-gamma tensor branch (`attention_kd_weight=1e5`) is also
complete on GLUE3:

| task | paper-gamma tensor | FP gap | gamma=100 tensor |
| --- | ---: | ---: | ---: |
| MNLI | `0.630260` | `0.177381` | `0.641671` |
| QNLI | `0.759656` | `0.139301` | `0.787846` |
| SST2 | `0.841743` | `0.083716` | `0.866972` |

The first strict paper-gamma LR-search branch, LR=`1e-5`, is complete and
also negative:

| task | paper gamma | paper gamma LR=`1e-5` | LR delta | FP gap at LR=`1e-5` |
| --- | ---: | ---: | ---: | ---: |
| MNLI | `0.630260` | `0.604381` | `-0.025879` | `0.203260` |
| QNLI | `0.759656` | `0.757459` | `-0.002197` | `0.141497` |
| SST2 | `0.841743` | `0.846330` | `+0.004587` | `0.079128` |

The LR=`5e-5` branch is complete. It improves over strict paper-gamma on
MNLI/QNLI, regresses on SST2, and still misses the FP16 target by wide
margins:

| task | paper gamma | paper gamma LR=`5e-5` | LR delta | FP gap at LR=`5e-5` |
| --- | ---: | ---: | ---: | ---: |
| MNLI | `0.630260` | `0.642384` | `+0.012124` | `0.165257` |
| QNLI | `0.759656` | `0.790957` | `+0.031301` | `0.107999` |
| SST2 | `0.841743` | `0.836009` | `-0.005734` | `0.089450` |

Strict paper-gamma teacher-head initialization is also complete and does not
close the gap:

| task | paper gamma | paper gamma head-init | head-init delta | FP gap after head-init |
| --- | ---: | ---: | ---: | ---: |
| MNLI | `0.630260` | `0.627815` | `-0.002445` | `0.179827` |
| QNLI | `0.759656` | `0.762951` | `+0.003295` | `0.136006` |
| SST2 | `0.841743` | `0.834862` | `-0.006881` | `0.090596` |

Paper-gamma row-scale results are now complete and do not rescue the
strict paper coefficient:

| task | paper-gamma tensor | paper-gamma row | row-tensor delta |
| --- | ---: | ---: | ---: |
| MNLI | `0.630260` | `0.617626` | `-0.012634` |
| QNLI | `0.759656` | `0.760937` | `+0.001281` |
| SST2 | `0.841743` | `0.837156` | `-0.004587` |

Under this local implementation and budget, the literal paper coefficient does
not close the quality gap. On MNLI, a coefficient sweep gives gamma=100
`0.641671`, gamma=1k `0.647275`, gamma=10k `0.635354`, and gamma=100k
`0.630260`. The best tensor point in that sweep is still `0.160366` accuracy
behind FP16-SFT and below the gamma=100 row-scale run (`0.653591`).

The paired prediction audit now passes `44/44` matched-example comparisons on
the full GLUE validation splits. BitNet-SFT trails FP16-SFT by `31.89`
accuracy points on MNLI, `29.95` on QNLI, and `14.79` on SST2. The best
gamma=100 row-scale BitDistill branch is still behind FP16-SFT by `15.46`
points on MNLI, `10.25` on QNLI, and `7.11` on SST2. All rows include paired
confidence intervals and exact McNemar tests in the evidence bundle.

These runs do **not** reproduce the paper target of being within 0.5-1.0
accuracy point of FP16-SFT. The early completed wave is labeled as a
short-budget diagnostic because it used the common KD convention of multiplying
logits KL by `temperature**2`; the BitDistill equations do not include that
multiplier. It also used a legacy Q/K/V mean for attention relation KD. The
strict paper-gamma, LR-search, head-init, gamma-sweep, and layer-sweep branches
now use paper-style logits scaling plus Q/K/V sum and record those settings in
each metrics file.

The strongest remaining known gap is still training budget:
the first completed Stage-2 diagnostic used `40.96M` effective token
presentations, and the current strict tensor-scale warm-up has completed
`163.84M` token presentations. The paper reports `10B`
continued-pretraining tokens. The completed short-budget GLUE results should
therefore be read as a failure boundary for direct or short-warm-up retrofit,
not as a disproof of BitDistill.

As of the current evidence snapshot, the strict tensor-scale warm-up has
finished and its ternary checkpoint passes integrity checks: `169/169`
BitLinear weights exported, `169` tensor scales, valid ternary codes, final CE
`3.738920`. The gamma=100 downstream long-warmup GLUE branch has completed
for MNLI, QNLI, and SST2. These runs are meaningful recoveries over BitNet-SFT,
but they still fail the BitDistill paper-reproduction threshold. Row-scale is
higher than tensor on MNLI (`+0.011921`, paired 95% CI `[0.004958, 0.018883]`)
and QNLI (`+0.009152`, paired 95% CI `[0.000093, 0.018212]`), but lower on
SST2 (`-0.012615`, paired 95% CI `[-0.027678, 0.002448]`). The strict
paper-gamma tensor branch is complete and negative. Gamma=100 teacher-head
initialization is also complete and mixed/negative overall: it improves QNLI
row from `0.796998` to `0.800476`, leaves SST2 tensor unchanged at
`0.866972`, and worsens MNLI row plus SST2 row. The first MNLI
attention-layer sweep result, layer `-1`, reaches `0.645950`, which is a
small improvement over tensor gamma=100 but still `0.161691` behind FP16-SFT.
The next layer-sweep points are worse: layer `-2` reaches `0.642894` and
layer `-4` reaches `0.640754`. Paper-gamma row is worse than tensor on MNLI
and essentially tied on QNLI, so it is not the missing fix. The LR=`1e-5`,
LR=`5e-5`, and strict paper-gamma head-init searches are complete on GLUE3 and
also below target. Clean row-warmup branches remain running or queued.
No paper-level GLUE success claim will be made until full-validation
candidates close the FP16 gap.

The first exportable causal-LM long-warmup downstream diagnostics have
completed for MNLI, QNLI, and SST2. On full validation, MNLI reaches `0.615181`
tensor / `0.608355` row versus causal FP16 `0.829852` and causal BitNet-SFT
`0.517983`; QNLI reaches `0.765697` tensor / `0.770822` row versus causal
FP16 `0.900970` and causal BitNet-SFT `0.614681`; SST2 reaches `0.833716`
tensor / `0.840596` row versus causal FP16 `0.939220` and causal BitNet-SFT
`0.831422`. This is useful recovery over BitNet-SFT on MNLI and QNLI, and a
small recovery on SST2, but it is not a paper-level reproduction and it is not
the strict `Qwen2ForSequenceClassification` branch.

Those causal checkpoints also export through the packed runtime path on the
Xeon: tensor-scale checkpoints emit `MOSTLY_I2_S`, row-scale checkpoints emit
`MOSTLY_I2_SR`, and the active packed export/runtime gate passes `6/6` rows
with `168` packed ternary tensors each. Runtime throughput is about
`1175-1260` prefill tok/s and `93.8-105.4` decode tok/s at about `0.70` GiB
max RSS, but WikiText PPL is catastrophic (`155,846-347,660`). Treat this as
proof that the format/runtime path works and that task-specific causal
BitDistill does not preserve general language-model quality in this
configuration.

A separate PyTorch CPU sequence-classification slice now passes as a scoped
runtime artifact on the Xeon: 15/15 rows across MNLI/QNLI/SST2 for FP16-SFT,
BitNet-SFT, gamma=100 tensor, gamma=100 row, and paper-gamma tensor have
valid load time, examples/s, RSS, and stored full-validation accuracy. This is
not a packed `I2_SR` claim. It shows that PyTorch BitLinear task inference is
memory-heavy and slower than FP16-SFT in this setup, which reinforces that the
product path must use packed GGUF kernels rather than Python-level BitLinear
execution.

Active follow-ups are now focused on clean row-scale warm-up, full CPU/I2_SR
producer gates, and a decision on whether to spend the much larger compute
needed for Qwen3/full-budget reproduction. Completed diagnostics already show
that gamma=100
teacher-head initialization, strict paper-gamma head-init, the MNLI layer
sweep, strict paper-gamma row, and the completed paper-gamma LR=`1e-5` /
LR=`5e-5` GLUE3 probes are not enough to recover paper-level accuracy. The
best LR=`5e-5` task improvement is QNLI, but it remains `0.107999` absolute
accuracy behind FP16-SFT. The completed MNLI gamma probes at `1e3` and `1e4`
also support the relation-loss scale audit: the paper's `1e5`
coefficient can dominate CE by orders of magnitude under this local
normalization.
The earlier completed BitDistill runs use attention KD weight `100`; those are
useful diagnostics but are not a strict match to the paper's reported
classification setting. Until the remaining gates close, the public claim
remains conservative:
**BitDistill is the right class of method, but this fork has not yet reproduced
paper-level task quality.**

Focused MNLI diagnostics after fixing logits-KL scaling and sweeping the
attention-distillation layer improved the best short-budget BitDistill result
to `0.535711` versus FP16-SFT `0.807641`. CE-only ablations stay near
`0.492-0.498`, so distillation helps, but the run is still far from
reproduction-quality. With long-warmup, MNLI layer `-1` now reaches
`0.645950`, layer `-2` reaches `0.642894`, and layer `-4` reaches `0.640754`,
all still below the row-scale gamma=100 best of `0.653591`.

Runtime boundary: the active paper-style GLUE reproduction uses
`Qwen2ForSequenceClassification`. Those checkpoints can be evaluated on CPU
with the PyTorch task benchmark in this fork, but they are **not** valid packed
llama.cpp / `I2_SR` exports today because the runtime path does not implement a
Qwen sequence-classification head. The stable `I2_SR` exporter is valid for
causal-LM BitDistill checkpoints; packed task inference requires either
causal prompt-scoring checkpoints or new classifier-head support in the
runtime. The causal prompt-scoring long-warmup branch under
`checkpoints/bitdistill-glue-causal-longwarmup-densehead` has completed and is
used only as an export/runtime diagnostic; it is not conflated with the
sequence-classification reproduction gate.

## Key Dense-Qwen Results

Quality measurements below are Qwen2.5-1.5B unless noted.

| run | WikiText PPL | FineWeb PPL | ten-task mean |
| --- | ---: | ---: | ---: |
| FP reference | `13.901` | `10.269` | `0.644169` |
| naive PTQ ternary | `3,813,121.803` | `9,582,923.269` | `0.348671` |
| QAT hidden-MSE | `86.414` | `40.398` | `0.464809` |
| QAT KL-only | `50.595` | `26.599` | `0.483438` |
| QAT KL-only, dense tied `lm_head` | `43.372` | `22.759` | `0.484378` |
| QAT KL-only, row-scale, dense tied `lm_head` | `38.580` | `21.333` | `0.499459` |

Paired ten-task deltas:

- Row-scale QAT minus naive PTQ: `+0.150788`, 95% CI `[+0.053427, +0.248149]`.
- Row-scale QAT minus FP: `-0.144710`, 95% CI `[-0.185756, -0.103664]`.
- Row-scale QAT minus tensor-scale dense-head QAT: `+0.015081`, 95% CI `[+0.009028, +0.021134]`.

Interpretation: row-scale distillation is a real recovery path, but it does not
close the gap to FP quality.

## Xeon CPU Runtime Snapshot

These are fixed-excerpt llama.cpp CPU runs on an Intel Xeon Silver 4116
portable-AVX2 build. Throughput should be compared only within this hardware
and build context.

| artifact | file MiB | PPL | prompt tok/s | decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| FP F16 | `2950.4` | `12.2808` | `114.47` | `5.56` |
| FP Q8_0 | `1570.3` | `12.3056` | `124.86` | `10.13` |
| FP Q4_K_M | `940.4` | `12.8112` | `92.08` | `16.01` |
| blind FP-to-I2_S | `766.1` | catastrophic | `204.57` | `18.34` |
| row-scale ternary TQ2_0 | `1218.6` | `38.8224` | `169.46` | `18.68` |
| row-scale ternary I2_S prototype | `1211.3` | `38.8832` | `218.17` | `18.97` |
| row-scale ternary I2_SR | `1211.3` | `38.8477` | `211.67` | `19.07` |

`I2_SR` is the stable row-scale path in this fork. It preserves the row-scale
QAT checkpoint quality while keeping the packed ternary CPU throughput benefit.
It does not make blind PTQ viable.

## What Is Proven

1. Naive post-training ternarization destroys dense pretrained model quality in
   the tested Qwen artifacts.
2. Training or distillation under ternary constraints recovers measurable
   quality.
3. Row-wise scales matter; collapsing row-wise scale information into one
   tensor scale breaks the best checkpoint.
4. A stable packed CPU runtime can preserve row-scale quality when the file
   format and kernels represent the same scale semantics.
5. The current dense-Qwen work is useful as a research MVP and evaluator, not as
   a universal model converter.
6. The current short-budget BitDistill reproduction fails GLUE3, reinforcing
   that continued pretraining and distillation budget are core parts of the
   method rather than implementation details.

## What Is Not Proven

- No evidence supports a one-click arbitrary FP16/BF16-to-ternary product.
- No evidence shows FP-quality 1.58-bit Qwen from this retrofit recipe.
- No completed local run yet reproduces BitDistill paper-level GLUE quality.
- No packed `I2_SR` path currently runs the sequence-classification GLUE heads.
- No evidence validates Kimi or a trained MoE model in this ternary runtime.
- The Kimi-K2 config path is specifically blocked on Kimi/DeepSeekV3 mapping,
  MLA/Q-LoRA attention metadata, shared experts, FP8 block import, and trained
  MoE quality/locality benchmarks.
- TL2 remains an engineering probe for row-scale Qwen, not a supported product
  path for the strongest checkpoint.

## Repository Map

| path | purpose |
| --- | --- |
| `benchmarks/` | benchmark, audit, conversion, and report-generation scripts |
| `benchmarks/results/` | public Markdown reports with parsed results and verdicts |
| `benchmark_results/` | raw JSON summaries and benchmark artifacts |
| `experiments/` | small mathematical audits and viability probes |
| `patches/` | historical/runtime patch records used by the audits |
| `utils/` | Hugging Face conversion and preprocessing utilities |
| `3rdparty/llama.cpp` | llama.cpp fork with active `I2_SR` support |
| `src/ggml-bitnet-mad.cpp` | BitNet CPU quantization/runtime integration |

## Reproducing The Current Gates

The active public reports use `BITNET_REPORT_DATE=2026-05-15`. They are
generated from checked-in scripts plus raw artifacts under `benchmark_results/`.
Clean row-warmup and the full CPU runtime benchmark are still running or
queued, while the active `I2_SR` export gate is complete. These commands
therefore keep the unfinished row-warmup and CPU gates partial rather than
success claims.

```bash
export BITNET_REPORT_DATE=2026-05-15

python benchmarks/monitor_bitdistill_jobs.py

python benchmarks/monitor_bitdistill_jobs.py \
  --job-table benchmark_results/bitdistill_rowwarmup_downstream_gamma100_20260515.tsv \
  --warmup-log logs/bitdistill-glue-10028.out \
  --output-json benchmark_results/bitdistill_row_warmup_monitor_2026-05-15.json \
  --output-md benchmarks/results/bitdistill_row_warmup_monitor_2026-05-15.md

python benchmarks/audit_bitdistill_warmup_health.py

python benchmarks/audit_bitdistill_warmup_health.py \
  --monitor-json benchmark_results/bitdistill_row_warmup_monitor_2026-05-15.json \
  --log-path logs/bitdistill-glue-10028.out \
  --output-json benchmark_results/bitdistill_row_warmup_health_2026-05-15.json \
  --output-md benchmarks/results/bitdistill_row_warmup_health_2026-05-15.md

python benchmarks/audit_bitdistill_snapshot_integrity.py \
  --monitor-json benchmark_results/bitdistill_row_warmup_monitor_2026-05-15.json \
  --validate-codes

python benchmarks/gate_bitdistill_reproduction.py
python benchmarks/gate_bitdistill_rowwarmup.py
python benchmarks/audit_bitdistill_paper_alignment.py
python benchmarks/audit_bitdistill_task_formulation.py
python benchmarks/gate_bitdistill_i2sr_export.py
python benchmarks/gate_bitdistill_cpu_benchmark.py
python benchmarks/gate_bitdistill_cpu_benchmark.py \
  --input-json benchmark_results/bitdistill_glue_cpu_fast_2026-05-15.json \
  --critical-runs \
    short:fp16_sft-tensor-layer-1 \
    short:bitnet_sft-tensor-layer-1 \
    longwarmup:bitdistill-longwarmup-tensor-layer-8 \
    longwarmup:bitdistill-longwarmup-row-layer-8 \
    papergamma:bitdistill-longwarmup-tensor-layer-8 \
  --output-json benchmark_results/bitdistill_glue_cpu_fast_gate_2026-05-15.json \
  --output-md benchmarks/results/bitdistill_glue_cpu_fast_gate_2026-05-15.md
python benchmarks/audit_tl2_row_scale_runtime_contract.py
python benchmarks/audit_product_scope.py
python benchmarks/audit_bitdistill_active_goal.py
```

Build smoke used for the active `I2_SR` runtime:

```bash
cmake --build build-portable-avx2 --target llama-cli llama-bench llama-perplexity llama-quantize -j 12
./build-portable-avx2/bin/llama-quantize --help | rg "I2_SR|I2_S"
```

## Primary Reports

- [Qwen side-by-side summary](benchmarks/results/qwen_side_by_side_2026-05-15.md)
- [BitDistill active goal audit](benchmarks/results/bitdistill_active_goal_audit_2026-05-15.md)
- [BitDistill active job monitor](benchmarks/results/bitdistill_job_monitor_2026-05-15.md)
- [BitDistill reproduction gap analysis](benchmarks/results/bitdistill_reproduction_gap_analysis_2026-05-15.md)
- [BitDistill warm-up health audit](benchmarks/results/bitdistill_warmup_health_2026-05-15.md)
- [BitDistill row-warmup monitor](benchmarks/results/bitdistill_row_warmup_monitor_2026-05-15.md)
- [BitDistill row-warmup health audit](benchmarks/results/bitdistill_row_warmup_health_2026-05-15.md)
- [BitDistill snapshot integrity audit](benchmarks/results/bitdistill_snapshot_integrity_2026-05-15.md)
- [BitDistill reproduction gate](benchmarks/results/bitdistill_reproduction_gate_2026-05-15.md)
- [BitDistill row-warmup gate](benchmarks/results/bitdistill_rowwarmup_gate_2026-05-15.md)
- [BitDistill paper alignment audit](benchmarks/results/bitdistill_paper_alignment_2026-05-15.md)
- [BitDistill task formulation audit](benchmarks/results/bitdistill_task_formulation_audit_2026-05-15.md)
- [BitDistill GLUE CPU gate](benchmarks/results/bitdistill_glue_cpu_gate_2026-05-15.md)
- [BitDistill scoped GLUE CPU gate](benchmarks/results/bitdistill_glue_cpu_fast_gate_2026-05-15.md)
- [BitDistill causal I2_SR export gate](benchmarks/results/bitdistill_i2sr_export_gate_2026-05-15.md)
- [BitDistill local causal I2_SR export gate](benchmarks/results/bitdistill_i2sr_export_gate_local_2026-05-15.md)
- [BitDistill producer script audit](benchmarks/results/bitdistill_producer_script_audit_2026-05-15.md)
- [Objective completion audit](benchmarks/results/objective_completion_audit_2026-05-15.md)
- [Evidence manifest](benchmarks/results/evidence_manifest_2026-05-15.md)
- [Benchmark coverage gate](benchmarks/results/benchmark_coverage_gate_2026-05-15.md)
- [Product scope gate](benchmarks/results/product_scope_gate_2026-05-15.md)
- [I2_SR submodule promotion audit](benchmarks/results/i2sr_submodule_promotion_audit_2026-05-13.md)
- [Row-scale qtype productization gate](benchmarks/results/row_scale_qtype_productization_gate_2026-05-13.md)
- [Direct packed GGUF support audit](benchmarks/results/direct_packed_gguf_support_2026-05-13.md)
- [TL2 row-scale design audit](benchmarks/results/tl2_row_scale_design_2026-05-13.md)
- [TL2 row-scale runtime contract](benchmarks/results/tl2_row_scale_runtime_contract_2026-05-15.md)
- [MoE support audit](benchmarks/results/moe_support_audit_2026-05-15.md)
- [MoE packing contract](benchmarks/results/moe_packing_contract_2026-05-15.md)
- [MoE TL2 runtime contract](benchmarks/results/moe_tl2_runtime_contract_2026-05-15.md)
- [Kimi config feasibility audit](benchmarks/results/kimi_config_feasibility_2026-05-15.md)
- [Tiny Qwen2MoE runtime fixture](benchmarks/results/tiny_qwen2moe_fixture_2026-05-14.md)
- [Unblock requirements audit](benchmarks/results/unblock_requirements_2026-05-14.md)

## Product Direction

The credible product is not "convert any model to 1.58-bit." The credible
product is a CPU-first retrofit evaluator and distillation pipeline:

1. Ingest a dense Hugging Face checkpoint.
2. Replace selected linear projections with BitLinear-compatible modules.
3. Distill with KL loss under ternary forward constraints.
4. Preserve learned scale semantics in GGUF using `I2_SR`.
5. Report quality, size, throughput, RSS, and paired statistical deltas before
   any model is advertised as usable.

MoE/Kimi should be treated as the next research milestone, requiring licensed
artifacts, router/expert distillation, ternary expert runtime validation,
expert-locality measurement, and CPU quality/throughput/RSS benchmarks. The
tiny Qwen2MoE fixture only proves generic FP16 converter/runtime plumbing.

## Upstream Attribution

This fork is based on Microsoft's BitNet / bitnet.cpp work and uses llama.cpp
as the CPU inference substrate. Upstream BitNet release claims apply to native
BitNet models, not to the arbitrary-retrofit experiments reported here.
