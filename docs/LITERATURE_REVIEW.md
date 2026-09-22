# Ternary LLM Literature Review

Cutoff: 2026-09-22

This review focuses on the question this fork can answer: can a pretrained
full-precision model be made into a useful ternary CPU artifact, and what
training and runtime representation does that require?

## The Landscape Has Changed

The 2024-2025 baseline story was simple: native ternary pretraining worked,
while direct post-training ternarization usually collapsed. By late 2026,
several optimized PTQ methods report substantially better results. The local
negative result therefore applies to **naive symmetric absmean rounding**, not
to the entire PTQ category.

The relevant method families are now:

| Family | Representative work | What changes relative to naive rounding |
| --- | --- | --- |
| Native ternary training | [BitNet b1.58](https://arxiv.org/abs/2402.17764), [2B4T](https://arxiv.org/abs/2504.12285) | Learns the entire model under the ternary forward constraint |
| Task-specific adaptation | [BitDistill](https://arxiv.org/abs/2510.13998), [BitNet Text Embeddings](https://arxiv.org/abs/2606.25674) | Adds normalization, continued training, and teacher alignment for a bounded task |
| Optimized ternary PTQ | [PT2-LLM](https://arxiv.org/abs/2510.03267), [CAT-Q](https://arxiv.org/abs/2606.26650), [ScaleQ-1.58](https://arxiv.org/abs/2608.01078), [TWLA](https://arxiv.org/abs/2606.13054) | Learns asymmetric grids, modulation, rotations, or calibration-aware corrections |
| More expressive low-bit codes | [PTQTP](https://arxiv.org/abs/2509.16989), [D2Quant](https://arxiv.org/abs/2602.02546) | Uses multiple ternary planes or richer sub-4-bit representations |
| Runtime/storage specialization | [bitnet.cpp](https://arxiv.org/abs/2502.11880), [BITCOS](https://arxiv.org/abs/2609.16338) | Exploits the resulting code distribution and target ISA after quality is established |

These families are not interchangeable. A method can report “ternary weights”
while requiring asymmetric positive/negative levels, group scales, FP16
activations, rotations, or two trit planes. Each choice changes effective bits,
kernel design, and whether this fork's symmetric row-scale `I2_SR` format can
represent it losslessly.

## Native BitNet

[The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) introduced BitNet
b1.58 with ternary weights and low-precision activations. The official
[BitNet b1.58 2B4T](https://arxiv.org/abs/2504.12285) model was trained from
scratch on four trillion tokens. This remains the cleanest evidence that the
ternary model family is viable when optimization sees the constraint from the
beginning.

Implication for this fork: native BitNet proves physical feasibility, but not
cheap retrofit. It is the quality upper reference for a model designed around
the representation.

## BitDistill

[BitNet Distillation](https://arxiv.org/abs/2510.13998) adapts pretrained
models for specific downstream tasks through:

1. SubLN architecture refinement;
2. 10B-token continued pretraining;
3. task CE plus logits and attention-relation distillation.

It reports task performance close to FP16 with substantial memory and CPU
benefits. That claim does not imply a general-purpose language-model retrofit;
the reported success is task-specific. As of this review, the official
repository's request for the referenced training implementation remains
[open](https://github.com/microsoft/BitNet/issues/354). This fork is therefore
an independent paper-inspired implementation, not verified source-equivalent
code.

Implication: our `0.755340-0.757039` matched MNLI means versus FP16 `0.808151`
are a failed local recovery, not a falsification of the paper. The missing exact
recipe and much smaller continued-pretraining budget remain material.

## BitNet Text Embeddings

[BitNet Text Embeddings](https://arxiv.org/abs/2606.25674) applies a similar
idea to Qwen3-0.6B and Gemma3-270M encoders: normalization refinement,
continued contrastive pretraining, supervised contrastive learning, similarity
distribution distillation, and attention-relation distillation. The official
[BitNet repository](https://github.com/microsoft/BitNet) now releases embedding
models and reports optimized x86 I2_S inference.

Implication: this strengthens the narrower product thesis. Embeddings,
rerankers, classifiers, and ASR components have bounded objectives and can
justify task-specific adaptation. They are a more credible initial product than
a universal chat-model converter.

## PT2-LLM

[PT2-LLM](https://arxiv.org/abs/2510.03267), accepted at ICLR 2026, combines:

- an asymmetric ternary quantizer;
- iterative ternary fitting of grids and rounding assignments;
- activation-aware grid alignment;
- GPTQ-style Hessian error compensation;
- structural-similarity column reordering.

Its [Apache-2.0 implementation](https://github.com/XIANGLONGYAN/PT2-LLM)
contains calibration, perplexity, and seven-task zero-shot evaluation code.

This is not equivalent to the local diagonal-Hessian LS initializer. Our test
only changed a local ternary fit before training and failed downstream quality.
PT2-LLM jointly uses asymmetric levels, activation alignment, blockwise error
feedback, and reordering. It must be benchmarked directly before drawing a
broader PTQ conclusion.

Runtime implication: asymmetric positive and negative levels cannot generally
be encoded as one symmetric row scale times `{-1,0,+1}`. A successful PT2 model
may require a new group descriptor and kernel.

## CAT-Q and ScaleQ-1.58

[CAT-Q](https://arxiv.org/abs/2606.26650), an ICML 2026 oral, uses learnable
weight/threshold modulation and softened ternarization. It reports conversion
from only 512 calibration samples across dense models and models up to 235B
parameters. The open [BitTern toolkit](https://github.com/IntelChina-AI/BitTern)
now advertises checkpoints, evaluation, and packed deployment code.

[ScaleQ-1.58](https://arxiv.org/abs/2608.01078) adds Attend to Your Own Thoughts
(AYOT): calibration contexts contain reasoning traces and answers generated by
the full-precision target model itself. The paper reports Qwen3 dense and MoE
experiments from 1.7B through 235B with 4M default calibration tokens, plus
llama.cpp deployment examples. It also reports that smaller models and MoE
models are more sensitive to ternarization than larger dense models.

Critical boundaries:

- ScaleQ-1.58 is a recent preprint; this repository has not independently
  reproduced it.
- Its default is W1.58A16, not the W1.58A8 BitNet path tested locally.
- Its MoE evidence is Qwen3 MoE, not Kimi/DeepSeek MLA compatibility.
- Reported conversion still consumes hours on eight A100-80GB GPUs; “PTQ” does
  not mean zero optimization cost.
- Group-wise learned scales and thresholds require a matching storage and
  kernel contract.

Implication: CAT-Q/ScaleQ is now the highest-priority external baseline. If its
released Qwen3 artifacts retain quality under our independent suite, the
project should integrate their representation rather than continue tuning the
weaker local BitDistill controller.

## TWLA

[TWLA](https://arxiv.org/abs/2606.13054), accepted at ICML 2026, targets W1.58A4
with an asymmetric ternary quantizer, Kronecker orthogonal shaping, and
inter-layer-aware activation mixed precision. Its
[implementation](https://github.com/Kishon-zzx/TWLA) is public.

Implication: activation precision is a separate design axis. If W1.58A4 quality
holds independently, the runtime opportunity is larger than this fork's A8
path, but the rotations and mixed-precision policy require a different graph
and kernel implementation.

## PTQTP and Other Two-Bit Methods

[PTQTP](https://arxiv.org/abs/2509.16989) represents a matrix as two ternary
planes. This has greater expressive capacity than one ternary plane but costs
roughly two trits per weight before scale overhead. It should be compared with
2-bit methods, not marketed as the same 1.58-bit storage contract.

[D2Quant](https://arxiv.org/abs/2602.02546) and related 2-bit methods show that
down projections and layer-specific deviations can dominate low-bit error.
They support a representation ladder rather than a ternary-or-FP binary choice:

```text
single symmetric ternary
asymmetric/group-scale ternary
dual ternary planes or 2-bit integers
Q4
FP16/BF16
```

The product should choose a Pareto point per layer or model, while reporting
the actual end-to-end bit cost.

## BITCOS

[Breaking the 1.58-bit Barrier for Ternary LLMs](https://arxiv.org/abs/2609.16338)
is a September 2026 preprint introducing BITCOS. It observes nonuniform ternary
symbols, with zeros reaching `51.5%` in the surveyed models, and stores a dense
presence bitmap plus compacted signs. The reported cost is `2 - z` bits per
weight at zero fraction `z`, with AVX-512, AVX2, and Intel Xe2 unpacking paths.

This is directly relevant to the Xeon Silver 4116 and to our finding that I2
arithmetic dominates the classifier kernel. It is not a quality-recovery
method. BITCOS should be tested only after a model passes the quality gate, and
against current upstream kernels under matched affinity and warm-up conditions.

## What the Literature Changes

### Superseded conclusion

```text
All post-training ternarization is infeasible.
```

The local experiments never proved that universal statement, and 2026 results
make it especially indefensible.

### Current conclusion

```text
Naive symmetric absmean PTQ fails catastrophically in the tested Qwen setup.
Modern optimized PTQ is plausible, method-dependent, and not yet independently
validated in this repository or mapped to its CPU runtime contracts.
```

### Highest-value research question

```text
Under a common quality suite, calibration budget, effective-bit accounting,
and CPU runtime, which pretrained models are best served by optimized PTQ,
task-specific QAT/distillation, or native ternary training?
```

That question is both useful and publishable because current papers use
different models, tasks, activation precisions, compute budgets, and kernels.

## Required Benchmark Matrix

| Axis | Required controls |
| --- | --- |
| Model | Start with Qwen3-1.7B; add Qwen3-4B only after the harness is stable |
| Methods | FP16, Q4_K_M, naive absmean, PT2-LLM, CAT-Q, ScaleQ calibration, best local QAT |
| General language | WikiText-2 and C4 PPL |
| Zero-shot | PIQA, ARC-e, ARC-c, HellaSwag, WinoGrande, OpenBookQA, BoolQ |
| Reasoning | Math-500, GSM8K, HumanEval+, MBPP+ with fixed decoding |
| Cost | calibration tokens, accelerator type/count, wall time, peak memory |
| Representation | code symbols, scales/shifts, group size, activation precision, total file bits |
| CPU | file size, peak RSS, prompt/decode tok/s, energy when instrumentation is available |
| Statistics | aligned examples, paired intervals, repeated pinned timings, artifact hashes |

The first milestone is independent checkpoint quality. Runtime engineering
starts only for methods that survive that gate.
