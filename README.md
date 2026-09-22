# BitNet Retrofit Lab

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Evidence: audited](https://img.shields.io/badge/evidence-audited-2f855a.svg)](benchmarks/results/current_evidence_2026-09-22.md)
[![Status: research](https://img.shields.io/badge/status-research-orange.svg)](docs/RESEARCH_STATUS.md)

An evidence-first testbed for converting pretrained language models to ternary
weights and running them on commodity CPUs. This fork combines quantization
experiments, BitDistill-style training, packed GGUF formats, native classifier
execution, and paired statistical audits.

## Verdict

> **No, naive FP16/BF16 to ternary conversion is not an acceptable universal
> retrofit.** In the tested Qwen setup, one-shot absmean rounding destroys
> general-language quality. Training or calibration under the discrete
> constraint is necessary, and the deployed format must preserve the scale
> semantics learned by that process.

That conclusion is narrower than "ternary PTQ is impossible." Newer optimized
methods such as [PT2-LLM](https://arxiv.org/abs/2510.03267),
[CAT-Q](https://arxiv.org/abs/2606.26650), and
[ScaleQ-1.58](https://arxiv.org/abs/2608.01078) directly challenge naive PTQ
with asymmetric grids, learned modulation, and task-aware calibration. The next
phase of this project is an independent, apples-to-apples test of those methods.

## Current Evidence

| Question | Result | What it establishes |
| --- | --- | --- |
| Can absmean PTQ retrofit Qwen2.5-1.5B? | **Rejected** | WikiText PPL `13.901` -> `3,813,121.803`; ten-task mean `0.644169` -> `0.348671`. |
| Does ternary QAT recover quality? | **Partly** | Best row-scale ten-task mean `0.499459`, `+0.150788` over PTQ but `-0.144710` below FP. |
| Did the local BitDistill study recover FP16 MNLI? | **No** | Best matched arm mean `0.757039` versus FP16 `0.808151`. |
| Did adaptive loss balancing beat fixed `gamma=60`? | **Inconclusive** | Mean delta `-0.001698`; seed-level paired 95% CI `[-0.010898, 0.007502]`. |
| Are row scales part of the model contract? | **Yes** | Collapsing to one scale gives relative RMS error `1.904230`; exact row scales give `0.000197`. |
| Does packed ternary beat FP16 for the classifier workload? | **No** | `I2_SR/FP16 = 0.650x` throughput on the Xeon 4116. |
| Is Kimi/MoE deployment proven? | **No** | Only tiny Qwen2MoE plumbing exists; no Kimi quality or routed CPU benchmark exists. |

Machine-readable source of truth:
[current evidence JSON](benchmarks/results/current_evidence_2026-09-22.json) and
[human summary](benchmarks/results/current_evidence_2026-09-22.md). Both record
the source artifacts and SHA-256 hashes.

## Dense Retrofit Result

The original hypothesis was tested on Qwen2.5-1.5B with held-out perplexity and
ten `lm-eval` tasks.

| Model path | WikiText PPL | Ten-task mean | Interpretation |
| --- | ---: | ---: | --- |
| FP reference | `13.901` | `0.644169` | Quality anchor |
| Naive tensor-absmean PTQ | `3,813,121.803` | `0.348671` | Catastrophic collapse |
| Best row-scale QAT | `38.580` | `0.499459` | Real recovery, still not competitive |

This rejects the **specific** one-click absmean converter. It does not reject
optimized PTQ algorithms that learn grids, thresholds, rotations, or
calibration-dependent corrections.

## Matched BitDistill Result

The strongest local task result uses Qwen2.5-0.5B on all `9,815` MNLI
`validation_matched` examples. Adaptive and fixed arms share the same Stage-2
checkpoint, objective, schedule, data, and three training seeds.

| Arm | Mean accuracy | Seed-level 95% t-CI | Gap to FP16 |
| --- | ---: | ---: | ---: |
| FP16-SFT | `0.808151` | - | - |
| Adaptive attention weight | `0.755340` | `[0.750811, 0.759870]` | `-0.052811` |
| Fixed attention weight `60` | `0.757039` | `[0.751434, 0.762643]` | `-0.051112` |

Adaptive minus fixed is `-0.001698`, with seed-level paired 95% CI
`[-0.010898, 0.007502]`. The preregistered superiority and paper-recovery gates
both fail. With only three seeds, the fixed-simplicity gate also remains
underpowered, so the formal method decision is **inconclusive**. There is no
evidence to promote the adaptive controller.

The [matched audit](benchmarks/results/bitdistill_adaptive_vs_fixed_matched_audit_2026-09-22.md)
and [compact prediction bundle](benchmarks/results/bitdistill_matched_prediction_bundle_2026-09-22.md)
retain enough public data to recompute every accuracy and paired test.

## CPU Results

These are different artifacts and workloads; causal decode and isolated
classification must not be conflated.

### Causal Qwen2.5-1.5B

| Format | File MiB | PPL | Prompt tok/s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| F16 | `2950.4` | `12.2808` | `114.47` | `5.56` |
| Q4_K_M | `940.4` | `12.8112` | `92.08` | `16.01` |
| Row-scale `I2_SR` | `1211.3` | `38.8477` | `211.67` | `19.07` |

`I2_SR` proves that learned row-scale ternary semantics can survive packing and
execute quickly. It is **not** a Q4 replacement here: Q4_K_M is smaller and far
closer to FP quality.

### Sequence Classification on Xeon Silver 4116

Four interleaved, affinity-pinned runs with 12 physical cores give:

| Artifact | Size | Throughput / FP16 | 95% CI |
| --- | ---: | ---: | ---: |
| `I2_SR` student | `352.62 MiB` | `0.650x` | `[0.646, 0.653]` |
| `I2_SR` + Q8 embedding | `230.90 MiB` | `0.605x` | `[0.603, 0.607]` |

The mixed artifact is `4.106x` smaller than the FP16 classifier, but ternary
arithmetic is slower for this short, sequence-isolated workload. Profiling
attributes `94.51%` of projection cost to packed I2 GEMM and only `5.49%` to A8
activation quantization; optimizing the activation prepass cannot fix the gap.

Experimental `TL2_SR` preserves outputs (`18/18` kernel cases) and reduces
projection storage by `12.862%`, but every tested tile is slower than `I2_SR`:
BM128 `0.853x`, BM64 `0.866x`, BM32 `0.919x`. This is a useful rejected
optimization, not a speed claim.

## What This Fork Contributes

- A reproducible negative result for naive ternary retrofit, with full
  perplexity and ten-task evaluation rather than generation anecdotes.
- An independent BitDistill-style Qwen implementation: SubLN, continual
  pretraining, logits KL, Q/K/V relation distillation, layer selection, loss
  telemetry, and sequence-classification or causal formulations.
- The row-scale `I2_SR` training/export/runtime contract and native Qwen2
  classifier-head execution in the linked
  [llama.cpp fork](https://github.com/sabdulmajid/llama.cpp).
- Experimental `TL2_SR`, including exact-shape code generation, layout guards,
  paired quality checks, and a negative speed result.
- Fail-closed reports, artifact hashes, aligned prediction traces, paired
  confidence intervals, and explicit claim boundaries.

BitDistill, BitNet, and optimized ternary PTQ are prior work. The adaptive
controller is not a demonstrated contribution. The defensible original systems
result is that **retrofit-specific scale semantics must be represented by the
packed runtime**, plus the measured boundary where that representation does and
does not pay off.

## Literature Update

The field moved materially in 2026:

- [CAT-Q](https://arxiv.org/abs/2606.26650) reports learned modulation and
  softened ternarization from 512 calibration samples; its open
  [BitTern toolkit](https://github.com/IntelChina-AI/BitTern) now includes
  checkpoints and packed deployment code.
- [ScaleQ-1.58](https://arxiv.org/abs/2608.01078) adds self-generated reasoning
  traces to CAT-Q calibration and reports dense and Qwen3 MoE results through
  235B parameters. It is a preprint and must be independently reproduced.
- [PT2-LLM](https://arxiv.org/abs/2510.03267) is an ICLR 2026 asymmetric,
  activation-aware, Hessian-compensated PTQ method with Apache-2.0 code.
- [TWLA](https://arxiv.org/abs/2606.13054) is an ICML 2026 W1.58A4 method using
  asymmetric ternarization, rotations, and mixed-precision activation planning.
- [BitNet Text Embeddings](https://arxiv.org/abs/2606.25674) strengthens the
  task-specific product thesis: continual task adaptation and distillation are
  a credible route for CPU embedding appliances.
- [BITCOS](https://arxiv.org/abs/2609.16338) exploits nonuniform ternary symbol
  frequencies with an AVX-512/AVX2-friendly sparse layout. This is directly
  relevant to the Xeon runtime after model quality is solved.

See [the literature review](docs/LITERATURE_REVIEW.md) for method boundaries,
publication status, and what each paper changes in this project.

## Next Phase

The next experiment is not another local loss-weight sweep. It is a controlled
advanced-PTQ falsification study:

1. Reproduce released CAT-Q/ScaleQ and PT2-LLM checkpoints on a common Qwen3
   backbone and calibration corpus.
2. Compare FP16, naive absmean, PT2-LLM, CAT-Q/ScaleQ, Q4_K_M, and the best local
   QAT artifact on the same PPL, seven-task zero-shot, and reasoning suites.
3. Report true end-to-end bits per weight, file size, RSS, prompt/decode speed,
   calibration GPU-hours, and paired quality deltas.
4. Port only a quality-surviving representation to CPU. Test current TQ/I2,
   row/group-scale, and BITCOS-style storage on the Xeon with pinned repetitions.
5. Attempt Qwen3 MoE only after the dense gate; Kimi remains out of scope until
   MLA, routing, expert locality, and checkpoint conversion are all validated.

Pre-registered decision rules and the product/publication path are in
[ROADMAP.md](docs/ROADMAP.md). The full interpretation is in
[RESEARCH_STATUS.md](docs/RESEARCH_STATUS.md).

## Reproduce and Validate

```bash
python3 benchmarks/build_current_evidence_snapshot.py \
  --created-utc 2026-09-22T00:00:00+00:00
python3 benchmarks/validate_public_docs.py
python3 -m pytest -q
```

Training, Slurm, export, and historical audit commands remain in
[EXPERIMENTS.md](EXPERIMENTS.md). Runtime semantics are specified in
[RUNTIME_CONTRACT.md](RUNTIME_CONTRACT.md); reporting rules are in
[REPORTING.md](REPORTING.md).

## Repository Map

| Path | Purpose |
| --- | --- |
| [docs/RESEARCH_STATUS.md](docs/RESEARCH_STATUS.md) | What has been proved, rejected, and left open |
| [docs/LITERATURE_REVIEW.md](docs/LITERATURE_REVIEW.md) | Current literature and implications |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Next experiments and decision gates |
| [CLAIMS.md](CLAIMS.md) | Concise public claim ledger |
| [benchmarks/](benchmarks/) | Evaluation, audit, export, and evidence builders |
| [benchmarks/results/](benchmarks/results/README.md) | Curated index of public, sanitized reports |
| [experiments/](experiments/) | Mathematical probes and synthetic tests |
| [3rdparty/llama.cpp](3rdparty/llama.cpp) | Row-scale runtime fork |

## Scope

This is a research fork of [microsoft/BitNet](https://github.com/microsoft/BitNet),
not an official Microsoft project and not a universal converter. The upstream
BitDistill training-code request remains
[open](https://github.com/microsoft/BitNet/issues/354), so this implementation
must be described as independent and paper-inspired until exact equivalence is
demonstrated.
