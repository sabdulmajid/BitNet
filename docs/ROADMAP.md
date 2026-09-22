# Research Roadmap

Last updated: 2026-09-22

## Objective

Build an independent, CPU-first benchmark that determines whether a pretrained
model should be deployed with optimized ternary PTQ, task-specific
QAT/distillation, a richer 2-bit code, Q4, or full precision. A method advances
only when one artifact passes both quality and systems gates.

## Decision Already Made

Stop investing in the adaptive attention-loss controller. In the completed
three-seed matched MNLI comparison, adaptive minus fixed-60 is `-0.001698` with
95% CI `[-0.010898, 0.007502]`. It neither improves quality nor closes the FP16
gap. Preserve it as an ablation; do not make it the default.

Do not scale the unchanged fixed-gamma BitDistill recipe to 10B token
presentations. The measured Stage-2 curve has sharply diminishing gains and the
unchanged recipe remains below the preregistered recovery floor.

## Phase 1: Advanced PTQ Reproduction

### Canonical model

- Primary: Qwen3-1.7B, because CAT-Q/ScaleQ publish this scale and it is small
  enough for repeated evaluation.
- Secondary: Qwen3-4B, only after the primary harness is deterministic.
- Existing Qwen2.5-1.5B remains the historical naive-PTQ control, not the sole
  basis for judging newer methods.

### Arms

| Arm | Purpose |
| --- | --- |
| FP16/BF16 | Quality and memory reference |
| Q4_K_M | Mature commodity-CPU baseline |
| Symmetric absmean ternary | Destructive control |
| PT2-LLM | Asymmetric, activation-aware, Hessian-compensated PTQ |
| CAT-Q | Learned modulation and softened ternarization |
| ScaleQ-1.58 | CAT-Q with self-generated reasoning calibration |
| Local best QAT | Training-based comparison, clearly labeled non-equivalent |

Every external repository must be commit-pinned, license-recorded, and run in
an isolated environment. Released checkpoints are evaluated before spending
compute on reproduction; checkpoint and local-conversion results remain
separate rows.

### Quality protocol

- WikiText-2 and C4 perplexity with identical tokenization and block selection.
- Seven zero-shot tasks: PIQA, ARC-e, ARC-c, HellaSwag, WinoGrande, OpenBookQA,
  and BoolQ.
- Math-500, GSM8K, HumanEval+, and MBPP+ for the ScaleQ branch.
- Fixed decoding configuration and full task splits unless a pilot is labeled
  explicitly.
- Store aligned predictions or generations, not only aggregate scores.

### Quality gate

A representation becomes a runtime candidate only when all conditions hold:

1. finite PPL on both corpora;
2. at least `95%` retention of the FP mean on the seven zero-shot tasks;
3. no individual task loses more than `10` absolute accuracy points;
4. no catastrophic repetition or invalid-output failure on the generation
   suite;
5. all artifacts, effective-bit accounting, and calibration costs are complete.

These are project decision thresholds, not universal definitions of acceptable
quality. Full results remain public even when an arm fails.

### Compute ladder

1. Evaluate published checkpoints.
2. Run a 128-sample calibration smoke test.
3. Reproduce the documented default calibration.
4. Replicate the winning arm with three calibration seeds or sample sets.

This order prevents expensive reproduction before basic compatibility and
quality are established.

## Phase 2: Representation Audit

For each quality-surviving arm, record:

- actual code alphabet and entropy;
- positive/negative levels, zero points, scales, shifts, and group size;
- all dense fallback tensors and embedding/output-head precision;
- activation precision and runtime transforms;
- payload bits, metadata bits, alignment padding, and complete file bytes;
- zero density by layer and projection type.

The output is a lossless mathematical contract. A method is not called
“1.58-bit” in product tables unless the complete effective-bit calculation is
shown. Two trit planes, dense outliers, and FP scales are counted.

### Runtime mapping decision

| Representation | Likely path |
| --- | --- |
| Symmetric per-row ternary | Existing `I2_SR` |
| Symmetric group-scale ternary | Extend scale indexing and packing |
| Asymmetric positive/negative levels | New two-scale group descriptor and dot kernel |
| Rotated W1.58A4 | Add transform and A4 graph support |
| High-zero-density ternary | Prototype BITCOS presence/sign layout |
| Two trit planes | Treat as a separate 3.17+ bpw family |

## Phase 3: Xeon CPU Pareto Test

Target hardware: Intel Xeon Silver 4116, 12 physical cores, AVX-512BW/DQ/F/VL,
AVX2, BMI2, and FMA.

### Protocol

- Pin to physical cores `0-11` and record sibling/core topology.
- Require idle preflight on pinned cores and siblings.
- Interleave candidate/reference order and use at least five paired repetitions.
- Warm model and kernels before timing.
- Report prompt and decode separately.
- Record GGUF bytes, peak RSS, context length, tokens, power/energy when RAPL is
  available, binary hash, model hash, compiler, and ISA dispatch.
- Verify logits or paired task predictions against the source checkpoint.

### Runtime gate

A ternary artifact is a useful CPU Pareto point only if:

1. task quality still passes the Phase-1 gate after export;
2. prediction agreement with the source checkpoint is at least `99%`, or any
   lower agreement is justified by a paired non-inferiority interval;
3. complete file size is smaller than Q4_K_M;
4. decode speed ratio versus Q4_K_M is at least `1.10x` and the paired 95%
   interval lower bound exceeds `1.0`;
5. peak RSS is lower than Q4_K_M for the same context;
6. no unsupported dense fallback dominates the claimed format.

Failure is still a publishable boundary result, but not a product claim.

## Phase 4: Task-Specific Product

If general-language quality fails while task adaptation succeeds, pivot to one
bounded product:

- text embeddings/retrieval;
- intent or document classification;
- reranking;
- domain-specific summarization with a fixed evaluation contract.

The first recommended target is text embeddings because BitNet Text Embeddings
provides a current external baseline and CPU I2_S path. The product must compare
against FP16, int8, and Q4 encoders on MTEB quality, vectors/second, RSS, file
size, and index-storage precision.

## Phase 5: MoE Boundary

Start with Qwen3-30B-A3B because CAT-Q/ScaleQ publish an MoE reference. Do not
start with Kimi.

Required milestones:

1. dense advanced-PTQ gate passes;
2. expert, shared-expert, and router tensors have explicit quantization rules;
3. routing decisions match the FP checkpoint at a measured rate;
4. expert activation histograms and locality traces are recorded;
5. CPU memory policy distinguishes resident, mmap, and paged experts;
6. quality is evaluated on the same tasks as the dense reference;
7. end-to-end throughput includes routing and page faults.

Kimi becomes eligible only after its architecture-specific MLA/KV metadata,
tokenizer, shared experts, routed experts, and checkpoint dtypes have explicit
conversion and runtime tests. A tiny synthetic MoE fixture is not sufficient.

## Publication Gates

### Artifact report

Ready after Phase 1. It can publish the naive-PTQ negative result, matched
BitDistill gap, row-scale contract, and independent optimized-PTQ comparison.

### Systems paper

Requires one quality-valid advanced-PTQ artifact, an exact representation
contract, at least two CPU architectures, energy measurements, and a meaningful
Pareto improvement over Q4.

### Algorithm paper

Requires a new method that beats reproduced CAT-Q/PT2/ScaleQ controls under
equal calibration and effective-bit budgets, with model-family and scale
generalization. Row-scale alone is no longer enough novelty.

### MoE paper

Requires a real open MoE checkpoint, router/expert fidelity, memory-locality
analysis, and end-to-end CPU results. Kimi claims require actual Kimi evidence.

## Product Milestones

| Milestone | Deliverable | Exit criterion |
| --- | --- | --- |
| M1 | Evidence compiler | One command emits quality, cost, format, and hardware manifest |
| M2 | Advanced-PTQ evaluator | Reproduced PT2/CAT-Q/ScaleQ matrix on Qwen3-1.7B |
| M3 | Runtime adapter | Quality-surviving artifact executes with source-checkpoint fidelity |
| M4 | CPU Pareto report | Pinned repeated comparison against Q4_K_M and FP16 |
| M5 | Task appliance | One embedding/classification model meets a declared product SLO |
| M6 | MoE feasibility | Qwen3 MoE routing, quality, memory, and throughput all measured |

## Immediate Work Queue

1. Pin and audit PT2-LLM and BitTern licenses, commits, model formats, and
   published checkpoints.
2. Add an external-checkpoint adapter to the existing PPL and `lm-eval`
   harness without changing method implementations.
3. Evaluate released Qwen3-1.7B artifacts before launching calibration jobs.
4. Freeze a common dataset manifest and generation configuration.
5. Publish Phase-1 preregistration, then run the calibration ladder.
6. Rebase runtime comparisons against current upstream BitNet/llama.cpp before
   attributing novelty to `I2_SR` or `TL2_SR`.
7. Prototype BITCOS only if a quality-valid artifact has sufficient zero
   density to beat five-trit packing after metadata and alignment.
