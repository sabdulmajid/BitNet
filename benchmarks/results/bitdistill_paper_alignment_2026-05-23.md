# BitDistill Paper Alignment Audit

Generated: `2026-05-23T18:40:11.967414+00:00`

Status: **not_exact_reproduction**.

Quality claim: **paper_alignment_not_new_benchmark**.

The local work is a paper-inspired Qwen2.5 MNLI reproduction-gap study with several implemented BitDistill components. It is not an exact paper reproduction.

## Alignment Matrix

| axis | paper recipe | local state | status | risk |
| --- | --- | --- | --- | --- |
| Goal | Task-specific finetuning of FP LLMs into 1.58-bit BitNet models. | The repo is now framed as task-specific ternary distillation plus runtime-contract testing. | aligned | Do not revert to a universal arbitrary converter claim. |
| Backbone | Qwen3 0.6B/1.7B/4B primary; Qwen2.5-0.5B and Gemma ablations. | Primary controlled reproduction is Qwen/Qwen2.5-0.5B; dense Qwen2.5-1.5B used for PTQ/runtime boundary study. | paper_ablation_backbone_not_primary_backbone | Good for Qwen2.5 ablation alignment, not exact Qwen3 main-table reproduction. |
| Tasks | MNLI, QNLI, SST2, and CNNDM. | Current controlled BitDistill gate is MNLI first; QNLI/SST2 are intentionally gated; CNNDM not run. | partial | Do not claim full GLUE/CNNDM reproduction. |
| Baselines | FP16-SFT, BitNet-SFT, and BitDistill. | MNLI FP16-SFT 0.808151; BitNet-SFT best 0.628935; BitDistill latest 0.720020. | mnli_present | BitDistill is still below FP; QNLI/SST2 baselines should wait for MNLI recovery gate. |
| Stage-1 SubLN | Insert SubLN before attention output projection and FFN down projection. | Source check passed: True; active Stage-2 USE_SUBLN=True. | implemented | Source presence does not itself prove paper-level optimization behavior. |
| Ternary weight quantization | Per-tensor absmean W1.58 in the paper equation. | Active controlled Stage-2 scale_mode=tensor; row-scale work is separately labeled retrofit variant. | active_tensor_matches_paper_equation | Row-scale I2_SR results must not be labeled as standard BitDistill. |
| Activation quantization | 8-bit activation quantization. | Active Stage-2 activation_quantization=True. | matched_in_active_gate | Kernel/runtime parity still needs same-artifact proof for product claims. |
| Stage-2 corpus | 10B tokens sampled from FALCON corpus. | Local Slurm defaults use HuggingFaceFW/fineweb-edu sample-10BT unless overridden. | mismatch | Corpus mismatch is a plausible reproduction gap and must be named. |
| Stage-2 token budget | 10B continued-pretraining tokens. | Completed controlled row 327,680,000 tokens (3.2768% of paper); active gate targets 655,360,000 tokens (6.5536% of paper). | under_budget_active_extension_running | Current non-reproduction cannot disprove paper-scale BitDistill. |
| Stage-3 loss terms | CE + logits KL + attention-relation KD over Q/K/V. | Source check passed: True. | implemented | Loss terms exist, but normalization and gradient balance remain suspect. |
| Logits distillation temperature | Temperature 5.0. | Active handoff downstream recipe sets LOGIT_TEMPERATURE=5.0. | matched_for_active_downstream | Only applies after handoff/downstream job runs. |
| Attention-relation coefficient | Classification uses large attention-KD coefficient; exact meaning depends on reductions. | Paper-gamma local telemetry showed attention/CE imbalance; gamma status is pending_gamma60_telemetry. | normalization_not_proven_equivalent | Copying gamma numerically is not enough unless reductions match. |
| Attention layer selection | Distill a single selected layer, often later layers. | Active downstream handoff uses DISTILL_LAYER=-1. | matched_strategy | Layer choice still requires sweep/replication evidence. |
| Sequence length | Max sequence length 512 for GLUE setup. | Active Stage-2/downstream configuration uses max_seq_len=512. | matched | Padding/tokenization details still need exact paper parity if claiming reproduction. |
| Batch size | Batch size 32. | Active Stage-2 effective local batch is 16 (4 per device x grad_accum 4). | mismatch_or_unproven | Optimizer dynamics may differ from paper setup. |
| Hardware | 8x AMD MI300X servers for paper experiments. | Local active gate is single midcard GPU; CPU runtime measured on Xeon Silver 4116. | mismatch | Throughput and feasible token budget are not paper-comparable. |
| Success criterion | BitDistill comparable to FP16-SFT on downstream tasks. | Latest completed MNLI delta vs FP16 is -0.088130; configured recovery gate is -0.010000. | not_met | No public claim of paper-level reproduction. |

## Highest-Risk Mismatches

- Stage-2 token budget is far below 10B and the 655M gate is still running.
- Stage-2 corpus differs from the paper's FALCON corpus unless explicitly overridden.
- Attention-KD coefficient equivalence is not proven because loss reductions may differ.
- Batch size/hardware differ materially from paper conditions.
- QNLI/SST2/CNNDM are not yet paper-level reproduction rows.

## Source Artifacts

| artifact | path |
| --- | --- |
| reproduction_gap | benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json |
| stage2_submission | benchmarks/results/stage2_655m_submission_2026-05-23.json |
| traceability | benchmarks/results/bitdistill_goal_traceability_2026-05-23.json |
| next_decision | benchmarks/results/bitdistill_next_decision_2026-05-23.json |
