# BitDistill Goal Traceability Audit

Generated: `2026-09-04T03:53:38.149650+00:00`

Quality claim: **traceability_from_existing_artifacts_not_new_benchmark**.

Objective achieved: **False**.

Completion status: **in_progress**.

The original universal retrofit thesis is disproven for the tested dense-Qwen setup. The active goal is now a bounded BitDistill/row-scale runtime study. The completed 655M gate remains below FP16, so the next test is a matched gamma-balanced downstream run.

## Live State

| job id | name | state | time | reason |
| --- | --- | --- | --- | --- |
| 10250 |  | not_in_squeue |  |  |
| 10259 |  | not_in_squeue |  |  |
| 10257 |  | not_in_squeue |  |  |

| stage2 field | value |
| --- | --- |
| latest_step | 40000 |
| max_steps | 40000 |
| progress | 1.000000 |
| latest_ce | 3.426713 |
| recent_ce_mean | 3.463437 |
| eta_hours | 0.000000 |
| log_path | logs/bd-s2-655m-10250.out |

## Requirement Traceability

| requirement | status | proof strength | evidence | remaining gap | next action |
| --- | --- | --- | --- | --- | --- |
| Post-training ternary math audit | proven_negative_for_tested_dense_qwen | strong empirical plus analytic probe | FP PPL 13.901475; PTQ PPL 3813121.803327; FP ten-task mean 0.644169; PTQ mean 0.348671; math test present True | The claim is scoped to tested dense Qwen checkpoints, not every possible architecture. | Keep this as the headline negative result; do not market universal conversion. |
| BitLinear/SubLN implementation | implemented_alignment_still_under_quality_audit | source evidence plus training artifacts | SubLN source check True; BitNet-SFT best budget row 0.628935; default row 0.487621 | Implementation exists, but paper-level BitDistill recovery is not proven. | The Stage-2 marginal gain is weak and gamma-60 rebalances updates. Run a matched 10k-step downstream MNLI row with the balanced coefficient before spending more Stage-2 tokens. |
| Stage-2 continued pretraining | completed_655m_curve | completed 655.36M row with paired predictions | completed latest tokens 655,360,000; latest MNLI 0.729903; live job None; live step 40000; ETA hours 0.00 | The completed row remains 7.825 accuracy points below FP16. | The Stage-2 marginal gain is weak and gamma-60 rebalances updates. Run a matched 10k-step downstream MNLI row with the balanced coefficient before spending more Stage-2 tokens. |
| Stage-3 downstream CE + logits KL + attention-relation KD | implemented_but_not_reproduced | source evidence plus MNLI curve | loss source check True; FP16-SFT MNLI 0.808151; 655.36M BitDistill MNLI 0.729903; delta -0.078248 | Not within the 0.5-1.0 point FP recovery target. | The Stage-2 marginal gain is weak and gamma-60 rebalances updates. Run a matched 10k-step downstream MNLI row with the balanced coefficient before spending more Stage-2 tokens. |
| MNLI/QNLI/SST2 paper-style baseline reproduction | partial_mnli_focused_not_complete | MNLI controlled rows and earlier GLUE audits | FP16-SFT MNLI 0.808151; BitNet-SFT best MNLI 0.628935; BitDistill latest MNLI 0.729903; scoreboard status mixed_supported_and_blocked | QNLI/SST2 should be run only after a credible MNLI recovery row or recipe fix. | Gate QNLI/SST2 on the MNLI recovery decision to avoid wasting compute. |
| Row-scale novelty vs paper-style tensor scale | supported_as_retrofit_variant | paired quality evidence plus runtime contract | row-scale QAT mean 0.499459; recovery vs PTQ +0.150788; row-scale RMS 0.000197; one-scale RMS 1.904230 | Row scale is not standard BitDistill and does not close FP gap yet. | Keep row-scale results labeled as retrofit-variant systems work. |
| I2_SR export and CPU benchmarking on Xeon | working_not_q4_quality_competitive | CPU benchmark artifact | I2_SR file 1211.3 MiB; PPL 38.8477; prompt 211.67; decode 19.07; Q4 PPL 12.8112 | Same-artifact task quality plus product-ready packed runtime remains unsolved. | Decide whether product target is packed classifier or causal prompt scorer. |
| At least ten benchmark comparisons | complete_for_existing_qwen15b_boundary_study | coverage gate | quality benchmarks 12; lm-eval tasks 10; coverage checks 108; failed checks 0 | The 655M paired MNLI row is now included; broader task coverage remains gated on recovery. | Keep the 655M result in the controlled curve and preserve task-specific claim boundaries. |
| MoE/Kimi feasibility | not_supported_beyond_tiny_plumbing | negative scope audit | local Kimi artifacts 0; MoE product gates passed 6/9 | No real Kimi mapping, trained MoE quality, routed expert-locality benchmark, or product CPU runtime. | Keep MoE/Kimi in future work until dense path is resolved. |
| Product-ready packed sequence classification | research_demo_not_product_ready | native classifier audit | MNLI 0.652165; agreement 0.976668; sequence-isolated 7.456204 ex/s; token-id runner 2.724140 ex/s | Agreement below 0.99 product gate and quality weak. | Choose product surface after MNLI recovery result: classifier runtime or causal prompt-scoring evaluator. |
| Publishable framing | publishable_as_boundary_study_not_converter | claim ledger plus scoreboard | product scope research_mvp_only; supported claims 5; unsupported claims 4; scoreboard publishable True | Paper-level BitDistill reproduction and product artifact remain incomplete. | Frame as negative PTQ result plus row-scale runtime contract; keep stronger claims gated. |

## Source Checks

| check | path | passed | missing patterns |
| --- | --- | --- | --- |
| SubLN wrapper implemented | train_bitdistill.py | true | none |
| Stage-3 loss combines CE, logits KD, and attention KD | train_bitdistill.py | true | none |
| component-gradient telemetry exists | train_bitdistill.py | true | none |
| math viability test exists | experiments/math_viability_test.py | true | none |
| row-scale scoreboard exists | benchmarks/results/bitdistill_benchmark_scoreboard_2026-05-23.json | true | none |

## What Is Solved

- Blind ternary PTQ is rejected for tested dense Qwen.
- BitDistill-style components and telemetry exist in source.
- Row-scale runtime semantics are proven material for current row-scale checkpoints.
- I2_SR packed CPU runtime works for compatible dense causal artifacts.
- MoE/Kimi is correctly scoped as not supported beyond tiny fixtures.

## What Is Being Tested Now

- Whether gamma-60's measured gradient rebalance improves full 10k-step MNLI quality from the 655M checkpoint.
- Whether loss normalization, rather than more Stage-2 tokens, explains the remaining FP16 gap.

## Publishability

| field | value |
| --- | --- |
| framing | Publishable as an independent negative/positive boundary study and systems-contract prototype, not as a universal BitNet converter and not yet as a paper-level BitDistill reproduction. |
| main_blocker | The same artifact still does not jointly satisfy paper-level task quality, general-LM quality, mature Q4-level storage/quality tradeoffs, and product-ready packed runtime. |
| publishable | true |
| strongest_contribution | Blind PTQ failure plus row-scale runtime-contract evidence showing that trained ternary scale semantics must be preserved in CPU formats such as I2_SR. |

## Source Artifacts

| artifact | path |
| --- | --- |
| canonical_bundle | benchmarks/results/canonical_evidence_bundle_2026-05-20.json |
| scoreboard | benchmarks/results/bitdistill_benchmark_scoreboard_2026-05-23.json |
| reproduction_gap | benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json |
| controlled_curve | benchmarks/results/bitdistill_controlled_curve_2026-05-23.json |
| product_scope | benchmark_results/product_scope_gate_2026-05-15.json |
| seqcls_gap | benchmark_results/seqcls_runtime_gap_2026-05-15.json |
| moe_support | benchmark_results/moe_support_audit_2026-05-15.json |
| next_decision | benchmarks/results/bitdistill_next_decision_2026-05-23.json |
| handoff_submission | benchmarks/results/stage2_655m_handoff_submission_2026-05-23.json |
