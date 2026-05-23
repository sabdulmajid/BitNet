# BitDistill Goal Traceability Audit

Generated: `2026-05-23T18:15:10.415885+00:00`

Quality claim: **traceability_from_existing_artifacts_not_new_benchmark**.

Objective achieved: **False**.

Completion status: **in_progress**.

The original universal retrofit thesis is disproven for the tested dense-Qwen setup. The active goal is now a bounded BitDistill/row-scale runtime study, and it is not complete until the 655M downstream gate and gamma-balance telemetry land.

## Live State

| job id | name | state | time | reason |
| --- | --- | --- | --- | --- |
| 10250 | bd-s2-655m | RUNNING | 2:37:45 | ece-nebula12 |
| 10255 | bd-655m-handoff | PENDING | 0:00 | (Dependency) |
| 10257 | bd-g60-telemetry | PENDING | 0:00 | (Dependency) |

| stage2 field | value |
| --- | --- |
| latest_step | 5170 |
| max_steps | 40000 |
| progress | 0.129250 |
| latest_ce | 3.931800 |
| recent_ce_mean | 3.613525 |
| eta_hours | 17.587915 |
| log_path | logs/bd-s2-655m-10250.out |

## Requirement Traceability

| requirement | status | proof strength | evidence | remaining gap | next action |
| --- | --- | --- | --- | --- | --- |
| Post-training ternary math audit | proven_negative_for_tested_dense_qwen | strong empirical plus analytic probe | FP PPL 13.901475; PTQ PPL 3813121.803327; FP ten-task mean 0.644169; PTQ mean 0.348671; math test present True | The claim is scoped to tested dense Qwen checkpoints, not every possible architecture. | Keep this as the headline negative result; do not market universal conversion. |
| BitLinear/SubLN implementation | implemented_alignment_still_under_quality_audit | source evidence plus training artifacts | SubLN source check True; BitNet-SFT best budget row 0.628935; default row 0.487621 | Implementation exists, but paper-level BitDistill recovery is not proven. | Use 655M and gamma telemetry to decide whether to continue budget scaling or audit recipe alignment. |
| Stage-2 continued pretraining | active_extension_running | completed 327.68M row plus live 655.36M job | completed latest tokens 327,680,000; latest MNLI 0.720020; live job RUNNING; live step 5170; ETA hours 17.59 | 655.36M downstream MNLI and paired prediction trace are pending. | Wait for job 10250, handoff 10255, and postprocess before changing experiment axes. |
| Stage-3 downstream CE + logits KL + attention-relation KD | implemented_but_not_reproduced | source evidence plus MNLI curve | loss source check True; FP16-SFT MNLI 0.808151; 327.68M BitDistill MNLI 0.720020; delta -0.088130 | Not within the 0.5-1.0 point FP recovery target. | Do not expand to claims; use the next decision gate after 655M/gamma evidence. |
| MNLI/QNLI/SST2 paper-style baseline reproduction | partial_mnli_focused_not_complete | MNLI controlled rows and earlier GLUE audits | FP16-SFT MNLI 0.808151; BitNet-SFT best MNLI 0.628935; BitDistill latest MNLI 0.720020; scoreboard status mixed_supported_and_blocked | QNLI/SST2 should be run only after a credible MNLI recovery row or recipe fix. | Gate QNLI/SST2 on the MNLI recovery decision to avoid wasting compute. |
| Row-scale novelty vs paper-style tensor scale | supported_as_retrofit_variant | paired quality evidence plus runtime contract | row-scale QAT mean 0.499459; recovery vs PTQ +0.150788; row-scale RMS 0.000197; one-scale RMS 1.904230 | Row scale is not standard BitDistill and does not close FP gap yet. | Keep row-scale results labeled as retrofit-variant systems work. |
| I2_SR export and CPU benchmarking on Xeon | working_not_q4_quality_competitive | CPU benchmark artifact | I2_SR file 1211.3 MiB; PPL 38.8477; prompt 211.67; decode 19.07; Q4 PPL 12.8112 | Same-artifact task quality plus product-ready packed runtime remains unsolved. | Decide whether product target is packed classifier or causal prompt scorer. |
| At least ten benchmark comparisons | complete_for_existing_qwen15b_boundary_study | coverage gate | quality benchmarks 12; lm-eval tasks 10; coverage checks 108; failed checks 0 | These benchmarks do not include the active 655M BitDistill row. | Append 655M downstream evidence after postprocess, do not preclaim it. |
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

- Whether 655.36M cumulative Stage-2 token presentations continue the MNLI recovery curve.
- Whether gamma-60 component-gradient telemetry fixes the local attention-KD imbalance.

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
| controlled_curve | benchmarks/results/bitdistill_controlled_curve_2026-05-20.json |
| product_scope | benchmark_results/product_scope_gate_2026-05-15.json |
| seqcls_gap | benchmark_results/seqcls_runtime_gap_2026-05-15.json |
| moe_support | benchmark_results/moe_support_audit_2026-05-15.json |
| next_decision | benchmarks/results/bitdistill_next_decision_2026-05-23.json |
