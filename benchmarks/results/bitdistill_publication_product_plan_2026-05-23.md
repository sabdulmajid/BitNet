# BitDistill Publication and Product Plan

Generated: `2026-05-23T18:15:10.553383+00:00`

Quality claim: **planning_from_existing_artifacts_not_new_benchmark**.

Status: **research_mvp_with_pending_quality_gate**.

The work is publishable as a rigorous boundary study and systems-contract prototype, not as a universal BitNet converter and not yet as a complete BitDistill reproduction.

## Publishable Units

| unit | claim | evidence | publishable now | risk |
| --- | --- | --- | --- | --- |
| Negative PTQ result | Blind FP/BF16-to-ternary projection is not viable for the tested dense-Qwen setup. | FP WikiText PPL 13.901475; naive PTQ PPL 3,813,121.803327; FP ten-task mean 0.644169; PTQ mean 0.348671 | true | Scope must remain tested dense Qwen; do not universalize to every architecture. |
| Row-scale runtime contract | Row scales are learned model semantics and must be preserved by CPU formats. | one-scale TL2 RMS error 1.904230; exact row-scale RMS error 0.000197 | true | This supports I2_SR/row-scale contracts, not TL2 row-scale support. |
| I2_SR CPU runtime prototype | A compatible row-scale ternary causal artifact can run through packed CPU I2_SR. | I2_SR file 1211.3 MiB, PPL 38.8477, prompt 211.67 tok/s, decode 19.07 tok/s; Q4_K_M file 940.4 MiB, PPL 12.8112 | true | Do not claim Q4-quality or Q4-storage competitiveness. |
| BitDistill reproduction gap | Local BitDistill-style training is improving but has not reproduced paper-level GLUE quality. | MNLI 40.96M 0.616607; 163.84M 0.691187; 327.68M 0.720020; delta vs FP -0.088130 | true | Must frame as a reproduction-gap study until a within-1pt row is reproduced. |
| Product-ready classifier | Native packed sequence classification is not product-ready yet. | MNLI 0.652165; PyTorch agreement 0.976668; sequence-isolated 7.456204 ex/s; token-id runner 2.724140 ex/s | false | Agreement and quality are below gate; useful only as a research demo. |
| MoE/Kimi | MoE/Kimi support is not proven beyond tiny Qwen2MoE plumbing. | local Kimi artifacts 0; MoE product gates passed 6/9 | false | No real Kimi artifact, quality benchmark, or routed expert locality proof exists. |

## Product MVP

| field | value |
| --- | --- |
| name | CPU-first ternary retrofit evaluator |
| target_user | Engineers deciding whether a model-task pair is worth ternary distillation and CPU deployment. |
| value | It prevents false converter claims by producing a fail-closed decision report: quality delta, paired confidence intervals, PPL, file size, RSS, prompt/decode speed, runtime compatibility, and claim label. |
| input | A Hugging Face model, task/eval dataset, and target CPU/runtime profile. |
| output | Pass/fail evidence bundle plus suggested path: reject PTQ, try BitDistill/QAT, use row-scale I2_SR, or stop. |
| current_readiness | research_mvp_only |
| why_useful_now | The negative PTQ and row-scale runtime findings are already actionable even when the answer is no. |

## Decision Gates

| gate | status | decision rule | success condition | failure condition |
| --- | --- | --- | --- | --- |
| 655M Stage-2 downstream MNLI | pending_655m_downstream | Wait for the active 655.36M Stage-2 producer, downstream MNLI, and postprocess reports. | Meaningful marginal gain or within-1pt FP recovery gate. | Saturation far below FP, requiring recipe/loss audit instead of broader sweeps. |
| Gamma-60 component-gradient telemetry | pending_gamma60_telemetry | If gamma-60 rebalances attention/CE updates, run matched downstream MNLI. | Attention-KD gradient no longer dominates CE by the current local threshold. | Attention remains dominant, indicating deeper loss-normalization or recipe mismatch. |
| Same-artifact task quality plus CPU runtime | not_ready | Choose packed classifier runtime or causal prompt-scoring product surface after MNLI recovery. | One artifact provides task quality, agreement, RSS, file size, and throughput. | Quality proof remains PyTorch-only while runtime proof remains causal-only. |

## Paper Outline

| section | content |
| --- | --- |
| Problem | Extreme ternary retrofit is a representation-learning and runtime-contract problem, not a file conversion problem. |
| Negative result | Blind PTQ collapses: FP PPL 13.901475, PTQ PPL 3813121.803327, FP mean 0.644169, PTQ mean 0.348671. |
| Recovery path | Row-scale QAT recovers +0.150788 over PTQ but remains -0.144710 below FP on the current ten-task mean. |
| BitDistill reproduction | Current MNLI curve: 40.96M 0.616607, 163.84M 0.691187, 327.68M 0.720020; 655M pending. |
| Runtime contract | One-scale TL2 relative RMS error 1.904230; exact row-scale error 0.000197. |
| CPU results | I2_SR file 1211.3 MiB, PPL 38.8477, decode 19.07 tok/s; native classifier MNLI 0.652165 is not product-ready. |

## Claim Rules

| field | value |
| --- | --- |
| safe_headline | Blind ternary PTQ fails for tested dense Qwen; row-scale ternary students require matching CPU runtime semantics; BitDistill-style recovery is still under gate. |
| avoid | universal converter, lossless retrofit, paper-level BitDistill reproduced, I2_SR beats Q4 on quality/storage, Kimi/MoE supported |
| minimum_for_stronger_claim | 655M downstream paired trace and decision report, gamma-balance telemetry report, replicated within-1pt MNLI recovery row, QNLI/SST2 rows only after MNLI gate, same-artifact task quality and CPU runtime proof |

## Source Artifacts

| artifact | path |
| --- | --- |
| canonical_bundle | benchmarks/results/canonical_evidence_bundle_2026-05-20.json |
| scoreboard | benchmarks/results/bitdistill_benchmark_scoreboard_2026-05-23.json |
| traceability | benchmarks/results/bitdistill_goal_traceability_2026-05-23.json |
| product_scope | benchmark_results/product_scope_gate_2026-05-15.json |
| next_decision | benchmarks/results/bitdistill_next_decision_2026-05-23.json |
