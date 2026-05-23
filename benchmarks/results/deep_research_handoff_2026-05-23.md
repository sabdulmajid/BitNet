# Deep Research Handoff

Generated: `2026-05-23T18:33:30.696732+00:00`

Status: **handoff_not_completion**.

## Thesis

- Original question: Can arbitrary pretrained FP16/BF16 models be post-hoc converted to BitNet-style W1.58A8 CPU inference?
- Current answer: No for the tested dense-Qwen setup; blind ternary PTQ collapses.
- Redirected question: Can task-specific ternary students be trained from pretrained teachers, and can CPU formats preserve the scale semantics those students learn?
- Core interpretation: Extreme ternary quantization is representation learning plus a runtime contract, not only compression or file conversion.

## Completed Findings

| finding | evidence | interpretation |
| --- | --- | --- |
| Blind ternary PTQ is a strong negative result in the tested dense-Qwen setup. | FP PPL 13.901 vs naive PTQ PPL 3813121.803; FP ten-task mean 0.644169 vs PTQ 0.348671. | The FP weight geometry is not preserved by a blind ternary projection. |
| QAT/distillation recovers signal but not FP quality. | Best row-scale QAT mean 0.499459; recovery over PTQ +0.150788; gap to FP -0.144710. | Training can move some function into the ternary family, but current runs do not close the gap. |
| BitDistill paper-level recovery remains governed by the latest completed Stage-2 row. | FP16-SFT MNLI 0.808151; latest 327.68M BitDistill 0.720020; delta -0.088130; status not_reproduced. | The local implementation remains below the paper recovery gate; more Stage-2 budget is being tested. |
| The earlier weak BitNet-SFT baseline was mostly undertraining, not the main blocker. | default BitNet-SFT 0.487621; best budget row 0.628935; delta vs paper anchor +0.020935. | The remaining problem is BitDistill recovery/loss dynamics, not merely BitLinear replacement. |
| Row-scale semantics are material to the learned function. | TL2 one-scale output RMS error 1.904230; exact row-scale RMS error 0.000197. | A row-scale ternary student represents W approximately as s_row times T, so scales are model semantics. |
| I2_SR is a working row-scale packed CPU path but not a Q4 replacement. | I2_SR file 1211.3 MiB, PPL 38.8477, prompt 211.67 tok/s, decode 19.07 tok/s; Q4_K_M PPL 12.8112, file 940.4 MiB. | The systems path is real, but quality/storage tradeoffs remain unfavorable versus mature Q4. |

## Novelty Boundary

| classification | description |
| --- | --- |
| Not novel | BitDistill as a concept: SubLN, continued pretraining, logits KD, and attention-relation KD are Microsoft paper contributions. |
| Potentially novel | Independent reproduction-gap study with fail-closed artifacts and paired evidence for where local BitDistill diverges. |
| Potentially novel | Row-scale ternary retrofit variant and the measured requirement that runtime formats preserve row-scale semantics. |
| Potentially novel | I2_SR packed CPU runtime extension for compatible row-scale causal artifacts. |
| Potentially novel | Boundary study separating task quality, LM perplexity, file size, RSS, prompt speed, and decode speed. |

## Active 655M Gate

| field | value |
| --- | --- |
| downstream_complete | False |
| downstream_status | waiting_for_handoff |
| eta_hours | 17.279719 |
| latest_ce | 3.394341 |
| latest_complete_snapshot_step | - |
| latest_step | 5780 |
| max_steps | 40000 |
| progress | 0.144500 |
| stage2_job_id | 10250 |
| stage2_slurm_state | RUNNING |
| stage2_status | running |
| telemetry_job_id | 10257 |
| telemetry_slurm_state | PENDING |

## Next Action Policy

| field | value |
| --- | --- |
| decision_status | pending_655m_downstream |
| recommendation | Wait for the active 655.36M Stage-2 producer, downstream MNLI, and postprocess reports. |
| blueprint_action | wait_and_watch_655m_gate |
| runnable_now | True |
| claim_boundary | status only; quality_claim remains none until ingestion is ingested_reports_rebuilt |
| required_evidence | 655M Stage-2 manifest, 655M downstream metrics.json, 655M downstream eval_predictions.jsonl, rebuilt controlled curve and next-decision report |
| commands | python benchmarks/run_active_gate_watchdog.py ; python benchmarks/audit_stage2_655m_ingestion.py |

## Open Research Questions

| question | evidence needed | current state |
| --- | --- | --- |
| Does the Stage-2 token-budget curve keep improving at 655.36M tokens? | Completed 655M Stage-2 manifest plus downstream MNLI metrics.json and eval_predictions.jsonl. | running; step 5780/40000; downstream waiting_for_handoff. |
| Is the remaining BitDistill gap mostly compute budget or loss-normalization mismatch? | 655M/longer budget curve and gamma-balanced component-gradient telemetry. | paper-gamma grad attention/CE 221.384986; gamma-60 telemetry queued. |
| Can the same artifact provide both quality and CPU runtime evidence? | Packed classifier or causal prompt-scoring artifact with task quality, RSS, file size, and throughput. | native classifier MNLI 0.652165, agreement 0.976668; not product-ready. |
| Do row-scale variants help generally or only in specific retrofit regimes? | Controlled tensor/row/group-scale comparisons across tasks/backbones with paired confidence intervals. | Row-scale runtime contract is strong; row-scale accuracy is not a universal guarantee. |
| Is MoE/Kimi feasible in this runtime path? | Real routed model mapping, expert layout, trained quality, and CPU expert-selection benchmarks. | Only tiny Qwen2MoE fixture/plumbing exists; no Kimi quality or routed CPU runtime is proven. |

## Nonclaims

- universal BitNet converter
- paper-level BitDistill reproduction
- Q4-quality I2_SR replacement
- Kimi/MoE runtime support

## Publishable Angles

- negative blind-ternary-PTQ result for tested dense Qwen models
- independent BitDistill reproduction-gap study
- row-scale ternary runtime-contract evidence
- I2_SR packed CPU row-scale extension for compatible causal artifacts
- boundary study separating task quality, LM perplexity, RSS, file size, and throughput

## Source Artifacts

| artifact | path | sha256 |
| --- | --- | --- |
| current_status | benchmarks/results/current_goal_status_2026-05-23.json | 6dd9314f74112ae1ab584321dbb4f69c69818e6e4699b94a86d024f06a475fb7 |
| canonical_bundle | benchmarks/results/canonical_evidence_bundle_2026-05-20.json | af9ec2e35931986c7caf63c178b7c482c3e93406f8d880774bbf8d114f27824c |
| reproduction_gap | benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json | b5a37266b33dc7318b55a23569673467d11fa7aa67ba6725baaa374210a42820 |
| next_decision | benchmarks/results/bitdistill_next_decision_2026-05-23.json | 34d4ceed2107b9df0fce4b919233ccf504e094253728454271d9cfd089e65ad4 |
| next_experiment_blueprint | benchmarks/results/bitdistill_next_experiment_blueprint_2026-05-23.json | 0c0eeb79ffb87014676d2cbb1cb38d1e5104909a7539c5b0d2051f8a8389bf22 |
