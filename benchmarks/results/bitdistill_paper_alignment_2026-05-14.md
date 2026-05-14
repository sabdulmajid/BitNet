# BitDistill Paper Alignment Audit, 2026-05-14

Verdict: local code contains the major BitDistill mechanisms, but the completed results are not a strict paper reproduction. The strict paper-hyperparameter branch is queued/pending.

## Alignment

| dimension | paper | local | status | note |
| --- | --- | --- | --- | --- |
| Backbone | Qwen3 0.6B/1.7B/4B primary, plus Qwen2.5/Gemma robustness | Qwen2.5-0.5B | partial | Useful robustness-style target, but not the paper's primary Qwen3 scale ladder. |
| Tasks | MNLI, QNLI, SST2 first for classification | MNLI, QNLI, SST2 | matched | Task set is aligned. |
| Baselines | FP16-SFT, BitNet-SFT, BitDistill | All three exist for short-budget GLUE3; long-warmup BitDistill is pending | partial | The final paper candidate is not complete until long-warmup downstream metrics exist. |
| Stage-1 SubLN | SubLN before attention output projection and FFN down projection | Implemented | matched | Implemented as RMSNorm wrappers around Qwen `o_proj` and `down_proj`. |
| Stage-2 warm-up | 10B-token continued pretraining | active target 163840000 token presentations | partial | Current target is 0.016384 of the paper token budget. |
| Stage-3 logits KD | temperature 5, lambda 10 | temperature 5, weight 10, no tau^2 scaling by default | matched | First completed wave used tau^2; current code and pending runs use paper-style scaling. |
| Stage-3 attention KD | single-layer Q/K/V relation KD, gamma 1e5 for classification | single-layer Q/K/V KD implemented; completed runs gamma 100; long-warmup gamma 1e3/1e4/1e5 branches pending | pending | The gamma sweep is intentional because local loss-scale probes show the paper gamma can dominate CE. |
| Hyperparameter search | greedy search over learning rate and epochs | fixed 1000-step downstream schedule plus selected diagnostics | partial | A strict reproduction needs at least a small LR/epoch search after the long warm-up. |
| Hardware/resources | 8x AMD MI300X training, CPU throughput with 16 threads | single-GPU Slurm jobs; Xeon CPU runtime for local inference | partial | Resource gap affects training budget and wall-clock, not the mathematical objective. |

## Warm-Up Budget

| field | value |
| --- | --- |
| paper warm-up tokens | 10000000000 |
| active target tokens | 163840000 |
| active target / paper | 0.016384 |
| active effective tokens | 59392000 |
| active effective / paper | 0.005939 |
| latest step | 7250 |
| max steps | 20000 |

## Current Accuracy Matrix

| task | run | family | exists | accuracy | FP16 | FP-run |
| --- | --- | --- | --- | --- | --- | --- |
| mnli | FP16-SFT | baseline | true | 0.807641 | 0.807641 | 0.000000 |
| mnli | BitNet-SFT | baseline | true | 0.487621 | 0.807641 | 0.320020 |
| mnli | BitDistill short tensor gamma100 | diagnostic | true | 0.525217 | 0.807641 | 0.282425 |
| mnli | BitDistill short row gamma100 | diagnostic | true | 0.516556 | 0.807641 | 0.291085 |
| mnli | BitDistill short tensor layer -8 | diagnostic | true | 0.535711 | 0.807641 | 0.271931 |
| mnli | BitDistill longwarmup tensor gamma100 | diagnostic_pending | false | - | 0.807641 | - |
| mnli | BitDistill longwarmup row gamma100 | novelty_pending | false | - | 0.807641 | - |
| mnli | BitDistill longwarmup tensor paper gamma | paper_candidate | false | - | 0.807641 | - |
| mnli | BitDistill longwarmup tensor gamma1k | mnli_gamma_sweep_pending | false | - | 0.807641 | - |
| mnli | BitDistill longwarmup tensor gamma10k | mnli_gamma_sweep_pending | false | - | 0.807641 | - |
| qnli | FP16-SFT | baseline | true | 0.898957 | 0.898957 | 0.000000 |
| qnli | BitNet-SFT | baseline | true | 0.596925 | 0.898957 | 0.302032 |
| qnli | BitDistill short tensor gamma100 | diagnostic | true | 0.596925 | 0.898957 | 0.302032 |
| qnli | BitDistill short row gamma100 | diagnostic | true | 0.618525 | 0.898957 | 0.280432 |
| qnli | BitDistill short tensor layer -8 | diagnostic | false | - | 0.898957 | - |
| qnli | BitDistill longwarmup tensor gamma100 | diagnostic_pending | false | - | 0.898957 | - |
| qnli | BitDistill longwarmup row gamma100 | novelty_pending | false | - | 0.898957 | - |
| qnli | BitDistill longwarmup tensor paper gamma | paper_candidate | false | - | 0.898957 | - |
| sst2 | FP16-SFT | baseline | true | 0.925459 | 0.925459 | 0.000000 |
| sst2 | BitNet-SFT | baseline | true | 0.770642 | 0.925459 | 0.154817 |
| sst2 | BitDistill short tensor gamma100 | diagnostic | true | 0.815367 | 0.925459 | 0.110092 |
| sst2 | BitDistill short row gamma100 | diagnostic | true | 0.808486 | 0.925459 | 0.116972 |
| sst2 | BitDistill short tensor layer -8 | diagnostic | false | - | 0.925459 | - |
| sst2 | BitDistill longwarmup tensor gamma100 | diagnostic_pending | false | - | 0.925459 | - |
| sst2 | BitDistill longwarmup row gamma100 | novelty_pending | false | - | 0.925459 | - |
| sst2 | BitDistill longwarmup tensor paper gamma | paper_candidate | false | - | 0.925459 | - |

## Code Feature Checks

| feature | status |
| --- | --- |
| subln_wrapper | pass |
| subln_o_proj | pass |
| subln_down_proj | pass |
| continued_pretrain_stage | pass |
| sequence_classification_stage | pass |
| logits_kd | pass |
| paper_logit_temperature_scale_default | pass |
| attention_relation_kd | pass |
| single_layer_selection | pass |
| row_scale_mode | pass |
| longwarmup_submitter | pass |
| strict_paper_gamma_gate | pass |

## Interpretation

- The existing negative GLUE result is a valid short-budget boundary result.
- It is not a disproof of BitDistill because warm-up budget, attention-KD gamma, backbone scale, and search are not paper-matched yet.
- The publishable angle remains independent reproduction plus a row-scale CPU-runtime extension if the strict branch closes the quality gap; otherwise the publishable angle becomes a resource-sensitivity and boundary study.
