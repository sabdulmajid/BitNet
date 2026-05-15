# BitDistill Paper Alignment Audit, 2026-05-15

Verdict: local code contains the major BitDistill mechanisms, but the completed results are not a strict paper reproduction. The strict paper-hyperparameter branch is tracked separately.

Full-evaluation contract: `{'mnli': 9815, 'qnli': 5463, 'sst2': 872}` examples. Accuracy rows expose whether each metric is full validation or partial.

## Alignment

| dimension | paper | local | status | note |
| --- | --- | --- | --- | --- |
| Backbone | Qwen3 0.6B/1.7B/4B primary, plus Qwen2.5/Gemma robustness | Qwen2.5-0.5B | partial | Useful robustness-style target, but not the paper's primary Qwen3 scale ladder. |
| Tasks | MNLI, QNLI, SST2 first for classification | MNLI, QNLI, SST2 | matched | Task set is aligned. |
| Baselines | FP16-SFT, BitNet-SFT, BitDistill | FP16-SFT, BitNet-SFT, and gamma=100 long-warmup BitDistill exist for GLUE3; strict paper-gamma candidates are tracked separately | partial | The completed gamma=100 branch improves over BitNet-SFT but remains below the FP16-gap target; strict paper-hyperparameter rows determine the strict reproduction gate. |
| Stage-1 SubLN | SubLN before attention output projection and FFN down projection | Implemented | matched | Implemented as RMSNorm wrappers around Qwen `o_proj` and `down_proj`. |
| Stage-2 warm-up | 10B-token continued pretraining | active target 163840000 token presentations | partial | Current target is 0.016384 of the paper token budget. |
| Stage-3 logits KD | temperature 5, lambda 10 | temperature 5, weight 10, no tau^2 scaling by default | matched | First completed wave used tau^2; current code and pending runs use paper-style scaling. |
| Stage-3 attention KD | single-layer Q/K/V relation KD, gamma 1e5 for classification | single-layer L2-normalized Q/K/V relation KD implemented with paper-style Q/K/V sum by default; completed runs include gamma 100 on GLUE3 plus MNLI tensor probes at gamma 1e3/1e4/1e5; paper-gamma row, LR search, headinit, and MNLI layer-sweep branches remain tracked separately | pending | The gamma sweep is intentional because local loss-scale probes show the paper gamma can dominate CE; queued jobs use the corrected paper-style Q/K/V reduction through the default. |
| Hyperparameter search | greedy search over learning rate and epochs | fixed 1000-step downstream schedule plus queued LR 1e-5/2e-5/5e-5 and output-head initialization diagnostics | pending | The local search is intentionally narrow; a strict paper reproduction still needs epoch/budget search if the queued LR candidates do not close the gap. |
| Hardware/resources | 8x AMD MI300X training, CPU throughput with 16 threads | single-GPU Slurm jobs; Xeon CPU runtime for local inference | partial | Resource gap affects training budget and wall-clock, not the mathematical objective. |

## Warm-Up Budget

| field | value |
| --- | --- |
| paper warm-up tokens | 10000000000 |
| active target tokens | 163840000 |
| active target / paper | 0.016384 |
| active effective tokens | 163840000 |
| active effective / paper | 0.016384 |
| latest step | 20000 |
| max steps | 20000 |

## Current Accuracy Matrix

| task | run | family | exists | accuracy | examples | expected | full eval | FP16 | FP16 full eval | FP-run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | FP16-SFT | baseline | true | 0.807641 | 9815 | 9815 | true | 0.807641 | true | 0.000000 |
| mnli | BitNet-SFT | baseline | true | 0.487621 | 9815 | 9815 | true | 0.807641 | true | 0.320020 |
| mnli | BitDistill short tensor gamma100 | diagnostic | true | 0.525217 | 9815 | 9815 | true | 0.807641 | true | 0.282425 |
| mnli | BitDistill short row gamma100 | diagnostic | true | 0.516556 | 9815 | 9815 | true | 0.807641 | true | 0.291085 |
| mnli | BitDistill short tensor layer -8 | diagnostic | true | 0.535711 | 9815 | 9815 | true | 0.807641 | true | 0.271931 |
| mnli | BitDistill longwarmup tensor gamma100 | diagnostic_gamma100 | true | 0.641671 | 9815 | 9815 | true | 0.807641 | true | 0.165970 |
| mnli | BitDistill longwarmup row gamma100 | novelty_gamma100 | true | 0.653591 | 9815 | 9815 | true | 0.807641 | true | 0.154050 |
| mnli | BitDistill longwarmup tensor paper gamma | paper_candidate | true | 0.630260 | 9815 | 9815 | true | 0.807641 | true | 0.177382 |
| mnli | BitDistill longwarmup row paper gamma | paper_row_candidate | true | 0.617626 | 9815 | 9815 | true | 0.807641 | true | 0.190015 |
| mnli | BitDistill longwarmup tensor paper gamma lr1e-5 | paper_lr_search_pending | true | 0.604381 | 9815 | 9815 | true | 0.807641 | true | 0.203260 |
| mnli | BitDistill longwarmup tensor paper gamma lr5e-5 | paper_lr_search_pending | false | - | - | 9815 | false | 0.807641 | true | - |
| mnli | BitDistill longwarmup tensor paper gamma headinit | paper_headinit_pending | false | - | - | 9815 | false | 0.807641 | true | - |
| mnli | BitDistill longwarmup tensor gamma1k | mnli_gamma_sweep_pending | true | 0.647275 | 9815 | 9815 | true | 0.807641 | true | 0.160367 |
| mnli | BitDistill longwarmup tensor gamma10k | mnli_gamma_sweep_pending | true | 0.635354 | 9815 | 9815 | true | 0.807641 | true | 0.172287 |
| mnli | BitDistill longwarmup tensor layer -1 | mnli_layer_sweep_pending | true | 0.645950 | 9815 | 9815 | true | 0.807641 | true | 0.161691 |
| mnli | BitDistill longwarmup tensor layer -2 | mnli_layer_sweep_pending | true | 0.642894 | 9815 | 9815 | true | 0.807641 | true | 0.164748 |
| mnli | BitDistill longwarmup tensor layer -4 | mnli_layer_sweep_pending | true | 0.640754 | 9815 | 9815 | true | 0.807641 | true | 0.166887 |
| qnli | FP16-SFT | baseline | true | 0.898957 | 5463 | 5463 | true | 0.898957 | true | 0.000000 |
| qnli | BitNet-SFT | baseline | true | 0.596925 | 5463 | 5463 | true | 0.898957 | true | 0.302032 |
| qnli | BitDistill short tensor gamma100 | diagnostic | true | 0.596925 | 5463 | 5463 | true | 0.898957 | true | 0.302032 |
| qnli | BitDistill short row gamma100 | diagnostic | true | 0.618525 | 5463 | 5463 | true | 0.898957 | true | 0.280432 |
| qnli | BitDistill short tensor layer -8 | diagnostic | false | - | - | 5463 | false | 0.898957 | true | - |
| qnli | BitDistill longwarmup tensor gamma100 | diagnostic_gamma100 | true | 0.787846 | 5463 | 5463 | true | 0.898957 | true | 0.111111 |
| qnli | BitDistill longwarmup row gamma100 | novelty_gamma100 | true | 0.796998 | 5463 | 5463 | true | 0.898957 | true | 0.101959 |
| qnli | BitDistill longwarmup tensor paper gamma | paper_candidate | true | 0.759656 | 5463 | 5463 | true | 0.898957 | true | 0.139301 |
| qnli | BitDistill longwarmup row paper gamma | paper_row_candidate | true | 0.760937 | 5463 | 5463 | true | 0.898957 | true | 0.138019 |
| qnli | BitDistill longwarmup tensor paper gamma lr1e-5 | paper_lr_search_pending | true | 0.757459 | 5463 | 5463 | true | 0.898957 | true | 0.141497 |
| qnli | BitDistill longwarmup tensor paper gamma lr5e-5 | paper_lr_search_pending | false | - | - | 5463 | false | 0.898957 | true | - |
| qnli | BitDistill longwarmup tensor paper gamma headinit | paper_headinit_pending | false | - | - | 5463 | false | 0.898957 | true | - |
| sst2 | FP16-SFT | baseline | true | 0.925459 | 872 | 872 | true | 0.925459 | true | 0.000000 |
| sst2 | BitNet-SFT | baseline | true | 0.770642 | 872 | 872 | true | 0.925459 | true | 0.154817 |
| sst2 | BitDistill short tensor gamma100 | diagnostic | true | 0.815367 | 872 | 872 | true | 0.925459 | true | 0.110092 |
| sst2 | BitDistill short row gamma100 | diagnostic | true | 0.808486 | 872 | 872 | true | 0.925459 | true | 0.116972 |
| sst2 | BitDistill short tensor layer -8 | diagnostic | false | - | - | 872 | false | 0.925459 | true | - |
| sst2 | BitDistill longwarmup tensor gamma100 | diagnostic_gamma100 | true | 0.866972 | 872 | 872 | true | 0.925459 | true | 0.058486 |
| sst2 | BitDistill longwarmup row gamma100 | novelty_gamma100 | true | 0.854358 | 872 | 872 | true | 0.925459 | true | 0.071101 |
| sst2 | BitDistill longwarmup tensor paper gamma | paper_candidate | true | 0.841743 | 872 | 872 | true | 0.925459 | true | 0.083716 |
| sst2 | BitDistill longwarmup row paper gamma | paper_row_candidate | true | 0.837156 | 872 | 872 | true | 0.925459 | true | 0.088303 |
| sst2 | BitDistill longwarmup tensor paper gamma lr1e-5 | paper_lr_search_pending | true | 0.846330 | 872 | 872 | true | 0.925459 | true | 0.079128 |
| sst2 | BitDistill longwarmup tensor paper gamma lr5e-5 | paper_lr_search_pending | false | - | - | 872 | false | 0.925459 | true | - |
| sst2 | BitDistill longwarmup tensor paper gamma headinit | paper_headinit_pending | false | - | - | 872 | false | 0.925459 | true | - |

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
| attention_relation_l2_normalization | pass |
| attention_qkv_sum_default | pass |
| single_layer_selection | pass |
| row_scale_mode | pass |
| longwarmup_submitter | pass |
| strict_paper_gamma_gate | pass |
| strict_paper_gamma_row_gate | pass |

## Interpretation

- The completed gamma=100 long-warmup GLUE result is a valid negative reproduction-boundary result.
- It is not a disproof of BitDistill because full hyperparameter search, backbone scale, and 10B-token warm-up are not paper-matched yet.
- The publishable angle remains independent reproduction plus a row-scale CPU-runtime extension if the remaining row/search branches close the quality gap; otherwise the publishable angle becomes a resource-sensitivity and boundary study.
