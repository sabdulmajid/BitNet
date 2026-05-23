# BitDistill Reproduction Gap Report

Status: **not reproduced**. This report separates the now-improved BitNet-SFT baseline from the remaining BitDistill/FP recovery gap.

| metric | value |
| --- | --- |
| FP16-SFT MNLI | 0.808151 |
| BitNet-SFT default MNLI | 0.487621 |
| BitNet-SFT best MNLI | 0.628935 |
| BitNet-SFT best vs default | 0.141314 |
| BitNet-SFT best vs paper anchor | 0.020935 |
| BitNet-SFT best vs FP16 | -0.179215 |
| BitDistill 40.96M MNLI | 0.616607 |
| BitDistill 163.84M MNLI | 0.691187 |
| BitDistill 327.68M MNLI | 0.720020 |
| BitDistill 327.68M vs BitNet-SFT best | 0.091085 |
| BitDistill 327.68M vs FP16 | -0.088130 |
| BitDistill 327.68M CI95 | [-0.0967494587122408, -0.07951136655520699] |
| 327.68M as paper Stage-2 fraction | 3.2768% |
| final grad attention/CE | 221.384986 |
| final loss attention/CE | 2549.206537 |
| controlled telemetry traces | 2 |

## Conclusions

| finding | evidence |
| --- | --- |
| The short BitNet-SFT default was undertrained. | default 0.487621; best budget row 0.628935; gain +0.141314 |
| The local BitNet-SFT anchor is no longer the primary blocker. | best BitNet-SFT 0.628935; paper BitNet-SFT anchor 0.608000; delta +0.020935 |
| BitDistill is still not reproduced. | 327.68M BitDistill MNLI 0.720020; FP16 0.808151; delta -0.088130 |
| Stage-2 budget helps, but current budget is still small relative to the paper. | 40.96M 0.616607; 163.84M 0.691187; 327.68M 0.720020; paper fraction 3.2768% |
| Local paper-gamma training dynamics are still suspect. | final grad attention/CE 221.384986; final loss attention/CE 2549.206537 |

## Next Gates

| gate | why | minimum next point |
| --- | --- | --- |
| Stage-2 token-budget curve | Determine whether MNLI continues improving toward FP or saturates far below it. | 655.36M cumulative token presentations with the same downstream recipe |
| Loss-normalization/gradient-balance sweep | Paper gamma is only comparable if CE, logits KD, and attention KD reductions match. | component-gradient telemetry for gamma near equalized and paper values |
| Same-artifact runtime quality | The strongest PyTorch classifier result and strongest packed causal runtime are still separate artifacts. | packed classifier head or primary causal prompt-scoring evaluation |
| Backbone alignment | Paper-scale claims need exact/closest public Qwen3/Qwen2.5 recipe alignment. | one Qwen3-0.6B or exact Qwen2.5-0.5B MNLI run with matched logging |

## Artifact Inventory

| label | path | sha256 |
| --- | --- | --- |
| bitnet_sft_budget | benchmarks/results/bitnet_sft_budget_sweep_2026-05-23.json | b9bec4ae74fbadf2e82488f7be53fbb2fb31ddeb375a61d42070bc93248191f6 |
| canonical_bundle | benchmarks/results/canonical_evidence_bundle_2026-05-20.json | af9ec2e35931986c7caf63c178b7c482c3e93406f8d880774bbf8d114f27824c |
| controlled_curve | benchmarks/results/bitdistill_controlled_curve_2026-05-20.json | d892817f387ff52f1105439e6b7a7c2417d7c0124b0d0dbf56adb3f7585f6356 |
| training_dynamics | benchmarks/results/bitdistill_training_dynamics_2026-05-23.json | 8861b2102e7b06f5967ae9497303bc66febab6d6a18ee21f553d8b0d5a57ca39 |
