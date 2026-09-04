# Native I2_SR Sequence-Classification Quality Equivalence

Generated: `2026-09-04T06:46:27.105491+00:00`

Status: **task_quality_preserved_for_artifact**.

## Paired Full-Split Result

| Metric | Result |
| --- | ---: |
| MNLI examples | `9815` |
| PyTorch correct / accuracy | `6415` / `0.653591` |
| native I2_SR correct / accuracy | `6401` / `0.652165` |
| native minus PyTorch accuracy | `-0.001426` |
| paired normal 95% CI | `[-0.004193, 0.001341]` |
| paired bootstrap 95% CI | `[-0.004177, 0.001325]` |
| runtime wins / PyTorch wins | `89` / `103` |
| exact McNemar p | `0.348171` |
| exact prediction agreement | `0.976668` |
| retrospective 0.5-point non-inferiority | `pass` |

## Interpretation

On all 9,815 MNLI validation examples, native I2_SR loses 14 net correct predictions relative to the saved PyTorch trace. The paired 95% interval includes zero, exact McNemar does not reject equal marginal accuracy, and the retrospective 0.5-point non-inferiority criterion passes. The runtime therefore preserves task accuracy for this artifact within the measured uncertainty, despite failing strict prediction identity.

## Claim Boundary

- This supports task-quality preservation for this one row-scale MNLI artifact, not numerical equivalence or general model equivalence.
- The 0.5-point non-inferiority margin was selected retrospectively and is labeled as such; the paired interval and raw discordance counts are the primary evidence.
- The reference trace was produced by GPU BF16 inference while I2_SR ran on CPU integer kernels, so exact prediction identity is not expected.
- The underlying ternary checkpoint remains far below the FP16 task model; runtime preservation does not repair model quality.
- Multi-prompt batching is still excluded. This result uses the verified sequence-isolated token-ID path.
