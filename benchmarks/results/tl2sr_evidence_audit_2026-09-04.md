# TL2_SR Evidence Audit

Generated: `2026-09-04T14:28:10.940787+00:00`. Status: **valid_runtime_no_speed_win**.

## Verdict

TL2_SR is a functionally valid row-scale ternary storage/runtime contract for the tested Qwen student. It reduces packed projection storage, but no tested tile layout proves a CPU throughput advantage over I2_SR.

## Correctness And Quality

| evidence | result |
| --- | ---: |
| generated kernel contracts passed | 3/3 |
| generated kernel cases passed | 18/18 |
| 512-sample I2_SR accuracy | 0.667969 |
| 512-sample TL2_SR accuracy | 0.667969 |
| cross-format prediction agreement | 0.988281 |
| I2-only / TL2-only correct | 3 / 3 |
| exact McNemar p | 1.000000 |

## Storage

| artifact region | I2_SR MiB | TL2_SR MiB | reduction |
| --- | ---: | ---: | ---: |
| complete GGUF | 352.617 | 341.495 | 3.154% |
| packed ternary projections | 86.478 | 75.355 | 12.862% |

## Xeon Tiling Sweep

| BM | mean TL2_SR tok/s | paired speed / I2_SR | ratio 95% CI |
| ---: | ---: | ---: | ---: |
| 128 | 211.988 | 0.853 | [0.848, 0.858] |
| 64 | 210.406 | 0.866 | [0.835, 0.897] |
| 32 | 228.638 | 0.919 | [0.917, 0.921] |

The speed ratios are paired within each build and must not be compared through their absolute
throughput across builds. The full-validation field remains separate from the 512-example
same-build format comparison.

## Gates

- **all kernel contracts:** pass
- **layout guard:** pass
- **sample accuracy within one point:** pass
- **sample prediction agreement at least 98 percent:** pass
- **projection storage reduced:** pass
- **repeated benchmarks valid:** pass
- **repeated benchmarks idle gated:** pass
- **kernel layout receipt matches:** pass
- **conversion output hashes match:** pass
- **artifact receipts match:** pass
- **speed superiority proven:** fail
- **full validation complete:** pass
- **full accuracy within one point:** pass
- **full prediction agreement at least 98 percent:** pass

## Full Validation

| evidence | result |
| --- | ---: |
| examples | 9815 |
| I2_SR accuracy | 0.651452 |
| TL2_SR accuracy | 0.652878 |
| TL2_SR minus I2_SR | +0.001426 |
| paired delta 95% bootstrap CI | [-0.000917, +0.003872] |
| cross-format prediction agreement | 0.982578 |
| I2-only / TL2-only correct | 65 / 79 |
| exact McNemar p | 0.278615 |

## Provenance

The JSON companion records SHA-256 identities for every generated kernel header/config,
conversion receipt, validation trace, and repeated benchmark consumed by this audit.
All correctness, identity, and full-validation gates must pass before this report is publication evidence.
