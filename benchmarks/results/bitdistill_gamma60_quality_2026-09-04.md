# Full MNLI Gamma-60 Quality Audit

Generated: `2026-09-04T06:35:56.955065+00:00`

Status: **matched_historical_control**.

## Result

In the matched historical 163.84M Stage-2 comparison, the run declaring local attention-KD coefficient 60 improves MNLI over the run declaring 100,000 by `+0.047275` with paired 95% CI `[0.039256, 0.055293]`. The exact step-1 execution fingerprint and matching serialized contract make this strong local evidence that loss-scale alignment matters, pending source-pinned seeded replication.

| Comparison | Candidate | Reference | Delta | Paired 95% CI | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| matched fixed gamma 163m | `0.738462` | `0.691187` | `+0.047275` | `[0.039256, 0.055293]` | `9.0707e-31` |
| fixed gamma 655m | `0.738462` | `0.729903` | `+0.008558` | `[0.000919, 0.016197]` | `0.0300297` |
| fp16 | `0.738462` | `0.808151` | `-0.069689` | `[-0.078431, -0.060947]` | `1.10193e-54` |

## Contract

The matched 163.84M comparison holds the checkpoint, model, task, 10,000-step budget, head initialization, tensor-scale W1.58A8 path, SubLN surgery, relation-head split, and all available serialized training-budget fields fixed. The logs declare only the attention-KD coefficient change from `100,000` to `60`. Their step-1 CE, logits-KD, and attention-KD values are exactly identical, providing an execution fingerprint for the same initialization and first batch.

## Limitations

- This is one historical run pair; neither metrics file serialized seed or source revision.
- The paired intervals measure validation-example uncertainty conditional on fixed checkpoints.
- Gamma 60 is implementation-specific and is not evidence that the paper's coefficient is wrong.
- The comparison against 655.36M changes both Stage-2 budget and gamma; it is not the one-axis test.
