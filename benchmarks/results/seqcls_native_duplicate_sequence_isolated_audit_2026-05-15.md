# Sequence-Classification Native I2_SR Duplicate-Prompt Batching Audit, 2026-05-15

This audit repeats the same rendered token-ID prompt within a single native llama-embedding batch. A mismatch here cannot be attributed to tokenizer round-trip differences, different text examples, or output-row swaps.

| field | value |
| --- | --- |
| status | pass |
| targets | [15, 35, 0, 1, 2] |
| repeat count | 4 |
| embedding sequential | true |
| same prompt repeated | true |
| all logits invariant | true |
| all predictions invariant | true |
| changed prediction count | 0 |
| max relative RMS vs alone | 0.000000 |
| formatting/tokenization ruled out | false |
| ready for batched product benchmark | false |
| ready for sequence-isolated product benchmark | true |

## Duplicate Positions

| target | position | pred | margin | rel RMS vs alone | pred matches alone | logits |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | 0 | 1 | 0.019177 | 0.000000 | true | [0.374451, 0.420180, 0.401003] |
| 15 | 1 | 1 | 0.019177 | 0.000000 | true | [0.374451, 0.420180, 0.401003] |
| 15 | 2 | 1 | 0.019177 | 0.000000 | true | [0.374451, 0.420180, 0.401003] |
| 15 | 3 | 1 | 0.019177 | 0.000000 | true | [0.374451, 0.420180, 0.401003] |
| 35 | 0 | 0 | 0.124395 | 0.000000 | true | [0.518326, 0.393931, 0.169148] |
| 35 | 1 | 0 | 0.124395 | 0.000000 | true | [0.518326, 0.393931, 0.169148] |
| 35 | 2 | 0 | 0.124395 | 0.000000 | true | [0.518326, 0.393931, 0.169148] |
| 35 | 3 | 0 | 0.124395 | 0.000000 | true | [0.518326, 0.393931, 0.169148] |
| 0 | 0 | 1 | 1.345433 | 0.000000 | true | [-0.546304, 1.516333, 0.170900] |
| 0 | 1 | 1 | 1.345433 | 0.000000 | true | [-0.546304, 1.516333, 0.170900] |
| 0 | 2 | 1 | 1.345433 | 0.000000 | true | [-0.546304, 1.516333, 0.170900] |
| 0 | 3 | 1 | 1.345433 | 0.000000 | true | [-0.546304, 1.516333, 0.170900] |
| 1 | 0 | 2 | 2.072655 | 0.000000 | true | [-1.487142, 0.974335, 3.046990] |
| 1 | 1 | 2 | 2.072655 | 0.000000 | true | [-1.487142, 0.974335, 3.046990] |
| 1 | 2 | 2 | 2.072655 | 0.000000 | true | [-1.487142, 0.974335, 3.046990] |
| 1 | 3 | 2 | 2.072655 | 0.000000 | true | [-1.487142, 0.974335, 3.046990] |
| 2 | 0 | 1 | 0.967149 | 0.000000 | true | [0.192682, 1.159831, -0.108849] |
| 2 | 1 | 1 | 0.967149 | 0.000000 | true | [0.192682, 1.159831, -0.108849] |
| 2 | 2 | 1 | 0.967149 | 0.000000 | true | [0.192682, 1.159831, -0.108849] |
| 2 | 3 | 1 | 0.967149 | 0.000000 | true | [0.192682, 1.159831, -0.108849] |

## Control Models

| model | logits invariant | argmax invariant | changed argmax | max rel RMS | gguf |
| --- | --- | --- | --- | --- | --- |

## Interpretation

Duplicate token-ID prompts are invariant when evaluated with --embd-sequential. This is a sequence-isolated mitigation, not proof that the multi-prompt batched I2_SR path is safe.
