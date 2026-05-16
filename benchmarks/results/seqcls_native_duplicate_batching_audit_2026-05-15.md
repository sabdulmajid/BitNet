# Sequence-Classification Native I2_SR Duplicate-Prompt Batching Audit, 2026-05-15

This audit repeats the same rendered token-ID prompt within a single native llama-embedding batch. A mismatch here cannot be attributed to tokenizer round-trip differences, different text examples, or output-row swaps.

| field | value |
| --- | --- |
| status | duplicate_batching_parity_mismatch |
| targets | [15, 35, 0, 1, 2] |
| repeat count | 4 |
| same prompt repeated | true |
| all logits invariant | false |
| all predictions invariant | false |
| changed prediction count | 4 |
| max relative RMS vs alone | 0.305153 |
| formatting/tokenization ruled out | true |
| ready for batched product benchmark | false |

## Duplicate Positions

| target | position | pred | margin | rel RMS vs alone | pred matches alone | logits |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | 0 | 1 | 0.019177 | 0.000000 | true | [0.374451, 0.420180, 0.401003] |
| 15 | 1 | 0 | 0.040229 | 0.111352 | false | [0.450472, 0.410243, 0.394397] |
| 15 | 2 | 1 | 0.019177 | 0.000000 | true | [0.374451, 0.420180, 0.401003] |
| 15 | 3 | 0 | 0.040229 | 0.111352 | false | [0.450472, 0.410243, 0.394397] |
| 35 | 0 | 0 | 0.124395 | 0.000000 | true | [0.518326, 0.393931, 0.169148] |
| 35 | 1 | 1 | 0.156405 | 0.305153 | false | [0.367728, 0.524132, 0.219143] |
| 35 | 2 | 1 | 0.084456 | 0.221412 | false | [0.426778, 0.511235, 0.162849] |
| 35 | 3 | 0 | 0.005010 | 0.129505 | true | [0.455583, 0.450573, 0.190206] |
| 0 | 0 | 1 | 1.345433 | 0.000000 | true | [-0.546304, 1.516333, 0.170900] |
| 0 | 1 | 1 | 1.315991 | 0.035818 | true | [-0.564050, 1.537818, 0.221827] |
| 0 | 2 | 1 | 1.277961 | 0.030402 | true | [-0.556369, 1.487624, 0.209663] |
| 0 | 3 | 1 | 1.319469 | 0.040701 | true | [-0.547699, 1.548143, 0.228675] |
| 1 | 0 | 2 | 2.072655 | 0.000000 | true | [-1.487142, 0.974335, 3.046990] |
| 1 | 1 | 2 | 2.052991 | 0.010047 | true | [-1.500928, 1.005058, 3.058049] |
| 1 | 2 | 2 | 2.053723 | 0.010313 | true | [-1.509440, 1.001790, 3.055513] |
| 1 | 3 | 2 | 2.066029 | 0.011555 | true | [-1.520669, 0.993705, 3.059734] |
| 2 | 0 | 1 | 0.967149 | 0.000000 | true | [0.192682, 1.159831, -0.108849] |
| 2 | 1 | 1 | 1.048502 | 0.074020 | true | [0.125255, 1.173757, -0.162687] |
| 2 | 2 | 1 | 1.121605 | 0.095216 | true | [0.102435, 1.224041, -0.128150] |
| 2 | 3 | 1 | 1.076745 | 0.075370 | true | [0.121887, 1.198632, -0.146296] |

## Interpretation

Duplicate token-ID prompts are not invariant across batch positions. Because every entry in each audited batch is the exact same rendered token-ID prompt, this rules out tokenizer round-trip differences, text formatting differences, and output-row swaps as sufficient explanations. Batched sequence-classification throughput must remain blocked until this native runtime position-dependent drift is fixed.
