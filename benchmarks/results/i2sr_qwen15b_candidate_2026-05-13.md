# I2_SR Qwen2.5-1.5B Candidate Result, 2026-05-13

This note records the first full applied-runtime benchmark of the candidate
stable row-scale ternary qtype `I2_SR`.

## Setup

| item | value |
| --- | --- |
| source checkpoint | `checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-row-notiehead-5000/step-5000` |
| runtime patch | `patches/llama-i2sr-row-scale-qtype.patch` |
| build | `build-portable-avx2` |
| CPU | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz |
| threads | `12` |
| context | `512` |
| perplexity tokens | `8192` |

The converter emitted
`models/qwen2.5-1.5b-klonly-row-notie-static-ternary-dense/qwen15b_klonly_row_notie_static_ternary_i2_sr.gguf`
with qtype `I2_SR` / numeric dtype `40` and file type
`MOSTLY_I2_SR` / numeric ftype `41`.

Conversion summary:

| field | value |
| --- | ---: |
| ternary tensors packed | `196` |
| row-scale tensors packed | `196` |
| copied tensors | `143` |
| output tensors | `339` |
| dense F16 output tensors | `0` |
| packed code bytes | `330,135,680` |
| GGUF file bytes | `1,270,157,888` |

## Result

| artifact | file MiB | PPL | PPL tok/s | prompt tok/s | decode tok/s | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `qwen15b_klonly_row_notie_static_ternary_i2_sr` | `1211.3` | `20,074,699.9423` | `149.90` | `212.10` | `19.01` | quality failed |

The deterministic smoke completion was nonsensical:

```text
di hopesallen学 getService$i_points方wartsihad_hppRepresentationihadetagetagewis_standieme＄lest\ 낡 senate.edu
```

## Interpretation

The candidate qtype is mechanically real: the patched runtime loads the file,
routes `196` tensors as `i2_sr`, and runs at roughly the same speed and file
size as the older row-scale packed prototype.

This first artifact was not semantically correct. The fixed-excerpt PPL was
catastrophic (`20,074,699.9423`) while the earlier quality-preserving row-scale
prototype on the same strong checkpoint reached PPL `38.8832`. This did not
disprove row-scale ternary execution; it exposed a direct-writer packing bug.

The follow-up fix changed the direct writer to use the active x86
`ACT_PARALLEL` chunk-128 packing layout. The fixed artifact reaches PPL
`38.8477`, prompt throughput `211.67` tok/s, and decode throughput `19.07`
tok/s. Details are recorded in
`benchmarks/results/i2sr_x86act_fix_2026-05-13.md`.
