# Artifact Prune Plan

Generated UTC: `2026-05-13T03:55:58+00:00`
Git HEAD: `2b92e84a5560`

This is a dry-run plan. No files were deleted by the generator.

## Guardrails

- Evidence manifest: `benchmarks/results/evidence_manifest_2026-05-13.json`
- Protected evidence files: `49`
- Protected benchmark-manifest paths: `27`
- Protected checkpoint directories: `12`

## Candidate Storage

| class | action | items | total |
| --- | --- | ---: | ---: |
| `keep_by_default_model_artifacts` | keep by default | 10 | 54.22 GiB |
| `prune_intermediate_checkpoints` | safe after review | 29 | 202.42 GiB |
| `remove_local_caches` | safe after review | 1 | 4.08 GiB |
| `remove_rebuildable_build_dirs` | safe after review | 3 | 82.11 MiB |

## Recommended Sequence

1. Rebuild the evidence manifest and confirm it still reports zero missing artifacts.
2. Remove only `prune_intermediate_checkpoints` first; this is the main storage win and preserves final checkpoints.
3. Optionally remove CMake build directories and `.hf_cache`; they are rebuildable but may cost time to recreate.
4. Do not remove `keep_by_default_model_artifacts` unless you intentionally accept rerunning GGUF conversion/quantization.
5. Rebuild the evidence manifest again after pruning.

## Manual Commands

Review these paths before running any command.

### prune_intermediate_checkpoints

```bash
rm -rf 'checkpoints/audit-save/step-7'
rm -rf 'checkpoints/qwen2.5-0.5b-fineweb-edu-12/step-250'
rm -rf 'checkpoints/qwen2.5-0.5b-fineweb-edu-12/step-500'
rm -rf 'checkpoints/qwen2.5-0.5b-fineweb-edu-12/step-750'
rm -rf 'checkpoints/qwen2.5-0.5b-fineweb-edu-klonly-1000/step-250'
rm -rf 'checkpoints/qwen2.5-0.5b-fineweb-edu-klonly-1000/step-500'
rm -rf 'checkpoints/qwen2.5-0.5b-fineweb-edu-klonly-1000/step-750'
rm -rf 'checkpoints/qwen2.5-0.5b-fineweb-edu-klonly-notiehead-1000/step-250'
rm -rf 'checkpoints/qwen2.5-0.5b-fineweb-edu-klonly-notiehead-1000/step-500'
rm -rf 'checkpoints/qwen2.5-0.5b-fineweb-edu-klonly-notiehead-1000/step-750'
rm -rf 'checkpoints/qwen2.5-0.5b-fineweb-edu-row-1000/step-250'
rm -rf 'checkpoints/qwen2.5-0.5b-fineweb-edu-row-1000/step-500'
rm -rf 'checkpoints/qwen2.5-0.5b-fineweb-edu-row-1000/step-750'
rm -rf 'checkpoints/qwen2.5-1.5b-fineweb-edu/step-1000'
rm -rf 'checkpoints/qwen2.5-1.5b-fineweb-edu/step-2000'
rm -rf 'checkpoints/qwen2.5-1.5b-fineweb-edu/step-3000'
rm -rf 'checkpoints/qwen2.5-1.5b-fineweb-edu/step-4000'
rm -rf 'checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-5000/step-1000'
rm -rf 'checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-5000/step-2000'
rm -rf 'checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-5000/step-3000'
rm -rf 'checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-5000/step-4000'
rm -rf 'checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-notiehead-5000/step-1000'
rm -rf 'checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-notiehead-5000/step-2000'
rm -rf 'checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-notiehead-5000/step-3000'
rm -rf 'checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-notiehead-5000/step-4000'
rm -rf 'checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-row-notiehead-5000/step-1000'
rm -rf 'checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-row-notiehead-5000/step-2000'
rm -rf 'checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-row-notiehead-5000/step-3000'
rm -rf 'checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-row-notiehead-5000/step-4000'
```

### remove_rebuildable_build_dirs

```bash
rm -rf 'build'
rm -rf 'build-portable-avx2'
rm -rf 'build-qwen05b-tl2'
```

### remove_local_caches

```bash
rm -rf '.hf_cache'
```

## Model Artifacts Kept By Default

- `models/eval-ternary-tiny-direct`: 24.19 MiB
- `models/qwen2.5-0.5b-direct-static-ternary`: 1.18 GiB
- `models/qwen2.5-0.5b-fp`: 2.02 GiB
- `models/qwen2.5-0.5b-qatsby`: 2.74 GiB
- `models/qwen2.5-1.5b-fp`: 6.84 GiB
- `models/qwen2.5-1.5b-klonly-notie-static-ternary-dense`: 9.01 GiB
- `models/qwen2.5-1.5b-klonly-row-notie-static-ternary-dense`: 10.19 GiB
- `models/qwen2.5-1.5b-klonly-static-ternary-dense`: 6.87 GiB
- `models/qwen2.5-1.5b-qatsby`: 7.30 GiB
- `models/qwen2.5-1.5b-static-ternary-dense`: 8.05 GiB

