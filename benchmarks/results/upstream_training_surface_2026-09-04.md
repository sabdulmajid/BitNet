# Upstream Training-Surface Audit

This audit distinguishes Microsoft Research's BitDistill method from the open
training implementation added by this fork.

| Field | Value |
| --- | --- |
| Upstream repository | `https://github.com/microsoft/BitNet.git` |
| Audited branch | `main` |
| Audited revision | `0b341e582afbf9e1011f24744b554c96a3477eb5` |
| Tracked paths | `82` |
| Python/shell paths | `29` |
| Training/distillation/QAT entrypoints found by filename scan | `0` |

The 29 Python/shell paths at this revision are inference, serving, conversion,
packing, kernel generation, setup, test, and benchmark utilities. Upstream
documentation describes how released embedding models were trained, but the
audited tree does not include a corresponding model-training entrypoint.
The public request for the BitDistill training code remains open as
[microsoft/BitNet#344](https://github.com/microsoft/BitNet/issues/344). That
issue corroborates the missing public surface; the pinned tree scan above is
the reproducible evidence for this audit.

Reproduce the scope check after configuring the upstream remote:

```bash
git fetch upstream main
git rev-parse upstream/main
git ls-tree -r --name-only upstream/main | wc -l
git ls-tree -r --name-only upstream/main | rg -i '\.(py|sh)$' | wc -l
git ls-tree -r --name-only upstream/main \
  | rg -i '(^|/)(train|distill|finetun|qat)[^/]*\.(py|sh)$'
git grep -n -E '\.backward\(|torch\.optim|optimizer\.step|loss\.backward' \
  upstream/main -- '*.py' '*.sh'
```

The final two commands produce no rows at the audited revision. A broader content
scan finds training and distillation descriptions in `README.md` and
`docs/bitnet-embeddings-i2s-guide.md`, not executable training code.

## Claim Boundary

BitDistill is Microsoft Research's method. This fork can claim an independent
open implementation and reproduction attempt at the pinned upstream revision;
it cannot claim invention of BitDistill. This absence result applies only to
the audited upstream branch and revision and must be rechecked if upstream
changes.
