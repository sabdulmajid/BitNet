#!/bin/bash
# Submit paper-style GLUE sequence-classification BitDistill jobs.
#
# This wrapper keeps sequence-classification reproduction runs separate from
# causal verbal-label experiments while reusing the same dependency handling.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export TASK_FORMAT="${TASK_FORMAT:-sequence_classification}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/bitdistill-glue-seqcls}"
export EXCLUDE_LINEAR_REGEX="${EXCLUDE_LINEAR_REGEX:-score|classifier}"
export LABEL_SCHEME="${LABEL_SCHEME:-letters}"
export CANDIDATE_SCORE="${CANDIDATE_SCORE:-mean}"
export JOB_KIND="${JOB_KIND:-bitdistill_glue_seqcls_jobs}"

exec "$ROOT_DIR/benchmarks/submit_bitdistill_glue_clean_causal_wave.sh"
