#!/usr/bin/env bash
# Run EuroEval on Mimir, with or without the prior-steering intervention.
#   ./src/run_euroeval.sh <delta> <outdir> [extra euroeval args...]
# The intervention is injected through src/patch/sitecustomize.py (vLLM's compute_logits).
set -euo pipefail
DELTA="${1:?delta}"; OUT="${2:?outdir}"; shift 2
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUT"
cd "$OUT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
PYTHONPATH="$REPO/src/patch" \
MIMIR_STEER_DELTA="$DELTA" \
MIMIR_STEER_PRIOR="${MIMIR_STEER_PRIOR:-mix}" \
MIMIR_SHIFT_FILE="$REPO/results/Wd_${MIMIR_STEER_PRIOR:-mix}.npy" \
"$REPO/.venv-eval/bin/euroeval" \
  -m danish-foundation-models/DFM-Mimir \
  -l da -l en \
  --num-iterations 1 --zero-shot --generative-type instruction_tuned \
  --device cuda --no-progress-bar --force \
  --cache-dir "$REPO/.euroeval_cache" "$@"
