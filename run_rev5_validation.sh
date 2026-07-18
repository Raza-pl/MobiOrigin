#!/bin/bash
set -euo pipefail
cd /sessions/sweet-epic-franklin/mnt/Plasflow

LOGFILE=data/models/candidates/clean_3class_rev5_hardneg_locked_20260717/evaluation/validation_run.log
mkdir -p data/models/candidates/clean_3class_rev5_hardneg_locked_20260717/evaluation

echo "$(date) START" >> "$LOGFILE"

python3 scripts/run_classifier_validation.py \
  --candidate-dir data/models/candidates/clean_3class_rev5_hardneg_locked_20260717 \
  --dataset-dir data/clean_3class_hardneg_experiment \
  --benchmark-dir data/benchmark \
  --tiers tier1 \
  >> "$LOGFILE" 2>&1

echo "$(date) DONE" >> "$LOGFILE"
