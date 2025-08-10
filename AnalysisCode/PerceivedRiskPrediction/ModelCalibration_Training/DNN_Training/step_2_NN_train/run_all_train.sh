#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for s in train_HB.py train_MB.py train_SVM.py train_LC_1.py train_LC_2.py train_LC_3.py; do
  echo "== Running $s =="
  python "$s" --num_epochs 2 --experiment_index "2025"
done