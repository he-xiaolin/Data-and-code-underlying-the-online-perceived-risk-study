#!/usr/bin/env bash
cd "$(dirname "$0")"

IDX="${1:-}"  # 可选：统一的 experiment_index；不传就用各脚本的默认值

run() {
  if [[ -z "$IDX" ]]; then
    python "$1"
  else
    python "$1" --experiment_index "$IDX"
  fi
}

run evaluation_HB.py
run evaluation_MB.py
run evaluation_SVM.py
run evaluation_LC_1.py
run evaluation_LC_2.py
run evaluation_LC_3.py