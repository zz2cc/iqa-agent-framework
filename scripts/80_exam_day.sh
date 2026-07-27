#!/usr/bin/env bash
# S7 考试日：五臂 × 两数据集，32B 全量推理（顺序执行防限流）
# 用法： bash scripts/80_exam_day.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
M="--model main"
LIB="--library runs/cke/final_library.json"

run() {
  echo "===== $(date +%H:%M:%S) $* ====="
  python scripts/30_run.py "$@"
  echo ""
}

# ---- KonIQ Val (2015) ----
run --route r1b --dataset koniq_val $M --outdir runs/final/r1b_koniq
run --route r1r --dataset koniq_val $M --outdir runs/final/r1r_koniq
run --route r2  --dataset koniq_val $M --outdir runs/final/r2_koniq
run --route r25 --dataset koniq_val $M --outdir runs/final/r25_koniq
run --route r25 --dataset koniq_val $M $LIB --outdir runs/final/r3_koniq

# ---- SPAQ Test (1125) ----
run --route r1b --dataset spaq_test $M --outdir runs/final/r1b_spaq
run --route r1r --dataset spaq_test $M --outdir runs/final/r1r_spaq
run --route r2  --dataset spaq_test $M --outdir runs/final/r2_spaq
run --route r25 --dataset spaq_test $M --outdir runs/final/r25_spaq
run --route r25 --dataset spaq_test $M $LIB --outdir runs/final/r3_spaq

echo "EXAM_INFERENCE_ALL_DONE"
