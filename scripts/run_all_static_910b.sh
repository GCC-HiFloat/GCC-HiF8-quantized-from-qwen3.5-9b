#!/usr/bin/env bash
set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tasks=(
  mmlu_pro
  mmlu_prox_en
  mmlu_prox_af
  mmlu_prox_ar
  mmlu_prox_bn
  mmlu_prox_cs
  global_piqa_nonparallel_generation
  global_piqa_parallel_generation
  longbench2
)
log_dir="$repo_root/benchmarks/qwen35-static-w8a8kv8-910b-20260826/logs"
mkdir -p "$log_dir"
# Override with HIF8_BATCH_SIZE when restarting this script; the value is
# inherited by each task and does not affect already-completed results. Set
# HIF8_DYNAMIC_BATCH=1 to use lm-eval's adaptive request batching while keeping
# HIF8_MAX_NUM_SEQS as the engine concurrency ceiling.
export HIF8_BATCH_SIZE="${HIF8_BATCH_SIZE:-2}"
for task in "${tasks[@]}"; do
  result_dir="$repo_root/benchmarks/qwen35-static-w8a8kv8-910b-20260826/hif8/$task"
  if find "$result_dir" -type f -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; then
    echo "[$(date -u +%FT%TZ)] skip completed $task"
    continue
  fi
  echo "[$(date -u +%FT%TZ)] start $task"
  attempt=1
  while (( attempt <= 3 )); do
    if "$repo_root/scripts/run_qwen35_static_benchmark_910b.sh" hif8 "$task" \
      >"$log_dir/$task.log" 2>&1; then
      echo "[$(date -u +%FT%TZ)] finished $task"
      break
    fi
    echo "[$(date -u +%FT%TZ)] $task failed (attempt $attempt), retrying" | tee -a "$log_dir/$task.log"
    ((attempt++))
  done
  if (( attempt > 3 )); then
    echo "[$(date -u +%FT%TZ)] giving up $task for now; rerun script to resume" | tee -a "$log_dir/$task.log"
  fi
done
echo "all tasks completed"
