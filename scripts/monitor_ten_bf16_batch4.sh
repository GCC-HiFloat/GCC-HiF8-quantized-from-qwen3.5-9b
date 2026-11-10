#!/usr/bin/env bash
set -euo pipefail

# Periodically record read-only progress for the long BF16 baseline run.
# Usage: monitor_ten_bf16_batch4.sh <orchestrator-pid> [root] [interval-seconds]
parent_pid=${1:?orchestrator PID is required}
root=${2:-/filesdir/vllm-ascend-hif8/benchmarks/qwen35-ten-bf16-batch4}
interval=${3:-7200}
log=${TEN_BF16_MONITOR_LOG:-$root/monitor-2h.log}
tasks=(arc_easy_local arc_challenge_ten piqa_local winogrande_local boolq_local hellaswag_local sciq_ten gsm8k_ten lambada_openai_ten mmlu_all_ten)

snapshot() {
  {
    printf '\n===== %s =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    if kill -0 "$parent_pid" 2>/dev/null; then
      echo "orchestrator_pid=$parent_pid state=running"
    else
      echo "orchestrator_pid=$parent_pid state=finished"
    fi
    ps -eo pid,ppid,etime,stat,cmd | awk '/run_ten_dataset_bf16_batch4|lm_eval run|summary-watcher/ && !/awk/' || true
    for task in "${tasks[@]}"; do
      result_count=$( { find "$root/$task" -type f -name 'results_*.json' 2>/dev/null || true; } | wc -l)
      if [[ "$result_count" -gt 0 ]]; then
        printf '%-24s results=%s\n' "$task" "$result_count"
      elif [[ -f "$root/$task.log" ]]; then
        progress=$(grep -E 'Running (loglikelihood|generate) requests:' "$root/$task.log" | tail -1 | sed -E $'s/\\x1b\\[[0-9;]*[mK]//g' || true)
        printf '%-24s results=0 progress=%s\n' "$task" "$progress"
      else
        printf '%-24s pending\n' "$task"
      fi
    done
  } >> "$log"
}

mkdir -p "$root"
snapshot
while kill -0 "$parent_pid" 2>/dev/null; do
  sleep "$interval"
  snapshot
done
snapshot
