#!/usr/bin/env bash
set -euo pipefail

# Accuracy-first static HiF8 W8A8KV8 batch sweep on Ascend 910B.
# Each batch has an independent output directory; completed tasks are skipped
# by run_eight_task_compare_910b.sh, so the sweep is safe to resume.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out_base=${EDU_SWEEP_ROOT:-$repo_root/benchmarks/qwen35-education-batch-sweep-910b}
tasks=${EDU_TASKS:-mmlu_elementary_mathematics_local,mmlu_high_school_computer_science_local,cmmlu_elementary_commonsense_local,cmmlu_chinese_driving_rule_local,cmmlu_college_mathematics_local,cmmlu_philosophy_local,cmmlu_college_actuarial_science_local,cmmlu_college_engineering_hydrology_local,cmmlu_college_medical_statistics_local,cmmlu_chinese_foreign_policy_local,cmmlu_high_school_geography_local,cmmlu_computer_science_local}

read -r -a batches <<< "${EDU_BATCHES:-8 16 32}"
for batch in "${batches[@]}"; do
  out_root="$out_base/batch${batch}"
  echo "[sweep] starting batch=${batch}; output=${out_root}"
  export HIF8_TASKS="$tasks"
  export HIF8_BATCH_SIZE="$batch"
  # Optionally cap scheduler microbatches (e.g. 8) while retaining a larger
  # lm-eval request batch; this is the precision-first workaround for 910B.
  export HIF8_MAX_NUM_SEQS="${EDU_MAX_NUM_SEQS:-$batch}"
  export HIF8_MAX_NUM_BATCHED_TOKENS=8192 HIF8_DYNAMIC_BATCH=0
  export HIF8_COMPARE_ROOT="$out_root" HIF8_FORCE="${HIF8_FORCE:-0}"
  bash "$repo_root/scripts/run_eight_task_compare_910b.sh" hif8 || {
    status=$?
    echo "[sweep] batch=${batch} exited with status ${status}; resume reruns unfinished tasks" >&2
    exit "$status"
  }
done
echo "[sweep] all batches complete"
