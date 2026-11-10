#!/usr/bin/env bash
set -euo pipefail

# Matched BF16/HIF8 endpoint sweep for batch-sensitivity diagnosis on 910B.
# Every task has an independent lm-eval output directory, so rerunning this
# script resumes at the first unfinished task without overwriting results.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out_base=${MATCHED_COMPARE_ROOT:-$repo_root/benchmarks/qwen35-twelve-task-matched-batch-910b}
tasks=${MATCHED_TASKS:-arc_easy_local,piqa_local,winogrande_local,boolq_local,mmlu_elementary_mathematics_local,mmlu_high_school_computer_science_local,cmmlu_elementary_commonsense_local,cmmlu_chinese_driving_rule_local,cmmlu_college_actuarial_science_local,cmmlu_college_engineering_hydrology_local,cmmlu_college_medical_statistics_local,cmmlu_philosophy_local}

read -r -a batches <<< "${MATCHED_BATCHES:-8 32}"
read -r -a modes <<< "${MATCHED_MODES:-hif8 bf16}"
for batch in "${batches[@]}"; do
  for mode in "${modes[@]}"; do
    echo "[matched] batch=${batch} mode=${mode} output=${out_base}/batch${batch}"
    export HIF8_TASKS="$tasks"
    export HIF8_BATCH_SIZE="$batch"
    export HIF8_MAX_NUM_SEQS="$batch"
    export HIF8_MAX_NUM_BATCHED_TOKENS=8192
    export HIF8_DYNAMIC_BATCH=0
    export HIF8_GROUP_TASKS="${MATCHED_GROUP_TASKS:-1}"
    export HIF8_LIMIT="${MATCHED_LIMIT:-100}"
    export HIF8_COMPARE_ROOT="$out_base/batch${batch}"
    export HIF8_FORCE="${HIF8_FORCE:-0}"
    bash "$repo_root/scripts/run_eight_task_compare_910b.sh" "$mode"
  done
done

"/root/miniconda3/envs/hifi/bin/python" \
  "$repo_root/scripts/summarize_matched_batch_compare.py" \
  --root "$out_base" \
  --output "$out_base/comparison.md"
echo "[matched] all requested runs complete"
