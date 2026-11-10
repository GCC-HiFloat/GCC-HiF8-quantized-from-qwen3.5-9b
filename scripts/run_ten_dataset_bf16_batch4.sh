#!/usr/bin/env bash
set -euo pipefail

# Full BF16 reference for ten medium/large evaluation datasets on one 910B.
# Each task has its own result directory so an interrupted run resumes without
# repeating completed tasks.  All runs use the same scheduler and batch=4.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
harness_root=${LMEVAL_ROOT:-/filesdir/lm-evaluation-harness}
out_root=${TEN_BF16_ROOT:-$repo_root/benchmarks/qwen35-ten-bf16-batch4}
model=${BF16_MODEL:-/filesdir/qwen3.5-9b}
batch_size=${TEN_BF16_BATCH_SIZE:-4}
max_num_seqs=${TEN_BF16_MAX_NUM_SEQS:-4}
max_num_batched_tokens=${TEN_BF16_MAX_NUM_BATCHED_TOKENS:-8192}
include_path=${TEN_BF16_INCLUDE_PATH:-$repo_root/local_tasks}

tasks=(
  arc_easy_local
  arc_challenge_ten
  piqa_local
  winogrande_local
  boolq_local
  hellaswag_local
  sciq_ten
  gsm8k_ten
  lambada_openai_ten
  mmlu_all_ten
)
if [[ -n "${TEN_BF16_TASKS:-}" ]]; then
  IFS=',' read -r -a tasks <<< "$TEN_BF16_TASKS"
fi

if [[ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export VLLM_ASCEND_DISABLE_CUSTOM_OP=${VLLM_ASCEND_DISABLE_CUSTOM_OP:-1}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONPATH="/tmp/vllm-d02:/tmp/vllm-ascend-v026:$repo_root/src:$harness_root:/usr/local/Ascend/cann-9.0.0/python/site-packages:${PYTHONPATH:-}"

mkdir -p "$out_root"
cd "$harness_root"
for task in "${tasks[@]}"; do
  task_out="$out_root/$task"
  mkdir -p "$task_out"
  if find "$task_out" -type f -name 'results_*.json' -print -quit | grep -q .; then
    echo "[bf16] skip completed $task"
    continue
  fi
  echo "[bf16] starting $task batch=$batch_size max_num_seqs=$max_num_seqs"
  /root/miniconda3/envs/hifi/bin/python -m lm_eval run --model vllm \
    --include_path "$include_path" \
    --model_args "pretrained=$model,dtype=bfloat16,trust_remote_code=True,tensor_parallel_size=1,max_model_len=16384,max_num_seqs=$max_num_seqs,max_num_batched_tokens=$max_num_batched_tokens,enforce_eager=True,async_scheduling=False,gpu_memory_utilization=0.8,kv_cache_memory_bytes=8589934592,enable_thinking=False" \
    --tasks "$task" --batch_size "$batch_size" --seed 42 \
    --trust_remote_code --confirm_run_unsafe_code --log_samples \
    --output_path "$task_out" 2>&1 | tee "$out_root/$task.log"
done
echo "[bf16] ten-dataset batch=4 baseline complete: $out_root"
