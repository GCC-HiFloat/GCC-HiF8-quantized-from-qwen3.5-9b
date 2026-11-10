#!/usr/bin/env bash
set -euo pipefail

# Full static W8A8KV8 HiFloat8 evaluation on one 910B.  Each task has its
# own output directory, so an interrupted invocation resumes at the first
# unfinished task without repeating completed datasets.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
harness_root=${LMEVAL_ROOT:-/filesdir/lm-evaluation-harness}
out_root=${TEN_HIF8_ROOT:-$repo_root/benchmarks/qwen35-ten-hif8-batch4}
model=${HIF8_MODEL:-/filesdir/qwen3.5-9b-hif8-static}
batch_size=${TEN_HIF8_BATCH_SIZE:-4}
max_num_seqs=${TEN_HIF8_MAX_NUM_SEQS:-4}
max_num_batched_tokens=${TEN_HIF8_MAX_NUM_BATCHED_TOKENS:-8192}
include_path=${TEN_HIF8_INCLUDE_PATH:-$repo_root/local_tasks}
opp_build=${HIF8_OPP_BUILD_DIR:-$repo_root/csrc/opp/build/ascend910b}
opp_vendor="$opp_build/package/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors/vllm_ascend_hif8"

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
if [[ -n "${TEN_HIF8_TASKS:-}" ]]; then
  IFS=',' read -r -a tasks <<< "$TEN_HIF8_TASKS"
fi

if [[ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export ASCEND_CUSTOM_OPP_PATH="$opp_vendor"
export HIF8_SOC=ascend910b
export HIF8_OPP_BUILD_DIR="$opp_build"
export HIF8_COMPUTE_BACKEND=soft_hif8
export HIF8_KV_CACHE=${HIF8_KV_CACHE:-1}
export HIF8_STATIC_FUSED_GENERATION=${HIF8_STATIC_FUSED_GENERATION:-1}
export HIF8_STABLE_M_TILE=${HIF8_STABLE_M_TILE:-1}
export VLLM_ASCEND_DISABLE_CUSTOM_OP=${VLLM_ASCEND_DISABLE_CUSTOM_OP:-1}
export VLLM_ASCEND_GDN_CUSTOM_OP=${VLLM_ASCEND_GDN_CUSTOM_OP:-1}
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
    echo "[hif8] skip completed $task"
    continue
  fi
  echo "[hif8] starting $task batch=$batch_size max_num_seqs=$max_num_seqs"
  /root/miniconda3/envs/hifi/bin/python -m lm_eval run --model vllm \
    --include_path "$include_path" \
    --model_args "pretrained=$model,dtype=bfloat16,trust_remote_code=True,tensor_parallel_size=1,max_model_len=16384,max_num_seqs=$max_num_seqs,max_num_batched_tokens=$max_num_batched_tokens,enforce_eager=True,async_scheduling=False,gpu_memory_utilization=0.8,kv_cache_memory_bytes=8589934592,enable_thinking=False" \
    --tasks "$task" --batch_size "$batch_size" --seed 42 \
    --trust_remote_code --confirm_run_unsafe_code --log_samples \
    --output_path "$task_out" 2>&1 | tee "$out_root/$task.log"
done
/root/miniconda3/envs/hifi/bin/python \
  "$repo_root/scripts/summarize_ten_hif8_compare.py" \
  --bf16-root "$repo_root/benchmarks/qwen35-ten-bf16-batch4" \
  --hif8-root "$out_root" \
  --output "$repo_root/benchmarks/ten-hif8-batch4-vs-bf16.md"
echo "[hif8] ten-dataset static W8A8KV8 batch=4 evaluation complete: $out_root"
