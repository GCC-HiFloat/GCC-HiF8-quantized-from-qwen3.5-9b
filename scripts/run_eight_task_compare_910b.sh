#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 {bf16|hif8}" >&2
  exit 2
fi
mode=$1
case "$mode" in
  bf16) model=${BF16_MODEL:-/filesdir/qwen3.5-9b} ;;
  hif8) model=${HIF8_MODEL:-/filesdir/qwen3.5-9b-hif8-static} ;;
  *) echo "unsupported mode: $mode" >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
harness_root=${LMEVAL_ROOT:-/filesdir/lm-evaluation-harness}
out_root=${HIF8_COMPARE_ROOT:-$repo_root/benchmarks/qwen35-eight-task-910b}
batch_size=${HIF8_BATCH_SIZE:-32}
dynamic_batch=${HIF8_DYNAMIC_BATCH:-0}
if [[ "$dynamic_batch" == 1 ]]; then
  eval_batch_size=auto
  # 32 is the largest scheduler window that matched the fixed-batch accuracy
  # reference on 910B while retaining adaptive request batching.
  max_num_seqs=${HIF8_MAX_NUM_SEQS:-32}
  # Adaptive lm-eval batches frequently issue single-row decode launches.
  # Keep those launches on the same GEMM implementation as larger rows so
  # accumulation order does not change with the scheduler's batch size.
  export HIF8_DISABLE_GEMV=${HIF8_DISABLE_GEMV:-1}
  export HIF8_STABLE_M_TILE=${HIF8_STABLE_M_TILE:-1}
else
  eval_batch_size=$batch_size
  max_num_seqs=${HIF8_MAX_NUM_SEQS:-$batch_size}
  # Keep Cube's M partition stable across request groupings.  This removes
  # batch-dependent accumulation-order changes while preserving the same
  # static W8A8KV8 scales.  GEMV remains available for true M=1 generation.
  export HIF8_STABLE_M_TILE=${HIF8_STABLE_M_TILE:-1}
  unset HIF8_DISABLE_GEMV || true
fi
# vLLM scheduler knobs.  Keep these independent from lm-eval's request
# batching: lm-eval controls how many prompts are submitted per call, while
# these limits control how many requests/tokens the single NPU engine can
# process concurrently.  They are intentionally opt-in so existing runs keep
# their exact behavior.
max_num_batched_tokens=${HIF8_MAX_NUM_BATCHED_TOKENS:-8192}
include_path=${HIF8_INCLUDE_PATH:-$repo_root/local_tasks}
opp_build=${HIF8_OPP_BUILD_DIR:-$repo_root/csrc/opp/build/ascend910b}
opp_vendor="$opp_build/package/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors/vllm_ascend_hif8"

if [[ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export ASCEND_CUSTOM_OPP_PATH="$opp_vendor"
export HIF8_SOC=ascend910b HIF8_OPP_BUILD_DIR="$opp_build"
export HIF8_COMPUTE_BACKEND=soft_hif8
export HIF8_STATIC_FUSED_GENERATION=${HIF8_STATIC_FUSED_GENERATION:-1}
if [[ "$mode" == hif8 ]]; then
  if [[ "${HIF8_DISABLE_KV_CACHE:-0}" == 1 ]]; then
    export HIF8_KV_CACHE=0
  else
    export HIF8_KV_CACHE=${HIF8_KV_CACHE:-1}
  fi
else
  unset HIF8_KV_CACHE || true
fi
export VLLM_ASCEND_DISABLE_CUSTOM_OP=1 VLLM_USE_V1=1 VLLM_NO_USAGE_STATS=1
# CANN 9.0 lacks unrelated optional custom ops such as AddRmsNormBias, so the
# global escape hatch stays enabled.  Load only the packaged 910B
# causal-convolution and recurrent-GDN AscendC operators.
export VLLM_ASCEND_GDN_CUSTOM_OP=${VLLM_ASCEND_GDN_CUSTOM_OP:-1}
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="/tmp/vllm-d02:/tmp/vllm-ascend-v026:$repo_root/src:$harness_root:/usr/local/Ascend/cann-9.0.0/python/site-packages:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE

default_tasks=(
  arc_easy_local
  piqa_local
  winogrande_local
  boolq_local
  mmlu_elementary_mathematics_local
  mmlu_high_school_computer_science_local
  cmmlu_elementary_commonsense_local
  cmmlu_chinese_driving_rule_local
)
if [[ -n "${HIF8_TASKS:-}" ]]; then
  IFS=',' read -r -a tasks <<< "$HIF8_TASKS"
else
  tasks=("${default_tasks[@]}")
fi

limit_args=()
if [[ -n "${HIF8_LIMIT:-}" ]]; then
  limit_args+=(--limit "$HIF8_LIMIT")
fi
mkdir -p "$out_root/$mode"
cd "$harness_root"
include_args=(--include_path "$include_path")

# Accuracy sweeps often contain many small task slices. Loading a 9B model
# once per slice dominates their runtime, so grouped mode submits all tasks
# to one lm-eval engine. The result file checkpoints the whole group.
if [[ "${HIF8_GROUP_TASKS:-0}" == 1 ]]; then
  group_out="$out_root/$mode/all_tasks"
  mkdir -p "$group_out"
  if [[ "${HIF8_FORCE:-0}" != 1 ]] && find "$group_out" -type f -name 'results_*.json' -print -quit | grep -q .; then
    echo "[$mode] skip completed grouped task run (set HIF8_FORCE=1 to rerun)"
    exit 0
  fi
  task_csv=$(IFS=,; echo "${tasks[*]}")
  echo "[$mode] starting grouped tasks with batch_size=$eval_batch_size max_num_seqs=$max_num_seqs"
  /root/miniconda3/envs/hifi/bin/python -m lm_eval run --model vllm \
    "${include_args[@]}" \
    "${limit_args[@]}" \
    --model_args "pretrained=$model,dtype=bfloat16,trust_remote_code=True,tensor_parallel_size=1,max_model_len=16384,max_num_seqs=$max_num_seqs,max_num_batched_tokens=$max_num_batched_tokens,enforce_eager=True,async_scheduling=False,gpu_memory_utilization=0.8,kv_cache_memory_bytes=8589934592,enable_thinking=False" \
    --tasks "$task_csv" --batch_size "$eval_batch_size" --seed 42 \
    --trust_remote_code --confirm_run_unsafe_code --log_samples \
    --output_path "$group_out"
  exit 0
fi

for task in "${tasks[@]}"; do
  task_out="$out_root/$mode/$task"
  mkdir -p "$task_out"
  if [[ "${HIF8_FORCE:-0}" != 1 ]] && find "$task_out" -type f -name 'results_*.json' -print -quit | grep -q .; then
    echo "[$mode] skip completed $task (set HIF8_FORCE=1 to rerun)"
    continue
  fi
  echo "[$mode] starting $task with batch_size=$eval_batch_size max_num_seqs=$max_num_seqs"
  /root/miniconda3/envs/hifi/bin/python -m lm_eval run --model vllm \
    "${include_args[@]}" \
    "${limit_args[@]}" \
    --model_args "pretrained=$model,dtype=bfloat16,trust_remote_code=True,tensor_parallel_size=1,max_model_len=16384,max_num_seqs=$max_num_seqs,max_num_batched_tokens=$max_num_batched_tokens,enforce_eager=True,async_scheduling=False,gpu_memory_utilization=0.8,kv_cache_memory_bytes=8589934592,enable_thinking=False" \
    --tasks "$task" --batch_size "$eval_batch_size" --seed 42 \
    --trust_remote_code --confirm_run_unsafe_code --log_samples \
    --output_path "$task_out"
done
