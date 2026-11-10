#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 {bf16|hif8} TASK" >&2; exit 2
fi
mode=$1; task=$2
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
harness_root=${LMEVAL_ROOT:-/filesdir/lm-evaluation-harness}
# Ensure ACLNN/TBE libraries resolve to the CANN release used for the 910B
# custom-operator build, even when the caller did not source set_env.sh.
if [[ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi
out_root=${HIF8_BENCHMARK_ROOT:-$repo_root/benchmarks/qwen35-static-w8a8kv8-910b-20260826}
batch_size=${HIF8_BATCH_SIZE:-2}
dynamic_batch=${HIF8_DYNAMIC_BATCH:-0}
if [[ "$dynamic_batch" == 1 ]]; then
  eval_batch_size=auto
  scheduler_max_seqs=${HIF8_MAX_NUM_SEQS:-32}
  export HIF8_DISABLE_GEMV=${HIF8_DISABLE_GEMV:-1}
  export HIF8_STABLE_M_TILE=${HIF8_STABLE_M_TILE:-1}
else
  eval_batch_size=$batch_size
  scheduler_max_seqs=${HIF8_MAX_NUM_SEQS:-$batch_size}
  unset HIF8_DISABLE_GEMV || true
  unset HIF8_STABLE_M_TILE || true
fi
scheduler_max_tokens=${HIF8_MAX_NUM_BATCHED_TOKENS:-8192}
opp_build=${HIF8_OPP_BUILD_DIR:-$repo_root/csrc/opp/build/ascend910b}
opp_vendor="$opp_build/package/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors/vllm_ascend_hif8"
case "$mode" in
  bf16) model=/filesdir/qwen3.5-9b; unset HIF8_KV_CACHE ;;
  hif8) model=/filesdir/qwen3.5-9b-hif8-static; export HIF8_KV_CACHE=1 ;;
  *) echo "unsupported mode: $mode" >&2; exit 2 ;;
esac
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export ASCEND_CUSTOM_OPP_PATH="$opp_vendor"
export HIF8_SOC=ascend910b HIF8_OPP_BUILD_DIR="$opp_build" HIF8_COMPUTE_BACKEND=soft_hif8
# Enable static dispatch: static_quant + GEMM for model projections.
export HIF8_STATIC_FUSED_GENERATION="${HIF8_STATIC_FUSED_GENERATION:-1}"
export VLLM_ASCEND_DISABLE_CUSTOM_OP=1 VLLM_USE_V1=1 VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ASCEND_GDN_CUSTOM_OP=${VLLM_ASCEND_GDN_CUSTOM_OP:-1}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# Prioritize the pinned upstream sources on PYTHONPATH.
export PYTHONPATH="/tmp/vllm-d02:/tmp/vllm-ascend-v026:$repo_root/src:$harness_root:/usr/local/Ascend/cann-9.0.0/python/site-packages:${PYTHONPATH:-}"
# Use the mirror for datasets missing from the local cache.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p "$out_root/$mode/$task"
cd "$harness_root"
include_args=()
if [[ -n "${HIF8_INCLUDE_PATH:-}" ]]; then
  include_args+=(--include_path "$HIF8_INCLUDE_PATH")
fi
limit_args=()
if [[ -n "${HIF8_LIMIT:-}" ]]; then
  limit_args+=(--limit "$HIF8_LIMIT")
fi
exec /root/miniconda3/envs/hifi/bin/python -m lm_eval run --model vllm \
  "${include_args[@]}" \
  "${limit_args[@]}" \
  --model_args "pretrained=$model,dtype=bfloat16,trust_remote_code=True,tensor_parallel_size=1,max_model_len=16384,max_num_seqs=$scheduler_max_seqs,max_num_batched_tokens=$scheduler_max_tokens,enforce_eager=True,async_scheduling=False,gpu_memory_utilization=0.8,kv_cache_memory_bytes=8589934592,enable_thinking=False" \
  --tasks "$task" --batch_size "$eval_batch_size" --apply_chat_template --fewshot_as_multiturn --seed 42 \
  --trust_remote_code --confirm_run_unsafe_code --log_samples --output_path "$out_root/$mode/$task"
