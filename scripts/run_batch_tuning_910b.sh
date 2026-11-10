#!/usr/bin/env bash
set -euo pipefail

# Small, resumable batch sweep for the single 910B card.  This deliberately
# runs one engine at a time: two 9B engines on the same NPU only contend for
# HBM/AIC and are slower than one scheduler with a larger in-flight batch.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out_root=${HIF8_BATCH_TUNING_ROOT:-$repo_root/benchmarks/qwen35-batch-tuning-910b}
mode=${1:-hif8}
case "$mode" in
  hif8) model=${HIF8_MODEL:-/filesdir/qwen3.5-9b-hif8-static} ;;
  bf16) model=${BF16_MODEL:-/filesdir/qwen3.5-9b} ;;
  *) echo "usage: $0 [hif8|bf16]" >&2; exit 2 ;;
esac
batch_list=${HIF8_BATCH_LIST:-"1 4 8 16 32"}
max_tokens=${HIF8_MAX_TOKENS:-32}

if [[ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi
opp_build=${HIF8_OPP_BUILD_DIR:-$repo_root/csrc/opp/build/ascend910b}
opp_vendor="$opp_build/package/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors/vllm_ascend_hif8"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export ASCEND_CUSTOM_OPP_PATH="$opp_vendor"
export HIF8_SOC=ascend910b HIF8_OPP_BUILD_DIR="$opp_build"
if [[ "$mode" == hif8 ]]; then
  export HIF8_COMPUTE_BACKEND=soft_hif8 HIF8_STATIC_FUSED_GENERATION=1 HIF8_KV_CACHE=1
  # Match the accuracy-evaluation path so throughput numbers are comparable
  # and do not hide batch-dependent Cube accumulation differences.
  export HIF8_STABLE_M_TILE=${HIF8_STABLE_M_TILE:-1}
else
  unset HIF8_COMPUTE_BACKEND HIF8_STATIC_FUSED_GENERATION HIF8_KV_CACHE || true
fi
export VLLM_ASCEND_DISABLE_CUSTOM_OP=1 VLLM_USE_V1=1 VLLM_NO_USAGE_STATS=1
export VLLM_ASCEND_GDN_CUSTOM_OP=${VLLM_ASCEND_GDN_CUSTOM_OP:-1}
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="/tmp/vllm-d02:/tmp/vllm-ascend-v026:$repo_root/src:/filesdir/lm-evaluation-harness:/usr/local/Ascend/cann-9.0.0/python/site-packages:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$out_root"
for batch in $batch_list; do
  output="$out_root/${mode}-batch${batch}.log"
  if [[ -s "$output" ]] && grep -q '^HIF8_BENCHMARK_JSON=' "$output"; then
    echo "skip completed batch=$batch"
    continue
  fi
  echo "starting $mode batch=$batch"
  /root/miniconda3/envs/hifi/bin/python "$repo_root/scripts/benchmark_model.py" "$model" \
    --max-model-len 128 --max-tokens "$max_tokens" --warmup 1 --repeats 2 \
    --kv-cache-mib 1024 --no-async-scheduling \
    --batch-size "$batch" --max-num-seqs "$batch" --max-num-batched-tokens 8192 \
    >"$output" 2>&1
done

echo "batch tuning complete: $out_root"
