#!/usr/bin/env bash
set -euo pipefail

# Isolate request-batch scaling from vLLM scheduling/model-loading overhead.
# Each point benchmarks a single representative projection in a fresh
# process because ACL/AscendC tiling caches are process-local.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out_root=${HIF8_GEMM_DIAG_ROOT:-$repo_root/benchmarks/qwen35-gemm-batch-diagnosis-910b}
opp_build=${HIF8_OPP_BUILD_DIR:-$repo_root/csrc/opp/build/ascend910b}
opp_vendor="$opp_build/package/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors/vllm_ascend_hif8"

if [[ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export ASCEND_CUSTOM_OPP_PATH="$opp_vendor"
export HIF8_SOC=ascend910b HIF8_OPP_BUILD_DIR="$opp_build"
export HIF8_COMPUTE_BACKEND=soft_hif8 HIF8_STATIC_FUSED_GENERATION=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$repo_root/src:/tmp/vllm-d02:/tmp/vllm-ascend-v026:/usr/local/Ascend/cann-9.0.0/python/site-packages:${PYTHONPATH:-}"
mkdir -p "$out_root"

run_one() {
  local mode=$1 label=$2 m=$3 n=$4 k=$5
  local output="$out_root/${mode}-${label}-m${m}-n${n}-k${k}.log"
  if [[ "${HIF8_FORCE:-0}" != 1 ]] && [[ -s "$output" ]] &&
     grep -q '^HIF8_BF16_BENCHMARK_JSON=' "$output"; then
    echo "skip completed $mode $label M=$m N=$n K=$k"
    return
  fi
  if [[ "$label" == stable ]]; then
    export HIF8_STABLE_M_TILE=1
  else
    unset HIF8_STABLE_M_TILE || true
  fi
  echo "starting $mode $label M=$m N=$n K=$k"
  /root/miniconda3/envs/hifi/bin/python \
    "$repo_root/scripts/benchmark_soft_hif8_vs_bf16.py" \
    --mode "$mode" --m "$m" --n "$n" --k "$k" \
    --warmup 5 --repeats 20 >"$output" 2>&1
}

# 4096x4096 is the common model-width projection.  Stable/adaptive tiling is
# compared directly because stable M was introduced only for reproducibility.
for m in 1 4 8 16 32; do
  run_one soft adaptive "$m" 4096 4096
  run_one soft stable "$m" 4096 4096
done

# The MLP gate/up projection dominates decoded weight traffic.
for m in 4 8 16 32; do
  run_one soft adaptive "$m" 24576 4096
done

# BF16 Cube reference for the common projection.
for m in 4 8 16 32; do
  run_one bf16 reference "$m" 4096 4096
done

echo "GEMM batch diagnosis complete: $out_root"
