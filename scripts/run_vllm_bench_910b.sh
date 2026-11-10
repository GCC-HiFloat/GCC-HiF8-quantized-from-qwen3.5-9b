#!/usr/bin/env bash
# No installation or background server; one offline run on an idle 910B.
set -eo pipefail
if [[ $# != 4 ]]; then
  echo "Usage: bash $0 {bf16|hif8} {1|4} MODEL NEW_OUTPUT_DIR" >&2
  echo "Set HIF8_SMI_CARD to the physical card shown by npu-smi (required)." >&2
  exit 2
fi
mode=$1; batch=$2; model=$3; out=$4
[[ "$mode" == bf16 || "$mode" == hif8 ]] || exit 2
[[ "$batch" == 1 || "$batch" == 4 ]] || exit 2
[[ -n "${HIF8_SMI_CARD:-}" ]] || { echo 'HIF8_SMI_CARD is required' >&2; exit 2; }
input_len=${HIF8_BENCH_INPUT_LEN:-128}
output_len=${HIF8_BENCH_OUTPUT_LEN:-32}
num_prompts=${HIF8_BENCH_NUM_PROMPTS:-8}
num_warmups=${HIF8_BENCH_NUM_WARMUPS:-4}
max_model_len=${HIF8_BENCH_MAX_MODEL_LEN:-512}
kv_bytes=${HIF8_BENCH_KV_BYTES:-1073741824}
for value in "$input_len" "$output_len" "$num_prompts" "$num_warmups" "$max_model_len" "$kv_bytes"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Benchmark sizes must be positive integers' >&2; exit 2; }
done
(( input_len + output_len <= max_model_len )) || { echo 'Input + output exceeds max_model_len' >&2; exit 2; }
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
py=${HIF8_PYTHON:-/root/miniconda3/envs/hifi/bin/python}
vllm_root=${HIF8_VLLM_ROOT:-$repo/.deps/vllm}
ascend_root=${HIF8_VLLM_ASCEND_ROOT:-$repo/.deps/vllm-ascend}
[[ $(git -C "$vllm_root" rev-parse HEAD) == d02df748bf9efd99022f1a062597dc3cb3808485 ]] || exit 1
[[ $(git -C "$ascend_root" rev-parse HEAD) == 53b489b78560363871ba5402dd72cfd124c4b209 ]] || exit 1
[[ -f "$model/config.json" && ! -e "$out" ]] || exit 1
# Resolve relative paths before changing the working directory.
model=$(realpath "$model")
out=$(realpath -m "$out")
"$py" -S - "$model" "$mode" <<'PY'
import json
from pathlib import Path
import sys
model, mode = Path(sys.argv[1]), sys.argv[2]
config = json.loads((model / 'config.json').read_text())
if mode == 'hif8':
    meta = json.loads((model / 'hif8_metadata.json').read_text())
    if meta.get('format') != 'W8A8HIF8_STATIC' or meta.get('kv_cache', {}).get('mode') != 'static_per_layer_per_head':
        raise SystemExit('Expected static W8A8KV8 checkpoint')
elif config.get('quantization_config'):
    raise SystemExit('BF16 checkpoint must not declare quantization_config')
PY
source "${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}/set_env.sh"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export VLLM_VERSION=0.26.0 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NO_USAGE_STATS=1
export VLLM_ASCEND_DISABLE_CUSTOM_OP=1 VLLM_ASCEND_GDN_CUSTOM_OP=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$vllm_root:$ascend_root:$repo/src:$repo/scripts:${PYTHONPATH:-}"
plugins=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_service_profiling
unset HIF8_DISABLE_KV_CACHE HIF8_STATIC_SCALE_FACTOR HIF8_DISABLE_GEMV
unset VLLM_ASCEND_GDN_REFERENCE
quant_args=()
if [[ "$mode" == hif8 ]]; then
  [[ -f "$model/hif8_metadata.json" ]] || exit 1
  export HIF8_SOC=ascend910b HIF8_COMPUTE_BACKEND=soft_hif8
  export HIF8_KV_CACHE=1 HIF8_STATIC_FUSED_GENERATION=1 HIF8_STABLE_M_TILE=1
  export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-$repo/csrc/opp/build/ascend910b/package/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors/vllm_ascend_hif8}
  export VLLM_PLUGINS=$plugins,ascend_hif8
  quant_args=(--quantization ascend)
else
  [[ ! -f "$model/hif8_metadata.json" && ! -f "$model/quant_model_description.json" ]] || exit 1
  unset HIF8_KV_CACHE HIF8_STATIC_FUSED_GENERATION HIF8_COMPUTE_BACKEND HIF8_STABLE_M_TILE
  export VLLM_PLUGINS=$plugins
fi
cd "$repo"
exec "$py" scripts/collect_vllm_bench.py --out "$out" --card "$HIF8_SMI_CARD" -- \
  "$py" -m vllm.entrypoints.cli.main bench throughput \
  --backend vllm --model "$model" --dtype bfloat16 --trust-remote-code \
  --tensor-parallel-size 1 --max-num-seqs "$batch" \
  --scheduler-cls bench_scheduler.BenchmarkScheduler \
  --max-model-len "$max_model_len" --max-num-batched-tokens 8192 \
  --kv-cache-memory-bytes "$kv_bytes" --gpu-memory-utilization 0.8 \
  --enforce-eager --no-async-scheduling --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --dataset-name random --random-input-len "$input_len" --random-output-len "$output_len" --random-range-ratio 0 \
  --num-prompts "$num_prompts" --num-warmups "$num_warmups" --seed 42 \
  "${quant_args[@]}" --output-json "$out/throughput.json"
