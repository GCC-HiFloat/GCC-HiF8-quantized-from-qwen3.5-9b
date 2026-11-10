#!/usr/bin/env bash
set -euo pipefail

# Deploy static W8A8KV8 Qwen3.5-9B on Ascend 910B.
# Build missing OPP/native extensions without changing dependency versions.

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${HIF8_MODEL:-/filesdir/qwen3.5-9b-hif8-static}
python_bin=${HIF8_PYTHON:-}
ascend_home=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}
opp_build=${HIF8_OPP_BUILD_DIR:-$repo_root/csrc/opp/build/ascend910b}
host=${HIF8_HOST:-0.0.0.0}
port=${HIF8_PORT:-8000}
served_model_name=${HIF8_SERVED_MODEL_NAME:-qwen3.5-9b-hif8-static}
max_model_len=${HIF8_MAX_MODEL_LEN:-16384}
max_num_seqs=${HIF8_MAX_NUM_SEQS:-16}
max_num_batched_tokens=${HIF8_MAX_NUM_BATCHED_TOKENS:-8192}
gpu_memory_utilization=${HIF8_GPU_MEMORY_UTILIZATION:-0.8}
kv_cache_memory_bytes=${HIF8_KV_CACHE_MEMORY_BYTES:-8589934592}
visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-0}
allow_unverified_npu=${HIF8_ALLOW_UNVERIFIED_NPU:-0}
build_opp=${HIF8_BUILD_OPP:-1}
build_extension=${HIF8_BUILD_EXTENSION:-1}
install_plugin=${HIF8_INSTALL_PLUGIN:-1}
check_only=0
detach=0
api_key=${HIF8_API_KEY:-}

usage() {
  cat <<'EOF'
Usage: scripts/deploy_910b_hif8.sh [options] [MODEL_DIR]

Deploy the validated static W8A8KV8 HIF8 checkpoint through the vLLM
OpenAI-compatible server.  The default command stays in the foreground.

Options:
  --check-only                 Validate NPU, CANN, model, OPP and Python env
  --no-build-opp               Fail if the 910B custom OPP is not built
  --no-build-extension         Fail if the local native extension is missing
  --no-install-plugin          Do not install this repository as editable
  --detach                     Start in background and write hif8-server.log
  --host HOST                  Listen address (default: 0.0.0.0)
  --port PORT                  Listen port (default: 8000)
  --max-model-len N            Maximum sequence length (default: 16384)
  --max-num-seqs N             Scheduler concurrency cap (default: 16)
  --max-num-batched-tokens N   Scheduler token cap (default: 8192)
  --api-key KEY                Require this API key (optional)
  -h, --help                   Show this help

Environment overrides:
  HIF8_MODEL, HIF8_PYTHON, HIF8_VLLM_ROOT, HIF8_VLLM_ASCEND_ROOT,
  VLLM_VERSION,
  ASCEND_HOME_PATH, HIF8_OPP_BUILD_DIR, ASCEND_RT_VISIBLE_DEVICES,
  HIF8_BUILD_OPP, HIF8_BUILD_EXTENSION, HIF8_INSTALL_PLUGIN,
  HIF8_GPU_MEMORY_UTILIZATION, HIF8_KV_CACHE_MEMORY_BYTES,
  HIF8_STABLE_M_TILE, HIF8_ALLOW_UNVERIFIED_NPU.
EOF
}

while (($#)); do
  case "$1" in
    --check-only) check_only=1; shift ;;
    --no-build-opp) build_opp=0; shift ;;
    --no-build-extension) build_extension=0; shift ;;
    --no-install-plugin) install_plugin=0; shift ;;
    --detach) detach=1; shift ;;
    --host) [[ $# -ge 2 ]] || { echo "--host requires a value" >&2; exit 2; }; host=$2; shift 2 ;;
    --port) [[ $# -ge 2 ]] || { echo "--port requires a value" >&2; exit 2; }; port=$2; shift 2 ;;
    --max-model-len) [[ $# -ge 2 ]] || { echo "--max-model-len requires a value" >&2; exit 2; }; max_model_len=$2; shift 2 ;;
    --max-num-seqs) [[ $# -ge 2 ]] || { echo "--max-num-seqs requires a value" >&2; exit 2; }; max_num_seqs=$2; shift 2 ;;
    --max-num-batched-tokens) [[ $# -ge 2 ]] || { echo "--max-num-batched-tokens requires a value" >&2; exit 2; }; max_num_batched_tokens=$2; shift 2 ;;
    --api-key) [[ $# -ge 2 ]] || { echo "--api-key requires a value" >&2; exit 2; }; api_key=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [[ "$model_dir" != "/filesdir/qwen3.5-9b-hif8-static" || -n "${HIF8_MODEL:-}" ]]; then
        echo "model path supplied more than once: $1" >&2; exit 2
      fi
      model_dir=$1; shift ;;
  esac
done

if [[ ! -d "$ascend_home" ]]; then
  echo "CANN toolkit not found: $ascend_home" >&2
  echo "Set ASCEND_HOME_PATH to the installed CANN 9.x directory." >&2
  exit 1
fi
if [[ -f "$ascend_home/set_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ascend_home/set_env.sh"
fi

if command -v npu-smi >/dev/null 2>&1; then
  npu_info=$(npu-smi info 2>&1 || true)
  printf '%s\n' "$npu_info" | sed -n '1,16p'
  if ! grep -Eiq '910B|910b' <<< "$npu_info" && [[ "$allow_unverified_npu" != 1 ]]; then
    echo "the visible accelerator is not identified as Ascend 910B" >&2
    echo "Set HIF8_ALLOW_UNVERIFIED_NPU=1 only for a controlled equivalent platform." >&2
    exit 1
  fi
elif [[ "$allow_unverified_npu" != 1 ]]; then
  echo "npu-smi is not available; cannot verify the 910B target" >&2
  echo "Set HIF8_ALLOW_UNVERIFIED_NPU=1 only for a controlled equivalent platform." >&2
  exit 1
fi

if [[ -z "$python_bin" ]]; then
  if [[ -x /root/miniconda3/envs/hifi/bin/python ]]; then
    python_bin=/root/miniconda3/envs/hifi/bin/python
  elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    python_bin="$CONDA_PREFIX/bin/python"
  else
    python_bin=$(command -v python3 || command -v python)
  fi
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Python executable not found: $python_bin" >&2
  exit 1
fi

if [[ ! -d "$model_dir" ]]; then
  echo "HIF8 checkpoint directory not found: $model_dir" >&2
  exit 1
fi
model_dir=$(cd "$model_dir" && pwd)

required_model_files=(config.json quant_model_description.json hif8_metadata.json model.safetensors.index.json)
for file in "${required_model_files[@]}"; do
  if [[ ! -f "$model_dir/$file" ]]; then
    echo "checkpoint is incomplete; missing $model_dir/$file" >&2
    exit 1
  fi
done

# Resolve the pinned upstream source pair before validating imported modules.
vllm_root=${HIF8_VLLM_ROOT:-}
if [[ -z "$vllm_root" ]]; then
  for candidate in "$repo_root/.deps/vllm" /tmp/vllm-d02 "$repo_root/../vllm" /opt/vllm; do
    if [[ -f "$candidate/vllm/__init__.py" ]]; then vllm_root=$candidate; break; fi
  done
fi
vllm_ascend_root=${HIF8_VLLM_ASCEND_ROOT:-}
if [[ -z "$vllm_ascend_root" ]]; then
  for candidate in "$repo_root/.deps/vllm-ascend" /tmp/vllm-ascend-v026 "$repo_root/../vllm-ascend" /opt/vllm-ascend; do
    if [[ -f "$candidate/vllm_ascend/__init__.py" ]]; then vllm_ascend_root=$candidate; break; fi
  done
fi

# VLLM_VERSION selects API patches for the pinned snapshot, not a wheel release.
expected_vllm_commit=d02df748bf9efd99022f1a062597dc3cb3808485
expected_vllm_ascend_commit=53b489b78560363871ba5402dd72cfd124c4b209
vllm_source_commit=""
vllm_ascend_source_commit=""
if [[ -n "$vllm_root" && -d "$vllm_root/.git" ]]; then
  vllm_source_commit=$(git -C "$vllm_root" rev-parse HEAD 2>/dev/null || true)
fi
if [[ -n "$vllm_ascend_root" && -d "$vllm_ascend_root/.git" ]]; then
  vllm_ascend_source_commit=$(git -C "$vllm_ascend_root" rev-parse HEAD 2>/dev/null || true)
fi
if [[ -z "${VLLM_VERSION:-}" && "$vllm_source_commit" == "$expected_vllm_commit" ]]; then
  export VLLM_VERSION=0.26.0
fi

opp_vendor="$opp_build/package/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors/vllm_ascend_hif8"
if [[ ! -d "$opp_vendor" ]]; then
  if [[ "$build_opp" != 1 ]]; then
    echo "910B custom OPP is missing: $opp_vendor" >&2
    echo "Run without --no-build-opp or set HIF8_OPP_BUILD_DIR." >&2
    exit 1
  fi
  echo "[hif8] building 910B custom OPP under $opp_build"
  ASCEND_HOME_PATH="$ascend_home" HIF8_OPP_BUILD_DIR="$opp_build" \
    bash "$repo_root/csrc/opp/build.sh" ascend910b
fi
for file in "$opp_vendor/op_api/lib/libvllm_ascend_hif8_opapi.so" \
            "$opp_vendor/op_proto/inc/op_proto.h"; do
  if [[ ! -e "$file" ]]; then
    echo "custom OPP payload is incomplete; missing $file" >&2
    exit 1
  fi
done

native_ext="$repo_root/src/vllm_ascend_hif8/_C$("$python_bin" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')"
if [[ ! -f "$native_ext" ]]; then
  if [[ "$build_extension" != 1 ]]; then
    echo "native extension is missing: $native_ext" >&2
    echo "Run without --no-build-extension or build with HIF8_SOC=ascend910b." >&2
    exit 1
  fi
  echo "[hif8] building native extension for $(basename "$native_ext")"
  ASCEND_HOME_PATH="$ascend_home" HIF8_SOC=ascend910b HIF8_OPP_BUILD_DIR="$opp_build" \
    "$python_bin" "$repo_root/setup.py" build_ext --inplace
fi

export ASCEND_RT_VISIBLE_DEVICES="$visible_devices"
export ASCEND_CUSTOM_OPP_PATH="$opp_vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export HIF8_SOC=ascend910b
export HIF8_OPP_BUILD_DIR="$opp_build"
export HIF8_COMPUTE_BACKEND=soft_hif8
export HIF8_KV_CACHE=1
export HIF8_STATIC_FUSED_GENERATION=1
export HIF8_STABLE_M_TILE=${HIF8_STABLE_M_TILE:-1}
export VLLM_ASCEND_DISABLE_CUSTOM_OP=${VLLM_ASCEND_DISABLE_CUSTOM_OP:-1}
export VLLM_ASCEND_GDN_CUSTOM_OP=${VLLM_ASCEND_GDN_CUSTOM_OP:-1}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_NO_USAGE_STATS=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HIF8_VLLM_SOURCE_COMMIT="$vllm_source_commit"
export HIF8_VLLM_ASCEND_SOURCE_COMMIT="$vllm_ascend_source_commit"

python_paths=()
[[ -n "$vllm_root" ]] && python_paths+=("$vllm_root")
[[ -n "$vllm_ascend_root" ]] && python_paths+=("$vllm_ascend_root")
python_paths+=("$repo_root/src")
if [[ -n "${PYTHONPATH:-}" ]]; then python_paths+=("$PYTHONPATH"); fi
joined_pythonpath=$(IFS=:; echo "${python_paths[*]}")
export PYTHONPATH="$joined_pythonpath"

echo "[hif8] checking validated environment"
"$python_bin" - "$model_dir" <<'PY'
import importlib.metadata as metadata
import json
import os
import sys
from pathlib import Path

model = Path(sys.argv[1])
meta = json.loads((model / "hif8_metadata.json").read_text())
quant = json.loads((model / "quant_model_description.json").read_text())
if meta.get("format") != "W8A8HIF8_STATIC":
    raise SystemExit(f"expected W8A8HIF8_STATIC checkpoint, got {meta.get('format')!r}")
if meta.get("byte_layout") != "ascend_hifloat8":
    raise SystemExit(f"unexpected checkpoint byte layout: {meta.get('byte_layout')!r}")
if not any(v == "W8A8HIF8_STATIC" for v in quant.values()):
    raise SystemExit("quant_model_description has no static HiFloat8 tensors")
kv = meta.get("kv_cache", {})
if kv.get("mode") != "static_per_layer_per_head":
    raise SystemExit(f"checkpoint KV metadata is not static_per_layer_per_head: {kv.get('mode')!r}")

import torch
import torch_npu
import triton
import vllm
try:
    from vllm_ascend._version import __version__ as vllm_ascend_version
except Exception:
    vllm_ascend_version = "unknown"

def version(mod):
    return getattr(mod, "__version__", "unknown")

print(f"python={sys.version.split()[0]}")
print(f"torch={version(torch)}")
print(f"torch_npu={version(torch_npu)}")
actual_vllm_version = version(vllm)
compat_vllm_version = os.environ.get("VLLM_VERSION", actual_vllm_version)
print(f"vllm_imported={actual_vllm_version}")
print(f"vllm_compatibility_target={compat_vllm_version}")
print(f"vllm_source_commit={os.environ.get('HIF8_VLLM_SOURCE_COMMIT') or 'unknown'}")
print(f"vllm_ascend_runtime={vllm_ascend_version}")
print(f"vllm_ascend_source_commit={os.environ.get('HIF8_VLLM_ASCEND_SOURCE_COMMIT') or 'unknown'}")
print(f"triton_runtime={version(triton)}")
for dist_name in ("vllm-ascend", "vllm-ascend-hif8", "triton-ascend"):
    try:
        print(f"{dist_name}_distribution={metadata.version(dist_name)}")
    except metadata.PackageNotFoundError:
        print(f"{dist_name}_distribution=not-installed")

if sys.version_info[:2] != (3, 11):
    raise SystemExit("validated runtime requires Python 3.11.x")
if not version(torch).startswith("2.10.0"):
    raise SystemExit("validated runtime requires torch 2.10.0")
if not version(torch_npu).startswith("2.10.0"):
    raise SystemExit("validated runtime requires torch-npu 2.10.0 (post release is also accepted)")
expected_vllm_commit = "d02df748bf9efd99022f1a062597dc3cb3808485"
expected_vllm_ascend_commit = "53b489b78560363871ba5402dd72cfd124c4b209"
vllm_source_commit = os.environ.get("HIF8_VLLM_SOURCE_COMMIT", "")
vllm_ascend_source_commit = os.environ.get("HIF8_VLLM_ASCEND_SOURCE_COMMIT", "")
if compat_vllm_version != "0.26.0":
    raise SystemExit(
        "validated vLLM Ascend compatibility target is 0.26.0; "
        f"set VLLM_VERSION=0.26.0 for the validated source snapshot, got {compat_vllm_version}"
    )
if actual_vllm_version != "0.26.0" and vllm_source_commit != expected_vllm_commit:
    raise SystemExit(
        "vLLM is neither the official 0.26.0 build nor the validated source "
        f"snapshot {expected_vllm_commit[:12]}; got version={actual_vllm_version}, "
        f"commit={vllm_source_commit or 'unknown'}"
    )
if vllm_ascend_version != "0.26.0" and vllm_ascend_source_commit != expected_vllm_ascend_commit:
    raise SystemExit(
        "vLLM Ascend is neither the 0.26.0 build nor the validated source "
        f"snapshot {expected_vllm_ascend_commit[:12]}; got version={vllm_ascend_version}, "
        f"commit={vllm_ascend_source_commit or 'unknown'}"
    )
if version(triton) != "3.2.0":
    raise SystemExit(f"validated Triton runtime is 3.2.0, got {version(triton)}")
PY

if [[ "$install_plugin" == 1 ]]; then
  if ! "$python_bin" -c 'import importlib.metadata as m; m.version("vllm-ascend-hif8")' >/dev/null 2>&1; then
    echo "[hif8] installing this repository's plugin without changing dependencies"
    HIF8_SOC=ascend910b HIF8_OPP_BUILD_DIR="$opp_build" \
      "$python_bin" -m pip install --no-deps --no-build-isolation -e "$repo_root"
  fi
fi

if [[ "$check_only" == 1 ]]; then
  echo "[hif8] environment and static checkpoint checks passed"
  exit 0
fi

server_args=(
  -m vllm.entrypoints.openai.api_server
  --model "$model_dir"
  --served-model-name "$served_model_name"
  --host "$host"
  --port "$port"
  --dtype bfloat16
  --trust-remote-code
  --tensor-parallel-size 1
  --max-model-len "$max_model_len"
  --max-num-seqs "$max_num_seqs"
  --max-num-batched-tokens "$max_num_batched_tokens"
  --gpu-memory-utilization "$gpu_memory_utilization"
  --kv-cache-memory-bytes "$kv_cache_memory_bytes"
  --enforce-eager
  --no-async-scheduling
  --disable-log-stats
)
if [[ -n "$api_key" ]]; then server_args+=(--api-key "$api_key"); fi

echo "[hif8] serving $model_dir on http://$host:$port/v1"
echo "[hif8] static W8A8KV8; HIF8_STATIC_FUSED_GENERATION=1; max_num_seqs=$max_num_seqs"
if [[ "$detach" == 1 ]]; then
  log_file=${HIF8_SERVER_LOG:-$repo_root/hif8-server.log}
  pid_file=${HIF8_SERVER_PID_FILE:-$repo_root/hif8-server.pid}
  nohup "$python_bin" "${server_args[@]}" >"$log_file" 2>&1 < /dev/null &
  server_pid=$!
  echo "$server_pid" > "$pid_file"
  echo "[hif8] started pid=$server_pid"
  echo "[hif8] log=$log_file"
else
  exec "$python_bin" "${server_args[@]}"
fi
