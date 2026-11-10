#!/usr/bin/env bash
set -euo pipefail

# Install pinned vLLM/vLLM-Ascend sources with --no-deps to preserve torch/npu.
# Reruns reuse clean checkouts; dirty checkouts are rejected.

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${HIF8_PYTHON:-}
ascend_home=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}
vllm_dir=${HIF8_VLLM_ROOT:-$repo_root/.deps/vllm}
vllm_ascend_dir=${HIF8_VLLM_ASCEND_ROOT:-$repo_root/.deps/vllm-ascend}
vllm_url=${HIF8_VLLM_REPO_URL:-https://github.com/vllm-project/vllm.git}
vllm_ascend_url=${HIF8_VLLM_ASCEND_REPO_URL:-https://github.com/vllm-project/vllm-ascend.git}
vllm_commit=${HIF8_VLLM_COMMIT:-d02df748bf9efd99022f1a062597dc3cb3808485}
vllm_ascend_commit=${HIF8_VLLM_ASCEND_COMMIT:-53b489b78560363871ba5402dd72cfd124c4b209}
compat_version=${VLLM_VERSION:-0.26.0}
patch_file=${HIF8_VLLM_ASCEND_PATCH:-$repo_root/patches/vllm-ascend-910b-hif8.patch}
offline=${HIF8_OFFLINE:-0}
allow_dirty=${HIF8_ALLOW_DIRTY_UPSTREAM:-0}
install_packages=1
apply_patch=1

usage() {
  cat <<'EOF'
Usage: scripts/install_vllm_ascend_910b.sh [options]

Clone or reuse the pinned source trees, apply the validated 910B patch, and
install both packages into the active Python environment without changing
torch/torch-npu or any other dependency.

Options:
  --python PATH             Python 3.11 executable (default: hifi env)
  --vllm-dir DIR            vLLM checkout (default: .deps/vllm)
  --vllm-ascend-dir DIR     vLLM-Ascend checkout (default: .deps/vllm-ascend)
  --offline                 Do not clone/fetch; both checkouts must exist
  --no-install              Prepare and patch source trees only
  --no-patch                Do not apply the repository's 910B patch
  --allow-dirty             Permit patching a dirty upstream checkout
  -h, --help                Show this help

Environment overrides:
  HIF8_VLLM_COMMIT, HIF8_VLLM_ASCEND_COMMIT, HIF8_VLLM_REPO_URL,
  HIF8_VLLM_ASCEND_REPO_URL, HIF8_VLLM_ASCEND_PATCH, HIF8_OFFLINE,
  HIF8_PYTHON, ASCEND_HOME_PATH, HIF8_ALLOW_DIRTY_UPSTREAM.
EOF
}

while (($#)); do
  case "$1" in
    --python) [[ $# -ge 2 ]] || { echo "--python requires a value" >&2; exit 2; }; python_bin=$2; shift 2 ;;
    --vllm-dir) [[ $# -ge 2 ]] || { echo "--vllm-dir requires a value" >&2; exit 2; }; vllm_dir=$2; shift 2 ;;
    --vllm-ascend-dir) [[ $# -ge 2 ]] || { echo "--vllm-ascend-dir requires a value" >&2; exit 2; }; vllm_ascend_dir=$2; shift 2 ;;
    --offline) offline=1; shift ;;
    --no-install) install_packages=0; shift ;;
    --no-patch) apply_patch=0; shift ;;
    --allow-dirty) allow_dirty=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$python_bin" ]]; then
  if [[ -x /root/miniconda3/envs/hifi/bin/python ]]; then
    python_bin=/root/miniconda3/envs/hifi/bin/python
  elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    python_bin="$CONDA_PREFIX/bin/python"
  else
    python_bin=$(command -v python3 || command -v python)
  fi
fi
[[ -x "$python_bin" ]] || { echo "Python executable not found: $python_bin" >&2; exit 1; }
[[ -d "$ascend_home" ]] || { echo "CANN toolkit not found: $ascend_home" >&2; exit 1; }
if [[ -f "$ascend_home/set_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ascend_home/set_env.sh"
fi
[[ -f "$patch_file" || "$apply_patch" == 0 ]] || {
  echo "validated upstream patch not found: $patch_file" >&2
  exit 1
}

clone_or_validate() {
  local dir=$1 url=$2 commit=$3 name=$4
  if [[ ! -d "$dir/.git" ]]; then
    if [[ "$offline" == 1 ]]; then
      echo "[$name] missing checkout in offline mode: $dir" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$dir")"
    echo "[$name] cloning $url -> $dir"
    git clone "$url" "$dir"
  fi
  local current
  current=$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)
  if [[ "$current" != "$commit" ]]; then
    if [[ -n "$(git -C "$dir" status --porcelain)" && "$allow_dirty" != 1 ]]; then
      echo "[$name] checkout is dirty; refusing to change $dir" >&2
      echo "Commit/stash it first, or rerun with --allow-dirty." >&2
      exit 1
    fi
    if ! git -C "$dir" cat-file -e "$commit^{commit}" 2>/dev/null; then
      [[ "$offline" == 0 ]] || { echo "[$name] commit unavailable offline: $commit" >&2; exit 1; }
      git -C "$dir" fetch --tags origin "$commit"
    fi
    echo "[$name] checking out pinned commit $commit"
    git -C "$dir" checkout --detach "$commit"
  fi
  [[ "$(git -C "$dir" rev-parse HEAD)" == "$commit" ]] || {
    echo "[$name] failed to select commit $commit" >&2
    exit 1
  }
}

clone_or_validate "$vllm_dir" "$vllm_url" "$vllm_commit" vllm
clone_or_validate "$vllm_ascend_dir" "$vllm_ascend_url" "$vllm_ascend_commit" vllm-ascend

if [[ "$apply_patch" == 1 ]]; then
  if git -C "$vllm_ascend_dir" apply --check "$patch_file" >/dev/null 2>&1; then
    echo "[vllm-ascend] applying 910B HIF8 patch"
    git -C "$vllm_ascend_dir" apply "$patch_file"
  elif git -C "$vllm_ascend_dir" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "[vllm-ascend] 910B HIF8 patch is already applied"
  else
    echo "[vllm-ascend] patch does not apply cleanly to $vllm_ascend_dir" >&2
    echo "Use the pinned commit $vllm_ascend_commit or inspect local changes." >&2
    exit 1
  fi
fi

before_versions=$("$python_bin" - <<'PY'
import torch
import torch_npu
print("HIF8_TORCH=" + torch.__version__)
print("HIF8_TORCH_NPU=" + torch_npu.__version__)
PY
)
before_torch=$(grep '^HIF8_TORCH=' <<<"$before_versions" | tail -1 | cut -d= -f2-)
before_torch_npu=$(grep '^HIF8_TORCH_NPU=' <<<"$before_versions" | tail -1 | cut -d= -f2-)
[[ -n "$before_torch" && -n "$before_torch_npu" ]] || {
  echo "unable to read torch versions from $python_bin" >&2
  exit 1
}

export VLLM_VERSION="$compat_version"
export VLLM_VERSION_OVERRIDE="$compat_version"
export SETUPTOOLS_SCM_PRETEND_VERSION="$compat_version"
export HIF8_SOC=${HIF8_SOC:-ascend910b}
export SOC_VERSION=${SOC_VERSION:-ascend910b1}
export ASCEND_PYTHON_EXECUTABLE="$python_bin"
# The HIF8 repository builds and loads its own 910B OPP. Building the full
# upstream vLLM-Ascend custom-op matrix is unnecessary for this path.
export COMPILE_CUSTOM_KERNELS=${COMPILE_CUSTOM_KERNELS:-0}
export PYTHONPATH="$vllm_dir:$vllm_ascend_dir:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$install_packages" == 1 ]]; then
  echo "[pip] installing vLLM source at $vllm_commit (compatibility target $compat_version)"
  "$python_bin" -m pip install --no-deps --no-build-isolation --no-cache-dir -e "$vllm_dir"
  echo "[pip] installing vLLM-Ascend source at $vllm_ascend_commit"
  "$python_bin" -m pip install --no-deps --no-build-isolation --no-cache-dir -e "$vllm_ascend_dir"
fi

after_versions=$("$python_bin" - <<'PY'
import torch
import torch_npu
import vllm
import vllm_ascend
import triton
try:
    from vllm_ascend._version import __version__ as vllm_ascend_version
except Exception:
    vllm_ascend_version = "unknown"
print("HIF8_TORCH=" + torch.__version__)
print("HIF8_TORCH_NPU=" + torch_npu.__version__)
print("HIF8_VLLM=" + vllm.__version__)
print("HIF8_VLLM_ASCEND=" + vllm_ascend_version)
print("HIF8_TRITON=" + triton.__version__)
PY
)
after_torch=$(grep '^HIF8_TORCH=' <<<"$after_versions" | tail -1 | cut -d= -f2-)
after_torch_npu=$(grep '^HIF8_TORCH_NPU=' <<<"$after_versions" | tail -1 | cut -d= -f2-)
after_vllm=$(grep '^HIF8_VLLM=' <<<"$after_versions" | tail -1 | cut -d= -f2-)
after_vllm_ascend=$(grep '^HIF8_VLLM_ASCEND=' <<<"$after_versions" | tail -1 | cut -d= -f2-)
after_triton=$(grep '^HIF8_TRITON=' <<<"$after_versions" | tail -1 | cut -d= -f2-)
[[ "$before_torch" == "$after_torch" ]] || {
  echo "torch changed during installation: $before_torch -> $after_torch" >&2
  exit 1
}
[[ "$before_torch_npu" == "$after_torch_npu" ]] || {
  echo "torch-npu changed during installation: $before_torch_npu -> $after_torch_npu" >&2
  exit 1
}

cat <<EOF

Installation complete.
  HIF8_VLLM_ROOT=$vllm_dir
  HIF8_VLLM_ASCEND_ROOT=$vllm_ascend_dir
  VLLM_VERSION=$compat_version (compatibility selector)
  vLLM source commit=$vllm_commit
  vLLM-Ascend source commit=$vllm_ascend_commit
  torch=$after_torch (unchanged)
  torch-npu=$after_torch_npu (unchanged)
  vLLM imported version=$after_vllm
  vLLM-Ascend imported version=$after_vllm_ascend
  Triton runtime=$after_triton

Run the server with:
  HIF8_VLLM_ROOT=$vllm_dir HIF8_VLLM_ASCEND_ROOT=$vllm_ascend_dir \\
    VLLM_VERSION=$compat_version scripts/deploy_910b_hif8.sh /path/to/model
EOF
