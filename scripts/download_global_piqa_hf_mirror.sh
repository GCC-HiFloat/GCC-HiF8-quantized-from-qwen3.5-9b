#!/usr/bin/env bash
set -euo pipefail

env_name="${HIFI_ENV:-hifi}"
destination="${GLOBAL_PIQA_DEST:-/filesdir/datasets/global-piqa}"
endpoint="${HF_ENDPOINT:-https://hf-mirror.com}"
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

command -v conda >/dev/null 2>&1 || {
    echo "未找到 conda；请先加载 conda 环境。" >&2
    exit 1
}
conda run -n "$env_name" python "$repo_root/scripts/download_global_piqa_hf_mirror.py" \
    --destination "$destination" \
    --endpoint "$endpoint"
echo "数据集目录：$destination"
