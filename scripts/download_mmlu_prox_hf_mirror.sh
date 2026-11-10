#!/usr/bin/env bash
set -euo pipefail

env_name="${HIFI_ENV:-hifi}"
root="${EVAL_DATA_ROOT:-/filesdir/datasets}"
destination="${MMLU_PROX_DEST:-$root/mmlu-prox}"
cache_dir="${HF_CACHE_DIR:-/filesdir/.cache/huggingface}"
endpoint="${HF_ENDPOINT:-https://hf-mirror.com}"
revision="${MMLU_PROX_REVISION:-8e6106a6c6ce1c5027e66cc338143cf997b2aa09}"
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

command -v conda >/dev/null 2>&1 || {
    echo "未找到 conda；请先加载 conda 环境。" >&2
    exit 1
}

if ! conda run -n "$env_name" python -c 'import huggingface_hub' >/dev/null 2>&1; then
    echo "hifi 环境尚未安装 huggingface_hub，正在安装..."
    conda run -n "$env_name" python -m pip install -U huggingface_hub
fi

echo "通过 Hugging Face 国内镜像 $endpoint 下载 li-lab/MMLU-ProX -> $destination"
conda run -n "$env_name" python "$repo_root/scripts/download_mmlu_prox_hf_mirror.py" \
    --destination "$destination" \
    --cache-dir "$cache_dir" \
    --endpoint "$endpoint" \
    --revision "$revision"

echo "数据集目录：$destination"
