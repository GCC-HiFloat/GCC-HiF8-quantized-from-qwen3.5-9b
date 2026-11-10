#!/usr/bin/env bash
set -euo pipefail

# Download the dataset missing from the HIF8 evaluation matrix.
# Usage:
#   bash scripts/download_missing_eval_datasets_modelscope.sh
#   bash scripts/download_missing_eval_datasets_modelscope.sh /filesdir/my-data

env_name="${HIFI_ENV:-hifi}"
dest="${1:-/filesdir/datasets/longbench-v2}"
repo_id="${MODELSCOPE_LONGBENCH_REPO:-ZhipuAI/LongBench-v2}"
# Pin the known complete snapshot (data.json is 465,490,535 bytes). Override
# this when intentionally tracking a newer ModelScope revision.
revision="${MODELSCOPE_LONGBENCH_REVISION:-6a226395b12a1438f40dfabbe0c5ea55f6a102bc}"
cache_dir="${MODELSCOPE_CACHE_DIR:-/filesdir/.cache/modelscope}"

command -v conda >/dev/null 2>&1 || {
    echo "未找到 conda；请先加载 conda 环境。" >&2
    exit 1
}

mkdir -p "$dest" "$cache_dir"
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# Keep the dependency and the download in the requested hifi environment.
if ! conda run -n "$env_name" python -c 'import modelscope' >/dev/null 2>&1; then
    echo "hifi 环境尚未安装 modelscope，正在安装..."
    conda run -n "$env_name" python -m pip install -U modelscope
fi

echo "通过魔搭下载 $repo_id -> $dest"
conda run -n "$env_name" python "$repo_root/scripts/download_modelscope_dataset.py" \
    --repo-id "$repo_id" \
    --revision "$revision" \
    --destination "$dest" \
    --cache-dir "$cache_dir"

echo "数据集目录：$dest"
