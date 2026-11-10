#!/usr/bin/env bash
set -euo pipefail

# Download the additional benchmark datasets that are available on ModelScope.
# The four other MMLU-ProX languages in the requested list do not currently
# have a corresponding ModelScope dataset repository (see the final report).

env_name="${HIFI_ENV:-hifi}"
root="${EVAL_DATA_ROOT:-/filesdir/datasets}"
cache_dir="${MODELSCOPE_CACHE_DIR:-/filesdir/.cache/modelscope}"

command -v conda >/dev/null 2>&1 || {
    echo "未找到 conda；请先加载 conda 环境。" >&2
    exit 1
}
mkdir -p "$root" "$cache_dir"

if ! conda run -n "$env_name" python -c 'import modelscope' >/dev/null 2>&1; then
    echo "hifi 环境尚未安装 modelscope，正在安装..."
    conda run -n "$env_name" python -m pip install -U modelscope
fi

download_one() {
    local repo_id=$1 revision=$2 destination=$3
    mkdir -p "$destination"
    echo "通过魔搭下载 $repo_id -> $destination"
    conda run -n "$env_name" python -c '
import sys
from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download

repo_id, revision, destination, cache_dir = sys.argv[1:]
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    revision=revision,
    local_dir=str(Path(destination).resolve()),
    cache_dir=str(Path(cache_dir).resolve()),
)
print("下载完成：", Path(destination).resolve())
' "$repo_id" "$revision" "$destination" "$cache_dir"
}

download_one \
    "${MODELSCOPE_MMLU_PRO_REPO:-modelscope/MMLU-Pro}" \
    "${MODELSCOPE_MMLU_PRO_REVISION:-b051592813cace369cae88078d94665afcf85760}" \
    "$root/mmlu-pro"

download_one \
    "${MODELSCOPE_MMLU_PROX_EN_REPO:-tokyotech-llm/MMLU-ProX-English}" \
    "${MODELSCOPE_MMLU_PROX_EN_REVISION:-cbd447f3868e1af02e54e0d86573ad5763aab1d6}" \
    "$root/mmlu-prox-english"

echo "全部可用的附加数据集已下载到：$root"
