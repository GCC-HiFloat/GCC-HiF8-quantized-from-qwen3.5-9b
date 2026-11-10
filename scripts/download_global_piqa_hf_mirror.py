#!/usr/bin/env python3
"""Download Global PIQA TSV shards through a Hugging Face mirror."""

import argparse
import concurrent.futures
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path


def download_one(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size >= 100:
        return
    incomplete = path.with_suffix(path.suffix + ".incomplete")
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "global-piqa-downloader/1.0"})
            with urllib.request.urlopen(request, timeout=120) as source, incomplete.open("wb") as target:
                shutil.copyfileobj(source, target)
            incomplete.replace(path)
            return
        except (OSError, urllib.error.URLError) as exc:
            if attempt == 3:
                raise RuntimeError(f"下载失败：{url}") from exc
            time.sleep(attempt)


def download_repo(repo: str, revision: str, endpoint: str, destination: Path) -> None:
    api = f"{endpoint.rstrip('/')}/api/datasets/{repo}/tree/{revision}?recursive=true"
    request = urllib.request.Request(api, headers={"User-Agent": "global-piqa-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        entries = json.load(response)
    files = [entry["path"] for entry in entries if entry.get("type") == "file" and entry.get("path", "").startswith("data/") and entry["path"].endswith(".tsv")]
    if not files:
        raise RuntimeError(f"镜像仓库没有找到 TSV 文件：{repo}")
    jobs = []
    for name in files:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{endpoint.rstrip('/')}/datasets/{repo}/resolve/{revision}/{name}"
        jobs.append((url, path))
    workers = min(16, len(jobs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_one, url, path) for url, path in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    print(f"{repo}: {len(files)} 个 TSV 已下载")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    download_repo("mrlbenchmarks/global-piqa-nonparallel", "6777742fa3634c0583cda3b7f8a482ea7b1b0937", args.endpoint, root / "nonparallel")
    download_repo("mrlbenchmarks/global-piqa-parallel", "b0b18516a8bc2cb1106bce3dd4db32848ca715ea", args.endpoint, root / "parallel")


if __name__ == "__main__":
    main()
