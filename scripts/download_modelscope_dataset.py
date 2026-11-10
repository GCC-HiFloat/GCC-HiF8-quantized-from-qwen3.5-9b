#!/usr/bin/env python3
"""Download and validate one ModelScope dataset snapshot."""

import argparse
import json
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    args = parser.parse_args()

    destination = args.destination.resolve()
    cache_dir = args.cache_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(destination),
        cache_dir=str(cache_dir),
    )

    # A plain git clone leaves a tiny Git-LFS pointer. Reject that case so an
    # apparently successful download cannot be mistaken for usable eval data.
    data_file = destination / "data.json"
    size = data_file.stat().st_size if data_file.exists() else 0
    if not data_file.is_file() or size < 1_000_000:
        raise RuntimeError(
            f"{data_file} is missing or still an LFS pointer ({size} bytes)"
        )
    try:
        with data_file.open(encoding="utf-8") as fp:
            json.load(fp)
    except Exception as exc:
        raise RuntimeError(f"无法解析 {data_file}，下载可能不完整") from exc

    print(f"下载完成：{data_file} ({size:,} bytes)")


if __name__ == "__main__":
    main()
