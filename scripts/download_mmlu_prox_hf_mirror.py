#!/usr/bin/env python3
"""Download the missing MMLU-ProX language shards from a HF mirror."""

import argparse
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    destination = args.destination.resolve()
    cache_dir = args.cache_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download only test and validation: lm-eval uses test for scoring and
    # validation for its 5-shot examples; no train split is needed.
    relative = [
        f"{language}/{split}"
        for language in ("en", "af", "ar", "bn", "cs")
        for split in ("test-00000-of-00001.parquet", "validation-00000-of-00001.parquet")
    ]
    for name in relative:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size >= 1000:
            continue
        url = f"{args.endpoint.rstrip('/')}/datasets/li-lab/MMLU-ProX/resolve/{args.revision}/{name}"
        incomplete = path.with_suffix(path.suffix + ".incomplete")
        for attempt in range(1, 4):
            try:
                print(f"下载 {name} ({url})")
                request = urllib.request.Request(url, headers={"User-Agent": "mmlu-prox-downloader/1.0"})
                with urllib.request.urlopen(request, timeout=120) as source, incomplete.open("wb") as target:
                    shutil.copyfileobj(source, target)
                incomplete.replace(path)
                break
            except (OSError, urllib.error.URLError) as exc:
                if attempt == 3:
                    raise RuntimeError(f"下载失败：{url}") from exc
                time.sleep(attempt)

    expected = [destination / name for name in relative]
    bad = []
    for path in expected:
        size = path.stat().st_size if path.exists() else 0
        with path.open("rb") if size >= 8 else open("/dev/null", "rb") as handle:
            first = handle.read(4)
            if size >= 8:
                handle.seek(-4, 2)
                last = handle.read(4)
            else:
                last = b""
        if size < 1000 or first != b"PAR1" or last != b"PAR1":
            bad.append(f"{path} ({size} bytes)")
    if bad:
        raise RuntimeError("下载文件校验失败：" + ", ".join(bad))
    print(f"下载完成：{destination}（{len(expected)} 个有效 Parquet 文件）")


if __name__ == "__main__":
    main()
