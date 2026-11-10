#!/usr/bin/env python3
"""Compare per-sample multiple-choice predictions across lm-eval batches."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def latest_samples(root: Path, task: str) -> Path | None:
    files = list(root.glob(f"{task}/**/samples_{task}_*.jsonl"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def load(path: Path) -> dict[str, tuple[int, float, list[float]]]:
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        scores = [float(x[0]) for x in row["filtered_resps"]]
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        pred = order[0]
        margin = scores[order[0]] - scores[order[1]]
        key = row.get("doc_hash") or str(row["doc_id"])
        rows[key] = (pred, margin, scores)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    batch_roots = {b: args.root / f"batch{b}" / "hif8" for b in (8, 16, 32)}
    tasks = sorted({p.name for root in batch_roots.values() if root.exists() for p in root.iterdir() if p.is_dir()})
    print("| task | samples | 8↔16 flips | 8↔32 flips | 16↔32 flips | max changed-margin |")
    print("|---|---:|---:|---:|---:|---:|")
    for task in tasks:
        data = {}
        for batch, root in batch_roots.items():
            p = latest_samples(root, task)
            if p:
                data[batch] = load(p)
        if len(data) < 2:
            continue
        common = set.intersection(*(set(v) for v in data.values()))
        parts = []
        changed_margins = []
        for a, b in ((8, 16), (8, 32), (16, 32)):
            if a not in data or b not in data:
                parts.append("—")
                continue
            changed = [k for k in common if data[a][k][0] != data[b][k][0]]
            parts.append(str(len(changed)))
            changed_margins.extend(min(data[a][k][1], data[b][k][1]) for k in changed)
        max_margin = max(changed_margins, default=0.0)
        print(f"| {task} | {len(common)} | {parts[0]} | {parts[1]} | {parts[2]} | {max_margin:.6f} |")


if __name__ == "__main__":
    main()
