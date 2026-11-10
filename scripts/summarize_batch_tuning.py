#!/usr/bin/env python3
"""Summarize resumable 910B batch-throughput logs into Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_log(path: Path) -> dict:
    rows = [line for line in path.read_text(errors="replace").splitlines()
            if line.startswith("HIF8_BENCHMARK_JSON=")]
    if not rows:
        raise ValueError(f"no completed benchmark in {path}")
    return json.loads(rows[-1].split("=", 1)[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    values: dict[str, dict[int, float]] = {"hif8": {}, "bf16": {}}
    for mode in values:
        for path in sorted(args.root.glob(f"{mode}-batch*.log")):
            try:
                batch = int(path.stem.rsplit("batch", 1)[1])
                values[mode][batch] = float(read_log(path)["aggregate_tokens_per_second"])
            except (ValueError, IndexError, KeyError, json.JSONDecodeError):
                continue
    batches = sorted(set(values["hif8"]) & set(values["bf16"]))
    lines = [
        "# 910B batch throughput: static W8A8KV8 HIF8 vs BF16", "",
        "HIF8 uses static activation scaling, physical U8 HiFloat8 weights/KV, "
        "`soft_hif8`, and `HIF8_STATIC_FUSED_GENERATION=1`.", "",
        "| Batch | HIF8 token/s | BF16 token/s | HIF8/BF16 |", "|---:|---:|---:|---:|",
    ]
    for batch in batches:
        h, b = values["hif8"][batch], values["bf16"][batch]
        lines.append(f"| {batch} | {h:.4f} | {b:.4f} | {h / b:.4f}x |")
    if batches:
        best = max(batches, key=lambda x: values["hif8"][x])
        lines += ["", f"HIF8 throughput peaks at batch **{best}** among measured points."]
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
