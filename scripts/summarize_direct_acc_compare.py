#!/usr/bin/env python3
"""Compare raw dataset accuracy between a BF16 run and HIF8 runs.

The report intentionally keeps the metric in the same units as lm-eval:
dataset accuracy and its relative error against BF16.  It does not aggregate
accuracy differences into percentage-point summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def newest_result(root: Path, task: str) -> Path | None:
    paths: list[Path] = []
    for path in root.glob("**/results_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if task in data.get("results", {}):
            paths.append(path)
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def accuracy(path: Path, task: str) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    row = data["results"][task]
    value = row.get("acc,none", row.get("acc_norm,none"))
    if value is None:
        raise ValueError(f"no accuracy metric for {task} in {path}")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-root", type=Path, required=True)
    parser.add_argument("--hif8-root", type=Path, action="append", required=True)
    parser.add_argument("--hif8-label", action="append")
    parser.add_argument("--task", action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    labels = args.hif8_label or [root.name for root in args.hif8_root]
    if len(labels) != len(args.hif8_root):
        parser.error("--hif8-label must be supplied once per --hif8-root")

    bf16_tasks: set[str] = set()
    for path in args.bf16_root.glob("**/results_*.json"):
        try:
            bf16_tasks.update(json.loads(path.read_text())["results"])
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    tasks = args.task or sorted(bf16_tasks)

    lines = [
        "# Raw dataset accuracy comparison",
        "",
        "Relative error is `abs(HIF8 acc - BF16 acc) / BF16 acc`; values are shown "
        "per dataset without aggregate accuracy-difference metrics.",
        "",
        "| Dataset | BF16 acc | "
        + " | ".join(f"HIF8[{label}] acc | HIF8[{label}] relative error" for label in labels)
        + " |",
        "|---|---:|"
        + "---:|---:|" * len(args.hif8_root),
    ]

    for task in tasks:
        bf_path = newest_result(args.bf16_root, task)
        if bf_path is None:
            lines.append(f"| {task} | missing | " + " | ".join("missing | missing" for _ in args.hif8_root) + " |")
            continue
        bf = accuracy(bf_path, task)
        cells = [f"{bf:.6f}"]
        for root in args.hif8_root:
            path = newest_result(root, task)
            if path is None:
                cells.extend(("missing", "missing"))
                continue
            value = accuracy(path, task)
            relative = abs(value - bf) / abs(bf) * 100.0 if bf else float("nan")
            cells.extend((f"{value:.6f}", f"{relative:.6f}%"))
        lines.append("| " + task + " | " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
