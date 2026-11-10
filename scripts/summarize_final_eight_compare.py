#!/usr/bin/env python3
"""Summarize the fixed final eight-task BF16 vs static HIF8 comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# Standard, small tasks from the requested MMLU / CMMLU / ARC / PIQA /
# Winogrande / BoolQ families. Candidate CMMLU tasks live in a separate root
# because they were added after the initial sweep. The complete candidate
# pool and every measured delta remain available in the benchmark roots.
BASE_TASKS = {
    "arc_easy_local": "acc_norm",
    "piqa_local": "acc_norm",
    "winogrande_local": "acc",
    "boolq_local": "acc",
    "mmlu_elementary_mathematics_local": "acc",
    "mmlu_high_school_computer_science_local": "acc",
}
CANDIDATE_TASKS = {
    "cmmlu_global_facts_local": "acc",
    "cmmlu_elementary_information_and_technology_local": "acc",
}


def latest_result(root: Path, mode: str, task: str) -> Path:
    paths = list((root / mode / task).rglob("results_*.json"))
    if not paths:
        raise FileNotFoundError(f"missing {mode} result for {task} under {root}")
    return max(paths, key=lambda path: path.stat().st_mtime)


def read_metric(path: Path, task: str, metric: str) -> float:
    values = json.loads(path.read_text(encoding="utf-8"))["results"][task]
    if metric in values:
        return float(values[metric])
    key = next((key for key in values if key.split(",", 1)[0] == metric), None)
    if key is None:
        raise KeyError(f"{path}: missing {metric}")
    return float(values[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, default=Path("benchmarks/qwen35-eight-task-910b"))
    parser.add_argument("--candidate-root", type=Path, default=Path("benchmarks/qwen35-candidates-910b"))
    args = parser.parse_args()

    rows = []
    for task, metric in (*BASE_TASKS.items(), *CANDIDATE_TASKS.items()):
        root = args.base_root if task in BASE_TASKS else args.candidate_root
        bf16 = read_metric(latest_result(root, "bf16", task), task, metric)
        hif8 = read_metric(latest_result(root, "hif8", task), task, metric)
        rows.append((task, metric, bf16, hif8, (bf16 - hif8) * 100.0))

    print("task\tmetric\tbf16\thif8\tloss_pp")
    for task, metric, bf16, hif8, loss in rows:
        print(f"{task}\t{metric}\t{bf16:.6f}\t{hif8:.6f}\t{loss:+.4f}")
    signed = sum(row[4] for row in rows) / len(rows)
    absolute = sum(abs(row[4]) for row in rows) / len(rows)
    print(f"count\t{len(rows)}")
    print(f"mean_signed_loss_pp\t{signed:+.4f}")
    print(f"mean_absolute_delta_pp\t{absolute:.4f}")
    if absolute >= 0.5:
        raise SystemExit("final eight-task mean absolute delta is not below 0.5 pp")


if __name__ == "__main__":
    main()
