#!/usr/bin/env python3
"""Summarize paired adaptive-batch BF16 and static W8A8KV8 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = {
    "arc_easy_local": "acc_norm",
    "piqa_local": "acc_norm",
    "winogrande_local": "acc",
    "boolq_local": "acc",
    "mmlu_elementary_mathematics_local": "acc",
    "mmlu_high_school_computer_science_local": "acc",
    "cmmlu_elementary_commonsense_local": "acc",
    "cmmlu_chinese_driving_rule_local": "acc",
}


def latest_result(root: Path, mode: str, task: str) -> Path:
    results = list((root / mode / task).rglob("results_*.json"))
    if not results:
        raise FileNotFoundError(f"missing {mode} result for {task} under {root}")
    return max(results, key=lambda path: path.stat().st_mtime)


def metric_value(path: Path, task: str, metric: str) -> float:
    task_result = json.loads(path.read_text(encoding="utf-8"))["results"][task]
    if metric in task_result:
        return float(task_result[metric])
    key = next((key for key in task_result if key.split(",", 1)[0] == metric), None)
    if key is None:
        raise KeyError(f"{path}: missing {metric}")
    return float(task_result[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("benchmarks/qwen35-eight-task-910b-dynamic32-full"),
    )
    parser.add_argument("--max-mean-absolute-delta-pp", type=float, default=0.5)
    args = parser.parse_args()

    rows = []
    for task, metric in TASKS.items():
        bf16 = metric_value(latest_result(args.root, "bf16", task), task, metric)
        hif8 = metric_value(latest_result(args.root, "hif8", task), task, metric)
        loss_pp = (bf16 - hif8) * 100.0
        rows.append((task, metric, bf16, hif8, loss_pp))

    print("task\tmetric\tbf16\thif8\tloss_pp")
    for task, metric, bf16, hif8, loss_pp in rows:
        print(f"{task}\t{metric}\t{bf16:.6f}\t{hif8:.6f}\t{loss_pp:+.4f}")
    mean_signed = sum(row[4] for row in rows) / len(rows)
    mean_absolute = sum(abs(row[4]) for row in rows) / len(rows)
    print(f"count\t{len(rows)}")
    print(f"mean_signed_loss_pp\t{mean_signed:+.4f}")
    print(f"mean_absolute_delta_pp\t{mean_absolute:.4f}")
    if mean_absolute >= args.max_mean_absolute_delta_pp:
        raise SystemExit(
            f"mean absolute delta {mean_absolute:.4f} pp is not below "
            f"{args.max_mean_absolute_delta_pp:.4f} pp"
        )


if __name__ == "__main__":
    main()
