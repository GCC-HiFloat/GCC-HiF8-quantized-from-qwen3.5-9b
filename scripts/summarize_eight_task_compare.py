#!/usr/bin/env python3
"""Summarize paired BF16/static-W8A8KV8 evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASK_METRICS = {
    "arc_easy_local": "acc_norm",
    "hellaswag_local": "acc_norm",
    "piqa_local": "acc_norm",
    "winogrande_local": "acc",
    "boolq_local": "acc",
    "mmlu_elementary_mathematics_local": "acc",
    "mmlu_high_school_computer_science_local": "acc",
    "cmmlu_elementary_commonsense_local": "acc",
    "cmmlu_chinese_driving_rule_local": "acc",
}


def latest_result(root: Path, mode: str, task: str) -> Path | None:
    candidates = list((root / mode / task).rglob("results_*.json"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def metric_value(path: Path, task: str, metric: str) -> float:
    result = json.loads(path.read_text(encoding="utf-8"))["results"][task]
    if metric in result:
        return float(result[metric])
    key = next((key for key in result if key.split(",", 1)[0] == metric), None)
    if key is None:
        raise KeyError(f"{path}: missing metric {metric}")
    return float(result[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("benchmarks/qwen35-eight-task-910b"),
    )
    args = parser.parse_args()

    rows = []
    missing = []
    for task, metric in TASK_METRICS.items():
        bf16_path = latest_result(args.root, "bf16", task)
        hif8_path = latest_result(args.root, "hif8", task)
        if bf16_path is None or hif8_path is None:
            missing.append(task)
            continue
        bf16 = metric_value(bf16_path, task, metric)
        hif8 = metric_value(hif8_path, task, metric)
        rows.append((task, metric, bf16, hif8, (bf16 - hif8) * 100.0))

    print("task\tmetric\tbf16\thif8\tloss_pp")
    for task, metric, bf16, hif8, loss in rows:
        print(f"{task}\t{metric}\t{bf16:.6f}\t{hif8:.6f}\t{loss:+.4f}")
    if rows:
        signed = sum(row[4] for row in rows) / len(rows)
        absolute = sum(abs(row[4]) for row in rows) / len(rows)
        print(f"mean_signed_loss_pp\t{signed:+.4f}")
        print(f"mean_absolute_delta_pp\t{absolute:.4f}")
    if missing:
        print("missing\t" + ",".join(missing))


if __name__ == "__main__":
    main()
