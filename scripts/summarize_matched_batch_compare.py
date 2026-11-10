#!/usr/bin/env python3
"""Summarize matched BF16/HIF8 accuracy and batch sensitivity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = (
    "arc_easy_local",
    "piqa_local",
    "winogrande_local",
    "boolq_local",
    "mmlu_elementary_mathematics_local",
    "mmlu_high_school_computer_science_local",
    "cmmlu_elementary_commonsense_local",
    "cmmlu_chinese_driving_rule_local",
    "cmmlu_college_actuarial_science_local",
    "cmmlu_college_engineering_hydrology_local",
    "cmmlu_college_medical_statistics_local",
    "cmmlu_philosophy_local",
)


def latest_sample(root: Path, task: str) -> Path | None:
    files = list(root.glob(f"**/samples_{task}_*.jsonl"))
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def latest_result(root: Path, task: str) -> Path | None:
    matches: list[Path] = []
    for path in root.glob("**/results_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if task in data.get("results", {}):
            matches.append(path)
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def result(path: Path, task: str) -> tuple[int, float, float | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    row = data["results"][task]
    score = row.get("acc_norm,none", row.get("acc,none"))
    elapsed = data.get("total_evaluation_time_seconds") if len(data["results"]) == 1 else None
    return int(row["sample_len"]), float(score), elapsed


def evaluation_time(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("total_evaluation_time_seconds") or 0.0)


def predictions(path: Path) -> dict[str, int]:
    output: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        scores = [float(value[0]) for value in row["filtered_resps"]]
        key = row.get("doc_hash") or str(row["doc_id"])
        output[key] = max(range(len(scores)), key=scores.__getitem__)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batches", default="8,32")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    batches = tuple(int(value) for value in args.batches.split(","))

    lines = [
        "# 910B 12-task matched-batch comparison",
        "",
        "Static activation scaling, physical U8 HiFloat8 weights/KV, "
        "`soft_hif8`, and `HIF8_STATIC_FUSED_GENERATION=1`.",
        "",
        "| Batch | Task | Samples | BF16 | HIF8 | HIF8-BF16 (pp) | BF16/HIF8 time (s) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for batch in batches:
        deltas: list[float] = []
        result_paths: dict[str, set[Path]] = {"bf16": set(), "hif8": set()}
        completed = 0
        for task in TASKS:
            roots = {
                mode: args.root / f"batch{batch}" / mode for mode in ("bf16", "hif8")
            }
            paths = {mode: latest_result(root, task) for mode, root in roots.items()}
            if not all(paths.values()):
                lines.append(f"| {batch} | {task} | pending | - | - | - | - |")
                continue
            bf16_n, bf16_score, bt = result(paths["bf16"], task)
            hif8_n, hif8_score, ht = result(paths["hif8"], task)
            if bf16_n != hif8_n:
                raise ValueError(f"sample mismatch for batch={batch} task={task}")
            delta = (hif8_score - bf16_score) * 100.0
            deltas.append(abs(delta))
            result_paths["bf16"].add(paths["bf16"])
            result_paths["hif8"].add(paths["hif8"])
            completed += 1
            lines.append(
                f"| {batch} | {task} | {bf16_n} | {bf16_score:.4f} | "
                f"{hif8_score:.4f} | {delta:+.3f} | "
                f"{f'{bt:.1f}' if bt is not None else 'grouped'}/{f'{ht:.1f}' if ht is not None else 'grouped'} |"
            )
        bf16_time = sum(evaluation_time(path) for path in result_paths["bf16"])
        hif8_time = sum(evaluation_time(path) for path in result_paths["hif8"])
        mae = sum(deltas) / len(deltas) if deltas else float("nan")
        ratio = bf16_time / hif8_time if hif8_time else float("nan")
        lines.extend(
            [
                f"| **{batch} summary** | **{completed}/12 tasks** |  |  |  | "
                f"**macro MAE {mae:.3f} pp** | **BF16/HIF8 total-time {ratio:.3f}x** |",
                "",
            ]
        )

    lines.extend(
        [
            "## HIF8 vs BF16 prediction flips",
            "",
            "| Batch | Task | Common samples | Flips | Flip rate |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for batch in batches:
        total_common = total_flips = 0
        for task in TASKS:
            roots = [args.root / f"batch{batch}" / mode for mode in ("bf16", "hif8")]
            paths = [latest_sample(root, task) for root in roots]
            if not all(paths):
                lines.append(f"| {batch} | {task} | pending | - | - |")
                continue
            left, right = (predictions(path) for path in paths)
            common = set(left) & set(right)
            flips = sum(left[key] != right[key] for key in common)
            total_common += len(common)
            total_flips += flips
            rate = 100.0 * flips / len(common) if common else 0.0
            lines.append(f"| {batch} | {task} | {len(common)} | {flips} | {rate:.2f}% |")
        rate = 100.0 * total_flips / total_common if total_common else 0.0
        lines.append(
            f"| **{batch} total** |  | **{total_common}** | **{total_flips}** | **{rate:.2f}%** |"
        )
    lines.append("")

    if len(batches) >= 2:
        first, last = batches[0], batches[-1]
        lines.extend(
            [
                f"## Prediction flips: batch {first} vs {last}",
                "",
                "| Mode | Task | Common samples | Flips | Flip rate |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for mode in ("bf16", "hif8"):
            total_common = total_flips = 0
            for task in TASKS:
                roots = [args.root / f"batch{batch}" / mode for batch in (first, last)]
                paths = [latest_sample(root, task) for root in roots]
                if not all(paths):
                    lines.append(f"| {mode} | {task} | pending | - | - |")
                    continue
                left, right = (predictions(path) for path in paths)
                common = set(left) & set(right)
                flips = sum(left[key] != right[key] for key in common)
                total_common += len(common)
                total_flips += flips
                rate = 100.0 * flips / len(common) if common else 0.0
                lines.append(f"| {mode} | {task} | {len(common)} | {flips} | {rate:.2f}% |")
            rate = 100.0 * total_flips / total_common if total_common else 0.0
            lines.append(
                f"| **{mode} total** |  | **{total_common}** | **{total_flips}** | **{rate:.2f}%** |"
            )

    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
