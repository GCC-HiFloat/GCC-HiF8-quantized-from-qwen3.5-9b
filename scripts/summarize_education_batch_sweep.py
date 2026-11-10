#!/usr/bin/env python3
"""Summarize static HiF8 education-task batch sweep against BF16 references."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

TASKS = [
    "mmlu_elementary_mathematics_local",
    "mmlu_high_school_computer_science_local",
    "cmmlu_elementary_commonsense_local",
    "cmmlu_chinese_driving_rule_local",
    "cmmlu_college_mathematics_local",
    "cmmlu_philosophy_local",
    "cmmlu_college_actuarial_science_local",
    "cmmlu_college_engineering_hydrology_local",
    "cmmlu_college_medical_statistics_local",
    "cmmlu_chinese_foreign_policy_local",
    "cmmlu_high_school_geography_local",
    "cmmlu_computer_science_local",
]


def latest_result(root: Path, task: str) -> Path | None:
    files = list(root.glob(f"{task}/**/results_*.json"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def read(path: Path) -> tuple[float | None, float | None]:
    data = json.loads(path.read_text())
    row = next(iter(data.get("results", {}).values()), {})
    score = row.get("acc_norm,none", row.get("acc,none"))
    t = data.get("total_evaluation_time_seconds")
    return (float(score) if score is not None else None,
            float(t) if t is not None else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", type=Path, default=Path("benchmarks/qwen35-education-batch-sweep-910b"))
    ap.add_argument("--bf16-root", type=Path, action="append")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    bf16_roots = args.bf16_root or [
        Path("benchmarks/qwen35-eight-task-910b-dynamic32-full/bf16"),
        Path("benchmarks/qwen35-extra-six-910b/bf16"),
        Path("benchmarks/qwen35-candidates-910b/bf16"),
    ]
    refs: dict[str, float] = {}
    for task in TASKS:
        paths = [p for root in bf16_roots if (p := latest_result(root, task))]
        if paths:
            refs[task], _ = read(max(paths, key=lambda p: p.stat().st_mtime))

    lines = ["# 910B 静态 HiF8 W8A8KV8 教育类 batch 对比", "",
             "配置：`HIF8_STATIC_FUSED_GENERATION=1`、`HIF8_COMPUTE_BACKEND=soft_hif8`、静态 scaling；BF16 参考来自 dynamic32-full。", "",
             "| batch | 任务 | HiF8 acc_norm | BF16 acc_norm | 差值 (pp) | 用时(s) |", "|---:|---|---:|---:|---:|---:|"]
    for batch in (8, 16, 32):
        vals: list[float] = []
        for task in TASKS:
            p = latest_result(args.sweep_root / f"batch{batch}" / "hif8", task)
            if not p:
                lines.append(f"| {batch} | {task} | 进行中 | — | — | — |")
                continue
            score, elapsed = read(p)
            ref = refs.get(task)
            delta = (score - ref) * 100 if score is not None and ref is not None else None
            if score is not None and ref is not None:
                vals.append(abs(delta))
            elapsed_s = f"{elapsed:.1f}" if elapsed is not None else "—"
            lines.append(f"| {batch} | {task} | {score:.6f} | {ref:.6f} | {delta:+.3f} | {elapsed_s} |" if score is not None and ref is not None else f"| {batch} | {task} | — | — | — | — |")
        if vals:
            avg = sum(vals) / len(vals)
            status = "通过" if avg < 0.5 else "未通过"
            lines.append(f"| **{batch} 平均绝对误差** |  |  |  | **{avg:.3f} pp ({status})** |  |")
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
