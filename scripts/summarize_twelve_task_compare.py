#!/usr/bin/env python3
"""Summarize the 12-task 910B static HiF8 run against matched BF16.

The first eight tasks and four supplemental CMMLU subjects are kept in
separate roots so an interrupted run can resume without repeating completed
subjects.  This script discovers the newest results JSON for each task,
reports per-task deltas and aggregate error, and emits a compact Markdown
table suitable for the benchmark report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = [
    "arc_easy_local", "piqa_local", "winogrande_local", "boolq_local",
    "mmlu_elementary_mathematics_local",
    "mmlu_high_school_computer_science_local",
    "cmmlu_elementary_commonsense_local",
    "cmmlu_chinese_driving_rule_local",
    "cmmlu_college_actuarial_science_local",
    "cmmlu_college_engineering_hydrology_local",
    "cmmlu_college_medical_statistics_local",
    "cmmlu_philosophy_local",
]


def latest(root: Path, task: str) -> Path | None:
    files = list(root.glob(f"{task}/**/results_*.json"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def read(path: Path) -> tuple[float | None, float | None, int | None]:
    data = json.loads(path.read_text())
    row = next(iter(data.get("results", {}).values()), {})
    score = row.get("acc_norm,none", row.get("acc,none"))
    elapsed = data.get("total_evaluation_time_seconds")
    samples = data.get("n-samples", {})
    if isinstance(samples, dict) and "effective" in samples:
        n = samples.get("effective")
    elif isinstance(samples, dict):
        # lm-eval stores task-specific counts under the task key.
        nested = next(iter(samples.values()), {})
        n = nested.get("effective") if isinstance(nested, dict) else None
    else:
        n = None
    return (float(score) if score is not None else None,
            float(elapsed) if elapsed is not None else None,
            int(n) if n is not None else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hif8-root", type=Path, action="append", required=True)
    ap.add_argument("--bf16-root", type=Path, action="append", required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    lines = [
        "# 910B static HIF8 W8A8KV8: 12-task matched comparison", "",
        "Configuration: static activation scaling, `soft_hif8`, "
        "`HIF8_STATIC_FUSED_GENERATION=1`, matched batch/scheduler.", "",
        "| Task | Samples | BF16 | HIF8 | HIF8−BF16 (pp) | BF16/HIF8 time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    abs_deltas: list[float] = []
    weighted_abs_delta = 0.0
    weighted_signed_delta = 0.0
    total_samples = 0
    total_hif8_seconds = 0.0
    total_bf16_seconds = 0.0
    for task in TASKS:
        hp = next((p for root in args.hif8_root if (p := latest(root, task))), None)
        bp = next((p for root in args.bf16_root if (p := latest(root, task))), None)
        if not hp or not bp:
            lines.append(f"| {task} | — | pending | pending | — | — |")
            continue
        hs, ht, hn = read(hp)
        bs, bt, bn = read(bp)
        n = hn or bn
        if hs is None or bs is None:
            lines.append(f"| {task} | {n or '—'} | — | — | — | — |")
            continue
        delta = (hs - bs) * 100
        abs_deltas.append(abs(delta))
        if n is not None:
            weighted_abs_delta += abs(delta) * n
            weighted_signed_delta += delta * n
            total_samples += n
        if ht is not None:
            total_hif8_seconds += ht
        if bt is not None:
            total_bf16_seconds += bt
        times = f"{bt:.1f}/{ht:.1f}" if bt is not None and ht is not None else (f"—/{ht:.1f}" if ht is not None else "—")
        lines.append(f"| {task} | {n or '—'} | {bs:.4f} | {hs:.4f} | {delta:+.3f} | {times} |")
    if abs_deltas:
        avg = sum(abs_deltas) / len(abs_deltas)
        lines += [
            "",
            f"**Matched tasks: {len(abs_deltas)}/{len(TASKS)}; "
            f"macro mean absolute delta: {avg:.3f} pp.**",
        ]
        if total_samples:
            lines += [
                "",
                f"- Sample-weighted mean absolute task delta: "
                f"**{weighted_abs_delta / total_samples:.3f} pp** "
                f"over **{total_samples}** samples.",
                f"- Pooled signed HIF8-BF16 accuracy delta: "
                f"**{weighted_signed_delta / total_samples:+.3f} pp**.",
            ]
        if total_hif8_seconds and total_bf16_seconds:
            lines += [
                f"- Summed evaluation time: BF16 **{total_bf16_seconds:.1f}s**, "
                f"HIF8 **{total_hif8_seconds:.1f}s**; BF16-time/HIF8-time "
                f"ratio **{total_bf16_seconds / total_hif8_seconds:.3f}x**.",
            ]
    else:
        lines += ["", "No matched result pairs are complete yet."]
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
