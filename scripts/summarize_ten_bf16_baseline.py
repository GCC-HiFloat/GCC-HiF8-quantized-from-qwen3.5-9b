#!/usr/bin/env python3
"""Create a row-count and raw-score report for the ten-task BF16 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = {
    "arc_easy_local": ("ARC-Easy", 2376),
    "arc_challenge_ten": ("ARC-Challenge", 1172),
    "piqa_local": ("PIQA", 1838),
    "winogrande_local": ("WinoGrande", 1267),
    "boolq_local": ("BoolQ", 3270),
    "hellaswag_local": ("HellaSwag", 10042),
    "sciq_ten": ("SciQ", 1000),
    "gsm8k_ten": ("GSM8K", 1319),
    "lambada_openai_ten": ("LAMBADA OpenAI", 5153),
    "mmlu_all_ten": ("MMLU (all)", 14042),
}

EXPECTED_MODEL = "/filesdir/qwen3.5-9b"
EXPECTED_BATCH_SIZE = "4"
EXPECTED_MAX_NUM_SEQS = 4


def result_file(root: Path, task: str) -> Path | None:
    paths = []
    for path in root.joinpath(task).glob("**/results_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if task in data.get("results", {}):
            paths.append(path)
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def score(row: dict) -> tuple[str, float] | tuple[None, None]:
    # Prefer the requested raw accuracy.  Generation-only GSM8K has exact_match.
    for key in ("acc,none", "acc_norm,none", "exact_match,none", "exact_match,strict-match", "exact_match,flexible-extract"):
        if key in row:
            return key, float(row[key])
    for key, value in row.items():
        if key.startswith("exact_match") and isinstance(value, (int, float)):
            return key, float(value)
    return None, None


def validate_result(data: dict, task: str, expected_samples: int) -> list[str]:
    issues = []
    config = data.get("config", {})
    model_args = config.get("model_args", {})
    row = data.get("results", {}).get(task, {})
    if model_args.get("pretrained") != EXPECTED_MODEL:
        issues.append(f"model={model_args.get('pretrained')!r}")
    if str(config.get("batch_size")) != EXPECTED_BATCH_SIZE:
        issues.append(f"batch_size={config.get('batch_size')!r}")
    if model_args.get("max_num_seqs") != EXPECTED_MAX_NUM_SEQS:
        issues.append(f"max_num_seqs={model_args.get('max_num_seqs')!r}")
    if config.get("limit") is not None:
        issues.append(f"limit={config.get('limit')!r}")
    if row.get("sample_len") != expected_samples:
        issues.append(
            f"sample_len={row.get('sample_len')!r}, expected={expected_samples}"
        )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lines = [
        "# Ten-dataset BF16 batch=4 baseline",
        "",
        "Reference model: `/filesdir/qwen3.5-9b`, dtype `bfloat16`, scheduler `max_num_seqs=4`, `--batch_size 4`, seed 42.",
        "Scores are the raw lm-eval metric; classification tasks use `acc,none` (not `acc_norm`). GSM8K is generation-only and therefore reports exact match.",
        "",
        "| Dataset | Task | Samples | Metric | BF16 score | Validation | Result file |",
        "|---|---|---:|---|---:|---|---|",
    ]
    complete = 0
    total_samples = 0
    for task, (label, expected_samples) in TASKS.items():
        path = result_file(args.root, task)
        if path is None:
            lines.append(f"| {label} | `{task}` | pending/{expected_samples} | - | - | pending | - |")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        row = data["results"][task]
        metric, value = score(row)
        samples = row.get("sample_len", "?")
        issues = validate_result(data, task, expected_samples)
        if metric is None:
            issues.append("metric missing")
            lines.append(
                f"| {label} | `{task}` | {samples}/{expected_samples} | missing | - | "
                f"FAIL: {'; '.join(issues)} | `{path}` |"
            )
        else:
            validation = "PASS" if not issues else f"FAIL: {'; '.join(issues)}"
            lines.append(
                f"| {label} | `{task}` | {samples}/{expected_samples} | `{metric}` | "
                f"{value:.6f} | {validation} | `{path}` |"
            )
            if not issues:
                complete += 1
                total_samples += expected_samples
    expected_total = sum(samples for _, samples in TASKS.values())
    lines += [
        "",
        f"Validated complete: **{complete}/{len(TASKS)}** tasks, "
        f"**{total_samples}/{expected_total}** samples.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
