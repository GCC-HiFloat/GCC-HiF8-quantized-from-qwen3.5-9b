#!/usr/bin/env python3
"""Summarize raw accuracy for the ten-task BF16/HIF8 batch-4 runs.

The acceptance criterion is aggregate accuracy degradation:
``max(0, BF16 - HIF8) * 100 <= 0.5`` percentage points. A higher HIF8 score
is not an accuracy loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = (
    "arc_easy_local",
    "arc_challenge_ten",
    "piqa_local",
    "winogrande_local",
    "boolq_local",
    "hellaswag_local",
    "sciq_ten",
    "gsm8k_ten",
    "lambada_openai_ten",
    "mmlu_all_ten",
)


def newest_result(root: Path, task: str) -> Path | None:
    candidates: list[Path] = []
    for path in root.glob("**/results_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if task in data.get("results", {}):
            candidates.append(path)
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def score(path: Path, task: str) -> tuple[float, str, int]:
    row = json.loads(path.read_text(encoding="utf-8"))["results"][task]
    if task == "gsm8k_ten":
        metric = "exact_match,strict-match"
    else:
        metric = "acc,none"
    if metric not in row:
        raise ValueError(f"{metric} missing for {task} in {path}")
    return float(row[metric]), metric, int(row.get("sample_len", 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-root", type=Path, required=True)
    parser.add_argument("--hif8-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[str] = [
        "# Ten-dataset raw acc comparison (batch=4)",
        "",
        "Reference: BF16 `/filesdir/qwen3.5-9b`; candidate: static W8A8KV8 HIF8 "
        "`/filesdir/qwen3.5-9b-hif8-static` on 910B.",
        "The comparison uses raw `acc,none`; GSM8K uses strict exact match. "
        "Acceptance uses aggregate HIF8 accuracy loss: "
        "`max(0, BF16 - HIF8) * 100 <= 0.5` percentage points.",
        "",
        "| Dataset | Samples | Metric | BF16 | HIF8 | HIF8-BF16 (pp) | HIF8 loss (pp) | Diagnostic |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    complete = 0
    total_samples = 0
    passed = 0
    weighted_bf16 = 0.0
    weighted_hif8 = 0.0
    macro_bf16 = 0.0
    macro_hif8 = 0.0
    for task in TASKS:
        bf_path = newest_result(args.bf16_root, task)
        hf_path = newest_result(args.hif8_root, task)
        if bf_path is None or hf_path is None:
            rows.append(f"| {task} | — | — | {'—' if bf_path is None else 'present'} | {'—' if hf_path is None else 'present'} | — | — | PENDING |")
            continue
        bf, metric, samples = score(bf_path, task)
        hf, hf_metric, hf_samples = score(hf_path, task)
        if metric != hf_metric:
            raise ValueError(f"metric mismatch for {task}: {metric} vs {hf_metric}")
        delta = (hf - bf) * 100.0
        loss = max(0.0, -delta)
        gate = "PASS" if loss <= 0.5 else "FAIL"
        passed += gate == "PASS"
        complete += 1
        total_samples += samples
        weighted_bf16 += bf * samples
        weighted_hif8 += hf * samples
        macro_bf16 += bf
        macro_hif8 += hf
        if hf_samples != samples:
            gate += f" (sample mismatch {hf_samples})"
        rows.append(
            f"| {task} | {samples} | `{metric}` | {bf:.6f} | {hf:.6f} | "
            f"{delta:+.4f} | {loss:.4f} | {gate} |"
        )

    if complete:
        weighted_bf16 /= total_samples
        weighted_hif8 /= total_samples
        macro_bf16 /= complete
        macro_hif8 /= complete
        weighted_delta = (weighted_hif8 - weighted_bf16) * 100.0
        macro_delta = (macro_hif8 - macro_bf16) * 100.0
        weighted_loss = max(0.0, -weighted_delta)
        macro_loss = max(0.0, -macro_delta)
        weighted_gate = "PASS" if weighted_loss <= 0.5 else "FAIL"
        macro_gate = "PASS" if macro_loss <= 0.5 else "FAIL"
        rows += [
            f"| **Sample-weighted aggregate** | **{total_samples:,}** | — | "
            f"**{weighted_bf16:.6f}** | **{weighted_hif8:.6f}** | "
            f"**{weighted_delta:+.4f}** | **{weighted_loss:.4f}** | **{weighted_gate}** |",
            f"| **Macro average** | **{complete} tasks** | — | "
            f"**{macro_bf16:.6f}** | **{macro_hif8:.6f}** | "
            f"**{macro_delta:+.4f}** | **{macro_loss:.4f}** | **{macro_gate}** |",
        ]

    rows += [
        "",
        f"Completed: **{complete}/{len(TASKS)}** tasks, **{total_samples:,}** BF16 samples.",
        (
            f"Aggregate accuracy gate: **{weighted_gate}** sample-weighted and "
            f"**{macro_gate}** macro average; {passed}/{complete} individual tasks "
            "pass the diagnostic degradation threshold."
            if complete
            else "Accuracy gate: pending HIF8 results."
        ),
        "",
        "Performance fields are intentionally omitted here; this report is an accuracy-only comparison.",
        "",
    ]
    text = "\n".join(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
