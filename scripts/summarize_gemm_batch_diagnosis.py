#!/usr/bin/env python3
"""Summarize the direct 910B HiFloat8 batch-shape microbenchmarks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PREFIX = "HIF8_BF16_BENCHMARK_JSON="
NAME = re.compile(
    r"(?P<mode>soft|bf16)-(?P<label>[a-z]+)-m(?P<m>\d+)-n(?P<n>\d+)-k(?P<k>\d+)\.log$"
)


def load_results(root: Path) -> dict[tuple[str, str, int, int, int], dict]:
    results: dict[tuple[str, str, int, int, int], dict] = {}
    for path in root.glob("*.log"):
        match = NAME.fullmatch(path.name)
        if match is None:
            continue
        payload = next(
            (line[len(PREFIX):] for line in path.read_text(errors="replace").splitlines() if line.startswith(PREFIX)),
            None,
        )
        if payload is None:
            continue
        key = (
            match["mode"], match["label"], int(match["m"]),
            int(match["n"]), int(match["k"]),
        )
        results[key] = json.loads(payload)
    return results


def median(result: dict | None, key: str) -> float | None:
    if result is None:
        return None
    value = result.get(key, {}).get("median_ms")
    return float(value) if value is not None else None


def cell(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("benchmarks/qwen35-gemm-batch-diagnosis-910b"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = load_results(args.root)

    lines = [
        "# 910B static HiFloat8 GEMM batch diagnosis", "",
        "All HiFloat8 points use physical U8 operands. `static total` is "
        "checkpoint-fixed activation quantization followed by the online-decode GEMM.", "",
        "## 4096×4096 projection", "",
        "| M | adaptive GEMM ms | adaptive static-quant ms | adaptive static total ms | stable static total ms | BF16 mm(view) ms | adaptive rows/s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in (1, 4, 8, 16, 32):
        adaptive = results.get(("soft", "adaptive", m, 4096, 4096))
        stable = results.get(("soft", "stable", m, 4096, 4096))
        bf16 = results.get(("bf16", "reference", m, 4096, 4096))
        gemm = median(adaptive, "soft_linear_static_input")
        quant = median(adaptive, "static_quant")
        total = median(adaptive, "static_quant_plus_soft_linear")
        stable_total = median(stable, "static_quant_plus_soft_linear")
        # The direct F.linear M=32 point can select an asynchronously cached
        # fast path and produced an implausible 0.0197 ms outlier.  The
        # transpose-view matmul is the same arithmetic and is stable at about
        # 0.094 ms for every tested M, so use it as the Cube reference.
        bf16_ms = median(bf16, "bf16_mm_transpose_view")
        rows_s = None if total in (None, 0.0) else m * 1000.0 / total
        lines.append(
            f"| {m} | {cell(gemm)} | {cell(quant)} | {cell(total)} | "
            f"{cell(stable_total)} | {cell(bf16_ms)} | "
            f"{'—' if rows_s is None else f'{rows_s:.1f}'} |"
        )

    lines += [
        "", "## 24576×4096 MLP gate/up projection", "",
        "| M | adaptive GEMM ms | static-quant ms | static total ms | rows/s |",
        "|---:|---:|---:|---:|---:|",
    ]
    for m in (4, 8, 16, 32):
        result = results.get(("soft", "adaptive", m, 24576, 4096))
        gemm = median(result, "soft_linear_static_input")
        quant = median(result, "static_quant")
        total = median(result, "static_quant_plus_soft_linear")
        rows_s = None if total in (None, 0.0) else m * 1000.0 / total
        lines.append(
            f"| {m} | {cell(gemm)} | {cell(quant)} | {cell(total)} | "
            f"{'—' if rows_s is None else f'{rows_s:.1f}'} |"
        )

    completed = len(results)
    lines += ["", f"Completed points: **{completed}/18**.", ""]
    text = "\n".join(lines)
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
