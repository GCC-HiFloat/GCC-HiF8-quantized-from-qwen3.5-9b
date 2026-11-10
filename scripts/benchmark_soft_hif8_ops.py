#!/usr/bin/env python3
"""Measure strict uint8 HiFloat8 operator latency on one Ascend device."""

from __future__ import annotations

import argparse
import statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--skip-fused",
        action="store_true",
        help="do not launch the experimental fused operator",
    )
    return parser.parse_args()


def elapsed_ms(torch, fn, repeats: int) -> float:
    starts = []
    ends = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        starts.append(start)
        ends.append(end)
    torch.npu.synchronize()
    return statistics.median(
        start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)
    )


def main() -> None:
    args = parse_args()
    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend_hif8 import native

    if not torch.npu.is_available():
        raise RuntimeError("no NPU is available")
    device = torch.device("npu:0")
    generator = torch.Generator().manual_seed(23)
    activation = torch.randn(
        args.m, args.k, generator=generator, dtype=torch.float32
    ).to(torch.bfloat16).to(device)
    weight = torch.randint(
        0, 127, (args.n, args.k), generator=generator, dtype=torch.uint8
    ).to(device)
    weight_scale = torch.rand(
        args.n, generator=generator, dtype=torch.float32
    ).add_(0.01).to(device)

    def quant_then_gemm() -> None:
        data, scale = native.dynamic_quant(activation)
        native.soft_linear(
            data,
            weight,
            scale,
            weight_scale,
            output_dtype=torch.bfloat16,
        )

    def dynamic_quant() -> None:
        native.dynamic_quant(activation)

    quantized, token_scale = native.dynamic_quant(activation)

    def gemm() -> None:
        native.soft_linear(
            quantized,
            weight,
            token_scale,
            weight_scale,
            output_dtype=torch.bfloat16,
        )

    def fused_gemm() -> None:
        native.soft_fused_linear(
            activation,
            weight,
            weight_scale,
            output_dtype=torch.bfloat16,
        )

    for _ in range(args.warmup):
        quant_then_gemm()
        if not args.skip_fused and args.m <= 2:
            fused_gemm()
    torch.npu.synchronize()
    print(
        {
            "shape": [args.m, args.n, args.k],
            "dynamic_quant_ms_median": elapsed_ms(torch, dynamic_quant, args.repeats),
            "soft_linear_ms_median": elapsed_ms(torch, gemm, args.repeats),
            "quant_plus_linear_ms_median": elapsed_ms(
                torch, quant_then_gemm, args.repeats
            ),
            "soft_fused_linear_ms_median": (
                elapsed_ms(torch, fused_gemm, args.repeats)
                if not args.skip_fused and args.m <= 2
                else None
            ),
        }
    )


if __name__ == "__main__":
    main()
