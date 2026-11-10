#!/usr/bin/env python3
"""Compare one strict soft-HiFloat8 linear with an NPU BF16 linear.

The script deliberately benchmarks one mode per process.  ACLNN/AscendC
executor and tiling caches are process-local, so running both modes in one
process can hide the first-use and allocator effects that matter when
comparing two backends.  Set ``--mode`` to ``bf16`` or ``soft`` and invoke
the script in separate processes for an A/B measurement.
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bf16", "soft"), required=True)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument(
        "--include-fused",
        action="store_true",
        help="also measure soft_hif8_fused_gemm for M <= 2",
    )
    return parser.parse_args()


def measure(torch, fn: Callable[[], object], warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    events = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        events.append((start, end))
    torch.npu.synchronize()
    values = [start.elapsed_time(end) for start, end in events]
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "p90_ms": ordered[max(0, (len(ordered) * 9) // 10 - 1)],
    }


def main() -> None:
    args = parse_args()
    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend_hif8 import native

    if not torch.npu.is_available():
        raise RuntimeError("no NPU is available")
    if min(args.m, args.n, args.k) <= 0:
        raise ValueError("M, N, and K must be positive")

    device = torch.device("npu:0")
    generator = torch.Generator().manual_seed(23)
    activation = torch.randn(
        args.m, args.k, generator=generator, dtype=torch.float32
    ).to(torch.bfloat16).to(device)

    result: dict[str, object] = {
        "mode": args.mode,
        "shape": [args.m, args.n, args.k],
        "warmup": args.warmup,
        "repeats": args.repeats,
    }

    if args.mode == "bf16":
        weight = torch.randn(
            args.n, args.k, generator=generator, dtype=torch.float32
        ).to(torch.bfloat16).to(device)
        weight_t = weight.transpose(0, 1)
        weight_t_contiguous = weight_t.contiguous()

        # Keep both forms visible: F.linear matches the usual [N, K] model
        # layout, while the contiguous transpose is a useful lower-bound for
        # implementations that cache a K-by-N weight view.
        result["bf16_linear"] = measure(
            torch,
            lambda: torch.nn.functional.linear(activation, weight),
            args.warmup,
            args.repeats,
        )
        result["bf16_mm_transpose_view"] = measure(
            torch,
            lambda: torch.mm(activation, weight_t),
            args.warmup,
            args.repeats,
        )
        result["bf16_mm_contiguous_transpose"] = measure(
            torch,
            lambda: torch.mm(activation, weight_t_contiguous),
            args.warmup,
            args.repeats,
        )
    else:
        # Restrict the synthetic payload to finite HiFloat8 codes.  This is
        # closer to an exported checkpoint and avoids timing NaN/Inf
        # propagation that production weights never contain.
        weight = torch.randint(
            0,
            111,
            (args.n, args.k),
            generator=generator,
            dtype=torch.uint8,
        ).to(device)
        weight_scale = torch.rand(
            args.n, generator=generator, dtype=torch.float32
        ).add_(0.01).to(device)
        quantized, token_scale = native.dynamic_quant(activation)
        # Compute the representative fixed scale once, outside timed regions.
        # Production reads this scalar from the checkpoint; using the setup
        # activation's amax keeps the synthetic payload finite without adding
        # a runtime reduction to the static path.
        fixed_scale = activation.float().abs().amax().reshape(1)
        static_quantized, static_token_scale = native.static_quant(
            activation, fixed_scale
        )

        result["dynamic_quant"] = measure(
            torch,
            lambda: native.dynamic_quant(activation),
            args.warmup,
            args.repeats,
        )
        result["static_quant"] = measure(
            torch,
            lambda: native.static_quant(activation, fixed_scale),
            args.warmup,
            args.repeats,
        )
        result["soft_linear"] = measure(
            torch,
            lambda: native.soft_linear(
                quantized,
                weight,
                token_scale,
                weight_scale,
                output_dtype=torch.bfloat16,
            ),
            args.warmup,
            args.repeats,
        )
        result["soft_linear_static_input"] = measure(
            torch,
            lambda: native.soft_linear(
                static_quantized,
                weight,
                static_token_scale,
                weight_scale,
                output_dtype=torch.bfloat16,
            ),
            args.warmup,
            args.repeats,
        )

        def quant_plus_soft_linear() -> object:
            data, scale = native.dynamic_quant(activation)
            return native.soft_linear(
                data,
                weight,
                scale,
                weight_scale,
                output_dtype=torch.bfloat16,
            )

        result["quant_plus_soft_linear"] = measure(
            torch,
            quant_plus_soft_linear,
            args.warmup,
            args.repeats,
        )

        def static_quant_plus_soft_linear() -> object:
            data, scale = native.static_quant(activation, fixed_scale)
            return native.soft_linear(
                data,
                weight,
                scale,
                weight_scale,
                output_dtype=torch.bfloat16,
            )

        result["static_quant_plus_soft_linear"] = measure(
            torch,
            static_quant_plus_soft_linear,
            args.warmup,
            args.repeats,
        )
        if args.include_fused and args.m <= 2:
            result["soft_fused_linear"] = measure(
                torch,
                lambda: native.soft_fused_linear(
                    activation,
                    weight,
                    weight_scale,
                    output_dtype=torch.bfloat16,
                ),
                args.warmup,
                args.repeats,
            )

    print("HIF8_BF16_BENCHMARK_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
