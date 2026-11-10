"""Numerical reference for software HiFloat8 W8A8 linear layers."""

from __future__ import annotations

from collections.abc import Sequence

from .codec import QuantizedRows, decode_ascend, quantize_rows


def linear_u8_u8(
    activation: QuantizedRows,
    weight: QuantizedRows,
    bias: Sequence[float] | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Compute a decoded U8-by-U8 linear operation.

    ``activation`` is [M, K], ``weight`` is [N, K], and the result is [M, N].
    This deliberately decodes individual elements inside the reduction instead
    of materializing a complete floating-point weight matrix.
    """
    m, k = activation.shape
    n, weight_k = weight.shape
    if k != weight_k:
        raise ValueError(f"incompatible K dimensions: activation={k}, weight={weight_k}")
    if bias is not None and len(bias) != n:
        raise ValueError(f"bias length must equal output dimension {n}")

    output: list[tuple[float, ...]] = []
    for row_index in range(m):
        output_row: list[float] = []
        activation_scale = activation.scale[row_index]
        for channel in range(n):
            accumulator = 0.0
            for inner in range(k):
                accumulator += decode_ascend(activation.data[row_index][inner]) * decode_ascend(
                    weight.data[channel][inner]
                )
            value = accumulator * activation_scale * weight.scale[channel]
            if bias is not None:
                value += float(bias[channel])
            output_row.append(value)
        output.append(tuple(output_row))
    return tuple(output)


def dynamic_linear(
    activation: Sequence[Sequence[float]],
    weight: QuantizedRows,
    bias: Sequence[float] | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Dynamically quantize activations per token and execute the reference op."""
    return linear_u8_u8(quantize_rows(activation), weight, bias)
