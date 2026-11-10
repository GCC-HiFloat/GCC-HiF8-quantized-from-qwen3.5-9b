"""Optional PyTorch implementation of the Ascend HiFloat8 codec."""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING

from .codec import ASCEND_DECODE_TABLE, ASCEND_NAN, ASCEND_PINF, MAX_FINITE, encode_ascend

if TYPE_CHECKING:
    import torch


def _float_from_bits(bits: int) -> float:
    return struct.unpack("!f", struct.pack("!I", bits))[0]


def generate_float32_encode_lut() -> tuple[int, ...]:
    """Generate the 4096-entry exponent/top-mantissa encode LUT."""
    output: list[int] = []
    for exponent_field in range(256):
        for top_mantissa in range(16):
            bits = (exponent_field << 23) | (top_mantissa << 19)
            output.append(encode_ascend(_float_from_bits(bits)) & 0x7F)
    return tuple(output)


FLOAT32_ENCODE_LUT = generate_float32_encode_lut()


def encode_tensor(x: "torch.Tensor") -> "torch.Tensor":
    """Encode a floating tensor to physical Ascend-layout HiFloat8 bytes."""
    import torch

    x_f32 = x.to(torch.float32).contiguous()
    bits = x_f32.view(torch.int32)
    exponent = torch.bitwise_and(torch.bitwise_right_shift(bits, 23), 0xFF)
    mantissa = torch.bitwise_and(bits, 0x007FFFFF)
    sign = torch.bitwise_and(torch.bitwise_right_shift(bits, 24), 0x80)
    index = exponent.to(torch.int64) * 16 + torch.bitwise_right_shift(mantissa, 19).to(torch.int64)
    lut = torch.tensor(FLOAT32_ENCODE_LUT, dtype=torch.uint8, device=x.device)
    magnitude = lut[index]
    signed = torch.bitwise_or(magnitude, sign.to(torch.uint8))
    signed = torch.where(magnitude == 0, magnitude, signed)
    special = exponent == 0xFF
    nan_value = torch.full_like(signed, ASCEND_NAN)
    infinity = torch.bitwise_or(sign.to(torch.uint8), torch.full_like(signed, ASCEND_PINF))
    return torch.where(special, torch.where(mantissa != 0, nan_value, infinity), signed)


def quantize_tensor_rows(
    x: "torch.Tensor",
    *,
    scale: "torch.Tensor | None" = None,
    amax_clip: float | None = None,
    scale_factor: float = 1.0,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Quantize the final dimension with one FP32 scale per row/token."""
    import torch

    if x.ndim == 0:
        raise ValueError("expected at least one tensor dimension")
    if scale_factor <= 0.0:
        raise ValueError("scale_factor must be greater than zero")
    x_f32 = x.to(torch.float32)
    if scale is None:
        scale = x_f32.abs().amax(dim=-1)
        if amax_clip is not None:
            scale = scale.clamp(max=amax_clip)
        scale = scale.clamp(min=1.0e-12).div(scale_factor)
    scale = scale.to(device=x.device, dtype=torch.float32)
    return encode_tensor(x_f32 / scale.unsqueeze(-1)), scale


def quantize_tensor_rows_mse(
    x: "torch.Tensor",
    *,
    candidate_factors: tuple[float, ...] = (
        0.50,
        0.625,
        0.75,
        0.875,
        1.0,
        1.125,
        1.25,
        1.5,
        2.0,
    ),
    search_columns: int = 128,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Quantize rows with a fixed per-row scale selected by codebook MSE.

    The existing exporter uses each row's amax as its scale.  That is robust,
    but it gives one outlier full control over the codebook range.  This
    optional exporter path evaluates a small, deterministic set of fixed
    ``amax / factor`` scales against the exact Ascend HiFloat8 codebook and
    keeps the candidate with the lowest row MSE.  The resulting scale is
    persisted in the checkpoint; no runtime/dynamic scale is introduced.

    ``candidate_factors`` are deliberately finite and positive.  Scale search
    uses evenly spaced columns (at most ``search_columns``) and then encodes
    the complete row once with the selected scale.  This keeps export cost
    bounded for the very wide Qwen matrices while retaining the exact physical
    codebook for the final bytes.
    """
    import torch

    if x.ndim != 2:
        raise ValueError(f"expected a 2-D row matrix, got shape {tuple(x.shape)}")
    if search_columns < 1:
        raise ValueError("search_columns must be positive")
    if not candidate_factors or any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in candidate_factors
    ):
        raise ValueError("candidate_factors must contain finite positive values")

    values = x.to(torch.float32).contiguous()
    amax = values.abs().amax(dim=1).clamp_min(1.0e-12)
    decode_lut = torch.tensor(ASCEND_DECODE_TABLE, dtype=torch.float32, device=values.device)
    best_mse = torch.full(
        (values.shape[0],), float("inf"), dtype=torch.float32, device=values.device
    )
    best_scale = torch.empty_like(amax)
    if values.shape[1] > search_columns:
        stride = math.ceil(values.shape[1] / search_columns)
        search_values = values[:, ::stride]
    else:
        search_values = values

    for factor in candidate_factors:
        scale = amax.div(float(factor))
        normalized = (search_values / scale.unsqueeze(-1)).clamp(
            min=-MAX_FINITE, max=MAX_FINITE
        )
        encoded = encode_tensor(normalized)
        reconstructed = decode_lut[encoded.to(torch.long)] * scale.unsqueeze(-1)
        mse = (reconstructed - search_values).square().mean(dim=1)
        choose = mse < best_mse
        if bool(choose.any()):
            best_mse = torch.where(choose, mse, best_mse)
            best_scale = torch.where(choose, scale, best_scale)

    full_normalized = (values / best_scale.unsqueeze(-1)).clamp(
        min=-MAX_FINITE, max=MAX_FINITE
    )
    best_data = encode_tensor(full_normalized)

    return best_data.contiguous(), best_scale.contiguous()


def quantize_tensor_rows_activation_mse(
    x: "torch.Tensor",
    channel_mean_square: "torch.Tensor",
    *,
    candidate_factors: tuple[float, ...] = (
        0.50,
        0.625,
        0.75,
        0.875,
        1.0,
        1.125,
        1.25,
        1.5,
        2.0,
    ),
    search_columns: int = 256,
    blend_alpha: float = 1.0,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Select fixed row scales using calibration activation energy.

    For one output row, diagonal activation covariance makes expected Linear
    output MSE proportional to ``sum_j E[x_j^2] * (Wq_j - W_j)^2``.  This
    search therefore optimizes the quantity consumed by the model instead of
    treating rarely used and high-energy input channels equally.

    Wide matrices are searched with deterministic systematic samples from
    the activation-energy distribution.  ``blend_alpha`` geometrically blends
    the selected scale with the existing amax scale: zero reproduces amax,
    one uses the full activation-MSE result, and intermediate values provide
    a fixed, conservative export-time interpolation.  Factor 1.0 (the
    existing amax scale) is always retained as a candidate, and every selected
    scale remains a checkpoint-fixed FP32 value.
    """
    import torch

    if x.ndim != 2:
        raise ValueError(f"expected a 2-D row matrix, got shape {tuple(x.shape)}")
    if search_columns < 1:
        raise ValueError("search_columns must be positive")
    if not math.isfinite(float(blend_alpha)) or not 0.0 <= float(blend_alpha) <= 1.0:
        raise ValueError("blend_alpha must be finite and in [0,1]")
    if not candidate_factors or any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in candidate_factors
    ):
        raise ValueError("candidate_factors must contain finite positive values")

    values = x.to(torch.float32).contiguous()
    importance = torch.as_tensor(
        channel_mean_square, dtype=torch.float32, device=values.device
    ).reshape(-1)
    if importance.numel() != values.shape[1]:
        raise ValueError(
            "channel_mean_square width does not match the weight input: "
            f"{importance.numel()} != {values.shape[1]}"
        )
    if bool((~torch.isfinite(importance) | (importance < 0)).any()):
        raise ValueError("channel_mean_square must be finite and non-negative")
    total_importance = importance.sum()
    if not bool(total_importance > 0):
        raise ValueError("channel_mean_square must contain positive energy")

    if values.shape[1] > search_columns:
        cumulative = importance.cumsum(dim=0)
        targets = (
            torch.arange(search_columns, dtype=torch.float32, device=values.device)
            + 0.5
        ) * (total_importance / search_columns)
        indices = torch.searchsorted(cumulative, targets).clamp_max(values.shape[1] - 1)
        search_values = values.index_select(1, indices)
        search_importance = None
    else:
        search_values = values
        search_importance = importance

    amax = values.abs().amax(dim=1).clamp_min(1.0e-12)
    decode_lut = torch.tensor(
        ASCEND_DECODE_TABLE, dtype=torch.float32, device=values.device
    )
    best_error = torch.full(
        (values.shape[0],), float("inf"), dtype=torch.float32, device=values.device
    )
    best_scale = torch.empty_like(amax)
    factors = tuple(dict.fromkeys((*candidate_factors, 1.0)))
    for factor in factors:
        scale = amax.div(float(factor))
        normalized = (search_values / scale.unsqueeze(-1)).clamp(
            min=-MAX_FINITE, max=MAX_FINITE
        )
        encoded = encode_tensor(normalized)
        reconstructed = decode_lut[encoded.to(torch.long)] * scale.unsqueeze(-1)
        squared_error = (reconstructed - search_values).square()
        if search_importance is None:
            error = squared_error.mean(dim=1)
        else:
            error = (squared_error * search_importance).sum(dim=1) / total_importance
        choose = error < best_error
        if bool(choose.any()):
            best_error = torch.where(choose, error, best_error)
            best_scale = torch.where(choose, scale, best_scale)

    if blend_alpha == 0.0:
        best_scale = amax
    elif blend_alpha != 1.0:
        alpha = float(blend_alpha)
        best_scale = torch.exp(
            (1.0 - alpha) * torch.log(amax) + alpha * torch.log(best_scale)
        )

    encoded = encode_tensor(
        (values / best_scale.unsqueeze(-1)).clamp(
            min=-MAX_FINITE, max=MAX_FINITE
        )
    )
    return encoded.contiguous(), best_scale.contiguous()
