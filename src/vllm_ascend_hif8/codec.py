"""Reference HiFloat8 codec for the Ascend byte layout.

The tables in this module are generated from the public HiFloat8 format rules.
They are intentionally not copied from a device implementation. This module is
the numerical oracle for exporters, kernel tests, and exhaustive byte checks.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

ASCEND_ZERO = 0x00
ASCEND_NAN = 0x80
ASCEND_PINF = 0x6F
ASCEND_NINF = 0xEF

MIN_FINITE = 2.0**-22
MAX_FINITE = 2.0**15
UNDERFLOW_TIE = 2.0**-23
OVERFLOW_TIE = 1.25 * MAX_FINITE
DEFAULT_SCALE_EPS = 1.0e-12


def _normal_code(exponent: int, mantissa_index: int) -> int:
    """Return the lower seven Ascend bits for a finite normal value."""
    if -3 <= exponent <= 3:
        base_by_exponent = {
            0: 0x08,
            1: 0x10,
            -1: 0x18,
            2: 0x20,
            3: 0x28,
            -2: 0x30,
            -3: 0x38,
        }
        return base_by_exponent[exponent] + mantissa_index

    if 4 <= exponent <= 7:
        return 0x40 + (exponent - 4) * 0x04 + mantissa_index
    if -7 <= exponent <= -4:
        return 0x50 + (-exponent - 4) * 0x04 + mantissa_index
    if 8 <= exponent <= 15:
        return 0x60 + (exponent - 8) * 0x02 + mantissa_index
    if -15 <= exponent <= -8:
        return 0x70 + (-exponent - 8) * 0x02 + mantissa_index
    raise ValueError(f"normal HiFloat8 exponent out of range: {exponent}")


def _mantissa_bits(exponent: int) -> int:
    absolute = abs(exponent)
    if absolute <= 3:
        return 3
    if absolute <= 7:
        return 2
    if absolute <= 15:
        return 1
    raise ValueError(f"normal HiFloat8 exponent out of range: {exponent}")


def _generate_positive_finite() -> tuple[tuple[float, int], ...]:
    entries: list[tuple[float, int]] = [(0.0, ASCEND_ZERO)]

    for exponent in range(-22, -15):
        entries.append((2.0**exponent, exponent + 23))

    for exponent in range(-15, 16):
        mantissa_bits = _mantissa_bits(exponent)
        for mantissa_index in range(1 << mantissa_bits):
            value = (1.0 + mantissa_index / (1 << mantissa_bits)) * 2.0**exponent
            if value >= OVERFLOW_TIE:
                continue
            entries.append((value, _normal_code(exponent, mantissa_index)))

    entries.sort(key=lambda item: item[0])
    if len(entries) != 127:
        raise AssertionError(f"expected 127 non-negative finite values, got {len(entries)}")
    if len({code for _, code in entries}) != len(entries):
        raise AssertionError("generated duplicate Ascend HiFloat8 byte codes")
    return tuple(entries)


POSITIVE_FINITE = _generate_positive_finite()
POSITIVE_VALUES = tuple(value for value, _ in POSITIVE_FINITE)
POSITIVE_CODES = tuple(code for _, code in POSITIVE_FINITE)


def _generate_decode_table() -> tuple[float, ...]:
    positive = {code: value for value, code in POSITIVE_FINITE}
    positive[ASCEND_PINF] = math.inf
    decoded: list[float] = []
    for byte in range(256):
        if byte == ASCEND_NAN:
            decoded.append(math.nan)
            continue
        magnitude = positive[byte & 0x7F]
        decoded.append(-magnitude if byte & 0x80 else magnitude)
    return tuple(decoded)


ASCEND_DECODE_TABLE = _generate_decode_table()


def decode_ascend_bf16_bits_arithmetic(byte: int) -> int:
    """Return the canonical BF16 payload for one physical HiFloat8 byte.

    The Ascend HiFloat8 finite value set is a subset of BF16, so the mapping
    can be expressed directly in terms of the BF16 exponent/fraction fields.
    Used as a host-side reference for device decoding.
    """
    if not isinstance(byte, int) or not 0 <= byte <= 0xFF:
        raise ValueError(f"HiFloat8 byte must be an integer in [0, 255], got {byte!r}")
    if byte == ASCEND_NAN:
        return 0x7FC0

    sign = 0x8000 if byte & 0x80 else 0
    magnitude = byte & 0x7F
    if magnitude == 0:
        return sign
    if magnitude == ASCEND_PINF:
        return sign | 0x7F80

    if magnitude <= 7:
        exponent_field = 104 + magnitude
        fraction = 0
    elif magnitude <= 63:
        group = magnitude >> 3
        exponent_by_group = (0, 0, 1, -1, 2, 3, -2, -3)
        exponent_field = 127 + exponent_by_group[group]
        fraction = (magnitude & 0x7) << 4
    elif magnitude <= 79:
        exponent_field = 131 + ((magnitude - 64) >> 2)
        fraction = (magnitude & 0x3) << 5
    elif magnitude <= 95:
        exponent_field = 123 - ((magnitude - 80) >> 2)
        fraction = (magnitude & 0x3) << 5
    elif magnitude <= 110:
        exponent_field = 135 + ((magnitude - 96) >> 1)
        fraction = (magnitude & 0x1) << 6
    else:
        exponent_field = 119 - ((magnitude - 112) >> 1)
        fraction = (magnitude & 0x1) << 6
    return sign | (exponent_field << 7) | fraction


def encode_ascend(value: float) -> int:
    """Encode one number as an Ascend-layout HiFloat8 byte.

    Rounding is nearest with ties away from zero. Values at or above the
    overflow tie (40960) encode as infinity. Signed zero and underflow both
    encode as the unique zero byte.
    """
    value = float(value)
    if math.isnan(value):
        return ASCEND_NAN
    if math.isinf(value):
        return ASCEND_NINF if value < 0.0 else ASCEND_PINF

    magnitude = abs(value)
    if magnitude < UNDERFLOW_TIE:
        return ASCEND_ZERO
    if magnitude >= OVERFLOW_TIE:
        return ASCEND_NINF if value < 0.0 else ASCEND_PINF

    upper = bisect.bisect_left(POSITIVE_VALUES, magnitude)
    if upper == 0:
        code = POSITIVE_CODES[0]
    elif upper == len(POSITIVE_VALUES):
        code = POSITIVE_CODES[-1]
    else:
        lower = upper - 1
        lower_distance = magnitude - POSITIVE_VALUES[lower]
        upper_distance = POSITIVE_VALUES[upper] - magnitude
        code = POSITIVE_CODES[upper if upper_distance <= lower_distance else lower]

    if code == ASCEND_ZERO:
        return code
    return code | (0x80 if value < 0.0 else 0x00)


def decode_ascend(byte: int) -> float:
    """Decode one Ascend-layout HiFloat8 byte to a Python float."""
    if not isinstance(byte, int) or not 0 <= byte <= 0xFF:
        raise ValueError(f"HiFloat8 byte must be an integer in [0, 255], got {byte!r}")
    return ASCEND_DECODE_TABLE[byte]


def fake_quantize(value: float) -> float:
    """Round one value to the HiFloat8 value set without changing storage."""
    return decode_ascend(encode_ascend(value))


@dataclass(frozen=True)
class QuantizedRows:
    data: tuple[tuple[int, ...], ...]
    scale: tuple[float, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.data), len(self.data[0]) if self.data else 0)


def choose_row_scale(
    row: Sequence[float],
    *,
    amax_clip: float | None = None,
    scale_factor: float = 1.0,
    eps: float = DEFAULT_SCALE_EPS,
) -> float:
    """Choose a scale that normally maps the row into [-1, 1]."""
    if scale_factor <= 0.0 or not math.isfinite(scale_factor):
        raise ValueError("scale_factor must be finite and greater than zero")
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and greater than zero")
    if amax_clip is not None and (amax_clip <= 0.0 or not math.isfinite(amax_clip)):
        raise ValueError("amax_clip must be finite and greater than zero")

    finite_magnitudes = [abs(float(value)) for value in row if math.isfinite(float(value))]
    amax = max(finite_magnitudes, default=0.0)
    if amax_clip is not None:
        amax = min(amax, amax_clip)
    return max(amax, eps) / scale_factor


def quantize_rows(
    rows: Iterable[Sequence[float]],
    *,
    scales: Sequence[float] | None = None,
    amax_clip: float | None = None,
    scale_factor: float = 1.0,
) -> QuantizedRows:
    """Quantize a rectangular matrix with one scale per row."""
    materialized = [tuple(float(value) for value in row) for row in rows]
    widths = {len(row) for row in materialized}
    if len(widths) > 1:
        raise ValueError("rows must form a rectangular matrix")

    if scales is None:
        row_scales = tuple(
            choose_row_scale(row, amax_clip=amax_clip, scale_factor=scale_factor) for row in materialized
        )
    else:
        row_scales = tuple(float(scale) for scale in scales)
        if len(row_scales) != len(materialized):
            raise ValueError("one scale is required for each row")
        if any(scale <= 0.0 or not math.isfinite(scale) for scale in row_scales):
            raise ValueError("all row scales must be finite and greater than zero")

    data = tuple(
        tuple(encode_ascend(value / scale) for value in row)
        for row, scale in zip(materialized, row_scales, strict=True)
    )
    return QuantizedRows(data=data, scale=row_scales)


def dequantize_rows(quantized: QuantizedRows) -> tuple[tuple[float, ...], ...]:
    """Decode and rescale a matrix produced by :func:`quantize_rows`."""
    return tuple(
        tuple(decode_ascend(byte) * scale for byte in row)
        for row, scale in zip(quantized.data, quantized.scale, strict=True)
    )
