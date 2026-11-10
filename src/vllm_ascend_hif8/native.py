"""Python wrappers for Ascend HiFloat8 operators."""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_EXTENSION_MODULE = "vllm_ascend_hif8._C"
_REQUIRED_SYMBOLS = (
    "hif8_dynamic_quant",
    "soft_hif8_gemv",
    "soft_hif8_gemm",
)
_extension: ModuleType | None = None
_load_error: BaseException | None = None

_SOFT_HIF8_GEMV_MAX_OUTPUT_COLUMNS = 1_536
# Limit scalar GEMV reduction to narrow, short-K single-row projections.
_SOFT_HIF8_GEMV_MAX_INPUT_COLUMNS = 8_192
_DISABLE_GEMV_ENV = "HIF8_DISABLE_GEMV"


class NativeExtensionUnavailable(RuntimeError):
    """Raised when production HiFloat8 execution cannot load native operators."""


def load_extension() -> ModuleType:
    """Load and validate the native extension, caching success or failure."""
    global _extension, _load_error
    if _extension is not None:
        return _extension
    if _load_error is not None:
        raise NativeExtensionUnavailable(_error_message()) from _load_error

    try:
        # OPP path should already be configured at import time
        candidate = importlib.import_module(_EXTENSION_MODULE)
        missing = [symbol for symbol in _REQUIRED_SYMBOLS if not callable(getattr(candidate, symbol, None))]
        if missing:
            raise ImportError(f"native extension is missing required symbols: {', '.join(missing)}")
        _extension = candidate
        return candidate
    except BaseException as error:
        _load_error = error
        raise NativeExtensionUnavailable(_error_message()) from error


def _error_message() -> str:
    detail = f": {_load_error}" if _load_error is not None else ""
    return (
        "Ascend HiFloat8 native operators are unavailable"
        f"{detail}. Build/install the custom OPP and {_EXTENSION_MODULE}; "
        "the production path intentionally has no decoded-weight or floating-point fallback."
    )


def is_available() -> bool:
    try:
        load_extension()
    except NativeExtensionUnavailable:
        return False
    return True


def reset_extension_cache() -> None:
    """Reset loader state for isolated tests."""
    global _extension, _load_error
    _extension = None
    _load_error = None


def dynamic_quant(x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    """Quantize BF16/FP16 activations into U8 HiFloat8 with FP32 token scales."""
    import torch

    extension = load_extension()
    operator = getattr(torch.ops.vllm_ascend_hif8, "hif8_dynamic_quant", None)
    if operator is None:
        operator = extension.hif8_dynamic_quant
    data, scale = operator(x)
    _validate_quantized_activation(x, data, scale)
    return data, scale


def static_quant(x: "torch.Tensor", fixed_scale: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    """Quantize with fixed FP32 scale(s), without an activation reduction.

    One scale is used for every row in a Linear activation.  For KV tensors,
    flatten ``[tokens, heads, dim]`` into ``[tokens * heads, dim]`` and pass one
    scale per head; the kernel repeats them in row-major head order.
    """
    import torch

    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"static HiFloat8 quant requires FP16/BF16 input, got {x.dtype}")
    if x.ndim < 1:
        raise ValueError("static HiFloat8 quant requires at least one dimension")
    if fixed_scale.dtype != torch.float32 or fixed_scale.ndim != 1:
        raise TypeError("fixed_scale must be a 1D torch.float32 tensor")
    rows = x.numel() // x.shape[-1]
    if fixed_scale.numel() <= 0 or rows % fixed_scale.numel():
        raise ValueError(f"fixed_scale count {fixed_scale.numel()} must divide {rows} activation rows")
    if fixed_scale.device != x.device:
        raise ValueError(f"fixed_scale must be on {x.device}, got {fixed_scale.device}")
    # ACL descriptors require zero offset, including contiguous KV views.
    x = _as_native_operand(x)
    fixed_scale = _as_native_operand(fixed_scale)
    extension = load_extension()
    operator = getattr(torch.ops.vllm_ascend_hif8, "hif8_static_quant", None)
    if operator is None:
        operator = getattr(extension, "hif8_static_quant", None)
    if not callable(operator):
        raise NativeExtensionUnavailable(
            "the installed native extension has no hif8_static_quant operator; "
            "rebuild the custom OPP and extension for W8A8HIF8_STATIC"
        )
    data, scale = operator(x, fixed_scale)
    _validate_quantized_activation(x, data, scale, operation="hif8_static_quant")
    return data, scale


def scaled_decode(data: "torch.Tensor", scale: "torch.Tensor") -> "torch.Tensor":
    """Decode U8 HiFloat8 rows with FP32 scales into a temporary BF16 tensor."""
    import torch

    if data.dtype != torch.uint8:
        raise TypeError(f"scaled_decode requires uint8 data, got {data.dtype}")
    if scale.dtype != torch.float32:
        raise TypeError(f"scaled_decode requires float32 scale, got {scale.dtype}")
    if data.ndim != 2:
        raise ValueError(f"scaled_decode requires 2D data, got shape {data.shape}")

    extension = load_extension()
    # Sidecar slices are strided; ACL operands must be contiguous and zero-offset.
    data = _as_native_operand(data)
    scale = _as_native_operand(scale.reshape(-1))
    operator = getattr(torch.ops.vllm_ascend_hif8, "hif8_scaled_decode", None)
    if operator is None:
        operator = extension.hif8_scaled_decode
    return operator(data, scale)


def _as_native_operand(tensor: "torch.Tensor") -> "torch.Tensor":
    """Return a tensor the native bridge accepts (contiguous, zero offset).

    ``reshape`` on a sliced activation can yield a view that is contiguous but
    starts at a nonzero storage offset, which the native bridge rejects because
    it builds ACL descriptors from ``data_ptr()`` with offset zero.
    """
    if tensor.is_contiguous() and tensor.storage_offset() == 0:
        return tensor
    return tensor.contiguous().clone()


def soft_linear(
    activation: "torch.Tensor",
    weight: "torch.Tensor",
    activation_scale: "torch.Tensor",
    weight_scale: "torch.Tensor",
    bias: "torch.Tensor | None" = None,
    *,
    output_dtype: "torch.dtype",
) -> "torch.Tensor":
    """Dispatch physical U8 operands to generation GEMV or prefill GEMM."""
    import torch

    if activation.dtype != torch.uint8 or weight.dtype != torch.uint8:
        raise TypeError("soft HiFloat8 linear requires uint8 activation and weight tensors")
    if activation.ndim < 2 or weight.ndim != 2:
        raise ValueError("activation must be at least 2D and weight must be 2D")
    if activation.shape[-1] != weight.shape[-1]:
        raise ValueError(f"incompatible K dimensions: activation={activation.shape[-1]}, weight={weight.shape[-1]}")
    if activation_scale.dtype != torch.float32 or weight_scale.dtype != torch.float32:
        raise TypeError("soft HiFloat8 linear requires float32 activation and weight scales")

    extension = load_extension()
    flattened = _as_native_operand(activation.reshape(-1, activation.shape[-1]))
    token_scale = _as_native_operand(activation_scale.reshape(-1))
    channel_scale = _as_native_operand(weight_scale.reshape(-1))
    if token_scale.numel() != flattened.shape[0]:
        raise ValueError("activation_scale must contain one value per token row")
    if channel_scale.numel() != weight.shape[0]:
        raise ValueError("weight_scale must contain one value per output channel")

    use_gemv = (
        os.environ.get(_DISABLE_GEMV_ENV) != "1"
        and flattened.shape[0] == 1
        and weight.shape[0] <= _SOFT_HIF8_GEMV_MAX_OUTPUT_COLUMNS
        and flattened.shape[1] <= _SOFT_HIF8_GEMV_MAX_INPUT_COLUMNS
    )
    operator_name = "soft_hif8_gemv" if use_gemv else "soft_hif8_gemm"
    operator = getattr(torch.ops.vllm_ascend_hif8, operator_name, None)
    if operator is None:
        operator = getattr(extension, operator_name)
    output = operator(flattened, weight, token_scale, channel_scale, bias, output_dtype)
    return output.reshape(*activation.shape[:-1], weight.shape[0])


def soft_fused_linear(
    x: "torch.Tensor",
    weight: "torch.Tensor",
    weight_scale: "torch.Tensor",
    bias: "torch.Tensor | None" = None,
    *,
    output_dtype: "torch.dtype",
) -> "torch.Tensor":
    """Fuse dynamic activation HiFP8 quant/decode with U8-weight online GEMM."""
    import torch

    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"fused soft HiFP8 linear requires FP16/BF16 activation, got {x.dtype}")
    if weight.dtype != torch.uint8 or weight.ndim != 2:
        raise TypeError("fused soft HiFP8 linear requires a 2D uint8 weight tensor")
    if x.ndim < 2 or x.shape[-1] != weight.shape[-1]:
        raise ValueError("activation and weight must have compatible final K dimensions")
    if weight_scale.dtype != torch.float32 or weight_scale.numel() != weight.shape[0]:
        raise TypeError("fused soft HiFP8 linear requires one float32 scale per output channel")

    flattened = _as_native_operand(x.reshape(-1, x.shape[-1]))
    if flattened.shape[0] > 2:
        raise ValueError("fused soft HiFP8 linear supports M <= 2")
    channel_scale = _as_native_operand(weight_scale.reshape(-1))
    extension = load_extension()
    operator = getattr(torch.ops.vllm_ascend_hif8, "soft_hif8_fused_gemm", None)
    if operator is None:
        operator = getattr(extension, "soft_hif8_fused_gemm", None)
    if not callable(operator):
        raise NativeExtensionUnavailable(
            "the optional soft_hif8_fused_gemm experiment is unavailable; "
            "rebuild the custom OPP and native extension to enable it"
        )
    output = operator(flattened, weight, channel_scale, bias, output_dtype)
    return output.reshape(*x.shape[:-1], weight.shape[0])


def soft_static_fused_linear(
    x: "torch.Tensor",
    weight: "torch.Tensor",
    fixed_scale: "torch.Tensor",
    weight_scale: "torch.Tensor",
    bias: "torch.Tensor | None" = None,
    *,
    output_dtype: "torch.dtype",
) -> "torch.Tensor":
    """Fuse fixed-scale activation encode/decode with U8-weight GEMM.

    Unlike :func:`soft_fused_linear`, this operator never derives a scale from
    the runtime activation.  The sole activation scale comes from the
    calibrated checkpoint tensor passed as ``fixed_scale``.
    """
    import torch

    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"static fused soft HiFP8 linear requires FP16/BF16 activation, got {x.dtype}")
    if weight.dtype != torch.uint8 or weight.ndim != 2:
        raise TypeError("static fused soft HiFP8 linear requires a 2D uint8 weight tensor")
    if x.ndim < 2 or x.shape[-1] != weight.shape[-1]:
        raise ValueError("activation and weight must have compatible final K dimensions")
    if fixed_scale.dtype != torch.float32 or fixed_scale.ndim != 1 or fixed_scale.numel() != 1:
        raise TypeError("static fused soft HiFP8 linear requires one float32 activation scale")
    if weight_scale.dtype != torch.float32 or weight_scale.numel() != weight.shape[0]:
        raise TypeError("static fused soft HiFP8 linear requires one float32 scale per output channel")
    if fixed_scale.device != x.device or weight_scale.device != x.device or weight.device != x.device:
        raise ValueError("activation, weight, fixed_scale, and weight_scale must be on the same device")

    flattened = _as_native_operand(x.reshape(-1, x.shape[-1]))
    fixed = _as_native_operand(fixed_scale.reshape(-1))
    channel_scale = _as_native_operand(weight_scale.reshape(-1))
    extension = load_extension()
    operator = getattr(torch.ops.vllm_ascend_hif8, "soft_hif8_static_fused_gemm", None)
    if operator is None:
        operator = getattr(extension, "soft_hif8_static_fused_gemm", None)
    if not callable(operator):
        raise NativeExtensionUnavailable(
            "the optional soft_hif8_static_fused_gemm experiment is unavailable; "
            "rebuild the custom OPP and native extension to enable it"
        )
    output = operator(flattened, weight, fixed, channel_scale, bias, output_dtype)
    return output.reshape(*x.shape[:-1], weight.shape[0])


def _validate_quantized_activation(
    source: "torch.Tensor",
    data: "torch.Tensor",
    scale: "torch.Tensor",
    *,
    operation: str = "hif8_dynamic_quant",
) -> None:
    import torch

    if data.dtype != torch.uint8:
        raise RuntimeError(f"{operation} returned {data.dtype}, expected torch.uint8")
    if data.shape != source.shape:
        raise RuntimeError(f"{operation} changed activation shape from {source.shape} to {data.shape}")
    if scale.dtype != torch.float32:
        raise RuntimeError(f"{operation} returned {scale.dtype} scale, expected torch.float32")
    expected_rows = source.numel() // source.shape[-1]
    if scale.numel() != expected_rows:
        raise RuntimeError(f"{operation} returned {scale.numel()} scales for {expected_rows} activation rows")
