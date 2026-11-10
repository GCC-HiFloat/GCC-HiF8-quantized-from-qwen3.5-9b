"""Static HiFloat8 W8A8 linears and checkpoint compatibility schemes."""

from __future__ import annotations

import math
import os
from typing import Any

import torch
from vllm_ascend.quantization.methods.base import AscendLinearScheme
from vllm_ascend.quantization.methods.registry import register_scheme

from . import native

DYNAMIC_QUANT_TYPE = "W8A8HIF8_SOFT"
STATIC_QUANT_TYPE = "W8A8HIF8_STATIC"
# Public alias for the W8A8HIF8_SOFT format.
QUANT_TYPE = DYNAMIC_QUANT_TYPE
_COMPUTE_BACKEND_ENV = "HIF8_COMPUTE_BACKEND"
_SOFT_HIF8_BACKEND = "soft_hif8"
_DEFAULT_COMPUTE_BACKEND = _SOFT_HIF8_BACKEND
_STATIC_FUSED_GENERATION_ENV = "HIF8_STATIC_FUSED_GENERATION"
_STATIC_SCALE_FACTOR_ENV = "HIF8_STATIC_SCALE_FACTOR"


def _static_scale_factor() -> float:
    """Return an optional multiplicative factor for calibrated static scales.

    Defaults to 1.0. The factor is constant across requests.
    """
    raw = os.environ.get(_STATIC_SCALE_FACTOR_ENV)
    if raw is None or raw == "":
        return 1.0
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{_STATIC_SCALE_FACTOR_ENV} must be a positive finite float") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{_STATIC_SCALE_FACTOR_ENV} must be a positive finite float")
    return value


@register_scheme(DYNAMIC_QUANT_TYPE, "linear")
class AscendW8A8HiFloat8SoftLinearMethod(AscendLinearScheme):
    """Dynamic per-token A8 and per-channel W8 with tile-local software decode."""

    supports_tp_weight_switch = False
    quant_type = DYNAMIC_QUANT_TYPE

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return {"weight": torch.empty(output_size, input_size, dtype=torch.uint8)}

    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return {"weight_scale": torch.empty(output_size, 1, dtype=torch.float32)}

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._process_hif8_weights(layer)

    def _process_hif8_weights(
        self,
        layer: torch.nn.Module,
    ) -> None:
        if layer.weight.dtype != torch.uint8:
            raise TypeError(
                f"{self.quant_type} checkpoint weight must be torch.uint8, "
                f"got {layer.weight.dtype}"
            )
        if layer.weight_scale.dtype != torch.float32:
            layer.weight_scale.data = layer.weight_scale.data.to(torch.float32)
        layer.weight.data = layer.weight.data.contiguous()
        layer.weight_scale.data = layer.weight_scale.data.reshape(-1).contiguous()
        backend = os.environ.get(_COMPUTE_BACKEND_ENV, _DEFAULT_COMPUTE_BACKEND)
        if backend != _SOFT_HIF8_BACKEND:
            raise ValueError(
                f"unsupported {_COMPUTE_BACKEND_ENV}={backend!r}; "
                f"only '{_SOFT_HIF8_BACKEND}' is supported"
            )
        layer._hif8_compute_backend = backend

        # Embedding lookup decodes selected rows with a 512-byte BF16 LUT.
        from .codec import ASCEND_DECODE_TABLE

        layer.register_buffer(
            "_hif8_decode_lut",
            torch.tensor(
                ASCEND_DECODE_TABLE,
                dtype=torch.bfloat16,
                device=layer.weight.device,
            ),
            persistent=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        del tp_rank
        token_rows = x.numel() // x.shape[-1]
        if token_rows <= 2 and os.environ.get("HIF8_SOFT_FUSED_GENERATION") == "1":
            return native.soft_fused_linear(
                x,
                layer.weight,
                layer.weight_scale,
                bias,
                output_dtype=x.dtype,
            )
        activation, activation_scale = native.dynamic_quant(x)
        return native.soft_linear(
            activation,
            layer.weight,
            activation_scale,
            layer.weight_scale,
            bias,
            output_dtype=x.dtype,
        )

    def embedding(
        self,
        layer: torch.nn.Module,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Gather and dequantize only the requested token embedding rows."""
        flat_ids = input_ids.reshape(-1)
        rows = torch.index_select(layer.weight, 0, flat_ids)
        scales = torch.index_select(layer.weight_scale, 0, flat_ids)
        rows = layer._hif8_decode_lut[rows.long()]
        output = rows * scales.to(torch.bfloat16).unsqueeze(-1)
        return output.reshape(*input_ids.shape, layer.weight.shape[1])


@register_scheme(STATIC_QUANT_TYPE, "linear")
class AscendW8A8HiFloat8StaticLinearMethod(
    AscendW8A8HiFloat8SoftLinearMethod
):
    """Calibrated per-tensor A8 and per-channel W8 HiFloat8 Linear.

    ``input_scale`` is loaded from the checkpoint and remains fixed for every
    invocation.  The native static encoder expands that scalar into the
    existing one-scale-per-row sidecar expected by ``soft_linear`` without an
    activation ``Abs``/``ReduceMax`` pass.
    """

    quant_type = STATIC_QUANT_TYPE

    def get_pertensor_param(
        self,
        params_dtype: torch.dtype,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del params_dtype, kwargs
        # Input scales use FP32 independently of the model parameter dtype.
        return {"input_scale": torch.empty(1, dtype=torch.float32)}

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer, "input_scale"):
            raise ValueError(f"{STATIC_QUANT_TYPE} checkpoint is missing input_scale")
        scale = layer.input_scale.data.to(dtype=torch.float32).reshape(-1)
        if scale.numel() != 1:
            raise ValueError(
                f"{STATIC_QUANT_TYPE} requires one input_scale value, got "
                f"shape {tuple(layer.input_scale.shape)}"
            )
        if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
            raise ValueError(
                f"{STATIC_QUANT_TYPE} input_scale must be finite and greater "
                "than zero"
            )
        layer.input_scale.data = scale.contiguous()
        self._process_hif8_weights(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        del tp_rank
        token_rows = x.numel() // x.shape[-1]
        # Real 910B projections use static quantization plus GEMM to avoid
        # the fused kernel's global-barrier cost; tiny shapes retain fusion.
        use_fused = (
            token_rows <= 4
            and os.environ.get(_STATIC_FUSED_GENERATION_ENV) == "1"
            and x.shape[-1] < 1024
        )
        scale_factor = _static_scale_factor()
        fixed_scale = layer.input_scale
        if scale_factor != 1.0:
            fixed_scale = fixed_scale * scale_factor
        if use_fused:
            return native.soft_static_fused_linear(
                x,
                layer.weight,
                fixed_scale,
                layer.weight_scale,
                bias,
                output_dtype=x.dtype,
            )
        activation, row_scale = native.static_quant(x, fixed_scale)
        return native.soft_linear(
            activation,
            layer.weight,
            row_scale,
            layer.weight_scale,
            bias,
            output_dtype=x.dtype,
        )
