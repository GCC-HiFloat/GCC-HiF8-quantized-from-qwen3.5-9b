"""Fake tensor implementations for torch.compile support."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def register_fake_tensor_impls() -> None:
    """Register fake tensor implementations for HiFP8 operators."""
    import torch

    # Use register_fake instead of impl_abstract for better compatibility
    @torch.library.register_fake("vllm_ascend_hif8::hif8_dynamic_quant")
    def hif8_dynamic_quant_fake(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fake implementation for dynamic quantization."""
        ctx = torch.library.get_ctx()
        data = ctx.new_empty(x.shape, dtype=torch.uint8, device=x.device)
        rows = x.numel() // x.shape[-1]
        scale = ctx.new_empty((rows,), dtype=torch.float32, device=x.device)
        return data, scale

    @torch.library.register_fake("vllm_ascend_hif8::hif8_static_quant")
    def hif8_static_quant_fake(x: torch.Tensor, fixed_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del fixed_scale
        data = torch.empty_like(x, dtype=torch.uint8)
        scale_shape = x.shape[:-1] or (1,)
        scale = torch.empty(scale_shape, dtype=torch.float32, device=x.device)
        return data, scale

    @torch.library.register_fake("vllm_ascend_hif8::hif8_scaled_decode")
    def hif8_scaled_decode_fake(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Fake implementation for scaled decode."""
        ctx = torch.library.get_ctx()
        return ctx.new_empty(data.shape, dtype=torch.bfloat16, device=data.device)

    @torch.library.register_fake("vllm_ascend_hif8::soft_hif8_gemv")
    def soft_hif8_gemv_fake(
        activation: torch.Tensor,
        weight: torch.Tensor,
        token_scale: torch.Tensor,
        channel_scale: torch.Tensor,
        bias: torch.Tensor | None,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Fake implementation for GEMV."""
        ctx = torch.library.get_ctx()
        M = activation.shape[0]
        N = weight.shape[0]
        return ctx.new_empty((M, N), dtype=output_dtype, device=activation.device)

    @torch.library.register_fake("vllm_ascend_hif8::soft_hif8_gemm")
    def soft_hif8_gemm_fake(
        activation: torch.Tensor,
        weight: torch.Tensor,
        token_scale: torch.Tensor,
        channel_scale: torch.Tensor,
        bias: torch.Tensor | None,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Fake implementation for GEMM."""
        ctx = torch.library.get_ctx()
        M = activation.shape[0]
        N = weight.shape[0]
        return ctx.new_empty((M, N), dtype=output_dtype, device=activation.device)

    @torch.library.register_fake("vllm_ascend_hif8::soft_hif8_fused_gemm")
    def soft_hif8_fused_gemm_fake(
        activation: torch.Tensor,
        weight: torch.Tensor,
        channel_scale: torch.Tensor,
        bias: torch.Tensor | None,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Fake implementation for fused generation GEMM."""
        del channel_scale, bias
        ctx = torch.library.get_ctx()
        return ctx.new_empty(
            (activation.shape[0], weight.shape[0]),
            dtype=output_dtype,
            device=activation.device,
        )

    @torch.library.register_fake("vllm_ascend_hif8::soft_hif8_static_fused_gemm")
    def soft_hif8_static_fused_gemm_fake(
        activation: torch.Tensor,
        weight: torch.Tensor,
        fixed_scale: torch.Tensor,
        channel_scale: torch.Tensor,
        bias: torch.Tensor | None,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Fake implementation for fixed-scale fused generation GEMM."""
        del fixed_scale, channel_scale, bias
        ctx = torch.library.get_ctx()
        return ctx.new_empty(
            (activation.shape[0], weight.shape[0]),
            dtype=output_dtype,
            device=activation.device,
        )
