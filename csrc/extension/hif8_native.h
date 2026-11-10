#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#include <tuple>

namespace vllm_ascend_hif8 {

std::tuple<at::Tensor, at::Tensor> Hif8DynamicQuant(const at::Tensor &x);

std::tuple<at::Tensor, at::Tensor> Hif8StaticQuant(
    const at::Tensor &x,
    const at::Tensor &fixedScale);

at::Tensor Hif8ScaledDecode(
    const at::Tensor &data,
    const at::Tensor &scale);

at::Tensor SoftHif8Gemv(
    const at::Tensor &activation,
    const at::Tensor &weight,
    const at::Tensor &tokenScale,
    const at::Tensor &channelScale,
    const c10::optional<at::Tensor> &bias,
    at::ScalarType outputType);

at::Tensor SoftHif8Gemm(
    const at::Tensor &activation,
    const at::Tensor &weight,
    const at::Tensor &tokenScale,
    const at::Tensor &channelScale,
    const c10::optional<at::Tensor> &bias,
    at::ScalarType outputType);

at::Tensor SoftHif8FusedGemm(
    const at::Tensor &activation,
    const at::Tensor &weight,
    const at::Tensor &channelScale,
    const c10::optional<at::Tensor> &bias,
    at::ScalarType outputType);

at::Tensor SoftHif8StaticFusedGemm(
    const at::Tensor &activation,
    const at::Tensor &weight,
    const at::Tensor &fixedScale,
    const at::Tensor &channelScale,
    const c10::optional<at::Tensor> &bias,
    at::ScalarType outputType);

}  // namespace vllm_ascend_hif8
