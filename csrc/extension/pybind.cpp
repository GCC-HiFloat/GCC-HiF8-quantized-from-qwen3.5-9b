#include <torch/extension.h>

#include "hif8_native.h"

// Register as PyTorch custom operators so torch.compile can handle them
TORCH_LIBRARY(vllm_ascend_hif8, m) {
    m.def("hif8_dynamic_quant(Tensor x) -> (Tensor, Tensor)");
    m.def("hif8_static_quant(Tensor x, Tensor fixed_scale) -> (Tensor, Tensor)");
    m.def("hif8_scaled_decode(Tensor data, Tensor scale) -> Tensor");
    m.def("soft_hif8_gemv(Tensor activation, Tensor weight, Tensor token_scale, Tensor channel_scale, Tensor? bias, ScalarType output_dtype) -> Tensor");
    m.def("soft_hif8_gemm(Tensor activation, Tensor weight, Tensor token_scale, Tensor channel_scale, Tensor? bias, ScalarType output_dtype) -> Tensor");
    m.def("soft_hif8_fused_gemm(Tensor activation, Tensor weight, Tensor channel_scale, Tensor? bias, ScalarType output_dtype) -> Tensor");
    m.def("soft_hif8_static_fused_gemm(Tensor activation, Tensor weight, Tensor fixed_scale, Tensor channel_scale, Tensor? bias, ScalarType output_dtype) -> Tensor");
}

TORCH_LIBRARY_IMPL(vllm_ascend_hif8, PrivateUse1, m) {
    m.impl("hif8_dynamic_quant", &vllm_ascend_hif8::Hif8DynamicQuant);
    m.impl("hif8_static_quant", &vllm_ascend_hif8::Hif8StaticQuant);
    m.impl("hif8_scaled_decode", &vllm_ascend_hif8::Hif8ScaledDecode);
    m.impl("soft_hif8_gemv", &vllm_ascend_hif8::SoftHif8Gemv);
    m.impl("soft_hif8_gemm", &vllm_ascend_hif8::SoftHif8Gemm);
    m.impl("soft_hif8_fused_gemm", &vllm_ascend_hif8::SoftHif8FusedGemm);
    m.impl("soft_hif8_static_fused_gemm", &vllm_ascend_hif8::SoftHif8StaticFusedGemm);
}

// Meta implementations for torch.compile fake tensor support
TORCH_LIBRARY_IMPL(vllm_ascend_hif8, Meta, m) {
    m.impl("hif8_dynamic_quant", [](const at::Tensor& x) -> std::tuple<at::Tensor, at::Tensor> {
        auto data = at::empty_like(x, x.options().dtype(at::kByte));
        // Compute rows symbolically without calling numel()
        auto x_sizes = x.sizes();
        std::vector<int64_t> scale_shape;
        for (int64_t i = 0; i < x.dim() - 1; ++i) {
            scale_shape.push_back(x_sizes[i]);
        }
        if (scale_shape.empty()) {
            scale_shape.push_back(1);
        }
        auto scale = at::empty(scale_shape, x.options().dtype(at::kFloat));
        return std::make_tuple(data, scale);
    });
    m.impl("hif8_static_quant", [](const at::Tensor& x, const at::Tensor& fixed_scale) -> std::tuple<at::Tensor, at::Tensor> {
        auto data = at::empty_like(x, x.options().dtype(at::kByte));
        auto x_sizes = x.sizes();
        std::vector<int64_t> scale_shape;
        for (int64_t i = 0; i < x.dim() - 1; ++i) {
            scale_shape.push_back(x_sizes[i]);
        }
        if (scale_shape.empty()) {
            scale_shape.push_back(1);
        }
        auto row_scale = at::empty(scale_shape, x.options().dtype(at::kFloat));
        return std::make_tuple(data, row_scale);
    });
    m.impl("hif8_scaled_decode", [](const at::Tensor& data, const at::Tensor& scale) -> at::Tensor {
        return at::empty_like(data, data.options().dtype(at::kBFloat16));
    });
    m.impl("soft_hif8_gemv", [](const at::Tensor& activation, const at::Tensor& weight,
                                 const at::Tensor& token_scale, const at::Tensor& channel_scale,
                                 const c10::optional<at::Tensor>& bias, at::ScalarType output_dtype) -> at::Tensor {
        auto M = activation.size(0);
        auto N = weight.size(0);
        return at::empty({M, N}, activation.options().dtype(output_dtype));
    });
    m.impl("soft_hif8_gemm", [](const at::Tensor& activation, const at::Tensor& weight,
                                 const at::Tensor& token_scale, const at::Tensor& channel_scale,
                                 const c10::optional<at::Tensor>& bias, at::ScalarType output_dtype) -> at::Tensor {
        auto M = activation.size(0);
        auto N = weight.size(0);
        return at::empty({M, N}, activation.options().dtype(output_dtype));
    });
    m.impl("soft_hif8_fused_gemm", [](const at::Tensor& activation, const at::Tensor& weight,
                                       const at::Tensor& channel_scale,
                                       const c10::optional<at::Tensor>& bias,
                                       at::ScalarType output_dtype) -> at::Tensor {
        auto M = activation.size(0);
        auto N = weight.size(0);
        return at::empty({M, N}, activation.options().dtype(output_dtype));
    });
    m.impl("soft_hif8_static_fused_gemm", [](const at::Tensor& activation, const at::Tensor& weight,
                                              const at::Tensor& fixed_scale,
                                              const at::Tensor& channel_scale,
                                              const c10::optional<at::Tensor>& bias,
                                              at::ScalarType output_dtype) -> at::Tensor {
        auto M = activation.size(0);
        auto N = weight.size(0);
        return at::empty({M, N}, activation.options().dtype(output_dtype));
    });
}

// Keep pybind11 bindings for backward compatibility and direct Python access
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "hif8_dynamic_quant",
        &vllm_ascend_hif8::Hif8DynamicQuant,
        pybind11::arg("x"));
    module.def(
        "hif8_static_quant",
        &vllm_ascend_hif8::Hif8StaticQuant,
        pybind11::arg("x"),
        pybind11::arg("fixed_scale"));
    module.def(
        "hif8_scaled_decode",
        &vllm_ascend_hif8::Hif8ScaledDecode,
        pybind11::arg("data"),
        pybind11::arg("scale"));
    module.def(
        "soft_hif8_gemv",
        &vllm_ascend_hif8::SoftHif8Gemv,
        pybind11::arg("activation"),
        pybind11::arg("weight"),
        pybind11::arg("token_scale"),
        pybind11::arg("channel_scale"),
        pybind11::arg("bias") = c10::nullopt,
        pybind11::arg("output_dtype") = at::ScalarType::Half);
    module.def(
        "soft_hif8_gemm",
        &vllm_ascend_hif8::SoftHif8Gemm,
        pybind11::arg("activation"),
        pybind11::arg("weight"),
        pybind11::arg("token_scale"),
        pybind11::arg("channel_scale"),
        pybind11::arg("bias") = c10::nullopt,
        pybind11::arg("output_dtype") = at::ScalarType::Half);
    module.def(
        "soft_hif8_fused_gemm",
        &vllm_ascend_hif8::SoftHif8FusedGemm,
        pybind11::arg("activation"),
        pybind11::arg("weight"),
        pybind11::arg("channel_scale"),
        pybind11::arg("bias") = c10::nullopt,
        pybind11::arg("output_dtype") = at::ScalarType::Half);
    module.def(
        "soft_hif8_static_fused_gemm",
        &vllm_ascend_hif8::SoftHif8StaticFusedGemm,
        pybind11::arg("activation"),
        pybind11::arg("weight"),
        pybind11::arg("fixed_scale"),
        pybind11::arg("channel_scale"),
        pybind11::arg("bias") = c10::nullopt,
        pybind11::arg("output_dtype") = at::ScalarType::Half);
}
