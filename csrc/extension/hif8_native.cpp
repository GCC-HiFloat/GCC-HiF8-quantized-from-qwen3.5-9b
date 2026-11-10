#include "hif8_native.h"

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/irange.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>

#include "aclnn_hif8_dynamic_quant.h"
#include "aclnn_hif8_static_quant.h"
#include "aclnn_hif8_scaled_decode.h"
#include "aclnn_soft_hif8_fused_gemm.h"
#include "aclnn_soft_hif8_gemm.h"
#include "aclnn_soft_hif8_gemv.h"
#include "aclnn_soft_hif8_prefill_gemm.h"
#include "aclnn_soft_hif8_static_fused_gemm.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace vllm_ascend_hif8 {
namespace {

constexpr int64_t kFp16Output = 0;
constexpr int64_t kBf16Output = 1;
// On C220, batched row DMA starts paying for its setup cost at M=64.
constexpr int64_t kBatchedEpilogueMinRows = 64;
constexpr int64_t kBatchedEpilogueColumnAlignment = 16;

bool BatchedEpilogueEnabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("HIF8_DISABLE_BATCHED_EPILOGUE");
        return value == nullptr || std::string(value) != "1";
    }();
    return enabled;
}

void CheckAclStatus(aclnnStatus status, const char *operation) {
    TORCH_CHECK(
        status == 0,
        operation,
        " failed with ACLNN status ",
        status);
}

aclDataType ToAclDataType(at::ScalarType type) {
    switch (type) {
        case at::ScalarType::Byte:
            return ACL_UINT8;
        case at::ScalarType::Half:
            return ACL_FLOAT16;
        case at::ScalarType::BFloat16:
            return ACL_BF16;
        case at::ScalarType::Float:
            return ACL_FLOAT;
        default:
            TORCH_CHECK(
                false,
                "unsupported ACL tensor dtype: ",
                c10::toString(type));
    }
}

class AclTensorHandle {
public:
    explicit AclTensorHandle(const at::Tensor &tensor) {
        const auto sizes = tensor.sizes();
        const auto strides = tensor.strides();
        tensor_ = aclCreateTensor(
            sizes.data(),
            sizes.size(),
            ToAclDataType(tensor.scalar_type()),
            strides.data(),
            0,
            ACL_FORMAT_ND,
            sizes.data(),
            sizes.size(),
            tensor.data_ptr());
        TORCH_CHECK(tensor_ != nullptr, "aclCreateTensor returned null");
    }

    AclTensorHandle(const AclTensorHandle &) = delete;
    AclTensorHandle &operator=(const AclTensorHandle &) = delete;

    ~AclTensorHandle() {
        if (tensor_ != nullptr) {
            aclDestroyTensor(tensor_);
        }
    }

    const aclTensor *get() const {
        return tensor_;
    }

private:
    aclTensor *tensor_ = nullptr;
};

class AclExecutorHandle {
public:
    AclExecutorHandle() = default;
    AclExecutorHandle(const AclExecutorHandle &) = delete;
    AclExecutorHandle &operator=(const AclExecutorHandle &) = delete;

    ~AclExecutorHandle() {
        if (executor_ != nullptr) {
            aclDestroyAclOpExecutor(executor_);
        }
    }

    aclOpExecutor **out() {
        return &executor_;
    }

    aclOpExecutor *get() const {
        return executor_;
    }

private:
    aclOpExecutor *executor_ = nullptr;
};

void CheckNpuTensor(const at::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(
        torch_npu::utils::is_npu(tensor),
        name,
        " must be an NPU tensor, got ",
        tensor.device());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(
        tensor.storage_offset() == 0,
        name,
        " must have storage_offset == 0");
    TORCH_CHECK(
        at_npu::native::get_npu_format(tensor) == ACL_FORMAT_ND,
        name,
        " must use ACL_FORMAT_ND storage");
}

void CheckSameDevice(
    const at::Tensor &tensor,
    const at::Device &device,
    const char *name) {
    TORCH_CHECK(
        tensor.device() == device,
        name,
        " must be on ",
        device,
        ", got ",
        tensor.device());
}

void CheckPositiveDimensions(const at::Tensor &tensor, const char *name) {
    for (const auto dim : c10::irange(tensor.dim())) {
        TORCH_CHECK(
            tensor.size(dim) > 0,
            name,
            " dimensions must be positive, got ",
            tensor.sizes());
    }
}

void CheckFitsUint32(int64_t value, const char *name) {
    TORCH_CHECK(
        value > 0 &&
            static_cast<uint64_t>(value) <=
                std::numeric_limits<uint32_t>::max(),
        name,
        " must fit a positive uint32, got ",
        value);
}

aclrtStream CurrentStream(const at::Device &device) {
    c10::OptionalDeviceGuard guard(device);
    auto stream =
        c10_npu::getCurrentNPUStream(device.index()).stream(false);
    if (c10_npu::check_enqueue_need_use(stream)) {
        CheckAclStatus(
            aclrtUseStreamResInCurrentThread(stream),
            "aclrtUseStreamResInCurrentThread");
    }
    return stream;
}

void PrepareDequeuedStream(aclrtStream stream) {
    if (c10_npu::check_dequeue_need_use(stream)) {
        CheckAclStatus(
            aclrtUseStreamResInCurrentThread(stream),
            "aclrtUseStreamResInCurrentThread");
    }
}

void *AllocateWorkspace(
    uint64_t workspaceSize,
    aclrtStream stream,
    at::Tensor &workspace) {
    if (workspaceSize == 0) {
        return nullptr;
    }
    workspace = at_npu::native::allocate_workspace(workspaceSize, stream);
    TORCH_CHECK(
        workspace.defined(),
        "torch-npu returned an undefined workspace tensor for ",
        workspaceSize,
        " bytes");
    void *workspaceAddress = const_cast<void *>(workspace.storage().data());
    TORCH_CHECK(
        workspaceAddress != nullptr,
        "torch-npu returned a null workspace storage pointer for ",
        workspaceSize,
        " bytes (tensor numel=",
        workspace.numel(),
        ", storage bytes=",
        workspace.storage().nbytes(),
        ")");
    return workspaceAddress;
}

using LinearWorkspaceFunction = aclnnStatus (*)(
    const aclTensor *,
    const aclTensor *,
    const aclTensor *,
    const aclTensor *,
    const aclTensor *,
    int64_t,
    const aclTensor *,
    uint64_t *,
    aclOpExecutor **);
using LinearExecuteFunction = aclnnStatus (*)(
    void *, uint64_t, aclOpExecutor *, aclrtStream);

void ValidateSoftLinear(
    const at::Tensor &activation,
    const at::Tensor &weight,
    const at::Tensor &tokenScale,
    const at::Tensor &channelScale,
    const c10::optional<at::Tensor> &bias,
    at::ScalarType outputType,
    bool gemv) {
    CheckNpuTensor(activation, "activation");
    CheckNpuTensor(weight, "weight");
    CheckNpuTensor(tokenScale, "token_scale");
    CheckNpuTensor(channelScale, "channel_scale");

    TORCH_CHECK(
        activation.scalar_type() == at::ScalarType::Byte,
        "activation must have dtype torch.uint8");
    TORCH_CHECK(
        weight.scalar_type() == at::ScalarType::Byte,
        "weight must have dtype torch.uint8");
    TORCH_CHECK(
        tokenScale.scalar_type() == at::ScalarType::Float,
        "token_scale must have dtype torch.float32");
    TORCH_CHECK(
        channelScale.scalar_type() == at::ScalarType::Float,
        "channel_scale must have dtype torch.float32");
    TORCH_CHECK(activation.dim() == 2, "activation must have shape [M, K]");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [N, K]");
    TORCH_CHECK(tokenScale.dim() == 1, "token_scale must have shape [M]");
    TORCH_CHECK(channelScale.dim() == 1, "channel_scale must have shape [N]");
    CheckPositiveDimensions(activation, "activation");
    CheckPositiveDimensions(weight, "weight");

    const int64_t m = activation.size(0);
    const int64_t k = activation.size(1);
    const int64_t n = weight.size(0);
    TORCH_CHECK(
        weight.size(1) == k,
        "activation and weight K dimensions differ: ",
        k,
        " != ",
        weight.size(1));
    TORCH_CHECK(
        tokenScale.numel() == m,
        "token_scale must contain one value per activation row");
    TORCH_CHECK(
        channelScale.numel() == n,
        "channel_scale must contain one value per output channel");
    TORCH_CHECK(
        gemv ? m <= 2 : m > 0,
        gemv ? "soft_hif8_gemv requires M <= 2"
             : "soft_hif8_gemm requires M > 0");
    CheckFitsUint32(m, "M");
    CheckFitsUint32(n, "N");
    CheckFitsUint32(k, "K");

    const auto device = activation.device();
    CheckSameDevice(weight, device, "weight");
    CheckSameDevice(tokenScale, device, "token_scale");
    CheckSameDevice(channelScale, device, "channel_scale");

    if (bias.has_value()) {
        CheckNpuTensor(*bias, "bias");
        CheckSameDevice(*bias, device, "bias");
        TORCH_CHECK(bias->dim() == 1, "bias must have shape [N]");
        TORCH_CHECK(bias->numel() == n, "bias must contain N values");
        TORCH_CHECK(
            bias->scalar_type() == at::ScalarType::Half ||
                bias->scalar_type() == at::ScalarType::BFloat16,
            "bias must have dtype torch.float16 or torch.bfloat16");
    }

    TORCH_CHECK(
        outputType == at::ScalarType::Half ||
            outputType == at::ScalarType::BFloat16,
        "output_dtype must be torch.float16 or torch.bfloat16");
}

at::Tensor SoftHif8Linear(
    const at::Tensor &activation,
    const at::Tensor &weight,
    const at::Tensor &tokenScale,
    const at::Tensor &channelScale,
    const c10::optional<at::Tensor> &bias,
    at::ScalarType outputType,
    bool gemv,
    const char *operationName,
    LinearWorkspaceFunction getWorkspaceSize,
    LinearExecuteFunction execute) {
    ValidateSoftLinear(
        activation,
        weight,
        tokenScale,
        channelScale,
        bias,
        outputType,
        gemv);

    // Production always executes the fused uint8-resident custom operator.
    // Full BF16 operand materialization exists only as an explicit diagnostic
    // path and is never selected implicitly.
    const char *decodedFallback = std::getenv("HIF8_USE_DECODED_LINEAR");
    if (decodedFallback != nullptr && std::string(decodedFallback) == "1") {
        auto decodedActivation = Hif8ScaledDecode(activation, tokenScale);
        auto decodedWeight = Hif8ScaledDecode(weight, channelScale);
        auto output = at::mm(decodedActivation, decodedWeight.transpose(0, 1));
        if (bias.has_value()) {
            output = output.add(bias->to(at::ScalarType::BFloat16));
        }
        if (outputType != at::ScalarType::BFloat16) {
            output = output.to(outputType);
        }
        return output;
    }

    c10::OptionalDeviceGuard guard(activation.device());
    auto output = at_npu::native::OpPreparation::apply_tensor_without_format(
        {activation.size(0), weight.size(0)},
        activation.options().dtype(outputType));
    const auto stream = CurrentStream(activation.device());
    const int64_t outputDtype =
        outputType == at::ScalarType::Half ? kFp16Output : kBf16Output;

    auto run = [activation,
                weight,
                tokenScale,
                channelScale,
                bias,
                output,
                outputDtype,
                gemv,
                stream,
                operationName,
                getWorkspaceSize,
                execute]() -> int {
        PrepareDequeuedStream(stream);
        AclTensorHandle activationAcl(activation);
        AclTensorHandle weightAcl(weight);
        AclTensorHandle tokenScaleAcl(tokenScale);
        AclTensorHandle channelScaleAcl(channelScale);
        AclTensorHandle outputAcl(output);
        c10::optional<AclTensorHandle> biasAcl;
        if (bias.has_value()) {
            biasAcl.emplace(*bias);
        }

        uint64_t workspaceSize = 0;
        AclExecutorHandle executor;
        const auto tilingStatus = getWorkspaceSize(
            activationAcl.get(),
            weightAcl.get(),
            tokenScaleAcl.get(),
            channelScaleAcl.get(),
            biasAcl.has_value() ? biasAcl->get() : nullptr,
            outputDtype,
            outputAcl.get(),
            &workspaceSize,
            executor.out());
        TORCH_CHECK(
            tilingStatus == 0,
            operationName,
            "GetWorkspaceSize failed with ACLNN status ",
            tilingStatus,
            ", workspace size = ",
            workspaceSize);
        TORCH_CHECK(
            gemv || workspaceSize != 0,
            operationName,
            " reported a zero-byte workspace for GEMM (M=",
            activation.size(0),
            ", N=",
            weight.size(0),
            ", K=",
            activation.size(1),
            ")");

        at::Tensor workspace;
        void *workspaceAddress =
            AllocateWorkspace(workspaceSize, stream, workspace);
        const auto status = execute(
            workspaceAddress,
            workspaceSize,
            executor.get(),
            stream);
        CheckAclStatus(status, operationName);
        return status;
    };
    at_npu::native::OpCommand::RunOpApiV2(operationName, run);
    return output;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> Hif8DynamicQuant(const at::Tensor &x) {
    CheckNpuTensor(x, "x");
    TORCH_CHECK(
        x.scalar_type() == at::ScalarType::Half ||
            x.scalar_type() == at::ScalarType::BFloat16,
        "x must have dtype torch.float16 or torch.bfloat16");
    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
    CheckPositiveDimensions(x, "x");
    const auto rows = x.numel() / x.size(-1);
    CheckFitsUint32(rows, "activation row count");
    CheckFitsUint32(x.size(-1), "K");

    c10::OptionalDeviceGuard guard(x.device());
    auto data = at_npu::native::OpPreparation::apply_tensor_without_format(
        x.sizes(), x.options().dtype(at::ScalarType::Byte));
    constexpr int64_t kScaleStride = 16;
    auto tokenScaleStorage =
        at_npu::native::OpPreparation::apply_tensor_without_format(
            {rows * kScaleStride}, x.options().dtype(at::ScalarType::Float));
    auto tokenScale = tokenScaleStorage.as_strided({rows}, {kScaleStride});
    const auto stream = CurrentStream(x.device());

    auto run = [x, data, tokenScale, stream]() -> int {
        PrepareDequeuedStream(stream);
        AclTensorHandle xAcl(x);
        AclTensorHandle dataAcl(data);
        AclTensorHandle tokenScaleAcl(tokenScale);
        uint64_t workspaceSize = 0;
        AclExecutorHandle executor;
        CheckAclStatus(
            aclnnHif8DynamicQuantGetWorkspaceSize(
                xAcl.get(),
                dataAcl.get(),
                tokenScaleAcl.get(),
                &workspaceSize,
                executor.out()),
            "aclnnHif8DynamicQuantGetWorkspaceSize");

        at::Tensor workspace;
        void *workspaceAddress =
            AllocateWorkspace(workspaceSize, stream, workspace);
        const auto status = aclnnHif8DynamicQuant(
            workspaceAddress,
            workspaceSize,
            executor.get(),
            stream);
        CheckAclStatus(status, "aclnnHif8DynamicQuant");
        return status;
    };
    at_npu::native::OpCommand::RunOpApiV2(
        "aclnnHif8DynamicQuant", run);
    return std::make_tuple(data, tokenScale);
}

std::tuple<at::Tensor, at::Tensor> Hif8StaticQuant(
    const at::Tensor &x,
    const at::Tensor &fixedScale) {
    CheckNpuTensor(x, "x");
    CheckNpuTensor(fixedScale, "fixed_scale");
    TORCH_CHECK(
        x.scalar_type() == at::ScalarType::Half ||
            x.scalar_type() == at::ScalarType::BFloat16,
        "x must have dtype torch.float16 or torch.bfloat16");
    TORCH_CHECK(
        fixedScale.scalar_type() == at::ScalarType::Float,
        "fixed_scale must have dtype torch.float32");
    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
    TORCH_CHECK(
        fixedScale.dim() == 1,
        "fixed_scale must have shape [scale_count]");
    CheckPositiveDimensions(x, "x");
    CheckPositiveDimensions(fixedScale, "fixed_scale");
    const auto rows = x.numel() / x.size(-1);
    TORCH_CHECK(
        rows % fixedScale.numel() == 0,
        "fixed_scale count must divide the activation row count");
    CheckFitsUint32(rows, "activation row count");
    CheckFitsUint32(x.size(-1), "K");
    CheckFitsUint32(fixedScale.numel(), "fixed scale count");
    CheckSameDevice(fixedScale, x.device(), "fixed_scale");
    c10::OptionalDeviceGuard guard(x.device());
    auto data = at_npu::native::OpPreparation::apply_tensor_without_format(
        x.sizes(), x.options().dtype(at::ScalarType::Byte));
    constexpr int64_t kScaleStride = 16;
    auto rowScaleStorage =
        at_npu::native::OpPreparation::apply_tensor_without_format(
            {rows * kScaleStride}, x.options().dtype(at::ScalarType::Float));
    auto rowScale = rowScaleStorage.as_strided({rows}, {kScaleStride});
    const auto stream = CurrentStream(x.device());

    auto run = [x, fixedScale, data, rowScale, stream]() -> int {
        PrepareDequeuedStream(stream);
        AclTensorHandle xAcl(x);
        AclTensorHandle fixedScaleAcl(fixedScale);
        AclTensorHandle dataAcl(data);
        AclTensorHandle rowScaleAcl(rowScale);
        uint64_t workspaceSize = 0;
        AclExecutorHandle executor;
        CheckAclStatus(
            aclnnHif8StaticQuantGetWorkspaceSize(
                xAcl.get(),
                fixedScaleAcl.get(),
                dataAcl.get(),
                rowScaleAcl.get(),
                &workspaceSize,
                executor.out()),
            "aclnnHif8StaticQuantGetWorkspaceSize");

        at::Tensor workspace;
        void *workspaceAddress =
            AllocateWorkspace(workspaceSize, stream, workspace);
        const auto status = aclnnHif8StaticQuant(
            workspaceAddress,
            workspaceSize,
            executor.get(),
            stream);
        CheckAclStatus(status, "aclnnHif8StaticQuant");
        return status;
    };
    at_npu::native::OpCommand::RunOpApiV2("aclnnHif8StaticQuant", run);
    return std::make_tuple(data, rowScale);
}

at::Tensor Hif8ScaledDecode(
    const at::Tensor &data,
    const at::Tensor &scale) {
    CheckNpuTensor(data, "data");
    CheckNpuTensor(scale, "scale");
    TORCH_CHECK(
        data.scalar_type() == at::ScalarType::Byte,
        "data must have dtype torch.uint8");
    TORCH_CHECK(
        scale.scalar_type() == at::ScalarType::Float,
        "scale must have dtype torch.float32");
    TORCH_CHECK(data.dim() == 2, "data must have shape [rows, K]");
    TORCH_CHECK(scale.dim() == 1, "scale must have shape [rows]");
    CheckPositiveDimensions(data, "data");
    TORCH_CHECK(
        scale.numel() == data.size(0),
        "scale must contain one value per data row");
    CheckFitsUint32(data.size(0), "row count");
    CheckFitsUint32(data.size(1), "K");
    CheckSameDevice(scale, data.device(), "scale");

    c10::OptionalDeviceGuard guard(data.device());
    auto output = at_npu::native::OpPreparation::apply_tensor_without_format(
        data.sizes(), data.options().dtype(at::ScalarType::BFloat16));
    const auto stream = CurrentStream(data.device());

    auto run = [data, scale, output, stream]() -> int {
        PrepareDequeuedStream(stream);
        AclTensorHandle dataAcl(data);
        AclTensorHandle scaleAcl(scale);
        AclTensorHandle outputAcl(output);
        uint64_t workspaceSize = 0;
        AclExecutorHandle executor;
        CheckAclStatus(
            aclnnHif8ScaledDecodeGetWorkspaceSize(
                dataAcl.get(),
                scaleAcl.get(),
                outputAcl.get(),
                &workspaceSize,
                executor.out()),
            "aclnnHif8ScaledDecodeGetWorkspaceSize");

        at::Tensor workspace;
        void *workspaceAddress =
            AllocateWorkspace(workspaceSize, stream, workspace);
        const auto status = aclnnHif8ScaledDecode(
            workspaceAddress,
            workspaceSize,
            executor.get(),
            stream);
        CheckAclStatus(status, "aclnnHif8ScaledDecode");
        return status;
    };
    at_npu::native::OpCommand::RunOpApiV2("aclnnHif8ScaledDecode", run);
    return output;
}

at::Tensor SoftHif8Gemv(
    const at::Tensor &activation,
    const at::Tensor &weight,
    const at::Tensor &tokenScale,
    const at::Tensor &channelScale,
    const c10::optional<at::Tensor> &bias,
    at::ScalarType outputType) {
    return SoftHif8Linear(
        activation,
        weight,
        tokenScale,
        channelScale,
        bias,
        outputType,
        true,
        "aclnnSoftHif8Gemv",
        aclnnSoftHif8GemvGetWorkspaceSize,
        aclnnSoftHif8Gemv);
}

at::Tensor SoftHif8Gemm(
    const at::Tensor &activation,
    const at::Tensor &weight,
    const at::Tensor &tokenScale,
    const at::Tensor &channelScale,
    const c10::optional<at::Tensor> &bias,
    at::ScalarType outputType) {
    const bool useBatchedEpilogue =
        BatchedEpilogueEnabled() && activation.dim() == 2 &&
        weight.dim() == 2 &&
        activation.size(0) >= kBatchedEpilogueMinRows &&
        weight.size(0) % kBatchedEpilogueColumnAlignment == 0;
    if (useBatchedEpilogue) {
        return SoftHif8Linear(
            activation,
            weight,
            tokenScale,
            channelScale,
            bias,
            outputType,
            false,
            "aclnnSoftHif8PrefillGemm",
            aclnnSoftHif8PrefillGemmGetWorkspaceSize,
            aclnnSoftHif8PrefillGemm);
    }
    return SoftHif8Linear(
        activation,
        weight,
        tokenScale,
        channelScale,
        bias,
        outputType,
        false,
        "aclnnSoftHif8Gemm",
        aclnnSoftHif8GemmGetWorkspaceSize,
        aclnnSoftHif8Gemm);
}

at::Tensor SoftHif8FusedGemm(
    const at::Tensor &activation,
    const at::Tensor &weight,
    const at::Tensor &channelScale,
    const c10::optional<at::Tensor> &bias,
    at::ScalarType outputType) {
    CheckNpuTensor(activation, "activation");
    CheckNpuTensor(weight, "weight");
    CheckNpuTensor(channelScale, "channel_scale");
    TORCH_CHECK(
        activation.scalar_type() == at::ScalarType::Half ||
            activation.scalar_type() == at::ScalarType::BFloat16,
        "activation must have dtype torch.float16 or torch.bfloat16");
    TORCH_CHECK(
        weight.scalar_type() == at::ScalarType::Byte,
        "weight must have dtype torch.uint8");
    TORCH_CHECK(
        channelScale.scalar_type() == at::ScalarType::Float,
        "channel_scale must have dtype torch.float32");
    TORCH_CHECK(activation.dim() == 2, "activation must have shape [M, K]");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [N, K]");
    TORCH_CHECK(
        channelScale.dim() == 1,
        "channel_scale must have shape [N]");
    CheckPositiveDimensions(activation, "activation");
    CheckPositiveDimensions(weight, "weight");
    TORCH_CHECK(
        activation.size(0) <= 2,
        "soft_hif8_fused_gemm requires M <= 2");
    TORCH_CHECK(
        activation.size(1) == weight.size(1),
        "activation and weight K dimensions differ: ",
        activation.size(1),
        " != ",
        weight.size(1));
    TORCH_CHECK(
        channelScale.numel() == weight.size(0),
        "channel_scale must contain one value per output channel");
    CheckFitsUint32(activation.size(0), "M");
    CheckFitsUint32(weight.size(0), "N");
    CheckFitsUint32(activation.size(1), "K");
    CheckSameDevice(weight, activation.device(), "weight");
    CheckSameDevice(channelScale, activation.device(), "channel_scale");
    if (bias.has_value()) {
        CheckNpuTensor(*bias, "bias");
        CheckSameDevice(*bias, activation.device(), "bias");
        TORCH_CHECK(bias->dim() == 1, "bias must have shape [N]");
        TORCH_CHECK(bias->numel() == weight.size(0), "bias must contain N values");
        TORCH_CHECK(
            bias->scalar_type() == at::ScalarType::Half ||
                bias->scalar_type() == at::ScalarType::BFloat16,
            "bias must have dtype torch.float16 or torch.bfloat16");
    }
    TORCH_CHECK(
        outputType == at::ScalarType::Half ||
            outputType == at::ScalarType::BFloat16,
        "output_dtype must be torch.float16 or torch.bfloat16");

    c10::OptionalDeviceGuard guard(activation.device());
    auto output = at_npu::native::OpPreparation::apply_tensor_without_format(
        {activation.size(0), weight.size(0)},
        activation.options().dtype(outputType));
    const auto stream = CurrentStream(activation.device());
    const int64_t outputDtype =
        outputType == at::ScalarType::Half ? kFp16Output : kBf16Output;
    auto run = [activation,
                weight,
                channelScale,
                bias,
                output,
                outputDtype,
                stream]() -> int {
        PrepareDequeuedStream(stream);
        AclTensorHandle activationAcl(activation);
        AclTensorHandle weightAcl(weight);
        AclTensorHandle channelScaleAcl(channelScale);
        AclTensorHandle outputAcl(output);
        c10::optional<AclTensorHandle> biasAcl;
        if (bias.has_value()) {
            biasAcl.emplace(*bias);
        }
        uint64_t workspaceSize = 0;
        AclExecutorHandle executor;
        CheckAclStatus(
            aclnnSoftHif8FusedGemmGetWorkspaceSize(
                activationAcl.get(),
                weightAcl.get(),
                channelScaleAcl.get(),
                biasAcl.has_value() ? biasAcl->get() : nullptr,
                outputDtype,
                outputAcl.get(),
                &workspaceSize,
                executor.out()),
            "aclnnSoftHif8FusedGemmGetWorkspaceSize");
        TORCH_CHECK(
            workspaceSize != 0,
            "soft_hif8_fused_gemm reported a zero-byte workspace");
        at::Tensor workspace;
        void *workspaceAddress =
            AllocateWorkspace(workspaceSize, stream, workspace);
        const auto status = aclnnSoftHif8FusedGemm(
            workspaceAddress,
            workspaceSize,
            executor.get(),
            stream);
        CheckAclStatus(status, "aclnnSoftHif8FusedGemm");
        return status;
    };
    at_npu::native::OpCommand::RunOpApiV2(
        "aclnnSoftHif8FusedGemm", run);
    return output;
}

at::Tensor SoftHif8StaticFusedGemm(
    const at::Tensor &activation,
    const at::Tensor &weight,
    const at::Tensor &fixedScale,
    const at::Tensor &channelScale,
    const c10::optional<at::Tensor> &bias,
    at::ScalarType outputType) {
    CheckNpuTensor(activation, "activation");
    CheckNpuTensor(weight, "weight");
    CheckNpuTensor(fixedScale, "fixed_scale");
    CheckNpuTensor(channelScale, "channel_scale");
    TORCH_CHECK(
        activation.scalar_type() == at::ScalarType::Half ||
            activation.scalar_type() == at::ScalarType::BFloat16,
        "activation must have dtype torch.float16 or torch.bfloat16");
    TORCH_CHECK(
        weight.scalar_type() == at::ScalarType::Byte,
        "weight must have dtype torch.uint8");
    TORCH_CHECK(
        fixedScale.scalar_type() == at::ScalarType::Float,
        "fixed_scale must have dtype torch.float32");
    TORCH_CHECK(
        channelScale.scalar_type() == at::ScalarType::Float,
        "channel_scale must have dtype torch.float32");
    TORCH_CHECK(activation.dim() == 2, "activation must have shape [M, K]");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [N, K]");
    TORCH_CHECK(
        fixedScale.dim() == 1 && fixedScale.numel() == 1,
        "fixed_scale must have shape [1]");
    TORCH_CHECK(
        channelScale.dim() == 1,
        "channel_scale must have shape [N]");
    CheckPositiveDimensions(activation, "activation");
    CheckPositiveDimensions(weight, "weight");
    CheckPositiveDimensions(fixedScale, "fixed_scale");
    TORCH_CHECK(
        activation.size(0) <= 4,
        "soft_hif8_static_fused_gemm requires M <= 4");
    TORCH_CHECK(
        activation.size(1) == weight.size(1),
        "activation and weight K dimensions differ: ",
        activation.size(1),
        " != ",
        weight.size(1));
    TORCH_CHECK(
        channelScale.numel() == weight.size(0),
        "channel_scale must contain one value per output channel");
    CheckFitsUint32(activation.size(0), "M");
    CheckFitsUint32(weight.size(0), "N");
    CheckFitsUint32(activation.size(1), "K");
    CheckSameDevice(weight, activation.device(), "weight");
    CheckSameDevice(fixedScale, activation.device(), "fixed_scale");
    CheckSameDevice(channelScale, activation.device(), "channel_scale");
    if (bias.has_value()) {
        CheckNpuTensor(*bias, "bias");
        CheckSameDevice(*bias, activation.device(), "bias");
        TORCH_CHECK(bias->dim() == 1, "bias must have shape [N]");
        TORCH_CHECK(
            bias->numel() == weight.size(0),
            "bias must contain N values");
        TORCH_CHECK(
            bias->scalar_type() == at::ScalarType::Half ||
                bias->scalar_type() == at::ScalarType::BFloat16,
            "bias must have dtype torch.float16 or torch.bfloat16");
    }
    TORCH_CHECK(
        outputType == at::ScalarType::Half ||
            outputType == at::ScalarType::BFloat16,
        "output_dtype must be torch.float16 or torch.bfloat16");

    c10::OptionalDeviceGuard guard(activation.device());
    auto output = at_npu::native::OpPreparation::apply_tensor_without_format(
        {activation.size(0), weight.size(0)},
        activation.options().dtype(outputType));
    const auto stream = CurrentStream(activation.device());
    const int64_t outputDtype =
        outputType == at::ScalarType::Half ? kFp16Output : kBf16Output;
    auto run = [activation,
                weight,
                fixedScale,
                channelScale,
                bias,
                output,
                outputDtype,
                stream]() -> int {
        PrepareDequeuedStream(stream);
        AclTensorHandle activationAcl(activation);
        AclTensorHandle weightAcl(weight);
        AclTensorHandle fixedScaleAcl(fixedScale);
        AclTensorHandle channelScaleAcl(channelScale);
        AclTensorHandle outputAcl(output);
        c10::optional<AclTensorHandle> biasAcl;
        if (bias.has_value()) {
            biasAcl.emplace(*bias);
        }
        uint64_t workspaceSize = 0;
        AclExecutorHandle executor;
        CheckAclStatus(
            aclnnSoftHif8StaticFusedGemmGetWorkspaceSize(
                activationAcl.get(),
                weightAcl.get(),
                fixedScaleAcl.get(),
                channelScaleAcl.get(),
                biasAcl.has_value() ? biasAcl->get() : nullptr,
                outputDtype,
                outputAcl.get(),
                &workspaceSize,
                executor.out()),
            "aclnnSoftHif8StaticFusedGemmGetWorkspaceSize");
        TORCH_CHECK(
            workspaceSize != 0,
            "soft_hif8_static_fused_gemm reported a zero-byte workspace");
        at::Tensor workspace;
        void *workspaceAddress =
            AllocateWorkspace(workspaceSize, stream, workspace);
        const auto status = aclnnSoftHif8StaticFusedGemm(
            workspaceAddress,
            workspaceSize,
            executor.get(),
            stream);
        CheckAclStatus(status, "aclnnSoftHif8StaticFusedGemm");
        return status;
    };
    at_npu::native::OpCommand::RunOpApiV2(
        "aclnnSoftHif8StaticFusedGemm", run);
    return output;
}

}  // namespace vllm_ascend_hif8
