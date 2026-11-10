#include <cstdint>
#include <cstdlib>
#include <limits>

#include "../op_kernel/soft_hif8_gemm_tiling.h"
#include "matmul/bmm_tiling.h"
#include "matmul/matmul_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr size_t kFixedScaleIndex = 2;
constexpr size_t kChannelScaleIndex = 3;
constexpr size_t kBiasIndex = 4;
constexpr uint32_t kTinyTileM = 16;
constexpr uint32_t kTileN = 128;
constexpr uint32_t kTileK = 2048;
constexpr uint32_t kCubeSplitN = 128;
constexpr uint32_t kCubeSplitK = 128;
constexpr uint64_t kWorkspaceAlignment = 512;
constexpr uint64_t kStagingGuardBytes = 64ULL * 1024;
constexpr uint64_t kAscendCReservedWorkspaceSize = 16ULL * 1024 * 1024;

uint64_t AlignUp(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

uint32_t CubeCoreNum(gert::TilingContext *context) {
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return 1;
    }
    const uint32_t cores =
        platform_ascendc::PlatformAscendC(platformInfo).GetCoreNumAic();
    return cores == 0 ? 1 : cores;
}

ge::graphStatus InferStaticShape(gert::InferShapeContext *context) {
    const gert::Shape *activationShape = context->GetInputShape(0);
    const gert::Shape *weightShape = context->GetInputShape(1);
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (activationShape == nullptr || weightShape == nullptr ||
        outputShape == nullptr || activationShape->GetDimNum() != 2 ||
        weightShape->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    outputShape->SetDimNum(2);
    outputShape->SetDim(0, activationShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(0));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferStaticDataType(gert::InferDataTypeContext *context) {
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *outputDtype = attrs == nullptr ? nullptr : attrs->GetInt(0);
    if (outputDtype == nullptr || (*outputDtype != 0 && *outputDtype != 1)) {
        return ge::GRAPH_FAILED;
    }
    return context->SetOutputDataType(
        0, *outputDtype == 0 ? ge::DT_FLOAT16 : ge::DT_BF16);
}

ge::graphStatus StaticTilingFunc(gert::TilingContext *context) {
    const gert::StorageShape *activationStorage = context->GetInputShape(0);
    const gert::StorageShape *weightStorage = context->GetInputShape(1);
    const gert::StorageShape *fixedScaleStorage =
        context->GetInputShape(kFixedScaleIndex);
    const gert::StorageShape *channelScaleStorage =
        context->GetInputShape(kChannelScaleIndex);
    const gert::CompileTimeTensorDesc *activationDesc =
        context->GetInputDesc(0);
    const gert::CompileTimeTensorDesc *weightDesc = context->GetInputDesc(1);
    const gert::CompileTimeTensorDesc *fixedScaleDesc =
        context->GetInputDesc(kFixedScaleIndex);
    const gert::CompileTimeTensorDesc *channelScaleDesc =
        context->GetInputDesc(kChannelScaleIndex);
    if (activationStorage == nullptr || weightStorage == nullptr ||
        fixedScaleStorage == nullptr || channelScaleStorage == nullptr ||
        activationDesc == nullptr || weightDesc == nullptr ||
        fixedScaleDesc == nullptr || channelScaleDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &activationShape = activationStorage->GetStorageShape();
    const gert::Shape &weightShape = weightStorage->GetStorageShape();
    const gert::Shape &fixedScaleShape = fixedScaleStorage->GetStorageShape();
    const gert::Shape &channelScaleShape =
        channelScaleStorage->GetStorageShape();
    if (activationShape.GetDimNum() != 2 || weightShape.GetDimNum() != 2 ||
        fixedScaleShape.GetDimNum() != 1 ||
        channelScaleShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const int64_t m = activationShape.GetDim(0);
    const int64_t k = activationShape.GetDim(1);
    const int64_t n = weightShape.GetDim(0);
    if (m <= 0 || n <= 0 || k <= 0 ||
        m > std::numeric_limits<uint32_t>::max() ||
        n > std::numeric_limits<uint32_t>::max() ||
        k > std::numeric_limits<uint32_t>::max() ||
        weightShape.GetDim(1) != k || fixedScaleShape.GetShapeSize() != 1 ||
        channelScaleShape.GetShapeSize() != n ||
        (activationDesc->GetDataType() != ge::DT_FLOAT16 &&
         activationDesc->GetDataType() != ge::DT_BF16) ||
        weightDesc->GetDataType() != ge::DT_UINT8 ||
        fixedScaleDesc->GetDataType() != ge::DT_FLOAT ||
        channelScaleDesc->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *biasStorage =
        context->GetOptionalInputShape(kBiasIndex);
    const gert::CompileTimeTensorDesc *biasDesc =
        context->GetOptionalInputDesc(kBiasIndex);
    if ((biasStorage == nullptr) != (biasDesc == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    if (biasStorage != nullptr) {
        const gert::Shape &biasShape = biasStorage->GetStorageShape();
        if (biasShape.GetDimNum() != 1 || biasShape.GetShapeSize() != n ||
            (biasDesc->GetDataType() != ge::DT_FLOAT16 &&
             biasDesc->GetDataType() != ge::DT_BF16)) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *outputDtype = attrs == nullptr ? nullptr : attrs->GetInt(0);
    const gert::CompileTimeTensorDesc *outputDesc = context->GetOutputDesc(0);
    if (outputDtype == nullptr || outputDesc == nullptr ||
        (*outputDtype != 0 && *outputDtype != 1) ||
        outputDesc->GetDataType() !=
            (*outputDtype == 0 ? ge::DT_FLOAT16 : ge::DT_BF16)) {
        return ge::GRAPH_FAILED;
    }

    auto *tiling =
        context->GetTilingData<SoftHif8StaticFusedGemmTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->m = static_cast<uint32_t>(m);
    tiling->n = static_cast<uint32_t>(n);
    tiling->k = static_cast<uint32_t>(k);
    tiling->hasBias = biasStorage == nullptr ? 0 : 1;
    tiling->biasDtype =
        biasDesc != nullptr && biasDesc->GetDataType() == ge::DT_BF16 ? 1 : 0;
    tiling->aicNum = CubeCoreNum(context);

    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    tiling->aivNum = platform.GetCoreNumAiv();
    tiling->tileM = kTinyTileM;
    tiling->tileN = kTileN;
    tiling->tileK = kTileK;
    tiling->mTileCount = (tiling->m + tiling->tileM - 1U) / tiling->tileM;
    tiling->nTileCount = (tiling->n + tiling->tileN - 1U) / tiling->tileN;
    tiling->kTileCount = (tiling->k + tiling->tileK - 1U) / tiling->tileK;

    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
    if (l1Size > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        l0cSize > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return ge::GRAPH_FAILED;
    }

    matmul_tiling::MatmulApiTiling mmTiling;
    mmTiling.SetAType(matmul_tiling::TPosition::GM,
                      matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_BF16, false);
    mmTiling.SetBType(matmul_tiling::TPosition::GM,
                      matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_BF16, true);
    mmTiling.SetCType(matmul_tiling::TPosition::GM,
                      matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_FLOAT);
    tiling->outputStride = tiling->n;
    mmTiling.SetOrgShape(tiling->tileM, tiling->tileN,
                         tiling->tileK, tiling->tileK);
    mmTiling.SetShape(tiling->tileM, tiling->tileN, tiling->tileK);
    mmTiling.SetBufferSpace(static_cast<int32_t>(l1Size),
                            static_cast<int32_t>(l0cSize));
    mmTiling.SetFixSplit(tiling->tileM, kCubeSplitN, kCubeSplitK);
    if (mmTiling.GetTiling(tiling->matmulTiling) == -1) {
        return ge::GRAPH_FAILED;
    }
    tiling->matmulTiling.shareL1Size = 0;
    // One AIC avoids overlapping vector-producer staging domains on 910B.
    tiling->aicNum = 1;
    const char *requested = std::getenv("HIF8_STATIC_FUSED_AIC_NUM");
    if (requested != nullptr && requested[0] != '\0') {
        char *end = nullptr;
        const unsigned long value = std::strtoul(requested, &end, 10);
        if (end != requested && *end == '\0' && value > 0) {
            tiling->aicNum = static_cast<uint32_t>(value);
            if (tiling->aicNum > CubeCoreNum(context)) {
                tiling->aicNum = CubeCoreNum(context);
            }
        }
    }
    if (tiling->aicNum == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t outputElements =
        static_cast<uint64_t>(tiling->m) * tiling->n;
    if (outputElements >
        std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        return ge::GRAPH_FAILED;
    }
    tiling->outputWorkspaceOffset = 0;
    const uint64_t outputWorkspaceSize = outputElements * sizeof(float);
    const uint64_t weightStageBytes =
        static_cast<uint64_t>(tiling->tileN) * tiling->tileK *
        sizeof(uint16_t);
    const uint64_t activationStageBytes =
        static_cast<uint64_t>(tiling->kTileCount) * tiling->tileM *
        tiling->tileK * sizeof(uint16_t);
    tiling->activationWorkspaceOffset =
        AlignUp(outputWorkspaceSize, kWorkspaceAlignment) +
        kStagingGuardBytes;
    if (tiling->activationWorkspaceOffset >
        std::numeric_limits<uint64_t>::max() - activationStageBytes) {
        return ge::GRAPH_FAILED;
    }
    tiling->tokenScaleWorkspaceOffset = AlignUp(
        tiling->activationWorkspaceOffset + activationStageBytes,
        kWorkspaceAlignment);
    const uint64_t tokenScaleBytes =
        static_cast<uint64_t>(tiling->m) * sizeof(float);
    if (tiling->tokenScaleWorkspaceOffset >
        std::numeric_limits<uint64_t>::max() - tokenScaleBytes) {
        return ge::GRAPH_FAILED;
    }
    tiling->weightWorkspaceOffset = AlignUp(
        tiling->tokenScaleWorkspaceOffset + tokenScaleBytes,
        kWorkspaceAlignment);
    tiling->weightWorkspaceStride =
        AlignUp(2ULL * weightStageBytes, kWorkspaceAlignment);
    const uint64_t weightWorkspaceSize =
        static_cast<uint64_t>(tiling->aicNum) *
        tiling->weightWorkspaceStride;
    if (tiling->weightWorkspaceOffset >
        std::numeric_limits<uint64_t>::max() - weightWorkspaceSize) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t userWorkspaceSize =
        tiling->weightWorkspaceOffset + weightWorkspaceSize;
    if (userWorkspaceSize >
            std::numeric_limits<uint64_t>::max() -
                kAscendCReservedWorkspaceSize ||
        userWorkspaceSize + kAscendCReservedWorkspaceSize >
            std::numeric_limits<size_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(tiling->aicNum);
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = static_cast<size_t>(
        kAscendCReservedWorkspaceSize + userWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

}  // namespace

namespace ops {

class SoftHif8StaticFusedGemm : public OpDef {
public:
    explicit SoftHif8StaticFusedGemm(const char *name) : OpDef(name) {
        this->Input("activation")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("fixed_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("channel_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("output_dtype").Int();

        this->SetInferShape(::InferStaticShape)
            .SetInferDataType(::InferStaticDataType);
        this->AICore()
            .SetTiling(::StaticTilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(SoftHif8StaticFusedGemm);

}  // namespace ops
