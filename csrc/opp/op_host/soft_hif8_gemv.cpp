#include <algorithm>
#include <cstdint>
#include <limits>

#include "../op_kernel/soft_hif8_gemv_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr size_t kBiasIndex = 4;

// The kernels are vector-only, so the AI vector core count is the usable width.
// A context without platform info still has to produce a launchable tiling.
uint32_t VectorCoreNum(gert::TilingContext *context) {
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return 1;
    }
    const uint32_t cores =
        platform_ascendc::PlatformAscendC(platformInfo).GetCoreNumAiv();
    return cores == 0 ? 1 : cores;
}

ge::graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *activationShape = context->GetInputShape(0);
    const gert::Shape *weightShape = context->GetInputShape(1);
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (activationShape == nullptr || weightShape == nullptr || outputShape == nullptr ||
        activationShape->GetDimNum() != 2 || weightShape->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    outputShape->SetDimNum(2);
    outputShape->SetDim(0, activationShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(0));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *outputDtype = attrs == nullptr ? nullptr : attrs->GetInt(0);
    if (outputDtype == nullptr || (*outputDtype != 0 && *outputDtype != 1)) {
        return ge::GRAPH_FAILED;
    }
    return context->SetOutputDataType(
        0, *outputDtype == 0 ? ge::DT_FLOAT16 : ge::DT_BF16);
}

ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::StorageShape *activationStorage = context->GetInputShape(0);
    const gert::StorageShape *weightStorage = context->GetInputShape(1);
    const gert::StorageShape *tokenScaleStorage = context->GetInputShape(2);
    const gert::StorageShape *channelScaleStorage = context->GetInputShape(3);
    if (activationStorage == nullptr || weightStorage == nullptr ||
        tokenScaleStorage == nullptr || channelScaleStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &activationShape = activationStorage->GetStorageShape();
    const gert::Shape &weightShape = weightStorage->GetStorageShape();
    const gert::Shape &tokenScaleShape = tokenScaleStorage->GetStorageShape();
    const gert::Shape &channelScaleShape = channelScaleStorage->GetStorageShape();
    if (activationShape.GetDimNum() != 2 || weightShape.GetDimNum() != 2 ||
        tokenScaleShape.GetDimNum() != 1 || channelScaleShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const int64_t m = activationShape.GetDim(0);
    const int64_t k = activationShape.GetDim(1);
    const int64_t n = weightShape.GetDim(0);
    if (m <= 0 || m > 2 || n <= 0 || k <= 0 ||
        m > std::numeric_limits<uint32_t>::max() ||
        n > std::numeric_limits<uint32_t>::max() ||
        k > std::numeric_limits<uint32_t>::max() ||
        weightShape.GetDim(1) != k || tokenScaleShape.GetShapeSize() != m ||
        channelScaleShape.GetShapeSize() != n) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *biasStorage = context->GetOptionalInputShape(kBiasIndex);
    const gert::CompileTimeTensorDesc *biasDesc = context->GetOptionalInputDesc(kBiasIndex);
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

    auto *tiling = context->GetTilingData<SoftHif8GemvTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->m = static_cast<uint32_t>(m);
    tiling->n = static_cast<uint32_t>(n);
    tiling->k = static_cast<uint32_t>(k);
    tiling->hasBias = biasStorage == nullptr ? 0 : 1;
    tiling->biasDtype =
        biasDesc != nullptr && biasDesc->GetDataType() == ge::DT_BF16 ? 1 : 0;
    const uint64_t outputElements = static_cast<uint64_t>(tiling->m) * tiling->n;
    context->SetBlockDim(static_cast<uint32_t>(
        std::min<uint64_t>(outputElements, VectorCoreNum(context))));
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace

namespace ops {

class SoftHif8Gemv : public OpDef {
public:
    explicit SoftHif8Gemv(const char *name) : OpDef(name) {
        this->Input("activation")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("token_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("channel_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("output_dtype").Int();

        this->SetInferShape(::InferShape).SetInferDataType(::InferDataType);
        this->AICore()
            .SetTiling(::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(SoftHif8Gemv);

}  // namespace ops
