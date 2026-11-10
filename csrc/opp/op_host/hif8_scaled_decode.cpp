#include <algorithm>
#include <cstdint>
#include <limits>

#include "../op_kernel/hif8_scaled_decode_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

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
    const gert::Shape *dataShape = context->GetInputShape(0);
    const gert::Shape *scaleShape = context->GetInputShape(1);
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (dataShape == nullptr || scaleShape == nullptr || outputShape == nullptr ||
        dataShape->GetDimNum() != 2 || scaleShape->GetDimNum() != 1 ||
        scaleShape->GetDim(0) != dataShape->GetDim(0)) {
        return ge::GRAPH_FAILED;
    }
    *outputShape = *dataShape;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    return context->SetOutputDataType(0, ge::DT_BF16);
}

ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::StorageShape *dataStorage = context->GetInputShape(0);
    const gert::StorageShape *scaleStorage = context->GetInputShape(1);
    const gert::CompileTimeTensorDesc *dataDesc = context->GetInputDesc(0);
    const gert::CompileTimeTensorDesc *scaleDesc = context->GetInputDesc(1);
    const gert::CompileTimeTensorDesc *outputDesc = context->GetOutputDesc(0);
    if (dataStorage == nullptr || scaleStorage == nullptr || dataDesc == nullptr ||
        scaleDesc == nullptr || outputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &dataShape = dataStorage->GetStorageShape();
    const gert::Shape &scaleShape = scaleStorage->GetStorageShape();
    if (dataShape.GetDimNum() != 2 || scaleShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const int64_t rows = dataShape.GetDim(0);
    const int64_t k = dataShape.GetDim(1);
    if (rows <= 0 || k <= 0 ||
        rows > std::numeric_limits<uint32_t>::max() ||
        k > std::numeric_limits<uint32_t>::max() ||
        scaleShape.GetShapeSize() != rows || dataDesc->GetDataType() != ge::DT_UINT8 ||
        scaleDesc->GetDataType() != ge::DT_FLOAT ||
        outputDesc->GetDataType() != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }

    auto *tiling = context->GetTilingData<Hif8ScaledDecodeTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->rows = static_cast<uint32_t>(rows);
    tiling->k = static_cast<uint32_t>(k);
    // Scalar tail writes update a whole cache line. Keep unaligned rows on one
    // core so adjacent rows cannot race; production K dimensions are aligned.
    const uint32_t blockDim = tiling->k % 32 == 0
        ? std::min<uint32_t>(tiling->rows, VectorCoreNum(context))
        : 1;
    context->SetBlockDim(blockDim);

    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace

namespace ops {

class Hif8ScaledDecode : public OpDef {
public:
    explicit Hif8ScaledDecode(const char *name) : OpDef(name) {
        this->Input("data")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(::InferShape).SetInferDataType(::InferDataType);
        this->AICore()
            .SetTiling(::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Hif8ScaledDecode);

}  // namespace ops
