#include <algorithm>
#include <cstdint>
#include <limits>

#include "../op_kernel/hif8_static_quant_tiling.h"
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
    const gert::Shape *xShape = context->GetInputShape(0);
    const gert::Shape *fixedScaleShape = context->GetInputShape(1);
    gert::Shape *dataShape = context->GetOutputShape(0);
    gert::Shape *rowScaleShape = context->GetOutputShape(1);
    if (xShape == nullptr || fixedScaleShape == nullptr ||
        dataShape == nullptr || rowScaleShape == nullptr ||
        xShape->GetDimNum() == 0 || fixedScaleShape->GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    *dataShape = *xShape;
    int64_t rows = 1;
    for (size_t axis = 0; axis + 1 < xShape->GetDimNum(); ++axis) {
        const int64_t dim = xShape->GetDim(axis);
        if (dim < 0) {
            rows = -1;
            break;
        }
        if (dim != 0 && rows > std::numeric_limits<int64_t>::max() / dim) {
            return ge::GRAPH_FAILED;
        }
        rows *= dim;
    }
    const int64_t scaleCount = fixedScaleShape->GetDim(0);
    if (rows >= 0 && (scaleCount <= 0 || rows % scaleCount != 0)) {
        return ge::GRAPH_FAILED;
    }
    rowScaleShape->SetDimNum(1);
    rowScaleShape->SetDim(0, rows);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    if (context->SetOutputDataType(0, ge::DT_UINT8) != ge::GRAPH_SUCCESS ||
        context->SetOutputDataType(1, ge::DT_FLOAT) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::StorageShape *xStorage = context->GetInputShape(0);
    const gert::StorageShape *fixedScaleStorage = context->GetInputShape(1);
    const gert::CompileTimeTensorDesc *xDesc = context->GetInputDesc(0);
    const gert::CompileTimeTensorDesc *fixedScaleDesc = context->GetInputDesc(1);
    if (xStorage == nullptr || fixedScaleStorage == nullptr ||
        xDesc == nullptr || fixedScaleDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &xShape = xStorage->GetStorageShape();
    const gert::Shape &fixedScaleShape = fixedScaleStorage->GetStorageShape();
    const size_t rank = xShape.GetDimNum();
    if (rank == 0 || fixedScaleShape.GetDimNum() != 1 ||
        fixedScaleDesc->GetDataType() != ge::DT_FLOAT ||
        (xDesc->GetDataType() != ge::DT_FLOAT16 &&
         xDesc->GetDataType() != ge::DT_BF16)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t k = xShape.GetDim(rank - 1);
    if (k <= 0 || k > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    uint64_t rows = 1;
    for (size_t axis = 0; axis + 1 < rank; ++axis) {
        const int64_t dim = xShape.GetDim(axis);
        if (dim <= 0 ||
            rows > std::numeric_limits<uint32_t>::max() /
                       static_cast<uint64_t>(dim)) {
            return ge::GRAPH_FAILED;
        }
        rows *= static_cast<uint64_t>(dim);
    }
    const int64_t scaleCount = fixedScaleShape.GetShapeSize();
    if (scaleCount <= 0 ||
        scaleCount > std::numeric_limits<uint32_t>::max() ||
        rows % static_cast<uint64_t>(scaleCount) != 0) {
        return ge::GRAPH_FAILED;
    }

    auto *tiling = context->GetTilingData<Hif8StaticQuantTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->rows = static_cast<uint32_t>(rows);
    tiling->k = static_cast<uint32_t>(k);
    tiling->scaleCount = static_cast<uint32_t>(scaleCount);
    const uint32_t activeCores =
        tiling->k % 32 == 0
            ? std::min<uint32_t>(tiling->rows, VectorCoreNum(context))
            : 1;
    context->SetBlockDim(activeCores);
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace

namespace ops {

class Hif8StaticQuant : public OpDef {
public:
    explicit Hif8StaticQuant(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("fixed_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("data")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("row_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(::InferShape).SetInferDataType(::InferDataType);
        this->AICore()
            .SetTiling(::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Hif8StaticQuant);

}  // namespace ops
