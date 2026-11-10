#include <algorithm>
#include <cstdint>
#include <limits>

#include "../op_kernel/hif8_dynamic_quant_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

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
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *dataShape = context->GetOutputShape(0);
    gert::Shape *scaleShape = context->GetOutputShape(1);
    if (xShape == nullptr || dataShape == nullptr || scaleShape == nullptr ||
        xShape->GetDimNum() == 0) {
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
    scaleShape->SetDimNum(1);
    scaleShape->SetDim(0, rows);
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
    if (xStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &xShape = xStorage->GetStorageShape();
    const size_t rank = xShape.GetDimNum();
    if (rank == 0) {
        return ge::GRAPH_FAILED;
    }

    const int64_t k = xShape.GetDim(rank - 1);
    if (k <= 0 || k > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    uint64_t rows = 1;
    for (size_t axis = 0; axis + 1 < rank; ++axis) {
        const int64_t dim = xShape.GetDim(axis);
        if (dim <= 0 || rows > std::numeric_limits<uint32_t>::max() /
                                   static_cast<uint64_t>(dim)) {
            return ge::GRAPH_FAILED;
        }
        rows *= static_cast<uint64_t>(dim);
    }

    auto *tiling = context->GetTilingData<Hif8DynamicQuantTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->rows = static_cast<uint32_t>(rows);
    tiling->k = static_cast<uint32_t>(k);
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

class Hif8DynamicQuant : public OpDef {
public:
    explicit Hif8DynamicQuant(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("data")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("token_scale")
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

OP_ADD(Hif8DynamicQuant);

}  // namespace ops
