#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "../op_kernel/soft_hif8_gemm_tiling.h"
#include "matmul/bmm_tiling.h"
#include "matmul/matmul_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

#ifndef HIF8_USE_TSCM_PIPELINE
#define HIF8_USE_TSCM_PIPELINE 0
#endif
#ifndef HIF8_WIDE_N_TILE
#define HIF8_WIDE_N_TILE 0
#endif

constexpr size_t kBiasIndex = 4;
constexpr uint32_t kTinyTileM = 16;
constexpr uint32_t kSmallTileM = 64;
constexpr uint32_t kLargeTileM = 128;
// Keep N tiles aligned to the fixed 128-column Cube split.  A 192-column
// tile leaves the second split only half populated; CANN 9.0 then writes
// invalid accumulators for columns 128..191.  Two complete splits preserve
// correctness and also reduce the number of staging/synchronization rounds.
constexpr uint32_t kTileN = 128;
constexpr uint32_t kWideTileN = 256;
// The merged path is a prefill optimization.  Below 32 rows, the extra
// second-slice setup and wider staging buffer are not consistently amortized.
constexpr uint32_t kWideMinM = 32;
// Wide-N tiles carry two Cube slices and therefore have enough work per
// producer once at least half of the AICs can receive a tile. Keep all AICs
// active in that case; otherwise a 256-column tile can lose the parallelism
// it was meant to recover (for example, N=4096 would otherwise use 16 of 24
// cores). Very small N still uses one block per tile to avoid idle barriers.
constexpr uint32_t kWideMinTilesForFullCores = 2;
// Bound UB usage for four-row byte/index/BF16 decode buffers.
constexpr uint32_t kTileK = 2048;
// C220 exposes 512 KiB of TSCM, with a small reserved tail.  The shared
// TSCM path stores one 16-row activation tile and one 128-row weight
// tile as BF16, so its footprint is tileK * (16 + 128) * 2 bytes.  1792 is
// the largest 128-aligned value that leaves headroom below the 512 KiB
// address limit; the regular GM-staging path keeps the faster 2048 tile.
constexpr uint32_t kTscmTileK = 1792;
constexpr uint32_t kCubeSplitN = 128;
constexpr uint32_t kCubeSplitK = 128;
constexpr uint64_t kWorkspaceAlignment = 512;
constexpr uint64_t kStagingGuardBytes = 64ULL * 1024;
constexpr uint64_t kAscendCReservedWorkspaceSize = 16ULL * 1024 * 1024;
// A decoded activation is invocation-local and shared by all Cube cores.
// Caching it removes repeated U8->BF16 unpack for every N tile/core while
// keeping the cache bounded for unusually large prefill requests.
constexpr uint64_t kActivationCacheMaxBytes = 32ULL * 1024 * 1024;
// Four or more N tiles amortize the one-time activation cache write even for
// the 8-row prefill shapes. Keep M=1/2 on the overlapped staging path, where
// a cache write costs more than the decode it avoids.
constexpr uint32_t kActivationCacheMinM = 8;
constexpr uint32_t kActivationCacheMinNTiles = 4;
constexpr bool kUseTscmPipeline = HIF8_USE_TSCM_PIPELINE != 0;
constexpr bool kEnableWideNTile = HIF8_WIDE_N_TILE != 0;
constexpr uint64_t kCubeInputBytes = 2U;

uint64_t AlignUp(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

uint32_t ActiveCubeCoreNum(uint32_t tileCount, uint32_t coreLimit,
                           bool preferBalancedRounds) {
    if (tileCount <= coreLimit) {
        return tileCount;
    }

    // Larger batches have enough decode and Cube work per N tile to benefit
    // from every available core.  The balanced-round choice is a generation
    // optimization: for one or two rows, avoiding a short tail round matters
    // more than the extra AIV/AIC parallelism.  Keep prefill on the original
    // full-core schedule, where reducing the number of producers regresses
    // the measured M=64 and M=128 shapes.
    if (!preferBalancedRounds) {
        return coreLimit;
    }

    // Keep the same number of N rounds while avoiding a small tail round.
    // For example, 32 tiles on 24 cores otherwise run as 24 + 8; 16 cores
    // produce two full 16-tile rounds with the same block lifetime. If no
    // divisor preserves the round count, retain the full core count because
    // adding rounds costs more than the tail imbalance.
    const uint32_t roundCount =
        (tileCount + coreLimit - 1U) / coreLimit;
    for (uint32_t cores = coreLimit; cores > 1; --cores) {
        if (tileCount % cores == 0 && tileCount / cores == roundCount) {
            return cores;
        }
    }
    return coreLimit;
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

bool EnvFlagEnabled(const char *name, bool defaultValue) {
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return defaultValue;
    }
    return value[0] != '0' || value[1] != '\0';
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
    if (m <= 0 || n <= 0 || k <= 0 ||
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

    auto *tiling = context->GetTilingData<SoftHif8GemmTilingData>();
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
    const bool stableMTile = EnvFlagEnabled("HIF8_STABLE_M_TILE", false);
    // Fix Cube row partitioning independently of request grouping.
    // Accept 16, 64, or 128; other values use the adaptive/stable policy.
    uint32_t forcedMTile = 0;
    if (const char *raw = std::getenv("HIF8_FORCE_TILE_M"); raw != nullptr) {
        const unsigned long parsed = std::strtoul(raw, nullptr, 10);
        if (parsed == kTinyTileM || parsed == kSmallTileM ||
            parsed == kLargeTileM) {
            forcedMTile = static_cast<uint32_t>(parsed);
        }
    }
    // Select the M tile from the runtime row count. Tiny and irregular
    // request batches stay on a 16-row tile to avoid padding three or four
    // requests up to a 64-row launch; medium/large prefill retains the wider
    // tile that amortizes staging and Cube setup.
    // Dynamic evaluation can opt into a stable 64-row tile for all small M
    // values. This keeps Cube accumulation partitioning independent of the
    // scheduler's request grouping while preserving the measured adaptive
    // 16-row path as the default for production throughput runs.
    tiling->tileM = kUseTscmPipeline
                        ? kTinyTileM
                        : (forcedMTile != 0
                               ? forcedMTile
                               : (stableMTile
                                      ? (tiling->m <= kSmallTileM ? kSmallTileM
                                                                 : kLargeTileM)
                                      : (tiling->m <= kTinyTileM
                                             ? kTinyTileM
                                             : (tiling->m <= kSmallTileM
                                                    ? kSmallTileM
                                                    : kLargeTileM))));
    // Wide-N merges two output slices into one online-decode/staging unit,
    // while the kernel still runs and writes each 128-column Cube slice
    // explicitly. Restrict it to at most two external K tiles so both decoded
    // planes remain resident in the two staging slots.
    const uint32_t narrowNTileCount =
        (tiling->n + kTileN - 1U) / kTileN;
    // Only merge columns when the narrow schedule would need more than one
    // AIC round. If the 128-column schedule already fits in one round, wide-N
    // merely reduces active blocks and cannot remove a synchronization round.
    const bool wideReducesRounds = narrowNTileCount > tiling->aicNum;
    const bool useWideNTile =
        kEnableWideNTile && !kUseTscmPipeline && tiling->m >= kWideMinM &&
        tiling->n >= kWideTileN && tiling->k <= 2U * kTileK &&
        wideReducesRounds;
    tiling->tileN = useWideNTile ? kWideTileN : kTileN;
    tiling->tileK = kUseTscmPipeline ? kTscmTileK : kTileK;
    tiling->mTileCount =
        (tiling->m + tiling->tileM - 1U) / tiling->tileM;
    tiling->nTileCount =
        (tiling->n + tiling->tileN - 1U) / tiling->tileN;
    tiling->kTileCount =
        (tiling->k + tiling->tileK - 1U) / tiling->tileK;

    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
    if (l1Size > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        l0cSize > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        ubSize > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return ge::GRAPH_FAILED;
    }

    matmul_tiling::MatmulApiTiling mmTiling;
#if HIF8_USE_TSCM_PIPELINE
    mmTiling.SetAType(matmul_tiling::TPosition::TSCM,
#else
    mmTiling.SetAType(matmul_tiling::TPosition::GM,
#endif
                      matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_BF16,
                      false);
#if HIF8_USE_TSCM_PIPELINE
    mmTiling.SetBType(matmul_tiling::TPosition::TSCM,
#else
    mmTiling.SetBType(matmul_tiling::TPosition::GM,
#endif
                      matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_BF16,
                      true);
    mmTiling.SetCType(matmul_tiling::TPosition::GM,
                      matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_FLOAT);
    tiling->outputStride = tiling->n;
    // Even in wide-N mode Matmul owns exactly one 128-column L0C result.
    // The kernel changes the staged B pointer and invokes it once per slice.
    mmTiling.SetOrgShape(tiling->tileM, kCubeSplitN,
                         tiling->tileK, tiling->tileK);
    mmTiling.SetShape(tiling->tileM, kCubeSplitN, tiling->tileK);
    mmTiling.SetBufferSpace(static_cast<int32_t>(l1Size),
                            static_cast<int32_t>(l0cSize));
    mmTiling.SetFixSplit(tiling->tileM, kCubeSplitN, kCubeSplitK);
    if (mmTiling.GetTiling(tiling->matmulTiling) == -1) {
        return ge::GRAPH_FAILED;
    }
    tiling->matmulTiling.shareL1Size = 0;
    const bool wideHasEnoughTilesForFullCores =
        useWideNTile &&
        tiling->nTileCount >= kWideMinTilesForFullCores &&
        tiling->nTileCount * 2U >= tiling->aicNum;
    tiling->aicNum = wideHasEnoughTilesForFullCores
                        ? CubeCoreNum(context)
                        : ActiveCubeCoreNum(tiling->nTileCount, tiling->aicNum,
                                            tiling->m <= 2);
    if (tiling->aicNum == 0) {
        return ge::GRAPH_FAILED;
    }
    tiling->outputStride = tiling->n;

    const uint64_t outputElements =
        static_cast<uint64_t>(tiling->m) * tiling->n;
    if (outputElements >
        std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        return ge::GRAPH_FAILED;
    }
    tiling->outputWorkspaceOffset = 0;
    const uint64_t outputWorkspaceSize = outputElements * sizeof(float);
    const uint64_t activationCacheTileElements =
        static_cast<uint64_t>(tiling->tileM) * tiling->tileK;
    const uint64_t activationCacheTiles =
        static_cast<uint64_t>(tiling->mTileCount) * tiling->kTileCount;
    if (activationCacheTileElements != 0 &&
        activationCacheTiles >
            std::numeric_limits<uint64_t>::max() /
                activationCacheTileElements) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t activationCacheElements =
        activationCacheTiles * activationCacheTileElements;
    if (activationCacheElements >
        std::numeric_limits<uint64_t>::max() / kCubeInputBytes) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t activationCacheBytes =
        activationCacheElements * kCubeInputBytes;
    const bool activationCacheFits =
        activationCacheBytes != 0 &&
        activationCacheBytes <= kActivationCacheMaxBytes;
    // All active AIV tasks build the cache in parallel. It pays off once the
    // activation has enough rows to occupy the vector pipeline and would
    // otherwise be decoded independently by several N tiles. Keep M=1/2
    // generation shapes on the overlapped per-core staging path. The
    // environment flag remains an explicit force-on/force-off rollback.
    const bool autoActivationCache =
        !kUseTscmPipeline && tiling->m >= kActivationCacheMinM &&
        tiling->nTileCount >= kActivationCacheMinNTiles;
    tiling->useActivationCache =
        activationCacheFits &&
                EnvFlagEnabled("HIF8_ENABLE_ACTIVATION_CACHE",
                               autoActivationCache)
            ? 1
            : 0;
    // Multi-block DataCopyPad removes four scalar DMA submissions per decode
    // chunk.  It is safe only when every K tile is complete; tail shapes keep
    // the proven row-by-row path and explicit padding semantics.  Setting the
    // flag to 0 provides a runtime rollback without rebuilding the OPP.
    tiling->useMultiLineDma =
        EnvFlagEnabled("HIF8_SOFT_MULTI_DMA", true) ? 1 : 0;
    tiling->activationCacheOffset =
        AlignUp(outputWorkspaceSize, kWorkspaceAlignment) +
        kStagingGuardBytes;
    const uint64_t activationCacheRegion =
        tiling->useActivationCache != 0 ? activationCacheBytes : 0;
    const uint64_t weightStageBytes =
        static_cast<uint64_t>(tiling->tileN) * tiling->tileK *
        kCubeInputBytes;
    const uint64_t activationStageBytes =
        static_cast<uint64_t>(tiling->tileM) * tiling->tileK *
        kCubeInputBytes;
    tiling->stagingWorkspaceOffset =
        AlignUp(tiling->activationCacheOffset + activationCacheRegion,
                kWorkspaceAlignment) +
        kStagingGuardBytes;
    tiling->stagingWorkspaceStride =
        AlignUp(2ULL * weightStageBytes + 2ULL * activationStageBytes,
                kWorkspaceAlignment);
    const uint64_t stagingWorkspaceSize =
        static_cast<uint64_t>(tiling->aicNum) *
        tiling->stagingWorkspaceStride;
    if (tiling->stagingWorkspaceOffset >
            std::numeric_limits<uint64_t>::max() - stagingWorkspaceSize) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t userWorkspaceSize =
        tiling->stagingWorkspaceOffset + stagingWorkspaceSize;
    if (userWorkspaceSize >
            std::numeric_limits<uint64_t>::max() -
                kAscendCReservedWorkspaceSize ||
        userWorkspaceSize + kAscendCReservedWorkspaceSize >
            std::numeric_limits<size_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t workspaceSize =
        kAscendCReservedWorkspaceSize + userWorkspaceSize;

    context->SetBlockDim(tiling->aicNum);
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = static_cast<size_t>(workspaceSize);
    if (std::getenv("HIF8_DUMP_TILING") != nullptr) {
        const auto &mm = tiling->matmulTiling;
        std::fprintf(
            stderr,
            "SoftHif8Gemm TCubeTiling: M=%d N=%d Ka=%d Kb=%d "
            "singleCoreM=%d singleCoreN=%d singleCoreK=%d "
            "baseM=%d baseN=%d baseK=%d depthA1=%d depthB1=%d "
            "stepM=%d stepN=%d stepKa=%d stepKb=%d "
            "shareL1Size=%d dbL0A=%d dbL0B=%d dbL0C=%d\n",
            mm.M, mm.N, mm.Ka, mm.Kb, mm.singleCoreM, mm.singleCoreN,
            mm.singleCoreK, mm.baseM, mm.baseN, mm.baseK, mm.depthA1,
            mm.depthB1, mm.stepM, mm.stepN, mm.stepKa, mm.stepKb,
            mm.shareL1Size, mm.dbL0A, mm.dbL0B, mm.dbL0C);
        std::fprintf(
            stderr,
            "SoftHif8Gemm choices: tileM=%u tileN=%u tileK=%u "
            "mTiles=%u nTiles=%u kTiles=%u aic=%u cache=%u "
            "cacheBytes=%llu multiDma=%u "
            "workspaceBytes=%llu\n",
            tiling->tileM, tiling->tileN, tiling->tileK,
            tiling->mTileCount, tiling->nTileCount, tiling->kTileCount,
            tiling->aicNum, tiling->useActivationCache,
            static_cast<unsigned long long>(activationCacheRegion),
            tiling->useMultiLineDma,
            static_cast<unsigned long long>(workspaceSize));
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace

namespace ops {

class SoftHif8Gemm : public OpDef {
public:
    explicit SoftHif8Gemm(const char *name) : OpDef(name) {
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

OP_ADD(SoftHif8Gemm);

class SoftHif8PrefillGemm : public SoftHif8Gemm {
public:
    explicit SoftHif8PrefillGemm(const char *name) : SoftHif8Gemm(name) {}
};

OP_ADD(SoftHif8PrefillGemm);

}  // namespace ops
