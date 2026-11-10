#include "hif8_codec.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "soft_hif8_gemm_tiling.h"

namespace vllm_ascend_hif8 {

#ifndef HIF8_DECODE_PROBE_STAGE
#define HIF8_DECODE_PROBE_STAGE 0
#endif
#ifndef HIF8_ROWS_PER_CHUNK
#define HIF8_ROWS_PER_CHUNK 4
#endif
#ifndef HIF8_USE_INT_OFFSET_SHIFT
#define HIF8_USE_INT_OFFSET_SHIFT 0
#endif
#ifndef HIF8_DECODE_IMPL
#define HIF8_DECODE_IMPL 0
#endif
#ifndef HIF8_DECODE_QUEUE_DEPTH
#define HIF8_DECODE_QUEUE_DEPTH 2
#endif
#ifndef HIF8_REUSE_DECODE_OUTPUT_AS_OFFSET
#define HIF8_REUSE_DECODE_OUTPUT_AS_OFFSET 0
#endif
#ifndef HIF8_USE_TSCM_PIPELINE
#define HIF8_USE_TSCM_PIPELINE 0
#endif
#ifndef HIF8_LOCAL_TILE_SYNC
#define HIF8_LOCAL_TILE_SYNC 0
#endif
#ifndef HIF8_SKIP_FINAL_TILE_BARRIER
#define HIF8_SKIP_FINAL_TILE_BARRIER 0
#endif
#ifndef HIF8_PAIRWISE_EPILOGUE_SYNC
#define HIF8_PAIRWISE_EPILOGUE_SYNC 0
#endif
#ifndef HIF8_BATCHED_EPILOGUE
#define HIF8_BATCHED_EPILOGUE 0
#endif
#ifndef HIF8_REUSE_WEIGHT_TILES
#define HIF8_REUSE_WEIGHT_TILES 1
#endif
#ifndef HIF8_WIDE_N_TILE
#define HIF8_WIDE_N_TILE 0
#endif
#ifndef HIF8_GEMM_KERNEL_CLASS
#define HIF8_GEMM_KERNEL_CLASS SoftHif8GemmKernel
#endif
#ifndef HIF8_GEMM_KERNEL_ENTRY
#define HIF8_GEMM_KERNEL_ENTRY soft_hif8_gemm
#endif

static_assert(HIF8_DECODE_PROBE_STAGE >= 0 &&
                  HIF8_DECODE_PROBE_STAGE <= 6,
              "HIF8_DECODE_PROBE_STAGE must be in [0, 6]");
static_assert(HIF8_ROWS_PER_CHUNK == 4 || HIF8_ROWS_PER_CHUNK == 8 ||
                  HIF8_ROWS_PER_CHUNK == 16,
              "HIF8_ROWS_PER_CHUNK must be 4, 8, or 16");
static_assert(HIF8_USE_INT_OFFSET_SHIFT == 0 ||
                  HIF8_USE_INT_OFFSET_SHIFT == 1,
              "HIF8_USE_INT_OFFSET_SHIFT must be 0 or 1");
static_assert(HIF8_DECODE_IMPL == 0 || HIF8_DECODE_IMPL == 1,
              "HIF8_DECODE_IMPL must be 0 or 1");
static_assert(HIF8_DECODE_QUEUE_DEPTH == 2 || HIF8_DECODE_QUEUE_DEPTH == 3 ||
                  HIF8_DECODE_QUEUE_DEPTH == 4,
              "HIF8_DECODE_QUEUE_DEPTH must be 2, 3, or 4");
static_assert(HIF8_REUSE_DECODE_OUTPUT_AS_OFFSET == 0 ||
                  HIF8_REUSE_DECODE_OUTPUT_AS_OFFSET == 1,
              "HIF8_REUSE_DECODE_OUTPUT_AS_OFFSET must be 0 or 1");
static_assert(HIF8_USE_TSCM_PIPELINE == 0 || HIF8_USE_TSCM_PIPELINE == 1,
              "HIF8_USE_TSCM_PIPELINE must be 0 or 1");
static_assert(HIF8_LOCAL_TILE_SYNC == 0 || HIF8_LOCAL_TILE_SYNC == 1,
              "HIF8_LOCAL_TILE_SYNC must be 0 or 1");
static_assert(HIF8_SKIP_FINAL_TILE_BARRIER == 0 ||
                  HIF8_SKIP_FINAL_TILE_BARRIER == 1,
              "HIF8_SKIP_FINAL_TILE_BARRIER must be 0 or 1");
static_assert(HIF8_PAIRWISE_EPILOGUE_SYNC == 0 ||
                  HIF8_PAIRWISE_EPILOGUE_SYNC == 1,
              "HIF8_PAIRWISE_EPILOGUE_SYNC must be 0 or 1");
static_assert(HIF8_BATCHED_EPILOGUE == 0 || HIF8_BATCHED_EPILOGUE == 1,
              "HIF8_BATCHED_EPILOGUE must be 0 or 1");
static_assert(HIF8_REUSE_WEIGHT_TILES == 0 || HIF8_REUSE_WEIGHT_TILES == 1,
              "HIF8_REUSE_WEIGHT_TILES must be 0 or 1");
static_assert(HIF8_WIDE_N_TILE == 0 || HIF8_WIDE_N_TILE == 1,
              "HIF8_WIDE_N_TILE must be 0 or 1");

constexpr uint32_t kRowsPerChunk = HIF8_ROWS_PER_CHUNK;
constexpr uint32_t kDecodeQueueDepth = HIF8_DECODE_QUEUE_DEPTH;
constexpr uint32_t kCubeKGroup = 16;
constexpr uint32_t kDmaBlockBytes = 32;
constexpr uint32_t kWeightBufferCount = 2;
constexpr uint32_t kActivationBufferCount = 2;
constexpr uint32_t kVectorSubBlockCount = 2;
constexpr uint32_t kCubeSplitN = 128;
constexpr uint32_t kEpilogueRowsPerBatch = 8;
constexpr uint8_t kAivAicSyncMode = 2;
constexpr uint16_t kWeightReadyEvent = 1;
constexpr uint16_t kWeightFreeEvent = 3;
constexpr uint16_t kOutputReadyEvent = 5;
constexpr uint16_t kActivationReadyEvent = 7;
constexpr uint16_t kActivationFreeEvent = 9;
constexpr uint16_t kSubBlockEventOffset = 16;
constexpr uint32_t kDecodeProbeStage = HIF8_DECODE_PROBE_STAGE;
constexpr bool kProbeDecodePipeline = kDecodeProbeStage != 0;
constexpr bool kProbeDirectOutput = kProbeDecodePipeline;
constexpr bool kProbeStagedActivation = false;
constexpr bool kProbeStagedWeight = kProbeDecodePipeline;
constexpr bool kProbeMatmulOutput = false;
// Select GM staging or TSCM at compile time to eliminate the unused path.
constexpr bool kUseTscmPipeline = HIF8_USE_TSCM_PIPELINE != 0;
using CubeInputType = bfloat16_t;
using CubeOutputType = float;
// The BF16-LUT Gather no longer needs its FP16 offsets once the uint32 byte
// offsets have been formed. On the GM-staging path, its output queue slot can
// safely provide that temporary storage until Gather overwrites it with the
// decoded BF16 values. Keep the FP16-LUT and TSCM paths separate because they
// still consume an independent FP16 scratch after Gather starts.
constexpr bool kReuseDecodeOutputAsOffset =
    HIF8_REUSE_DECODE_OUTPUT_AS_OFFSET != 0 && HIF8_DECODE_IMPL == 0 &&
    !kUseTscmPipeline;
// GM staging slots already carry paired AIV/AIC ready/free events.  The
// local-sync mode relies on those per-core events between M
// tiles and retains only the invocation-wide barriers around the pipeline.
constexpr bool kUseLocalTileSync = HIF8_LOCAL_TILE_SYNC != 0;
// FinishGmStagingPath has an invocation-wide barrier after mm_.End(). The
// final per-tile barrier immediately before it cannot protect a subsequent
// staging round, so it can be removed without changing the event protocol.
constexpr bool kSkipFinalTileBarrier =
    HIF8_SKIP_FINAL_TILE_BARRIER != 0;
// Each output tile is owned by one AIC and consumed only by its two paired
// AIV tasks. A mode-2 event can therefore publish completed FIX writes to the
// local vector epilogue without making unrelated core groups rendezvous.
constexpr bool kUsePairwiseEpilogueSync =
    HIF8_PAIRWISE_EPILOGUE_SYNC != 0;
constexpr bool kUseBatchedEpilogue = HIF8_BATCHED_EPILOGUE != 0;
// In the GM-staging path, the end-of-M-tile barrier means both
// Cube sub-blocks have finished consuming the two weight slots. Reusing those
// immutable slots across M tiles removes repeated U8->BF16 weight decode while
// retaining the event protocol. Probe and local-sync builds decode per tile
// because their synchronization contracts differ.
constexpr bool kReuseWeightTiles =
    HIF8_REUSE_WEIGHT_TILES != 0 && !kUseLocalTileSync &&
    !kProbeDecodePipeline && !kProbeStagedActivation &&
    !kProbeStagedWeight && !kProbeMatmulOutput;
constexpr bool kEnableWideNTile = HIF8_WIDE_N_TILE != 0;

constexpr MatmulConfig kMatmulConfig =
    GetNormalConfig(true, false, false, BatchMode::BATCH_LESS_THAN_L1,
                    true, IterateOrder::UNDEF,
                    ScheduleType::INNER_PRODUCT, false);

__aicore__ inline float DecodeGemmBias(uint16_t payload, uint32_t biasDtype) {
    if (biasDtype == 1) {
        const bfloat16_t value =
            AscendC::GetScalarBitcodeValue<uint16_t, bfloat16_t>(payload);
        return DeviceToFloat(value);
    }
    const half value = AscendC::GetScalarBitcodeValue<uint16_t, half>(payload);
    return static_cast<float>(value);
}

class HIF8_GEMM_KERNEL_CLASS {
public:
#if HIF8_USE_TSCM_PIPELINE
    using AType = matmul::MatmulType<AscendC::TPosition::TSCM,
                                     CubeFormat::ND,
                                     CubeInputType, false,
                                     LayoutMode::NONE, false>;
    using BType = matmul::MatmulType<AscendC::TPosition::TSCM,
                                     CubeFormat::ND,
                                     CubeInputType, true,
                                     LayoutMode::NONE, false>;
#else
    using AType = matmul::MatmulType<AscendC::TPosition::GM,
                                     CubeFormat::ND,
                                     CubeInputType, false,
                                     LayoutMode::NONE, false>;
    using BType = matmul::MatmulType<AscendC::TPosition::GM,
                                     CubeFormat::ND,
                                     CubeInputType, true,
                                     LayoutMode::NONE, false>;
#endif
    using CType = matmul::MatmulType<AscendC::TPosition::GM,
                                     CubeFormat::ND, CubeOutputType>;
    using BiasType = matmul::MatmulType<AscendC::TPosition::GM,
                                        CubeFormat::ND, CubeOutputType>;

    __aicore__ inline void Init(GM_ADDR activation, GM_ADDR weight,
                                GM_ADDR tokenScale, GM_ADDR channelScale,
                                GM_ADDR bias, GM_ADDR output, GM_ADDR workspace,
                                const SoftHif8GemmTilingData &tiling) {
        tiling_ = tiling;
        activationGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t *>(activation));
        weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(weight));
        tokenScaleGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(tokenScale));
        channelScaleGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(channelScale));
        if (tiling_.hasBias != 0) {
            biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(bias));
        }
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(output));

        __gm__ uint8_t *workspaceBytes =
            reinterpret_cast<__gm__ uint8_t *>(workspace);
        matmulOutputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ CubeOutputType *>(
            workspaceBytes + tiling_.outputWorkspaceOffset));
        const uint64_t coreStageOffset =
            tiling_.stagingWorkspaceOffset +
            static_cast<uint64_t>(BlockIndex()) *
                tiling_.stagingWorkspaceStride;
        const uint64_t weightStageElements =
            static_cast<uint64_t>(tiling_.tileN) * tiling_.tileK;
        const uint64_t activationStageElements =
            static_cast<uint64_t>(tiling_.tileM) * tiling_.tileK;
        activationStageGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ CubeInputType *>(
                workspaceBytes + coreStageOffset));
        activationStageGm1_.SetGlobalBuffer(
            reinterpret_cast<__gm__ CubeInputType *>(
                workspaceBytes + coreStageOffset +
                activationStageElements * sizeof(CubeInputType)));
        weightStageGm0_.SetGlobalBuffer(
            reinterpret_cast<__gm__ CubeInputType *>(
                workspaceBytes + coreStageOffset +
                2ULL * activationStageElements * sizeof(CubeInputType)));
        weightStageGm1_.SetGlobalBuffer(
            reinterpret_cast<__gm__ CubeInputType *>(
                workspaceBytes + coreStageOffset +
                (2ULL * activationStageElements + weightStageElements) *
                    sizeof(CubeInputType)));
        if (tiling_.useActivationCache != 0) {
            activationCacheGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ CubeInputType *>(
                    workspaceBytes + tiling_.activationCacheOffset));
        }

        if ASCEND_IS_AIV {
            const uint32_t decodeChunk = kRowsPerChunk * tiling_.tileK;
            pipe_.InitBuffer(rawInputQueue_, kDecodeQueueDepth,
                             decodeChunk * sizeof(uint8_t) + kDmaBlockBytes);
            pipe_.InitBuffer(decodedOutputQueue_, kDecodeQueueDepth,
                             decodeChunk * sizeof(CubeInputType));
            // These buffers are reused by the vector epilogue after staging.
            // The hot decode path uses the two queues above.
            uint32_t rawBufferBytes =
                tiling_.tileN * sizeof(DTYPE_Y) + kDmaBlockBytes;
            uint32_t halfIndexBytes = decodeChunk * sizeof(half);
            uint32_t indexBufferBytes = decodeChunk * sizeof(uint32_t);
            pipe_.InitBuffer(rawBuffer_, rawBufferBytes);
            pipe_.InitBuffer(decodedBuffer_, tiling_.tileN * sizeof(float));
            if constexpr (kReuseDecodeOutputAsOffset) {
                halfIndexBytes =
                    kEpilogueRowsPerBatch * tiling_.tileN * sizeof(float);
            }
            pipe_.InitBuffer(halfIndexBuffer_, halfIndexBytes);
            pipe_.InitBuffer(indexBuffer_, indexBufferBytes);
            pipe_.InitBuffer(bf16Buffer_, tiling_.tileN * sizeof(float));
            pipe_.InitBuffer(decodeTableBuffer_,
                             256 * sizeof(bfloat16_t));
            if constexpr (kUseTscmPipeline) {
                InitSharedTscm();
            }
        }
        if ASCEND_IS_AIC {
            if constexpr (kUseTscmPipeline) {
                InitSharedTscm();
            }
            // The Matmul interface uses the current sub-block selection when
            // it lays out the Cube input/output state.  Keep it explicit for
            // the TSCM/NZ path; omitting it makes the result depend on the
            // runtime's default sub-block and, on some CANN revisions, drops
            // subsequent K planes.
            mm_.SetSubBlockIdx(0);
            mm_.Init(&tiling_.matmulTiling, &pipe_);
        }
    }

    __aicore__ inline void Process() {
        if ASCEND_IS_AIV {
            // Every LUT variant initializes the table once per AIV task. Keep
            // that fixed work in every cumulative probe stage as well, so the
            // difference between stages 4 and 5 measures Gather itself rather
            // than Gather plus 128 scalar table stores.
            InitDecodeTable();
            if (tiling_.useActivationCache != 0) {
                ProduceActivationCache();
            }
        }
        if constexpr (kUseTscmPipeline) {
            if ASCEND_IS_AIV {
                ProduceTiles();
            }
            if ASCEND_IS_AIC {
                RunTiledMatmul();
            }
            AscendC::SyncAll<false>();
            FreeSharedTscm();
            AscendC::SyncAll<false>();
            if ASCEND_IS_AIV {
                RunVectorEpilogue();
            }
            return;
        }
        AscendC::SyncAll<false>();

        if constexpr (kEnableWideNTile) {
            if (tiling_.tileN > kCubeSplitN) {
                RunWideNTiledMatmul();
                FinishGmStagingPath();
                return;
            }
        }

        // The two AIV sub-blocks jointly populate one per-AIC GM stage.  Mode
        // 2 events provide the paired AIV<->AIC rendezvous without forcing
        // unrelated cores through a device-wide barrier for every K tile.
        // Alternating the two existing stages lets vector decode of tile t+1
        // overlap Cube consumption of tile t.
        uint32_t tileSequence = 0;
        // The two staging slots can retain a weight tile across M tiles only
        // when the complete K reduction fits in those slots. For a third K
        // tile, slot 0 would be overwritten before the next M tile reuses it;
        // decode per tile for those shapes.
        const bool reuseWeightTiles =
            kReuseWeightTiles && tiling_.kTileCount <= kWeightBufferCount;
        const uint32_t nTileRounds =
            tiling_.nTileCount == 0
                ? 0
                : (tiling_.nTileCount - 1U) / tiling_.aicNum + 1U;
        for (uint32_t nRound = 0; nRound < nTileRounds; ++nRound) {
            const uint32_t nTile =
                BlockIndex() + nRound * tiling_.aicNum;
            const bool hasNTile = nTile < tiling_.nTileCount;
            const uint32_t nBegin = nTile * tiling_.tileN;
            for (uint32_t mTile = 0; mTile < tiling_.mTileCount; ++mTile) {
                const uint32_t mBegin = mTile * tiling_.tileM;
                if (hasNTile) {
                  for (uint32_t kTile = 0; kTile < tiling_.kTileCount;
                       ++kTile) {
                    const uint32_t kBegin = kTile * tiling_.tileK;
                    const uint32_t slot =
                        reuseWeightTiles
                            ? kTile % kActivationBufferCount
                            : tileSequence % kActivationBufferCount;
                    if ASCEND_IS_AIV {
                        const bool slotMayBeBusy =
                            reuseWeightTiles
                                ? (nRound != 0 || mTile != 0 ||
                                   kTile >= kActivationBufferCount)
                                : tileSequence >= kActivationBufferCount;
                        if (slotMayBeBusy) {
                            AscendC::CrossCoreWaitFlag<kAivAicSyncMode,
                                                       PIPE_MTE3>(
                                ActivationFreeEvent(0, slot));
                        }
                        if constexpr (kProbeStagedActivation) {
                            StageActivationTile(mBegin, kBegin, slot);
                        } else if constexpr (kProbeStagedWeight) {
                            StageWeightTile(nBegin, kBegin, slot);
                        } else if (tiling_.useActivationCache != 0) {
                            if (reuseWeightTiles) {
                                if (mTile == 0) {
                                    StageWeightTile(nBegin, kBegin, slot);
                                }
                            } else {
                                StageWeightTile(nBegin, kBegin, slot);
                            }
                        } else {
                            if (reuseWeightTiles) {
                                if (mTile == 0) {
                                    StageWeightTile(nBegin, kBegin, slot);
                                }
                            } else {
                                StageWeightTile(nBegin, kBegin, slot);
                            }
                            StageActivationTile(mBegin, kBegin, slot);
                        }
                        AscendC::CrossCoreSetFlag<kAivAicSyncMode,
                                                  PIPE_MTE3>(
                            ActivationReadyEvent(0, slot));
                    }
                    if ASCEND_IS_AIC {
                        AscendC::CrossCoreWaitFlag<kAivAicSyncMode,
                                                   PIPE_MTE1>(
                            ActivationReadyEvent(0, slot));
                        if constexpr (!kProbeStagedActivation &&
                                      !kProbeStagedWeight) {
                            AccumulateTile(mBegin, nBegin, kBegin,
                                           kTile != 0, slot, 0,
                                           kCubeSplitN, true);
                        } else {
                            AscendC::CrossCoreSetFlag<kAivAicSyncMode,
                                                      PIPE_MTE1>(
                                ActivationFreeEvent(0, slot));
                        }
                    }
                    ++tileSequence;
                  }
                }
                if (hasNTile) {
                    if ASCEND_IS_AIC {
                        if constexpr (!kProbeStagedActivation &&
                                      !kProbeStagedWeight) {
                            const uint64_t outputOffset =
                                static_cast<uint64_t>(mBegin) *
                                    tiling_.outputStride +
                                nBegin;
                            mm_.GetTensorC(matmulOutputGm_[outputOffset]);
                        }
                    }
                }
                if constexpr (!kUseLocalTileSync) {
                    const bool hasFollowingTile =
                        mTile + 1U < tiling_.mTileCount ||
                        nRound + 1U < nTileRounds;
                    if (!kSkipFinalTileBarrier || hasFollowingTile) {
                        AscendC::SyncAll<false>();
                    }
                }
            }
        }

        FinishGmStagingPath();
    }

private:
    __aicore__ inline void FinishGmStagingPath() {
        if ASCEND_IS_AIC {
            if constexpr (!kProbeStagedActivation && !kProbeStagedWeight) {
                mm_.End();
            }
            if constexpr (kUsePairwiseEpilogueSync) {
                if constexpr (!kProbeStagedActivation &&
                              !kProbeStagedWeight) {
                    AscendC::CrossCoreSetFlag<kAivAicSyncMode, PIPE_FIX>(
                        OutputReadyEvent(0));
                } else {
                    AscendC::CrossCoreSetFlag<kAivAicSyncMode, PIPE_MTE1>(
                        OutputReadyEvent(0));
                }
            }
        }
        if constexpr (kUsePairwiseEpilogueSync) {
            if ASCEND_IS_AIV {
                AscendC::CrossCoreWaitFlag<kAivAicSyncMode, PIPE_MTE2>(
                    OutputReadyEvent(0));
            }
        } else {
            AscendC::SyncAll<false>();
        }
        if ASCEND_IS_AIV {
            if constexpr (kProbeDecodePipeline) {
                ProbeDecodeOutput();
            } else if constexpr (kProbeDirectOutput) {
                ProbeDirectOutput();
            } else if constexpr (kProbeStagedActivation) {
                ProbeStagedActivation();
            } else if constexpr (kProbeStagedWeight) {
                ProbeStagedWeight();
            } else if constexpr (kProbeMatmulOutput) {
                ProbeMatmulOutput();
            } else {
                RunVectorEpilogue();
            }
        }
    }

    __aicore__ inline void RunWideNTiledMatmul() {
        uint32_t tileSequence = 0;
        const bool reuseWeightTiles =
            kReuseWeightTiles && tiling_.kTileCount <= kWeightBufferCount;
        const uint32_t nTileRounds =
            tiling_.nTileCount == 0
                ? 0
                : (tiling_.nTileCount - 1U) / tiling_.aicNum + 1U;

        for (uint32_t nRound = 0; nRound < nTileRounds; ++nRound) {
            const uint32_t nTile =
                BlockIndex() + nRound * tiling_.aicNum;
            const bool hasNTile = nTile < tiling_.nTileCount;
            const uint32_t nBegin = nTile * tiling_.tileN;
            for (uint32_t mTile = 0; mTile < tiling_.mTileCount; ++mTile) {
                const uint32_t mBegin = mTile * tiling_.tileM;
                const uint32_t tileSequenceBegin = tileSequence;
                const uint32_t tileN =
                    hasNTile
                        ? (tiling_.n - nBegin < tiling_.tileN
                               ? tiling_.n - nBegin
                               : tiling_.tileN)
                        : 0;
                const bool hasSecondSlice = tileN > kCubeSplitN;

                if (hasNTile) {
                    // Decode each K plane once at the full logical width. The
                    // first Cube slice consumes planes as they become ready;
                    // both decoded slots stay resident for the second slice.
                    for (uint32_t kTile = 0; kTile < tiling_.kTileCount;
                         ++kTile) {
                        const uint32_t kBegin = kTile * tiling_.tileK;
                        const uint32_t slot =
                            reuseWeightTiles
                                ? kTile % kActivationBufferCount
                                : (tileSequenceBegin + kTile) %
                                      kActivationBufferCount;
                        if ASCEND_IS_AIV {
                            const bool slotMayBeBusy =
                                reuseWeightTiles
                                    ? (nRound != 0 || mTile != 0 ||
                                       kTile >= kActivationBufferCount)
                                    : tileSequenceBegin + kTile >=
                                          kActivationBufferCount;
                            if (slotMayBeBusy) {
                                AscendC::CrossCoreWaitFlag<kAivAicSyncMode,
                                                           PIPE_MTE3>(
                                    ActivationFreeEvent(0, slot));
                            }
                            if constexpr (kProbeStagedActivation) {
                                StageActivationTile(mBegin, kBegin, slot);
                            } else if constexpr (kProbeStagedWeight) {
                                StageWeightTile(nBegin, kBegin, slot);
                            } else if (tiling_.useActivationCache != 0) {
                                if (!reuseWeightTiles || mTile == 0) {
                                    StageWeightTile(nBegin, kBegin, slot);
                                }
                            } else {
                                if (!reuseWeightTiles || mTile == 0) {
                                    StageWeightTile(nBegin, kBegin, slot);
                                }
                                StageActivationTile(mBegin, kBegin, slot);
                            }
                            AscendC::CrossCoreSetFlag<kAivAicSyncMode,
                                                      PIPE_MTE3>(
                                ActivationReadyEvent(0, slot));
                        }
                        if ASCEND_IS_AIC {
                            AscendC::CrossCoreWaitFlag<kAivAicSyncMode,
                                                       PIPE_MTE1>(
                                ActivationReadyEvent(0, slot));
                            if constexpr (!kProbeStagedActivation &&
                                          !kProbeStagedWeight) {
                                AccumulateTile(
                                    mBegin, nBegin, kBegin, kTile != 0, slot,
                                    0, kCubeSplitN, !hasSecondSlice);
                            } else {
                                AscendC::CrossCoreSetFlag<kAivAicSyncMode,
                                                          PIPE_MTE1>(
                                    ActivationFreeEvent(0, slot));
                            }
                        }
                    }
                    tileSequence += tiling_.kTileCount;
                }

                if (hasNTile) {
                    if ASCEND_IS_AIC {
                        if constexpr (!kProbeStagedActivation &&
                                      !kProbeStagedWeight) {
                            const uint64_t firstOutputOffset =
                                static_cast<uint64_t>(mBegin) *
                                    tiling_.outputStride +
                                nBegin;
                            mm_.GetTensorC(
                                matmulOutputGm_[firstOutputOffset]);

                            if (hasSecondSlice) {
                                const uint32_t secondN =
                                    tileN - kCubeSplitN;
                                for (uint32_t kTile = 0;
                                     kTile < tiling_.kTileCount; ++kTile) {
                                    const uint32_t slot =
                                        reuseWeightTiles
                                            ? kTile % kActivationBufferCount
                                            : (tileSequenceBegin + kTile) %
                                                  kActivationBufferCount;
                                    AccumulateTile(
                                        mBegin, nBegin + kCubeSplitN,
                                        kTile * tiling_.tileK, kTile != 0,
                                        slot, kCubeSplitN, secondN, false);
                                }
                                const uint64_t secondOutputOffset =
                                    firstOutputOffset + kCubeSplitN;
                                mm_.GetTensorC(
                                    matmulOutputGm_[secondOutputOffset]);
                                for (uint32_t kTile = 0;
                                     kTile < tiling_.kTileCount; ++kTile) {
                                    const uint32_t slot =
                                        reuseWeightTiles
                                            ? kTile % kActivationBufferCount
                                            : (tileSequenceBegin + kTile) %
                                                  kActivationBufferCount;
                                    AscendC::CrossCoreSetFlag<
                                        kAivAicSyncMode, PIPE_MTE1>(
                                        ActivationFreeEvent(0, slot));
                                }
                            }
                        }
                    }
                }
                if constexpr (!kUseLocalTileSync) {
                    const bool hasFollowingTile =
                        mTile + 1U < tiling_.mTileCount ||
                        nRound + 1U < nTileRounds;
                    if (!kSkipFinalTileBarrier || hasFollowingTile) {
                        AscendC::SyncAll<false>();
                    }
                }
            }
        }
    }

    __aicore__ inline void InitSharedTscm() {
        const uint32_t activationBytes =
            tiling_.tileM * tiling_.tileK * sizeof(CubeInputType);
        const uint32_t weightBytes =
            tiling_.tileN * tiling_.tileK * sizeof(CubeInputType);
        // scmBlockGroup (mask bit 2) gives the paired AIV/AIC tasks the same
        // physical L1 address. Queue metadata is task-local, so every task
        // completes its own Alloc/EnQue/DeQue lifecycle and cross-core events
        // exclusively govern when the shared payload is ready or reusable.
        pipe_.InitBuffer(sharedL1Buffer_, 1,
                         activationBytes + weightBytes);
        sharedL1_ = sharedL1Buffer_.AllocTensor<CubeInputType>();
        sharedL1Buffer_.EnQue(sharedL1_);
        sharedL1_ = sharedL1Buffer_.DeQue<CubeInputType>();
        activationL1_ = sharedL1_;
        weightL1_ = sharedL1_[static_cast<uint64_t>(tiling_.tileM) *
                              tiling_.tileK];
    }

    __aicore__ inline void FreeSharedTscm() {
        sharedL1Buffer_.FreeTensor(sharedL1_);
    }

    __aicore__ inline uint16_t SubBlockEvent(uint16_t base,
                                             uint32_t subBlock,
                                             uint32_t slot = 0) const {
        return base + static_cast<uint16_t>(slot) +
               static_cast<uint16_t>(subBlock) * kSubBlockEventOffset;
    }

    __aicore__ inline uint16_t WeightReadyEvent(uint32_t subBlock,
                                                uint32_t slot) const {
        return SubBlockEvent(kWeightReadyEvent, subBlock, slot);
    }

    __aicore__ inline uint16_t WeightFreeEvent(uint32_t subBlock,
                                               uint32_t slot) const {
        return SubBlockEvent(kWeightFreeEvent, subBlock, slot);
    }

    __aicore__ inline uint16_t OutputReadyEvent(uint32_t subBlock) const {
        return SubBlockEvent(kOutputReadyEvent, subBlock);
    }

    __aicore__ inline uint16_t ActivationReadyEvent(
        uint32_t subBlock, uint32_t slot) const {
        return SubBlockEvent(kActivationReadyEvent, subBlock, slot);
    }

    __aicore__ inline uint16_t ActivationFreeEvent(
        uint32_t subBlock, uint32_t slot) const {
        return SubBlockEvent(kActivationFreeEvent, subBlock, slot);
    }

    __aicore__ inline uint32_t BlockIndex() const {
        return AscendC::GetBlockIdx() / AscendC::GetTaskRation();
    }

    __aicore__ inline bfloat16_t DecodeBf16(uint8_t payload) const {
        if (payload == kNan) {
            return AscendC::GetScalarBitcodeValue<uint16_t, bfloat16_t>(
                0x7fc0u);
        }
        return FloatToDevice<bfloat16_t>(Decode(payload));
    }

    __aicore__ inline void InitDecodeTable() {
        AscendC::LocalTensor<uint32_t> decodeTable =
            decodeTableBuffer_.Get<uint32_t>();
        for (uint32_t word = 0; word < 128; ++word) {
                decodeTable.SetValue(word, kAscendDecodeHalfU16Packed[word]);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    }


    __aicore__ inline void GatherDecode(
        const AscendC::LocalTensor<CubeInputType> &dst,
        const AscendC::LocalTensor<half> &scratch,
        AscendC::LocalTensor<uint32_t> &offsets, uint32_t count) {
            AscendC::LocalTensor<half> table =
                decodeTableBuffer_.Get<half>();
            AscendC::LocalTensor<float> floatScratch =
                offsets.ReinterpretCast<float>();
            // The offset scratch is no longer needed after the int32 offsets
            // have been built, so reuse it as the FP16 Gather destination and
            // then as the float conversion scratch. C220 has no direct
            // half->BF16 Cast, but it does support half->float->BF16.
            AscendC::Gather(scratch, table, offsets, 0, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(floatScratch, scratch,
                          AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(dst, floatScratch, AscendC::RoundMode::CAST_RINT,
                          count);
    }

    __aicore__ inline void ZeroCubeInput(
        const AscendC::LocalTensor<CubeInputType> &dst, uint32_t count) {
        AscendC::Duplicate(dst, FloatToDevice<CubeInputType>(0.0f), count);
    }

    __aicore__ inline void DecodeChunk(
        const AscendC::GlobalTensor<uint8_t> &src,
        AscendC::GlobalTensor<CubeInputType> &dst, uint64_t srcOffset,
        uint64_t dstOffset, uint32_t count) {
        AscendC::LocalTensor<uint8_t> raw = rawBuffer_.Get<uint8_t>();
        AscendC::LocalTensor<half> halfOffsets =
            halfIndexBuffer_.Get<half>();
        AscendC::LocalTensor<uint32_t> offsets =
            indexBuffer_.Get<uint32_t>();
        AscendC::LocalTensor<CubeInputType> decoded =
            decodedOutputQueue_.AllocTensor<CubeInputType>();

        AscendC::DataCopy(raw, src[srcOffset], count);
        AscendC::PipeBarrier<PIPE_MTE2>();
        AscendC::Cast(halfOffsets, raw, AscendC::RoundMode::CAST_NONE, count);
#if HIF8_USE_INT_OFFSET_SHIFT
        // Gather takes byte offsets.  Convert the exact byte code to int32
        // first and shift there, avoiding the half-precision multiply.
        AscendC::LocalTensor<int32_t> signedOffsets =
            offsets.ReinterpretCast<int32_t>();
        AscendC::Cast(signedOffsets, halfOffsets,
                      AscendC::RoundMode::CAST_RINT, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(signedOffsets, signedOffsets,
                           static_cast<int32_t>(1), count);
#else
        const half tableElementBytes = 2.0;
        AscendC::Muls(halfOffsets, halfOffsets, tableElementBytes, count);
        AscendC::LocalTensor<int32_t> signedOffsets =
            offsets.ReinterpretCast<int32_t>();
        AscendC::Cast(signedOffsets, halfOffsets,
                      AscendC::RoundMode::CAST_RINT, count);
#endif
        GatherDecode(decoded, halfOffsets, offsets, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(dst[dstOffset], decoded, count);
        AscendC::PipeBarrier<PIPE_MTE3>();
        decodedOutputQueue_.FreeTensor(decoded);
    }

    __aicore__ inline void DecodeActivationBlock(uint64_t srcOffset,
                                                  uint32_t rowCount,
                                                  uint32_t validColumns) {
        AscendC::LocalTensor<uint8_t> raw = rawBuffer_.Get<uint8_t>();
        AscendC::LocalTensor<half> halfOffsets =
            halfIndexBuffer_.Get<half>();
        AscendC::LocalTensor<uint32_t> offsets =
            indexBuffer_.Get<uint32_t>();
        AscendC::LocalTensor<CubeInputType> decoded =
            bf16Buffer_.Get<CubeInputType>();

        const uint32_t count = rowCount * tiling_.tileK;
        AscendC::LocalTensor<uint16_t> rawWords =
            raw.ReinterpretCast<uint16_t>();
        AscendC::Duplicate(rawWords, static_cast<uint16_t>(0),
                           (count + 1U) / 2U);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
        AscendC::DataCopyExtParams copyParams{1, validColumns, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint8_t> padParams{true, 0, 0, 0};
        for (uint32_t rowIndex = 0; rowIndex < rowCount; ++rowIndex) {
            AscendC::DataCopyPad(
                raw[static_cast<uint64_t>(rowIndex) * tiling_.tileK],
                activationGm_[srcOffset +
                              static_cast<uint64_t>(rowIndex) * tiling_.k],
                copyParams, padParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        AscendC::LocalTensor<int32_t> signedOffsets =
            offsets.ReinterpretCast<int32_t>();
        if (validColumns == tiling_.tileK) {
            // Full rows are contiguous in UB after the GM-to-UB copies above.
            // Decode the whole 16-row chunk with one vector instruction chain
            // instead of issuing five instructions per row.
            AscendC::Cast(halfOffsets, raw, AscendC::RoundMode::CAST_NONE,
                          count);
            AscendC::PipeBarrier<PIPE_V>();
#if HIF8_USE_INT_OFFSET_SHIFT
            AscendC::Cast(signedOffsets, halfOffsets,
                          AscendC::RoundMode::CAST_RINT, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftLeft(signedOffsets, signedOffsets,
                               static_cast<int32_t>(1), count);
#else
            const half tableElementBytes = 2.0;
            AscendC::Muls(halfOffsets, halfOffsets, tableElementBytes, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(signedOffsets, halfOffsets,
                          AscendC::RoundMode::CAST_RINT, count);
#endif
            AscendC::PipeBarrier<PIPE_V>();
            GatherDecode(decoded, halfOffsets, offsets, count);
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }

        // A tail K tile has padding between rows in UB. Decode its valid
        // columns row by row so stale padding bytes can never reach Matmul.
        for (uint32_t rowIndex = 0; rowIndex < rowCount; ++rowIndex) {
            const uint32_t rowOffset = rowIndex * tiling_.tileK;
            AscendC::Cast(halfOffsets, raw[rowOffset],
                          AscendC::RoundMode::CAST_NONE, validColumns);
            AscendC::PipeBarrier<PIPE_V>();
#if HIF8_USE_INT_OFFSET_SHIFT
            AscendC::Cast(signedOffsets, halfOffsets,
                          AscendC::RoundMode::CAST_RINT, validColumns);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftLeft(signedOffsets, signedOffsets,
                               static_cast<int32_t>(1), validColumns);
#else
            const half tableElementBytes = 2.0;
            AscendC::Muls(halfOffsets, halfOffsets, tableElementBytes,
                          validColumns);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(signedOffsets, halfOffsets,
                          AscendC::RoundMode::CAST_RINT, validColumns);
#endif
            AscendC::PipeBarrier<PIPE_V>();
            GatherDecode(decoded[rowOffset], halfOffsets, offsets,
                         validColumns);
            AscendC::PipeBarrier<PIPE_V>();
        }
        (void)count;
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyActivationBlockToNd(
        const AscendC::LocalTensor<CubeInputType> &src,
        AscendC::LocalTensor<CubeInputType> &dst, uint32_t localRow,
        uint32_t rowCount) {
        const uint64_t dstOffset =
            static_cast<uint64_t>(localRow) * tiling_.tileK;
        const uint32_t copyBytes =
            rowCount * tiling_.tileK * sizeof(CubeInputType);
        AscendC::DataCopyParams copyParams{
            1, static_cast<uint16_t>(copyBytes / kDmaBlockBytes), 0, 0};
        AscendC::DataCopy(dst[dstOffset], src, copyParams);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void ProduceActivationTile(
        uint32_t mBegin, uint32_t kBegin, uint32_t slot) {
        AscendC::LocalTensor<CubeInputType> decoded =
            bf16Buffer_.Get<CubeInputType>();
        AscendC::LocalTensor<CubeInputType> activationL1 = activationL1_;
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        const uint32_t rowsPerSubBlock = tiling_.tileM / 2U;
        const uint32_t localRowBegin = subBlock * rowsPerSubBlock;
        const uint32_t localRowEnd = localRowBegin + rowsPerSubBlock;
        const uint32_t validColumns =
            tiling_.k - kBegin < tiling_.tileK ? tiling_.k - kBegin
                                               : tiling_.tileK;
        (void)slot;

        for (uint32_t localRow = localRowBegin; localRow < localRowEnd;
             localRow += kRowsPerChunk) {
            const uint32_t remainingRows = localRowEnd - localRow;
            const uint32_t chunkRows =
                remainingRows < kRowsPerChunk ? remainingRows : kRowsPerChunk;
            const uint32_t row = mBegin + localRow;
            const uint32_t validRows =
                row < tiling_.m
                    ? (tiling_.m - row < chunkRows ? tiling_.m - row
                                                   : chunkRows)
                    : 0;
            const uint32_t chunkElements = chunkRows * tiling_.tileK;
            AscendC::LocalTensor<CubeInputType> chunk =
                decoded[static_cast<uint64_t>(localRow - localRowBegin) *
                        tiling_.tileK];
            ZeroCubeInput(chunk, chunkElements);
            AscendC::PipeBarrier<PIPE_V>();
            if (validRows != 0 && validColumns != 0) {
                DecodeActivationBlock(
                    static_cast<uint64_t>(row) * tiling_.k + kBegin,
                    validRows, validColumns);
            }
            CopyActivationBlockToNd(decoded, activationL1, localRow,
                                    chunkRows);
        }
    }

    __aicore__ inline void DecodeWeightBlock(
        uint64_t srcOffset, uint32_t rowCount, uint32_t validColumns,
        const AscendC::LocalTensor<CubeInputType> &decoded) {
        AscendC::LocalTensor<uint8_t> raw = rawBuffer_.Get<uint8_t>();
        AscendC::LocalTensor<half> halfOffsets =
            halfIndexBuffer_.Get<half>();
        AscendC::LocalTensor<uint32_t> offsets =
            indexBuffer_.Get<uint32_t>();

        AscendC::DataCopyExtParams copyParams{1, validColumns, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint8_t> padParams{true, 0, 0, 0};
        for (uint32_t rowIndex = 0; rowIndex < rowCount; ++rowIndex) {
            AscendC::DataCopyPad(
                raw[static_cast<uint64_t>(rowIndex) * tiling_.tileK],
                weightGm_[srcOffset +
                          static_cast<uint64_t>(rowIndex) * tiling_.k],
                copyParams, padParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        AscendC::LocalTensor<int32_t> signedOffsets =
            offsets.ReinterpretCast<int32_t>();
        const uint32_t count = rowCount * tiling_.tileK;
        if (validColumns == tiling_.tileK) {
            AscendC::Cast(halfOffsets, raw, AscendC::RoundMode::CAST_NONE,
                          count);
            AscendC::PipeBarrier<PIPE_V>();
#if HIF8_USE_INT_OFFSET_SHIFT
            AscendC::Cast(signedOffsets, halfOffsets,
                          AscendC::RoundMode::CAST_RINT, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftLeft(signedOffsets, signedOffsets,
                               static_cast<int32_t>(1), count);
#else
            const half tableElementBytes = 2.0;
            AscendC::Muls(halfOffsets, halfOffsets, tableElementBytes, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(signedOffsets, halfOffsets,
                          AscendC::RoundMode::CAST_RINT, count);
#endif
            AscendC::PipeBarrier<PIPE_V>();
            GatherDecode(decoded, halfOffsets, offsets, count);
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }

        for (uint32_t rowIndex = 0; rowIndex < rowCount; ++rowIndex) {
            const uint32_t rowOffset = rowIndex * tiling_.tileK;
            AscendC::Cast(halfOffsets, raw[rowOffset],
                          AscendC::RoundMode::CAST_NONE, validColumns);
            AscendC::PipeBarrier<PIPE_V>();
#if HIF8_USE_INT_OFFSET_SHIFT
            AscendC::Cast(signedOffsets, halfOffsets,
                          AscendC::RoundMode::CAST_RINT, validColumns);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftLeft(signedOffsets, signedOffsets,
                               static_cast<int32_t>(1), validColumns);
#else
            const half tableElementBytes = 2.0;
            AscendC::Muls(halfOffsets, halfOffsets, tableElementBytes,
                          validColumns);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(signedOffsets, halfOffsets,
                          AscendC::RoundMode::CAST_RINT, validColumns);
#endif
            AscendC::PipeBarrier<PIPE_V>();
            GatherDecode(decoded[rowOffset], halfOffsets, offsets,
                         validColumns);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ZeroWeightBlock(uint32_t count) {
        AscendC::LocalTensor<uint8_t> raw = rawBuffer_.Get<uint8_t>();
        AscendC::LocalTensor<CubeInputType> decoded =
            bf16Buffer_.Get<CubeInputType>();
        ZeroCubeInput(decoded, count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyWeightBlockToNz(
        const AscendC::LocalTensor<CubeInputType> &src,
        AscendC::LocalTensor<CubeInputType> &dst, uint32_t rowCount) {
        AscendC::Nd2NzParams copyParams{
            1,
            static_cast<uint16_t>(rowCount),
            static_cast<uint16_t>(tiling_.tileK),
            0,
            static_cast<uint16_t>(tiling_.tileK),
            static_cast<uint16_t>(tiling_.tileN /
                                  kVectorSubBlockCount),
            1,
            0};
        AscendC::DataCopy(dst, src, copyParams);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void CopyWeightTileToNd(
        const AscendC::LocalTensor<CubeInputType> &src,
        AscendC::LocalTensor<CubeInputType> &dst, uint32_t localRowBegin) {
        const uint32_t rowCount =
            tiling_.tileN / kVectorSubBlockCount;
        const uint32_t chunkBytes =
            kRowsPerChunk * tiling_.tileK * sizeof(CubeInputType);
        AscendC::DataCopyParams copyParams{
            1, static_cast<uint16_t>(chunkBytes / kDmaBlockBytes), 0, 0};
        for (uint32_t row = 0; row < rowCount; row += kRowsPerChunk) {
            const uint64_t srcOffset =
                static_cast<uint64_t>(row) * tiling_.tileK;
            const uint64_t dstOffset =
                static_cast<uint64_t>(localRowBegin + row) * tiling_.tileK;
            AscendC::DataCopy(dst[dstOffset], src[srcOffset], copyParams);
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void PrefetchDecodeBlock(
        const AscendC::GlobalTensor<uint8_t> &src, uint64_t srcOffset,
        uint32_t rowCount, uint32_t validColumns) {
        AscendC::LocalTensor<uint8_t> raw =
            rawInputQueue_.AllocTensor<uint8_t>();
        if (rowCount != 0 && validColumns != 0) {
            if (tiling_.useMultiLineDma != 0 &&
                validColumns == tiling_.tileK) {
                // DataCopyExt strides are byte gaps after each block.  This
                // branch is safe for complete K tiles: no padding is needed
                // and each source row has a regular GM stride. A final tail
                // tile takes the row-by-row branch below.
                AscendC::DataCopyExtParams copyParams{
                    static_cast<uint16_t>(rowCount), validColumns,
                    tiling_.k - validColumns, tiling_.tileK - validColumns,
                    0};
                AscendC::DataCopyPadExtParams<uint8_t> padParams{
                    false, 0, 0, 0};
                AscendC::DataCopyPad(raw, src[srcOffset], copyParams,
                                     padParams);
            } else {
                AscendC::DataCopyExtParams copyParams{
                    1, validColumns, 0, 0, 0};
                AscendC::DataCopyPadExtParams<uint8_t> padParams{
                    true, 0, 0, 0};
                for (uint32_t row = 0; row < rowCount; ++row) {
                    AscendC::DataCopyPad(
                        raw[static_cast<uint64_t>(row) * tiling_.tileK],
                        src[srcOffset +
                            static_cast<uint64_t>(row) * tiling_.k],
                        copyParams, padParams);
                }
            }
        }
        rawInputQueue_.EnQue(raw);
    }

    __aicore__ inline uint32_t WeightChunkValidRows(
        uint32_t nBegin, uint32_t localRowBegin, uint32_t chunk) const {
        const uint64_t channel =
            static_cast<uint64_t>(nBegin) + localRowBegin +
            static_cast<uint64_t>(chunk) * kRowsPerChunk;
        return channel < tiling_.n
                   ? (tiling_.n - channel < kRowsPerChunk
                          ? static_cast<uint32_t>(tiling_.n - channel)
                          : kRowsPerChunk)
                   : 0;
    }

    __aicore__ inline void PrefetchWeightChunk(
        uint32_t nBegin, uint32_t kBegin, uint32_t localRowBegin,
        uint32_t chunk, uint32_t validColumns) {
        const uint32_t channel =
            nBegin + localRowBegin + chunk * kRowsPerChunk;
        PrefetchDecodeBlock(
            weightGm_, static_cast<uint64_t>(channel) * tiling_.k + kBegin,
            WeightChunkValidRows(nBegin, localRowBegin, chunk), validColumns);
    }

    __aicore__ inline uint32_t ActivationChunkValidRows(
        uint32_t localRows, uint32_t chunk) const {
        const uint32_t localRow = chunk * kRowsPerChunk;
        return localRow < localRows
                   ? (localRows - localRow < kRowsPerChunk
                          ? localRows - localRow
                          : kRowsPerChunk)
                   : 0;
    }

    __aicore__ inline void PrefetchActivationChunk(
        uint32_t mBegin, uint32_t kBegin, uint32_t localRowBegin,
        uint32_t localRows, uint32_t chunk, uint32_t validColumns) {
        const uint32_t localRow = localRowBegin + chunk * kRowsPerChunk;
        const uint32_t validRows =
            ActivationChunkValidRows(localRows, chunk);
        if (validRows != 0 && validColumns != 0) {
            PrefetchDecodeBlock(
                activationGm_,
                static_cast<uint64_t>(mBegin + localRow) * tiling_.k + kBegin,
                validRows, validColumns);
        } else {
            // Keep the queue sequence full for padded rows and zero-width
            // tails. DecodePrefetchedBlock will produce a zero chunk.
            PrefetchDecodeBlock(activationGm_, 0, 0, 0);
        }
    }

    __aicore__ inline void PrefetchActivationCacheChunk(
        uint32_t mBegin, uint32_t kBegin, uint32_t tileRows,
        uint32_t chunk, uint32_t validColumns) {
        const uint32_t localRow = chunk * kRowsPerChunk;
        const uint32_t validRows =
            ActivationChunkValidRows(tileRows, chunk);
        if (validRows != 0 && validColumns != 0) {
            PrefetchDecodeBlock(
                activationGm_,
                static_cast<uint64_t>(mBegin + localRow) * tiling_.k + kBegin,
                validRows, validColumns);
        } else {
            PrefetchDecodeBlock(activationGm_, 0, 0, 0);
        }
    }

    __aicore__ inline void DecodePrefetchedBlock(uint32_t rowCount,
                                                  uint32_t validColumns) {
        AscendC::LocalTensor<uint8_t> raw =
            rawInputQueue_.DeQue<uint8_t>();
        AscendC::LocalTensor<CubeInputType> decoded =
            decodedOutputQueue_.AllocTensor<CubeInputType>();
        AscendC::LocalTensor<half> halfOffsets;
        if constexpr (kReuseDecodeOutputAsOffset) {
            halfOffsets = decoded.ReinterpretCast<half>();
        } else {
            halfOffsets = halfIndexBuffer_.Get<half>();
        }
        AscendC::LocalTensor<uint32_t> offsets =
            indexBuffer_.Get<uint32_t>();
        AscendC::LocalTensor<int32_t> signedOffsets =
            offsets.ReinterpretCast<int32_t>();
        const uint32_t chunkElements = kRowsPerChunk * tiling_.tileK;

        // A full chunk is overwritten entirely by Gather.  Only tails need
        // zero fill so padded rows/columns cannot enter Matmul.
        if (rowCount != kRowsPerChunk || validColumns != tiling_.tileK) {
            ZeroCubeInput(decoded, chunkElements);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (rowCount != 0 && validColumns != 0) {
            if constexpr (kDecodeProbeStage == 1) {
                // No vector consumer exists in the DMA-only probe to create
                // the usual MTE2->V dependency. Wait explicitly so freeing
                // the queue slot cannot make GM-to-UB traffic look free.
                AscendC::PipeBarrier<PIPE_MTE2>();
                decodedOutputQueue_.EnQue(decoded);
                rawInputQueue_.FreeTensor(raw);
                return;
            }
            if (validColumns == tiling_.tileK) {
                const uint32_t count = rowCount * tiling_.tileK;
                AscendC::Cast(halfOffsets, raw,
                              AscendC::RoundMode::CAST_NONE, count);
                AscendC::PipeBarrier<PIPE_V>();
                if constexpr (kDecodeProbeStage == 2) {
                    decodedOutputQueue_.EnQue(decoded);
                    rawInputQueue_.FreeTensor(raw);
                    return;
                }
#if HIF8_USE_INT_OFFSET_SHIFT
                AscendC::Cast(signedOffsets, halfOffsets,
                              AscendC::RoundMode::CAST_RINT, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ShiftLeft(signedOffsets, signedOffsets,
                                   static_cast<int32_t>(1), count);
                AscendC::PipeBarrier<PIPE_V>();
                if constexpr (kDecodeProbeStage == 3) {
                    decodedOutputQueue_.EnQue(decoded);
                    rawInputQueue_.FreeTensor(raw);
                    return;
                }
#else
                const half tableElementBytes = 2.0;
                AscendC::Muls(halfOffsets, halfOffsets, tableElementBytes,
                              count);
                AscendC::PipeBarrier<PIPE_V>();
                if constexpr (kDecodeProbeStage == 3) {
                    decodedOutputQueue_.EnQue(decoded);
                    rawInputQueue_.FreeTensor(raw);
                    return;
                }
                AscendC::Cast(signedOffsets, halfOffsets,
                              AscendC::RoundMode::CAST_RINT, count);
#endif
                AscendC::PipeBarrier<PIPE_V>();
                if constexpr (kDecodeProbeStage == 4) {
                    decodedOutputQueue_.EnQue(decoded);
                    rawInputQueue_.FreeTensor(raw);
                    return;
                }
                GatherDecode(decoded, halfOffsets, offsets, count);
                if constexpr (kDecodeProbeStage == 5) {
                    // Stage 6 relies on VECOUT EnQue to order Gather before
                    // MTE3. Stage 5 has no GM consumer, so explicitly wait
                    // here to measure completed Gather work.
                    AscendC::PipeBarrier<PIPE_V>();
                }
            } else {
                // Tail rows are tileK-strided in UB.  Restrict every vector
                // operation to validColumns so the zero-filled output keeps
                // the padded K region out of Matmul. When the output slot is
                // also the offset scratch, keep each row's offsets inside
                // that row so preparing a later row cannot overwrite an
                // already gathered BF16 result.
                for (uint32_t row = 0; row < rowCount; ++row) {
                    const uint32_t rowOffset = row * tiling_.tileK;
                    AscendC::LocalTensor<half> rowHalfOffsets =
                        halfOffsets[rowOffset];
                    AscendC::Cast(rowHalfOffsets, raw[rowOffset],
                                  AscendC::RoundMode::CAST_NONE,
                                  validColumns);
                    AscendC::PipeBarrier<PIPE_V>();
                    if constexpr (kDecodeProbeStage == 2) {
                        continue;
                    }
#if HIF8_USE_INT_OFFSET_SHIFT
                    AscendC::Cast(signedOffsets, rowHalfOffsets,
                                  AscendC::RoundMode::CAST_RINT,
                                  validColumns);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::ShiftLeft(signedOffsets, signedOffsets,
                                       static_cast<int32_t>(1), validColumns);
                    AscendC::PipeBarrier<PIPE_V>();
                    if constexpr (kDecodeProbeStage == 3) {
                        continue;
                    }
#else
                    const half tableElementBytes = 2.0;
                    AscendC::Muls(rowHalfOffsets, rowHalfOffsets,
                                  tableElementBytes, validColumns);
                    AscendC::PipeBarrier<PIPE_V>();
                    if constexpr (kDecodeProbeStage == 3) {
                        continue;
                    }
                    AscendC::Cast(signedOffsets, rowHalfOffsets,
                                  AscendC::RoundMode::CAST_RINT,
                                  validColumns);
#endif
                    AscendC::PipeBarrier<PIPE_V>();
                    if constexpr (kDecodeProbeStage == 4) {
                        continue;
                    }
                    GatherDecode(decoded[rowOffset], rowHalfOffsets, offsets,
                                 validColumns);
                    if constexpr (kDecodeProbeStage == 5) {
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                }
            }
        }
        decodedOutputQueue_.EnQue(decoded);
        rawInputQueue_.FreeTensor(raw);
    }

    __aicore__ inline void WriteDecodedBlock(
        AscendC::GlobalTensor<CubeInputType> &dst, uint64_t dstOffset) {
        AscendC::LocalTensor<CubeInputType> decoded =
            decodedOutputQueue_.DeQue<CubeInputType>();
        AscendC::DataCopy(dst[dstOffset], decoded,
                          kRowsPerChunk * tiling_.tileK);
        // DataCopy is asynchronous on MTE3.  The caller raises a cross-core
        // ready flag immediately after staging a tile; without an explicit
        // MTE3 barrier the AIC can consume stale bytes, producing
        // batch-/timing-dependent logits.  This was exposed by repeated
        // identical evaluations differing by several percentage points.
        AscendC::PipeBarrier<PIPE_MTE3>();
        decodedOutputQueue_.FreeTensor(decoded);
    }

    __aicore__ inline void DiscardDecodedBlock() {
        AscendC::LocalTensor<CubeInputType> decoded =
            decodedOutputQueue_.DeQue<CubeInputType>();
        decodedOutputQueue_.FreeTensor(decoded);
    }

    __aicore__ inline void WriteDecodedBlockToTscm(
        AscendC::LocalTensor<CubeInputType> &dst, uint64_t dstOffset) {
        AscendC::LocalTensor<CubeInputType> decoded =
            decodedOutputQueue_.DeQue<CubeInputType>();
        const uint32_t copyBytes =
            kRowsPerChunk * tiling_.tileK * sizeof(CubeInputType);
        AscendC::DataCopyParams copyParams{
            1, static_cast<uint16_t>(copyBytes / kDmaBlockBytes), 0, 0};
        AscendC::DataCopy(dst[dstOffset], decoded, copyParams);
        AscendC::PipeBarrier<PIPE_MTE3>();
        decodedOutputQueue_.FreeTensor(decoded);
    }

    __aicore__ inline void ProduceTscmOperand(
        const AscendC::GlobalTensor<uint8_t> &src, uint32_t globalRowBegin,
        uint32_t globalRowLimit, uint32_t kBegin, uint32_t tileRows,
        AscendC::LocalTensor<CubeInputType> &dst) {
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        const uint32_t rowsPerSubBlock = tileRows / kVectorSubBlockCount;
        const uint32_t localRowBegin = subBlock * rowsPerSubBlock;
        const uint32_t localRowEnd = localRowBegin + rowsPerSubBlock;
        const uint32_t validColumns =
            tiling_.k - kBegin < tiling_.tileK ? tiling_.k - kBegin
                                               : tiling_.tileK;
        for (uint32_t localRow = localRowBegin; localRow < localRowEnd;
             localRow += kRowsPerChunk) {
            const uint32_t row = globalRowBegin + localRow;
            const uint32_t chunkRows =
                localRowEnd - localRow < kRowsPerChunk
                    ? localRowEnd - localRow
                    : kRowsPerChunk;
            const uint32_t validRows =
                row < globalRowLimit
                    ? (globalRowLimit - row < chunkRows
                           ? globalRowLimit - row
                           : chunkRows)
                    : 0;
            if (validRows != 0 && validColumns != 0) {
                PrefetchDecodeBlock(
                    src, static_cast<uint64_t>(row) * tiling_.k + kBegin,
                    validRows, validColumns);
            } else {
                PrefetchDecodeBlock(src, 0, 0, 0);
            }
            DecodePrefetchedBlock(validRows, validColumns);
            WriteDecodedBlockToTscm(
                dst, static_cast<uint64_t>(localRow) * tiling_.tileK);
        }
    }

    __aicore__ inline void StageWeightTile(uint32_t nBegin,
                                          uint32_t kBegin, uint32_t slot) {
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        const uint32_t rowsPerSubBlock =
            tiling_.tileN / kVectorSubBlockCount;
        const uint32_t localRowBegin = subBlock * rowsPerSubBlock;
        const uint32_t validColumns =
            tiling_.k - kBegin < tiling_.tileK ? tiling_.k - kBegin
                                               : tiling_.tileK;
        AscendC::GlobalTensor<CubeInputType> &stage =
            slot == 0 ? weightStageGm0_ : weightStageGm1_;
        const uint32_t chunkCount = rowsPerSubBlock / kRowsPerChunk;

        // Leave one queue slot free for the first lookahead submission.  The
        // consumer below follows the same distance, so depth=2 provides
        // one-chunk lookahead while depth=3/4 can expose more MTE2
        // work to the vector pipeline.
        const uint32_t lookahead = kDecodeQueueDepth - 1U;
        const uint32_t initialPrefetch =
            chunkCount < lookahead ? chunkCount : lookahead;
        for (uint32_t chunk = 0; chunk < initialPrefetch; ++chunk) {
            PrefetchWeightChunk(nBegin, kBegin, localRowBegin, chunk,
                                validColumns);
        }
        for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
            const uint32_t localRow = localRowBegin + chunk * kRowsPerChunk;
            const uint32_t nextChunk = chunk + lookahead;
            if (nextChunk < chunkCount) {
                PrefetchWeightChunk(nBegin, kBegin, localRowBegin, nextChunk,
                                    validColumns);
            }
            DecodePrefetchedBlock(
                WeightChunkValidRows(nBegin, localRowBegin, chunk),
                validColumns);
            if constexpr (kDecodeProbeStage >= 1 &&
                          kDecodeProbeStage <= 5) {
                DiscardDecodedBlock();
            } else {
                WriteDecodedBlock(
                    stage, static_cast<uint64_t>(localRow) * tiling_.tileK);
            }
        }
    }

    __aicore__ inline void StageActivationTile(uint32_t mBegin,
                                               uint32_t kBegin,
                                               uint32_t slot) {
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        const uint32_t rowsPerSubBlock =
            tiling_.tileM / kVectorSubBlockCount;
        const uint32_t localRowBegin = subBlock * rowsPerSubBlock;
        const uint32_t validColumns =
            tiling_.k - kBegin < tiling_.tileK ? tiling_.k - kBegin
                                               : tiling_.tileK;
        AscendC::GlobalTensor<CubeInputType> &stage =
            slot == 0 ? activationStageGm_ : activationStageGm1_;
        const uint32_t tileRows =
            tiling_.m - mBegin < tiling_.tileM ? tiling_.m - mBegin
                                               : tiling_.tileM;
        const uint32_t localRows =
            tileRows > localRowBegin
                ? (tileRows - localRowBegin < rowsPerSubBlock
                       ? tileRows - localRowBegin
                       : rowsPerSubBlock)
                : 0;
        const uint32_t chunkCount =
            (localRows + kRowsPerChunk - 1U) / kRowsPerChunk;

        if (chunkCount == 0) {
            return;
        }

        const uint32_t lookahead = kDecodeQueueDepth - 1U;
        const uint32_t initialPrefetch =
            chunkCount < lookahead ? chunkCount : lookahead;
        for (uint32_t chunk = 0; chunk < initialPrefetch; ++chunk) {
            PrefetchActivationChunk(mBegin, kBegin, localRowBegin, localRows,
                                    chunk, validColumns);
        }
        for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
            const uint32_t localRow = localRowBegin + chunk * kRowsPerChunk;
            const uint32_t nextChunk = chunk + lookahead;
            if (nextChunk < chunkCount) {
                PrefetchActivationChunk(mBegin, kBegin, localRowBegin,
                                        localRows, nextChunk, validColumns);
            }
            DecodePrefetchedBlock(ActivationChunkValidRows(localRows, chunk),
                                  validColumns);
            WriteDecodedBlock(
                stage, static_cast<uint64_t>(localRow) * tiling_.tileK);
        }
    }

    // Decode activation exactly once per invocation into a tile-major GM
    // workspace. Every active AIV owns a contiguous range of four-row chunks;
    // the device-wide barrier in Process() makes the completed cache visible
    // before any AIC consumes it. Each padding write has a single owner.
    __aicore__ inline void ProduceActivationCache() {
        if (tiling_.useActivationCache == 0) {
            return;
        }
        const uint32_t chunksPerTile =
            (tiling_.tileM + kRowsPerChunk - 1U) / kRowsPerChunk;
        const uint32_t worker =
            BlockIndex() * kVectorSubBlockCount +
            AscendC::GetSubBlockIdx();
        const uint32_t workerCount =
            tiling_.aicNum * kVectorSubBlockCount;
        const uint64_t totalChunks =
            static_cast<uint64_t>(tiling_.mTileCount) *
            tiling_.kTileCount * chunksPerTile;
        const uint64_t workBegin = totalChunks * worker / workerCount;
        const uint64_t workEnd =
            totalChunks * (worker + 1U) / workerCount;
        const uint32_t lookahead = kDecodeQueueDepth - 1U;

        uint64_t initialEnd = workBegin + lookahead;
        if (initialEnd > workEnd) {
            initialEnd = workEnd;
        }
        for (uint64_t work = workBegin; work < initialEnd; ++work) {
            PrefetchActivationCacheWork(work, chunksPerTile);
        }
        for (uint64_t work = workBegin; work < workEnd; ++work) {
            const uint64_t nextWork = work + lookahead;
            if (nextWork < workEnd) {
                PrefetchActivationCacheWork(nextWork, chunksPerTile);
            }

            const uint64_t tile = work / chunksPerTile;
            const uint32_t chunk = static_cast<uint32_t>(
                work - tile * chunksPerTile);
            const uint32_t mTile =
                static_cast<uint32_t>(tile / tiling_.kTileCount);
            const uint32_t kTile = static_cast<uint32_t>(
                tile - static_cast<uint64_t>(mTile) *
                           tiling_.kTileCount);
            const uint32_t mBegin = mTile * tiling_.tileM;
            const uint32_t kBegin = kTile * tiling_.tileK;
            const uint32_t tileRows =
                tiling_.m - mBegin < tiling_.tileM
                    ? tiling_.m - mBegin
                    : tiling_.tileM;
            const uint32_t validColumns =
                tiling_.k - kBegin < tiling_.tileK
                    ? tiling_.k - kBegin
                    : tiling_.tileK;
            const uint64_t tileOffset =
                tile * static_cast<uint64_t>(tiling_.tileM) *
                tiling_.tileK;
            DecodePrefetchedBlock(
                ActivationChunkValidRows(tileRows, chunk), validColumns);
            WriteDecodedBlock(
                activationCacheGm_,
                tileOffset + static_cast<uint64_t>(chunk) *
                                 kRowsPerChunk * tiling_.tileK);
        }
    }

    __aicore__ inline void PrefetchActivationCacheWork(
        uint64_t work, uint32_t chunksPerTile) {
        const uint64_t tile = work / chunksPerTile;
        const uint32_t chunk =
            static_cast<uint32_t>(work - tile * chunksPerTile);
        const uint32_t mTile =
            static_cast<uint32_t>(tile / tiling_.kTileCount);
        const uint32_t kTile = static_cast<uint32_t>(
            tile - static_cast<uint64_t>(mTile) * tiling_.kTileCount);
        const uint32_t mBegin = mTile * tiling_.tileM;
        const uint32_t kBegin = kTile * tiling_.tileK;
        const uint32_t tileRows =
            tiling_.m - mBegin < tiling_.tileM
                ? tiling_.m - mBegin
                : tiling_.tileM;
        const uint32_t validColumns =
            tiling_.k - kBegin < tiling_.tileK
                ? tiling_.k - kBegin
                : tiling_.tileK;
        PrefetchActivationCacheChunk(mBegin, kBegin, tileRows, chunk,
                                     validColumns);
    }

    __aicore__ inline void LoadStagedTiles() {
        // The staging workspace contains exactly one ND matrix for each
        // operand.  Keep the matrix strides zero for ndNum == 1; on C310
        // these fields are encoded into the loop-4 stride and non-zero values
        // are unnecessary (and can make the NZ address state ambiguous).
        const uint32_t alignedActivationRows =
            (tiling_.tileM + 15U) / 16U * 16U;
        const uint32_t alignedWeightRows =
            (tiling_.tileN + 15U) / 16U * 16U;
        AscendC::Nd2NzParams activationCopy;
        activationCopy.ndNum = 1;
        activationCopy.nValue = tiling_.tileM;
        activationCopy.dValue = tiling_.tileK;
        activationCopy.srcDValue = tiling_.tileK;
        activationCopy.dstNzC0Stride = alignedActivationRows;
        activationCopy.dstNzNStride = 1;
        activationCopy.srcNdMatrixStride = 0;
        activationCopy.dstNzMatrixStride = 0;
        AscendC::DataCopy(activationL1_, activationStageGm_, activationCopy);

        AscendC::Nd2NzParams weightCopy;
        weightCopy.ndNum = 1;
        weightCopy.nValue = tiling_.tileN;
        weightCopy.dValue = tiling_.tileK;
        weightCopy.srcDValue = tiling_.tileK;
        weightCopy.dstNzC0Stride = alignedWeightRows;
        weightCopy.dstNzNStride = 1;
        weightCopy.srcNdMatrixStride = 0;
        weightCopy.dstNzMatrixStride = 0;
        AscendC::DataCopy(weightL1_, weightStageGm0_, weightCopy);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
    }

    __aicore__ inline void AccumulateTile(uint32_t mBegin,
                                         uint32_t nBegin,
                                         uint32_t kBegin,
                                         bool partialSum,
                                         uint32_t slot,
                                         uint32_t weightRowBegin,
                                         uint32_t maxTileN,
                                         bool releaseSlot) {
        const uint32_t tileM =
            tiling_.m - mBegin < tiling_.tileM ? tiling_.m - mBegin
                                               : tiling_.tileM;
        const uint32_t tileN =
            tiling_.n - nBegin < maxTileN ? tiling_.n - nBegin : maxTileN;
        const uint32_t tileK =
            tiling_.k - kBegin < tiling_.tileK ? tiling_.k - kBegin
                                               : tiling_.tileK;
        mm_.SetOrgShape(tiling_.tileM, kCubeSplitN, tiling_.tileK,
                        tiling_.tileK, tiling_.outputStride);
        if (tiling_.useActivationCache != 0) {
            const uint32_t mTile = mBegin / tiling_.tileM;
            const uint32_t kTile = kBegin / tiling_.tileK;
            const uint64_t activationOffset =
                (static_cast<uint64_t>(mTile) * tiling_.kTileCount + kTile) *
                static_cast<uint64_t>(tiling_.tileM) * tiling_.tileK;
            mm_.SetTensorA(activationCacheGm_[activationOffset], false);
        } else if (slot == 0) {
            mm_.SetTensorA(activationStageGm_, false);
        } else {
            mm_.SetTensorA(activationStageGm1_, false);
        }
        if (slot == 0) {
            mm_.SetTensorB(
                weightStageGm0_[static_cast<uint64_t>(weightRowBegin) *
                                tiling_.tileK],
                true);
        } else {
            mm_.SetTensorB(
                weightStageGm1_[static_cast<uint64_t>(weightRowBegin) *
                                tiling_.tileK],
                true);
        }
        mm_.SetSingleShape(tileM, tileN, tileK);
        mm_.Iterate(partialSum);
        if (releaseSlot) {
            AscendC::CrossCoreSetFlag<kAivAicSyncMode, PIPE_MTE1>(
                ActivationFreeEvent(0, slot));
        }
    }

    __aicore__ inline void ProduceWeightTile(uint32_t nBegin,
                                             uint32_t kBegin,
                                             uint32_t slot) {
        AscendC::LocalTensor<CubeInputType> decoded =
            bf16Buffer_.Get<CubeInputType>();
        AscendC::LocalTensor<CubeInputType> weightL1 = weightL1_;
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        const uint32_t rowsPerSubBlock = tiling_.tileN / 2U;
        const uint32_t localRowBegin = subBlock * rowsPerSubBlock;
        const uint32_t localRowEnd = localRowBegin + rowsPerSubBlock;
        const uint32_t validColumns =
            tiling_.k - kBegin < tiling_.tileK ? tiling_.k - kBegin
                                               : tiling_.tileK;

        for (uint32_t localRow = localRowBegin; localRow < localRowEnd;
             localRow += kRowsPerChunk) {
            const uint32_t remainingRows = localRowEnd - localRow;
            const uint32_t chunkRows =
                remainingRows < kRowsPerChunk ? remainingRows : kRowsPerChunk;
            const uint32_t channel = nBegin + localRow;
            const uint32_t validRows =
                channel < tiling_.n
                    ? (tiling_.n - channel < chunkRows
                           ? tiling_.n - channel
                           : chunkRows)
                    : 0;
            const uint32_t chunkElements = chunkRows * tiling_.tileK;
            AscendC::LocalTensor<CubeInputType> chunk =
                decoded[static_cast<uint64_t>(localRow - localRowBegin) *
                        tiling_.tileK];
            ZeroCubeInput(chunk, chunkElements);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::PipeBarrier<PIPE_V>();
            if (validRows != 0 && validColumns != 0) {
                DecodeWeightBlock(
                    static_cast<uint64_t>(channel) * tiling_.k + kBegin,
                    validRows, validColumns, chunk);
            }
        }
        CopyWeightTileToNd(decoded, weightL1, localRowBegin);
    }

    __aicore__ inline void ProduceTiles() {
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        uint32_t tileSequence = 0;
        for (uint32_t nTile = BlockIndex(); nTile < tiling_.nTileCount;
             nTile += tiling_.aicNum) {
            const uint32_t nBegin = nTile * tiling_.tileN;
            for (uint32_t mTile = 0; mTile < tiling_.mTileCount; ++mTile) {
                const uint32_t mBegin = mTile * tiling_.tileM;
                for (uint32_t kTile = 0; kTile < tiling_.kTileCount;
                     ++kTile) {
                    if (tileSequence != 0) {
                        AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(
                            WeightFreeEvent(subBlock, 0));
                        AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(
                            ActivationFreeEvent(subBlock, 0));
                    }
                    const uint32_t kBegin = kTile * tiling_.tileK;
                    ProduceTscmOperand(weightGm_, nBegin, tiling_.n, kBegin,
                                       tiling_.tileN, weightL1_);
                    AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(
                        WeightReadyEvent(subBlock, 0));
                    ProduceTscmOperand(activationGm_, mBegin, tiling_.m,
                                       kBegin, tiling_.tileM,
                                       activationL1_);
                    AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(
                        ActivationReadyEvent(subBlock, 0));
                    ++tileSequence;
                }
            }
        }
    }

    __aicore__ inline void RunTiledMatmul() {
        AscendC::LocalTensor<CubeInputType> activationL1 = activationL1_;
        for (uint32_t nTile = BlockIndex(); nTile < tiling_.nTileCount;
             nTile += tiling_.aicNum) {
            const uint32_t nBegin = nTile * tiling_.tileN;
            const uint32_t tileN =
                tiling_.n - nBegin < tiling_.tileN ? tiling_.n - nBegin
                                                   : tiling_.tileN;
            for (uint32_t mTile = 0; mTile < tiling_.mTileCount; ++mTile) {
                const uint32_t mBegin = mTile * tiling_.tileM;
                const uint32_t tileM =
                    tiling_.m - mBegin < tiling_.tileM ? tiling_.m - mBegin
                                                       : tiling_.tileM;
                for (uint32_t kTile = 0; kTile < tiling_.kTileCount;
                     ++kTile) {
                    AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(
                        ActivationReadyEvent(0, 0));
                    AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(
                        ActivationReadyEvent(1, 0));
                    AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(
                        WeightReadyEvent(0, 0));
                    AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(
                        WeightReadyEvent(1, 0));
                    const uint32_t kBegin = kTile * tiling_.tileK;
                    const uint32_t tileK =
                        tiling_.k - kBegin < tiling_.tileK
                            ? tiling_.k - kBegin
                            : tiling_.tileK;
                    mm_.SetOrgShape(tiling_.tileM, tiling_.tileN,
                                    tiling_.tileK, tiling_.tileK,
                                    tiling_.outputStride);
                    mm_.SetTensorA(activationL1, false);
                    mm_.SetTensorB(weightL1_, true);
                    mm_.SetSingleShape(tileM, tileN, tileK);
                    mm_.Iterate(kTile != 0);
                    AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(
                        ActivationFreeEvent(0, 0));
                    AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(
                        ActivationFreeEvent(1, 0));
                    AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(
                        WeightFreeEvent(0, 0));
                    AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(
                        WeightFreeEvent(1, 0));
                }
                const uint64_t outputOffset =
                    static_cast<uint64_t>(mBegin) * tiling_.outputStride +
                    nBegin;
                mm_.GetTensorC(matmulOutputGm_[outputOffset]);
            }
            AscendC::CrossCoreSetFlag<4, PIPE_FIX>(OutputReadyEvent(0));
            AscendC::CrossCoreSetFlag<4, PIPE_FIX>(OutputReadyEvent(1));
        }
        mm_.End();
    }

    // Decode probes only need one observable write to keep the kernel's
    // execution contract intact. Writing a full logical row here would both
    // add shape-dependent epilogue cost and exceed rawBuffer_ when N is
    // larger than tileN.
    __aicore__ inline void ProbeDecodeOutput() {
        if (BlockIndex() != 0 || AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        AscendC::LocalTensor<DTYPE_Y> output = rawBuffer_.Get<DTYPE_Y>();
        constexpr uint32_t kProbeOutputElements =
            kDmaBlockBytes / sizeof(DTYPE_Y);
        AscendC::Duplicate(output, FloatToDevice<DTYPE_Y>(0.0f),
                           kProbeOutputElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(outputGm_, output, kProbeOutputElements);
    }

    __aicore__ inline void ProbeDirectOutput() {
        AscendC::LocalTensor<DTYPE_Y> output = rawBuffer_.Get<DTYPE_Y>();
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        const float marker =
            (AscendC::GetBlockIdx() == 0 ? 0.0f : 100.0f) +
            (AscendC::GetTaskRation() == 2 ? 20.0f : 40.0f) +
            (subBlock == 0 ? 0.0f : 1.0f);
        for (uint32_t row = subBlock; row < tiling_.m;
             row += kVectorSubBlockCount) {
            AscendC::Duplicate(output, FloatToDevice<DTYPE_Y>(marker),
                               tiling_.n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopyExtParams outputCopy{
                1, static_cast<uint32_t>(tiling_.n * sizeof(DTYPE_Y)), 0, 0,
                0};
            AscendC::DataCopyPad(
                outputGm_[static_cast<uint64_t>(row) * tiling_.n], output,
                outputCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        }
    }

    // Echoes the raw Cube accumulator without any scaling so that Cube
    // correctness can be judged independently of the epilogue.
    __aicore__ inline void ProbeMatmulOutput() {
        AscendC::LocalTensor<float> values = decodedBuffer_.Get<float>();
        AscendC::LocalTensor<DTYPE_Y> output = rawBuffer_.Get<DTYPE_Y>();
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        for (uint32_t row = subBlock; row < tiling_.m;
             row += kVectorSubBlockCount) {
            const uint64_t offset =
                static_cast<uint64_t>(row) * tiling_.n;
            AscendC::DataCopyExtParams valueCopy{
                1, static_cast<uint32_t>(tiling_.n * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> valuePad{false, 0, 0, 0};
            AscendC::DataCopyPad(values, matmulOutputGm_[offset], valueCopy,
                                 valuePad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
            if constexpr (AscendC::IsSameType<DTYPE_Y, bfloat16_t>::value) {
                AscendC::Cast(output, values, AscendC::RoundMode::CAST_RINT,
                              tiling_.n);
            } else {
                AscendC::Cast(output, values, AscendC::RoundMode::CAST_NONE,
                              tiling_.n);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(4);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(4);
            AscendC::DataCopyExtParams outputCopy{
                1, static_cast<uint32_t>(tiling_.n * sizeof(DTYPE_Y)), 0, 0, 0};
            AscendC::DataCopyPad(outputGm_[offset], output, outputCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(4);
        }
    }

    __aicore__ inline void ProbeStagedActivation() {
        AscendC::LocalTensor<CubeInputType> staged =
            bf16Buffer_.Get<CubeInputType>();
        AscendC::LocalTensor<float> values = decodedBuffer_.Get<float>();
        AscendC::LocalTensor<DTYPE_Y> output = rawBuffer_.Get<DTYPE_Y>();
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        const uint32_t rowBegin = subBlock;
        const uint32_t rowEnd = tiling_.m;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(1);
        for (uint32_t row = rowBegin; row < rowEnd;
             row += kVectorSubBlockCount) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::DataCopy(
                staged,
                activationStageGm_[static_cast<uint64_t>(row) *
                                   tiling_.tileK],
                kDmaBlockBytes / sizeof(CubeInputType));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::Cast(values, staged, AscendC::RoundMode::CAST_NONE,
                          tiling_.k);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (AscendC::IsSameType<DTYPE_Y, bfloat16_t>::value) {
                AscendC::Cast(output, values, AscendC::RoundMode::CAST_RINT,
                              tiling_.k);
            } else {
                AscendC::Cast(output, values, AscendC::RoundMode::CAST_NONE,
                              tiling_.k);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
            AscendC::DataCopyExtParams outputCopy{
                1, static_cast<uint32_t>(tiling_.n * sizeof(DTYPE_Y)), 0, 0, 0};
            AscendC::DataCopyPad(
                outputGm_[static_cast<uint64_t>(row) * tiling_.n], output,
                outputCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        }
    }

    __aicore__ inline void ProbeStagedWeight() {
        AscendC::LocalTensor<CubeInputType> staged =
            bf16Buffer_.Get<CubeInputType>();
        AscendC::LocalTensor<float> values = decodedBuffer_.Get<float>();
        AscendC::LocalTensor<DTYPE_Y> output = rawBuffer_.Get<DTYPE_Y>();
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(1);
        for (uint32_t row = subBlock; row < tiling_.m;
             row += kVectorSubBlockCount) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::DataCopy(
                staged,
                weightStageGm0_[static_cast<uint64_t>(row) * tiling_.tileK],
                kDmaBlockBytes / sizeof(CubeInputType));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::Cast(values, staged, AscendC::RoundMode::CAST_NONE,
                          tiling_.k);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (AscendC::IsSameType<DTYPE_Y, bfloat16_t>::value) {
                AscendC::Cast(output, values, AscendC::RoundMode::CAST_RINT,
                              tiling_.k);
            } else {
                AscendC::Cast(output, values, AscendC::RoundMode::CAST_NONE,
                              tiling_.k);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
            AscendC::DataCopyExtParams outputCopy{
                1, static_cast<uint32_t>(tiling_.n * sizeof(DTYPE_Y)), 0, 0, 0};
            AscendC::DataCopyPad(
                outputGm_[static_cast<uint64_t>(row) * tiling_.n], output,
                outputCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        }
    }


    __aicore__ inline void RunVectorEpilogue() {
        if constexpr (kUseBatchedEpilogue) {
            if (tiling_.m >= kEpilogueRowsPerBatch &&
                tiling_.n % (kDmaBlockBytes / sizeof(DTYPE_Y)) == 0) {
                RunBatchedVectorEpilogue();
                return;
            }
        }

        AscendC::LocalTensor<float> values = decodedBuffer_.Get<float>();
        AscendC::LocalTensor<float> channelScale = indexBuffer_.Get<float>();
        AscendC::LocalTensor<uint16_t> biasPayload =
            halfIndexBuffer_.Get<uint16_t>();
        AscendC::LocalTensor<float> bias = bf16Buffer_.Get<float>();
        AscendC::LocalTensor<DTYPE_Y> output = rawBuffer_.Get<DTYPE_Y>();
        const uint32_t subBlock = AscendC::GetSubBlockIdx();

        for (uint32_t nTile = BlockIndex(); nTile < tiling_.nTileCount;
             nTile += tiling_.aicNum) {
            const uint32_t nBegin = nTile * tiling_.tileN;
            const uint32_t tileN =
                tiling_.n - nBegin < tiling_.tileN ? tiling_.n - nBegin
                                                   : tiling_.tileN;

            AscendC::DataCopyExtParams scaleCopy{1,
                                                 static_cast<uint32_t>(tileN * sizeof(float)), 0, 0,
                                                 0};
            AscendC::DataCopyPadExtParams<float> scalePad{false, 0, 0, 0};
            AscendC::DataCopyPad(channelScale, channelScaleGm_[nBegin],
                                scaleCopy, scalePad);
            if (tiling_.hasBias != 0) {
                AscendC::DataCopyExtParams biasCopy{
                    1, static_cast<uint32_t>(tileN * sizeof(uint16_t)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<uint16_t> biasPad{false, 0, 0,
                                                               0};
                AscendC::DataCopyPad(biasPayload, biasGm_[nBegin], biasCopy,
                                    biasPad);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);

            if (tiling_.hasBias != 0) {
                if (tiling_.biasDtype == 1) {
                    AscendC::Cast(bias, biasPayload.ReinterpretCast<bfloat16_t>(),
                                  AscendC::RoundMode::CAST_NONE, tileN);
                } else {
                    AscendC::Cast(bias, biasPayload.ReinterpretCast<half>(),
                                  AscendC::RoundMode::CAST_NONE, tileN);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }

            for (uint32_t row = subBlock; row < tiling_.m; row += 2U) {
                const uint64_t offset =
                    static_cast<uint64_t>(row) * tiling_.n + nBegin;
                AscendC::DataCopyExtParams valueCopy{
                    1, static_cast<uint32_t>(tileN * sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<float> valuePad{false, 0, 0, 0};
                AscendC::DataCopyPad(values, matmulOutputGm_[offset], valueCopy,
                                    valuePad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);

                AscendC::Mul(values, values, channelScale, tileN);
                AscendC::Muls(values, values, tokenScaleGm_.GetValue(row),
                              tileN);
                if (tiling_.hasBias != 0) {
                    AscendC::Add(values, values, bias, tileN);
                }
                if constexpr (AscendC::IsSameType<DTYPE_Y,
                                                  bfloat16_t>::value) {
                    AscendC::Cast(output, values, AscendC::RoundMode::CAST_RINT,
                                  tileN);
                } else {
                    AscendC::Cast(output, values, AscendC::RoundMode::CAST_NONE,
                                  tileN);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::DataCopyExtParams outputCopy{
                    1, static_cast<uint32_t>(tileN * sizeof(DTYPE_Y)), 0, 0, 0};
                AscendC::DataCopyPad(outputGm_[offset], output, outputCopy);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(3);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(3);
            }
        }
    }

    __aicore__ inline void RunBatchedVectorEpilogue() {
        // Decode scratch is dead after Cube publishes the FP32 output. Reuse
        // its larger buffers so an AIV can move several complete rows with
        // one MTE2/MTE3 submission without increasing the kernel's UB use.
        AscendC::LocalTensor<float> values =
            halfIndexBuffer_.Get<float>();
        AscendC::LocalTensor<float> channelScale =
            decodedBuffer_.Get<float>();
        AscendC::LocalTensor<uint16_t> biasPayload =
            rawBuffer_.Get<uint16_t>();
        AscendC::LocalTensor<float> bias = bf16Buffer_.Get<float>();
        AscendC::LocalTensor<DTYPE_Y> output =
            indexBuffer_.Get<DTYPE_Y>();
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        const uint32_t rowBegin =
            static_cast<uint64_t>(tiling_.m) * subBlock /
            kVectorSubBlockCount;
        const uint32_t rowEnd =
            static_cast<uint64_t>(tiling_.m) * (subBlock + 1U) /
            kVectorSubBlockCount;

        for (uint32_t nTile = BlockIndex(); nTile < tiling_.nTileCount;
             nTile += tiling_.aicNum) {
            const uint32_t nBegin = nTile * tiling_.tileN;
            const uint32_t tileN =
                tiling_.n - nBegin < tiling_.tileN ? tiling_.n - nBegin
                                                   : tiling_.tileN;

            AscendC::DataCopyExtParams scaleCopy{
                1, static_cast<uint32_t>(tileN * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> scalePad{false, 0, 0, 0};
            AscendC::DataCopyPad(channelScale, channelScaleGm_[nBegin],
                                scaleCopy, scalePad);
            if (tiling_.hasBias != 0) {
                AscendC::DataCopyExtParams biasCopy{
                    1, static_cast<uint32_t>(tileN * sizeof(uint16_t)), 0, 0,
                    0};
                AscendC::DataCopyPadExtParams<uint16_t> biasPad{false, 0, 0,
                                                               0};
                AscendC::DataCopyPad(biasPayload, biasGm_[nBegin], biasCopy,
                                    biasPad);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);

            if (tiling_.hasBias != 0) {
                if (tiling_.biasDtype == 1) {
                    AscendC::Cast(
                        bias, biasPayload.ReinterpretCast<bfloat16_t>(),
                        AscendC::RoundMode::CAST_NONE, tileN);
                } else {
                    AscendC::Cast(bias, biasPayload.ReinterpretCast<half>(),
                                  AscendC::RoundMode::CAST_NONE, tileN);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }

            for (uint32_t batchBegin = rowBegin; batchBegin < rowEnd;
                 batchBegin += kEpilogueRowsPerBatch) {
                const uint32_t batchRows =
                    rowEnd - batchBegin < kEpilogueRowsPerBatch
                        ? rowEnd - batchBegin
                        : kEpilogueRowsPerBatch;
                const uint64_t offset =
                    static_cast<uint64_t>(batchBegin) * tiling_.n + nBegin;
                AscendC::DataCopyExtParams valueCopy{
                    static_cast<uint16_t>(batchRows),
                    static_cast<uint32_t>(tileN * sizeof(float)),
                    static_cast<uint32_t>((tiling_.n - tileN) *
                                          sizeof(float)),
                    0, 0};
                AscendC::DataCopyPadExtParams<float> valuePad{false, 0, 0,
                                                              0};
                AscendC::DataCopyPad(values, matmulOutputGm_[offset],
                                    valueCopy, valuePad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);

                for (uint32_t localRow = 0; localRow < batchRows;
                     ++localRow) {
                    const uint32_t localOffset = localRow * tileN;
                    const uint32_t row = batchBegin + localRow;
                    AscendC::Mul(values[localOffset], values[localOffset],
                                 channelScale, tileN);
                    AscendC::Muls(values[localOffset], values[localOffset],
                                  tokenScaleGm_.GetValue(row), tileN);
                    if (tiling_.hasBias != 0) {
                        AscendC::Add(values[localOffset],
                                    values[localOffset], bias, tileN);
                    }
                    if constexpr (AscendC::IsSameType<
                                      DTYPE_Y, bfloat16_t>::value) {
                        AscendC::Cast(output[localOffset],
                                      values[localOffset],
                                      AscendC::RoundMode::CAST_RINT, tileN);
                    } else {
                        AscendC::Cast(output[localOffset],
                                      values[localOffset],
                                      AscendC::RoundMode::CAST_NONE, tileN);
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::DataCopyExtParams outputCopy{
                    static_cast<uint16_t>(batchRows),
                    static_cast<uint32_t>(tileN * sizeof(DTYPE_Y)), 0,
                    static_cast<uint32_t>((tiling_.n - tileN) *
                                          sizeof(DTYPE_Y)),
                    0};
                AscendC::DataCopyPad(outputGm_[offset], output, outputCopy);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(3);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(3);
            }
        }
    }

    SoftHif8GemmTilingData tiling_{};
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, HIF8_DECODE_QUEUE_DEPTH>
        rawInputQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, HIF8_DECODE_QUEUE_DEPTH>
        decodedOutputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> decodedBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfIndexBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bf16Buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> decodeTableBuffer_;
    AscendC::TBuf<AscendC::TPosition::TSCM> activationL1Buffer_;
    AscendC::TBuf<AscendC::TPosition::TSCM> weightL1Buffer0_;
    AscendC::TBuf<AscendC::TPosition::TSCM> weightL1Buffer1_;
    AscendC::TSCM<AscendC::TPosition::VECCALC, 1, 4> sharedL1Buffer_;
    AscendC::LocalTensor<CubeInputType> sharedL1_;
    AscendC::LocalTensor<CubeInputType> activationL1_;
    AscendC::LocalTensor<CubeInputType> weightL1_;
    AscendC::GlobalTensor<uint8_t> activationGm_;
    AscendC::GlobalTensor<uint8_t> weightGm_;
    AscendC::GlobalTensor<float> tokenScaleGm_;
    AscendC::GlobalTensor<float> channelScaleGm_;
    AscendC::GlobalTensor<uint16_t> biasGm_;
    AscendC::GlobalTensor<DTYPE_Y> outputGm_;
    AscendC::GlobalTensor<CubeOutputType> matmulOutputGm_;
    AscendC::GlobalTensor<CubeInputType> activationStageGm_;
    AscendC::GlobalTensor<CubeInputType> activationStageGm1_;
    AscendC::GlobalTensor<CubeInputType> activationCacheGm_;
    AscendC::GlobalTensor<CubeInputType> weightStageGm0_;
    AscendC::GlobalTensor<CubeInputType> weightStageGm1_;
    matmul::MatmulImpl<AType, BType, CType, BiasType, kMatmulConfig> mm_;
};

}  // namespace vllm_ascend_hif8

extern "C" __global__ __aicore__ void HIF8_GEMM_KERNEL_ENTRY(
    GM_ADDR activation, GM_ADDR weight, GM_ADDR token_scale,
    GM_ADDR channel_scale, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
    GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SoftHif8GemmTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    vllm_ascend_hif8::HIF8_GEMM_KERNEL_CLASS kernel;
    kernel.Init(activation, weight, token_scale, channel_scale, bias, y,
                workspace, tilingData);
    kernel.Process();
}
