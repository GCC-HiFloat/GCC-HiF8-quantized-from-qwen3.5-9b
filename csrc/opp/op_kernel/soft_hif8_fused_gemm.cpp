#include "hif8_codec.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "soft_hif8_gemm_tiling.h"

namespace vllm_ascend_hif8 {

#ifndef HIF8_FUSED_KERNEL_CLASS
#define HIF8_FUSED_KERNEL_CLASS SoftHif8FusedGemmKernel
#endif
#ifndef HIF8_FUSED_KERNEL_ENTRY
#define HIF8_FUSED_KERNEL_ENTRY soft_hif8_fused_gemm
#endif
#ifndef HIF8_FUSED_TILING_DATA
#define HIF8_FUSED_TILING_DATA SoftHif8FusedGemmTilingData
#endif
#ifndef HIF8_FUSED_STATIC_SCALE
#define HIF8_FUSED_STATIC_SCALE 0
#endif

constexpr uint32_t kRowsPerChunk = 4;
constexpr uint32_t kWeightBufferCount = 2;
constexpr uint32_t kVectorSubBlockCount = 2;
#ifndef HIF8_FUSED_VECTOR_ENCODE
#if HIF8_FUSED_STATIC_SCALE
// Use vector LUT encoding for fixed-scale activation quantization.
#define HIF8_FUSED_VECTOR_ENCODE 1
#else
#define HIF8_FUSED_VECTOR_ENCODE 0
#endif
#endif
// Dynamic scaling defaults to the scalar encoder.
constexpr bool kUseVectorActivationEncode = HIF8_FUSED_VECTOR_ENCODE != 0;
constexpr uint32_t kDmaBlockBytes = 32;
constexpr uint8_t kAivAicSyncMode = 2;
constexpr uint16_t kWeightReadyEvent = 1;
constexpr uint16_t kWeightFreeEvent = 3;
constexpr uint16_t kSubBlockEventOffset = 16;

constexpr MatmulConfig kMatmulConfig =
    GetNormalConfig(true, false, false, BatchMode::BATCH_LESS_THAN_L1,
                    true, IterateOrder::UNDEF,
                    ScheduleType::INNER_PRODUCT, false);

__aicore__ inline float DecodeFusedBias(uint16_t payload,
                                        uint32_t biasDtype) {
    if (biasDtype == 1) {
        const bfloat16_t value =
            AscendC::GetScalarBitcodeValue<uint16_t, bfloat16_t>(payload);
        return DeviceToFloat(value);
    }
    const half value =
        AscendC::GetScalarBitcodeValue<uint16_t, half>(payload);
    return static_cast<float>(value);
}

class HIF8_FUSED_KERNEL_CLASS {
public:
    using AType = matmul::MatmulType<AscendC::TPosition::GM,
                                     CubeFormat::ND,
                                     bfloat16_t, false,
                                     LayoutMode::NONE, false>;
    using BType = matmul::MatmulType<AscendC::TPosition::GM,
                                     CubeFormat::ND,
                                     bfloat16_t, true,
                                     LayoutMode::NONE, false>;
    using CType = matmul::MatmulType<AscendC::TPosition::GM,
                                     CubeFormat::ND, float>;
    using BiasType = matmul::MatmulType<AscendC::TPosition::GM,
                                        CubeFormat::ND, float>;

    __aicore__ inline void Init(
        GM_ADDR activation, GM_ADDR weight, GM_ADDR fixedScale,
        GM_ADDR channelScale,
        GM_ADDR bias, GM_ADDR output, GM_ADDR workspace,
        const HIF8_FUSED_TILING_DATA &tiling) {
        tiling_ = tiling;
        activationGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_ACTIVATION *>(activation));
        weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(weight));
#if HIF8_FUSED_STATIC_SCALE
        fixedScaleGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(fixedScale));
#else
        (void)fixedScale;
#endif
        channelScaleGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(channelScale));
        if (tiling_.hasBias != 0) {
            biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(bias));
        }
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(output));

        __gm__ uint8_t *workspaceBytes =
            reinterpret_cast<__gm__ uint8_t *>(workspace);
        matmulOutputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspaceBytes + tiling_.outputWorkspaceOffset));
        activationStageGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(
                workspaceBytes + tiling_.activationWorkspaceOffset));
        tokenScaleStageGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspaceBytes + tiling_.tokenScaleWorkspaceOffset));

        const uint64_t coreWeightStageOffset =
            tiling_.weightWorkspaceOffset +
            static_cast<uint64_t>(BlockIndex()) *
                tiling_.weightWorkspaceStride;
        const uint64_t weightStageElements =
            static_cast<uint64_t>(tiling_.tileN) * tiling_.tileK;
        weightStageGm0_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(
                workspaceBytes + coreWeightStageOffset));
        weightStageGm1_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(
                workspaceBytes + coreWeightStageOffset +
                weightStageElements * sizeof(bfloat16_t)));

        if ASCEND_IS_AIV {
            const uint32_t decodeChunk = kRowsPerChunk * tiling_.tileK;
            const uint32_t vectorWidth =
                tiling_.tileK > tiling_.tileN ? tiling_.tileK : tiling_.tileN;
            pipe_.InitBuffer(rawInputQueue_, 2,
                             decodeChunk * sizeof(uint8_t) + kDmaBlockBytes);
            pipe_.InitBuffer(decodedOutputQueue_, 2,
                             decodeChunk * sizeof(bfloat16_t));
            // The activation producer and epilogue both need a K-wide vector.
            // The weight path uses its own queues, so do not alias a K-wide
            // local view onto an N-wide allocation.
            pipe_.InitBuffer(rawBuffer_,
                             vectorWidth * sizeof(DTYPE_ACTIVATION) +
                                 kDmaBlockBytes);
            pipe_.InitBuffer(decodedBuffer_, vectorWidth * sizeof(float));
            pipe_.InitBuffer(halfIndexBuffer_, decodeChunk * sizeof(half));
            pipe_.InitBuffer(indexBuffer_, decodeChunk * sizeof(uint32_t));
            pipe_.InitBuffer(bf16Buffer_,
                             vectorWidth * sizeof(bfloat16_t));
            pipe_.InitBuffer(decodeTableBuffer_, 256 * sizeof(bfloat16_t));
            // Direct FP32-index -> decoded BF16 lookup used by activation
            // quantization.  The table has 8192 uint16 entries (the sign bit
            // is part of the FP32 index), packed two per generated uint32.
            // Keep this allocation separate from the byte decode table.
            pipe_.InitBuffer(
                encodeDecodeTableBuffer_,
#if HIF8_FUSED_STATIC_SCALE
                sizeof(kFloat32StaticEncodeDecodeBf16U16Packed));
#else
                sizeof(kFloat32EncodeDecodeBf16U16Packed));
#endif
            pipe_.InitBuffer(scaleBuffer_, kDmaBlockBytes);
        }
        if ASCEND_IS_AIC {
            mm_.SetSubBlockIdx(0);
            mm_.Init(&tiling_.matmulTiling, &pipe_);
        }
    }

    __aicore__ inline void Process() {
        if ASCEND_IS_AIV {
            InitTables();
            // A single producer quantizes/decode-stages the activation.  The
            // output lives only in this invocation's workspace, whereas every
            // AIC block retains its own online-unpacked weight double-buffer.
            if (IsActivationProducer()) {
                QuantizeAndStageSharedActivation();
            }
        }

#if HIF8_FUSED_STATIC_SCALE
        ProcessStaticBarrierPipeline();
        return;
#else

        // This is the only global rendezvous before compute. It makes the
        // shared activation planes and token scales visible to every N tile.
        AscendC::SyncAll<false>();

        uint32_t tileSequence = 0;
        for (uint32_t nTile = BlockIndex(); nTile < tiling_.nTileCount;
             nTile += tiling_.aicNum) {
            const uint32_t nBegin = nTile * tiling_.tileN;
            for (uint32_t kTile = 0; kTile < tiling_.kTileCount; ++kTile) {
                const uint32_t kBegin = kTile * tiling_.tileK;
                const uint32_t slot = tileSequence % kWeightBufferCount;
                if ASCEND_IS_AIV {
                    if (tileSequence >= kWeightBufferCount) {
                        AscendC::CrossCoreWaitFlag<kAivAicSyncMode,
                                                   PIPE_MTE3>(
                            WeightFreeEvent(0, slot));
                        AscendC::CrossCoreWaitFlag<kAivAicSyncMode,
                                                   PIPE_MTE3>(
                            WeightFreeEvent(1, slot));
                    }
                    StageWeightTile(nBegin, kBegin, slot);
                }
                if ASCEND_IS_AIC {
                    AccumulateTile(nBegin, kBegin, kTile != 0, slot);
                }
                ++tileSequence;
            }
            if ASCEND_IS_AIC {
                mm_.GetTensorC(matmulOutputGm_[nBegin]);
            }
        }

        if ASCEND_IS_AIC {
            mm_.End();
        }

        // Matmul output has to be committed before vector epilogue reads it.
        // This mirrors the proven strict GEMM completion rendezvous.
        AscendC::SyncAll<false>();
        if ASCEND_IS_AIV {
            RunVectorEpilogue();
        }
#endif
    }

private:
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

    __aicore__ inline uint32_t BlockIndex() const {
        return AscendC::GetBlockIdx() / AscendC::GetTaskRation();
    }

    __aicore__ inline bool IsActivationProducer() const {
        return BlockIndex() == 0 && AscendC::GetSubBlockIdx() == 0;
    }

    __aicore__ inline bfloat16_t DecodeBf16(uint8_t payload) const {
        if (payload == kNan) {
            return AscendC::GetScalarBitcodeValue<uint16_t, bfloat16_t>(
                0x7fc0u);
        }
        return FloatToDevice<bfloat16_t>(Decode(payload));
    }

    __aicore__ inline void InitTables() {
        AscendC::LocalTensor<bfloat16_t> decodeTable =
            decodeTableBuffer_.Get<bfloat16_t>();
        for (uint32_t code = 0; code < 256; ++code) {
            decodeTable.SetValue(code, DecodeBf16(static_cast<uint8_t>(code)));
        }
        // Only block 0/sub-block 0 encodes the shared activation.  The other
        // AIV tasks only unpack weight bytes and therefore do not need to
        // initialize the 16 KiB FP32-index table.  Avoiding the redundant
        // 4096 scalar writes on every AIV is particularly important for the
        // one-token static path this fused operator targets.
        if (IsActivationProducer()) {
            AscendC::LocalTensor<uint32_t> encodeDecodeTable =
                encodeDecodeTableBuffer_.Get<uint32_t>();
            for (uint32_t index = 0;
                 index <
#if HIF8_FUSED_STATIC_SCALE
                             sizeof(kFloat32StaticEncodeDecodeBf16U16Packed) /
#else
                             sizeof(kFloat32EncodeDecodeBf16U16Packed) /
#endif
                             sizeof(uint32_t);
                 ++index) {
                encodeDecodeTable.SetValue(
                    index,
#if HIF8_FUSED_STATIC_SCALE
                    kFloat32StaticEncodeDecodeBf16U16Packed[index]);
#else
                    kFloat32EncodeDecodeBf16U16Packed[index]);
#endif
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    }

    __aicore__ inline void DecodePrefetchedBlock(
        uint32_t rowCount, uint32_t validColumns) {
        AscendC::LocalTensor<uint8_t> raw =
            rawInputQueue_.DeQue<uint8_t>();
        AscendC::LocalTensor<bfloat16_t> bf16 =
            decodedOutputQueue_.AllocTensor<bfloat16_t>();
        AscendC::LocalTensor<half> halfOffsets =
            halfIndexBuffer_.Get<half>();
        AscendC::LocalTensor<uint32_t> offsets =
            indexBuffer_.Get<uint32_t>();
        AscendC::LocalTensor<int32_t> signedOffsets =
            offsets.ReinterpretCast<int32_t>();
        AscendC::LocalTensor<bfloat16_t> decodeTable =
            decodeTableBuffer_.Get<bfloat16_t>();
        const uint32_t chunkElements = kRowsPerChunk * tiling_.tileK;
        if (rowCount != kRowsPerChunk || validColumns != tiling_.tileK) {
            AscendC::Duplicate(bf16, FloatToDevice<bfloat16_t>(0.0f),
                               chunkElements);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (rowCount != 0 && validColumns != 0) {
            const half tableElementBytes = 2.0;
            if (validColumns == tiling_.tileK) {
                const uint32_t count = rowCount * tiling_.tileK;
                AscendC::Cast(halfOffsets, raw,
                              AscendC::RoundMode::CAST_NONE, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(halfOffsets, halfOffsets, tableElementBytes,
                              count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(signedOffsets, halfOffsets,
                              AscendC::RoundMode::CAST_RINT, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Gather(bf16, decodeTable, offsets, 0, count);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                for (uint32_t row = 0; row < rowCount; ++row) {
                    const uint32_t rowOffset = row * tiling_.tileK;
                    AscendC::Cast(halfOffsets, raw[rowOffset],
                                  AscendC::RoundMode::CAST_NONE,
                                  validColumns);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(halfOffsets, halfOffsets,
                                  tableElementBytes, validColumns);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Cast(signedOffsets, halfOffsets,
                                  AscendC::RoundMode::CAST_RINT,
                                  validColumns);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Gather(bf16[rowOffset], decodeTable, offsets, 0,
                                    validColumns);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
        }
        decodedOutputQueue_.EnQue(bf16);
        rawInputQueue_.FreeTensor(raw);
    }

    __aicore__ inline void WriteDecodedBlock(
        AscendC::GlobalTensor<bfloat16_t> &dst, uint64_t dstOffset) {
        AscendC::LocalTensor<bfloat16_t> bf16 =
            decodedOutputQueue_.DeQue<bfloat16_t>();
        AscendC::DataCopy(dst[dstOffset], bf16,
                          kRowsPerChunk * tiling_.tileK);
        decodedOutputQueue_.FreeTensor(bf16);
    }

    __aicore__ inline void PrefetchWeightBlock(
        uint64_t srcOffset, uint32_t rowCount, uint32_t validColumns) {
        AscendC::LocalTensor<uint8_t> raw =
            rawInputQueue_.AllocTensor<uint8_t>();
        if (rowCount != 0 && validColumns != 0) {
            AscendC::DataCopyExtParams copyParams{1, validColumns, 0, 0, 0};
            AscendC::DataCopyPadExtParams<uint8_t> padParams{true, 0, 0, 0};
            for (uint32_t row = 0; row < rowCount; ++row) {
                AscendC::DataCopyPad(
                    raw[static_cast<uint64_t>(row) * tiling_.tileK],
                    weightGm_[srcOffset + static_cast<uint64_t>(row) *
                                             tiling_.k],
                    copyParams, padParams);
            }
        }
        rawInputQueue_.EnQue(raw);
    }

    __aicore__ inline void StageWeightTile(uint32_t nBegin,
                                          uint32_t kBegin,
                                          uint32_t slot) {
#if HIF8_FUSED_STATIC_SCALE
        // C220's mixed-task BlockIndex/task-ratio mapping is not stable for
        // two independent AIV writers into the same GM stage when the static
        // operator is launched with more than one AIC tile.  The barrier
        // schedule below therefore gives ownership of the complete weight
        // tile to AIV sub-block 0.  AIC tiles still run in parallel across N;
        // only the intra-tile vector decode is serialized for this opt-in
        // correctness path.
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        const uint32_t subBlock = 0;
        const uint32_t rowsPerSubBlock = tiling_.tileN;
#else
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        const uint32_t rowsPerSubBlock =
            tiling_.tileN / kVectorSubBlockCount;
#endif
        const uint32_t localRowBegin = subBlock * rowsPerSubBlock;
        const uint32_t validColumns =
            tiling_.k - kBegin < tiling_.tileK ? tiling_.k - kBegin
                                               : tiling_.tileK;
        AscendC::GlobalTensor<bfloat16_t> &stage =
            slot == 0 ? weightStageGm0_ : weightStageGm1_;
        const uint32_t chunkCount = rowsPerSubBlock / kRowsPerChunk;

        uint32_t channel = nBegin + localRowBegin;
        uint32_t validRows =
            channel < tiling_.n
                ? (tiling_.n - channel < kRowsPerChunk
                       ? tiling_.n - channel
                       : kRowsPerChunk)
                : 0;
        PrefetchWeightBlock(
            static_cast<uint64_t>(channel) * tiling_.k + kBegin,
            validRows, validColumns);
        for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
            const uint32_t localRow = localRowBegin + chunk * kRowsPerChunk;
            if (chunk + 1U < chunkCount) {
                const uint32_t nextChannel = channel + kRowsPerChunk;
                const uint32_t nextValidRows =
                    nextChannel < tiling_.n
                        ? (tiling_.n - nextChannel < kRowsPerChunk
                               ? tiling_.n - nextChannel
                               : kRowsPerChunk)
                        : 0;
                PrefetchWeightBlock(
                    static_cast<uint64_t>(nextChannel) * tiling_.k + kBegin,
                    nextValidRows, validColumns);
            }
            DecodePrefetchedBlock(validRows, validColumns);
            WriteDecodedBlock(
                stage, static_cast<uint64_t>(localRow) * tiling_.tileK);
            channel += kRowsPerChunk;
            validRows =
                channel < tiling_.n
                    ? (tiling_.n - channel < kRowsPerChunk
                           ? tiling_.n - channel
                           : kRowsPerChunk)
                : 0;
        }
#if HIF8_FUSED_STATIC_SCALE
        // SyncAll does not drain vector GM writes on 910B; fence MTE3 first.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(2);
#endif
#if !HIF8_FUSED_STATIC_SCALE
        AscendC::CrossCoreSetFlag<kAivAicSyncMode, PIPE_MTE3>(
            WeightReadyEvent(subBlock, slot));
#endif
    }

    __aicore__ inline void StageQuantizedActivationRow(
        uint32_t row, uint32_t kTile, float scale) {
        const uint32_t kBegin = kTile * tiling_.tileK;
        const uint32_t validColumns =
            tiling_.k - kBegin < tiling_.tileK ? tiling_.k - kBegin
                                               : tiling_.tileK;
        const uint64_t dstOffset =
            (static_cast<uint64_t>(kTile) * tiling_.tileM + row) *
            tiling_.tileK;
        AscendC::LocalTensor<DTYPE_ACTIVATION> source =
            rawBuffer_.Get<DTYPE_ACTIVATION>();
        AscendC::LocalTensor<float> values = decodedBuffer_.Get<float>();
        AscendC::LocalTensor<bfloat16_t> decoded =
            bf16Buffer_.Get<bfloat16_t>();
        AscendC::DataCopyExtParams inputCopy{
            1, static_cast<uint32_t>(validColumns *
                                     sizeof(DTYPE_ACTIVATION)),
            0, 0, 0};
        AscendC::DataCopyPadExtParams<DTYPE_ACTIVATION> inputPad{
            true, 0, 0, 0};
        AscendC::DataCopyPad(
            source,
            activationGm_[static_cast<uint64_t>(row) * tiling_.k + kBegin],
            inputCopy, inputPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
        AscendC::Cast(values, source, AscendC::RoundMode::CAST_NONE,
                      validColumns);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (kUseVectorActivationEncode) {
            AscendC::LocalTensor<uint32_t> offsets =
                halfIndexBuffer_.Get<uint32_t>();
            AscendC::LocalTensor<bfloat16_t> encoded =
                bf16Buffer_.Get<bfloat16_t>();
            AscendC::LocalTensor<bfloat16_t> encodeDecodeTable =
                encodeDecodeTableBuffer_.Get<bfloat16_t>();
            AscendC::Muls(values, values, 1.0f / scale, validColumns);
            AscendC::PipeBarrier<PIPE_V>();
            // The top 12 FP32 bits plus sign form an 8192-entry index.  The
            // generated table is stored as uint16 BF16 payloads, so the
            // gather offset is index * sizeof(uint16_t).
            AscendC::LocalTensor<uint32_t> fp32Bits =
                values.ReinterpretCast<uint32_t>();
            AscendC::ShiftRight(offsets, fp32Bits, kEncodeMantissaShift,
                                validColumns);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftLeft(offsets, offsets, 1u, validColumns);
            AscendC::PipeBarrier<PIPE_V>();
            if (validColumns != tiling_.tileK) {
                AscendC::Duplicate(encoded, FloatToDevice<bfloat16_t>(0.0f),
                                   tiling_.tileK);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Gather(encoded, encodeDecodeTable, offsets, 0,
                            validColumns);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            const bfloat16_t nanValue =
                AscendC::GetScalarBitcodeValue<uint16_t, bfloat16_t>(0x7fc0u);
            for (uint32_t kk = 0; kk < validColumns; ++kk) {
                const uint8_t encoded = Encode(values.GetValue(kk) / scale);
                decoded.SetValue(
                    kk, encoded == kNan ? nanValue : DecodeBf16(encoded));
            }
            for (uint32_t kk = validColumns; kk < tiling_.tileK; ++kk) {
                decoded.SetValue(kk, FloatToDevice<bfloat16_t>(0.0f));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
        AscendC::DataCopy(
            activationStageGm_[dstOffset],
            decoded,
            tiling_.tileK);
        if constexpr (kUseVectorActivationEncode) {
            // The vector encoder reuses the BF16 destination for every K
            // plane. Complete the GM write before issuing the next MTE2 load;
            // MTE3->V alone allowed the middle plane to be overwritten on
            // C220 when three or more planes were staged back-to-back.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
        }
    }

#if !HIF8_FUSED_STATIC_SCALE
    __aicore__ inline float ScalarActivationAmax(uint32_t row) {
        float amax = 0.0f;
        const uint64_t rowOffset = static_cast<uint64_t>(row) * tiling_.k;
        for (uint32_t kk = 0; kk < tiling_.k; ++kk) {
            const float value =
                DeviceToFloat(activationGm_.GetValue(rowOffset + kk));
            if (IsFinite(value)) {
                const float magnitude = value < 0.0f ? -value : value;
                if (magnitude > amax) {
                    amax = magnitude;
                }
            }
        }
        return amax;
    }

    __aicore__ inline float VectorActivationAmax(uint32_t row) {
        // Scalar reduction handles arbitrary K in the dynamic-scale path.
        return ScalarActivationAmax(row);
    }
#endif

    __aicore__ inline void QuantizeAndStageSharedActivation() {
        for (uint32_t row = 0; row < tiling_.m; ++row) {
#if HIF8_FUSED_STATIC_SCALE
            // Read the fixed scale from calibration metadata.
            const float scale = fixedScaleGm_.GetValue(0);
#else
            const float amax = VectorActivationAmax(row);
            const float scale = amax > kScaleEpsilon ? amax : kScaleEpsilon;
#endif
            tokenScaleStageGm_.SetValue(row, scale);
            for (uint32_t kTile = 0; kTile < tiling_.kTileCount; ++kTile) {
                StageQuantizedActivationRow(row, kTile, scale);
            }
        }
    }

    __aicore__ inline void AccumulateTile(uint32_t nBegin,
                                         uint32_t kBegin,
                                         bool partialSum,
                                         uint32_t slot) {
        const uint32_t tileN =
            tiling_.n - nBegin < tiling_.tileN ? tiling_.n - nBegin
                                               : tiling_.tileN;
        const uint32_t tileK =
            tiling_.k - kBegin < tiling_.tileK ? tiling_.k - kBegin
                                               : tiling_.tileK;
#if !HIF8_FUSED_STATIC_SCALE
        // Mode-2 cross-core events pair one AIC task with its matching AIV
        // sub-block. C220 does not expose the other AIV event in this domain.
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        AscendC::CrossCoreWaitFlag<kAivAicSyncMode, PIPE_MTE1>(
            WeightReadyEvent(subBlock, slot));
#endif
        const uint64_t activationOffset =
            static_cast<uint64_t>(kBegin / tiling_.tileK) *
            tiling_.tileM * tiling_.tileK;
        mm_.SetOrgShape(tiling_.tileM, tiling_.tileN, tiling_.tileK,
                        tiling_.tileK, tiling_.outputStride);
        mm_.SetTensorA(activationStageGm_[activationOffset], false);
        if (slot == 0) {
            mm_.SetTensorB(weightStageGm0_, true);
        } else {
            mm_.SetTensorB(weightStageGm1_, true);
        }
        mm_.SetSingleShape(tiling_.m, tileN, tileK);
        mm_.Iterate(partialSum);
#if !HIF8_FUSED_STATIC_SCALE
        AscendC::CrossCoreSetFlag<kAivAicSyncMode, PIPE_MTE1>(
            WeightFreeEvent(0, slot));
        AscendC::CrossCoreSetFlag<kAivAicSyncMode, PIPE_MTE1>(
            WeightFreeEvent(1, slot));
#endif
    }

    // C220 mode-2 events cover one AIV sub-block. Global barriers protect
    // both halves of each weight tile before reuse. All cores, including
    // inactive tail cores, must execute the same number of barriers.
    __aicore__ inline void ProcessStaticBarrierPipeline() {
        AscendC::SyncAll<false>();

        const uint32_t nTileRounds =
            tiling_.nTileCount == 0
                ? 0
                : (tiling_.nTileCount - 1U) / tiling_.aicNum + 1U;
        uint32_t tileSequence = 0;
        for (uint32_t nRound = 0; nRound < nTileRounds; ++nRound) {
            const uint32_t nTile =
                BlockIndex() + nRound * tiling_.aicNum;
            const bool hasNTile = nTile < tiling_.nTileCount;
            const uint32_t nBegin = nTile * tiling_.tileN;
            for (uint32_t kTile = 0; kTile < tiling_.kTileCount;
                 ++kTile) {
                const uint32_t kBegin = kTile * tiling_.tileK;
                const uint32_t slot = tileSequence % kWeightBufferCount;
                if ASCEND_IS_AIV {
                    if (hasNTile) {
                        StageWeightTile(nBegin, kBegin, slot);
                    }
                }
                // All AIV producers must finish staging before Cube reads the
                // shared weight tile.  This barrier is required on 910B even
                // when the producer uses a local MTE3->S fence.
                AscendC::SyncAll<false>();
                if ASCEND_IS_AIC {
                    if (hasNTile) {
                        AccumulateTile(nBegin, kBegin, kTile != 0, slot);
                    }
                }
                // All AIV producers have finished the complete N half-tile
                // before Cube consumes it, and Cube has consumed it before
                // the next K tile can reuse the stage.
                AscendC::SyncAll<false>();
                ++tileSequence;
            }
            if ASCEND_IS_AIC {
                if (hasNTile) {
                    const uint64_t outputOffset =
                        static_cast<uint64_t>(0) * tiling_.outputStride +
                        nBegin;
                    mm_.GetTensorC(matmulOutputGm_[outputOffset]);
                }
            }
            AscendC::SyncAll<false>();
        }
        if ASCEND_IS_AIC {
            mm_.End();
        }
        AscendC::SyncAll<false>();
        if ASCEND_IS_AIV {
            RunVectorEpilogue();
        }
    }

    __aicore__ inline void RunVectorEpilogue() {
        AscendC::LocalTensor<float> values = decodedBuffer_.Get<float>();
        AscendC::LocalTensor<float> channelScale = indexBuffer_.Get<float>();
        AscendC::LocalTensor<uint16_t> biasPayload =
            halfIndexBuffer_.Get<uint16_t>();
        AscendC::LocalTensor<float> bias = bf16Buffer_.Get<float>();
        AscendC::LocalTensor<DTYPE_Y> output = rawBuffer_.Get<DTYPE_Y>();
        const uint32_t subBlock = AscendC::GetSubBlockIdx();

        // The activation producer is one AIV task. Keep the tiny M<=4
        // epilogue on the first AIV sub-block as well: it avoids a second
        // vector task observing a scalar token-scale write through the GM
        // cache path. This does not reduce Cube/weight-unpack parallelism and
        // is negligible beside an N-wide vector epilogue.
        if (subBlock != 0) {
            return;
        }

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
                    1, static_cast<uint32_t>(tileN * sizeof(uint16_t)),
                    0, 0, 0};
                AscendC::DataCopyPadExtParams<uint16_t> biasPad{
                    false, 0, 0, 0};
                AscendC::DataCopyPad(biasPayload, biasGm_[nBegin], biasCopy,
                                     biasPad);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
            if (tiling_.hasBias != 0) {
                if (tiling_.biasDtype == 1) {
                    AscendC::Cast(
                        bias,
                        biasPayload.ReinterpretCast<bfloat16_t>(),
                        AscendC::RoundMode::CAST_NONE, tileN);
                } else {
                    AscendC::Cast(bias, biasPayload.ReinterpretCast<half>(),
                                  AscendC::RoundMode::CAST_NONE, tileN);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }

            for (uint32_t row = 0; row < tiling_.m; ++row) {
                const uint64_t offset =
                    static_cast<uint64_t>(row) * tiling_.n + nBegin;
                AscendC::DataCopyExtParams valueCopy{
                    1, static_cast<uint32_t>(tileN * sizeof(float)),
                    0, 0, 0};
                AscendC::DataCopyPadExtParams<float> valuePad{
                    false, 0, 0, 0};
                AscendC::DataCopyPad(values, matmulOutputGm_[offset],
                                     valueCopy, valuePad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
                AscendC::Mul(values, values, channelScale, tileN);
                AscendC::Muls(values, values, tokenScaleStageGm_.GetValue(row),
                              tileN);
                if (tiling_.hasBias != 0) {
                    AscendC::Add(values, values, bias, tileN);
                }
                if constexpr (AscendC::IsSameType<DTYPE_Y,
                                                  bfloat16_t>::value) {
                    AscendC::Cast(output, values,
                                  AscendC::RoundMode::CAST_RINT, tileN);
                } else {
                    AscendC::Cast(output, values,
                                  AscendC::RoundMode::CAST_NONE, tileN);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::DataCopyExtParams outputCopy{
                    1, static_cast<uint32_t>(tileN * sizeof(DTYPE_Y)),
                    0, 0, 0};
                AscendC::DataCopyPad(outputGm_[offset], output, outputCopy);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(3);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(3);
            }
        }
    }

    HIF8_FUSED_TILING_DATA tiling_{};
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> rawInputQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> decodedOutputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> decodedBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfIndexBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bf16Buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> decodeTableBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> encodeDecodeTableBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scaleBuffer_;
    AscendC::GlobalTensor<DTYPE_ACTIVATION> activationGm_;
    AscendC::GlobalTensor<uint8_t> weightGm_;
    AscendC::GlobalTensor<float> fixedScaleGm_;
    AscendC::GlobalTensor<float> channelScaleGm_;
    AscendC::GlobalTensor<uint16_t> biasGm_;
    AscendC::GlobalTensor<DTYPE_Y> outputGm_;
    AscendC::GlobalTensor<float> matmulOutputGm_;
    AscendC::GlobalTensor<bfloat16_t> activationStageGm_;
    AscendC::GlobalTensor<float> tokenScaleStageGm_;
    AscendC::GlobalTensor<bfloat16_t> weightStageGm0_;
    AscendC::GlobalTensor<bfloat16_t> weightStageGm1_;
    matmul::MatmulImpl<AType, BType, CType, BiasType, kMatmulConfig> mm_;
};

}  // namespace vllm_ascend_hif8

extern "C" __global__ __aicore__ void HIF8_FUSED_KERNEL_ENTRY(
    GM_ADDR activation, GM_ADDR weight,
#if HIF8_FUSED_STATIC_SCALE
    GM_ADDR fixed_scale,
#endif
    GM_ADDR channel_scale, GM_ADDR bias,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(HIF8_FUSED_TILING_DATA);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    vllm_ascend_hif8::HIF8_FUSED_KERNEL_CLASS kernel;
    kernel.Init(activation, weight,
#if HIF8_FUSED_STATIC_SCALE
                fixed_scale,
#else
                nullptr,
#endif
                channel_scale, bias, y, workspace,
                tilingData);
    kernel.Process();
}
