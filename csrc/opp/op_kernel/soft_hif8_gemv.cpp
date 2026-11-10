#include "hif8_codec.h"
#include "kernel_operator.h"
#include "soft_hif8_gemv_tiling.h"

namespace vllm_ascend_hif8 {

constexpr uint32_t kGemvTileK = 1024;
constexpr uint32_t kGemvChannelBatch = 16;
// Gather<float> uses an 8-bit repeat count on this target.  Eight 1024-wide
// rows are 128 repeats, while 16 rows would overflow at 256.
constexpr uint32_t kGemvDecodeBatch = 8;
constexpr uint32_t kGemvOutputBatch = 128;
constexpr uint32_t kDmaBlockBytes = 32;
constexpr uint32_t kGlobalWriteAlignBytes = 64;

__aicore__ inline float DecodeGemvBias(uint16_t payload,
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

class SoftHif8GemvKernel {
public:
    __aicore__ inline void Init(GM_ADDR activation, GM_ADDR weight,
                                GM_ADDR tokenScale, GM_ADDR channelScale,
                                GM_ADDR bias, GM_ADDR output,
                                const SoftHif8GemvTilingData &tiling) {
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

        pipe_.InitBuffer(activationRawBuffer_,
                         kGemvTileK * sizeof(uint8_t) + kDmaBlockBytes);
        pipe_.InitBuffer(weightRawBuffer_,
                         kGemvChannelBatch * kGemvTileK * sizeof(uint8_t) +
                             kDmaBlockBytes);
        pipe_.InitBuffer(activationDecodedBuffer_,
                         kGemvTileK * sizeof(float));
        pipe_.InitBuffer(weightDecodedBuffer_,
                         kGemvDecodeBatch * kGemvTileK * sizeof(float));
        pipe_.InitBuffer(halfIndexBuffer_,
                         kGemvDecodeBatch * kGemvTileK * sizeof(half));
        pipe_.InitBuffer(indexBuffer_,
                         kGemvDecodeBatch * kGemvTileK * sizeof(uint32_t));
        pipe_.InitBuffer(productBuffer_,
                         kGemvTileK * sizeof(float));
        pipe_.InitBuffer(reduceBuffer_, kGemvTileK * sizeof(float));
        pipe_.InitBuffer(sumBuffer_, kDmaBlockBytes);
        pipe_.InitBuffer(decodeTableBuffer_, 256 * sizeof(float));
    }

    __aicore__ inline void Process() {
        InitDecodeTable();

        const uint64_t outputElements =
            static_cast<uint64_t>(tiling_.m) * tiling_.n;
        const uint32_t elementsPerGroup =
            kGlobalWriteAlignBytes / sizeof(DTYPE_Y);
        const uint64_t groups =
            (outputElements + elementsPerGroup - 1U) / elementsPerGroup;
        const uint64_t groupsPerCore =
            (groups + AscendC::GetBlockNum() - 1U) /
            AscendC::GetBlockNum();
        const uint64_t begin = static_cast<uint64_t>(AscendC::GetBlockIdx()) *
                               groupsPerCore * elementsPerGroup;
        const uint64_t rawEnd = begin + groupsPerCore * elementsPerGroup;
        const uint64_t end =
            rawEnd < outputElements ? rawEnd : outputElements;

        uint64_t outputIndex = begin;
        while (outputIndex < end) {
            const uint32_t row = outputIndex / tiling_.n;
            const uint32_t channel = outputIndex % tiling_.n;
            const uint32_t channelsLeftInRow = tiling_.n - channel;
            const uint32_t outputsLeft = static_cast<uint32_t>(end - outputIndex);
            uint32_t batch = channelsLeftInRow < kGemvOutputBatch
                                 ? channelsLeftInRow
                                 : kGemvOutputBatch;
            batch = outputsLeft < batch ? outputsLeft : batch;
            ProcessBatch(row, channel, batch, outputIndex);
            outputIndex += batch;
        }
    }

private:
    __aicore__ inline void InitDecodeTable() {
        AscendC::LocalTensor<float> table = decodeTableBuffer_.Get<float>();
        for (uint32_t code = 0; code < 256; ++code) {
            table.SetValue(code, Decode(static_cast<uint8_t>(code)));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    }

    __aicore__ inline void DecodeTile(
        const AscendC::GlobalTensor<uint8_t> &source, uint64_t sourceOffset,
        uint32_t count, const AscendC::LocalTensor<uint8_t> &raw,
        const AscendC::LocalTensor<float> &decoded) {
        AscendC::LocalTensor<half> halfOffsets = halfIndexBuffer_.Get<half>();
        AscendC::LocalTensor<uint32_t> offsets = indexBuffer_.Get<uint32_t>();
        AscendC::LocalTensor<float> table = decodeTableBuffer_.Get<float>();

        AscendC::DataCopyExtParams copyParams{1, count, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint8_t> padParams{true, 0, 0, 0};
        AscendC::DataCopyPad(raw, source[sourceOffset], copyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        AscendC::Cast(halfOffsets, raw, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(halfOffsets, halfOffsets, half(4.0), count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<int32_t> signedOffsets =
            offsets.ReinterpretCast<int32_t>();
        AscendC::Cast(signedOffsets, halfOffsets,
                      AscendC::RoundMode::CAST_RINT, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(decoded, table, offsets, 0, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DecodeLocal(
        const AscendC::LocalTensor<uint8_t> &raw, uint32_t count,
        const AscendC::LocalTensor<float> &decoded) {
        AscendC::LocalTensor<half> halfOffsets = halfIndexBuffer_.Get<half>();
        AscendC::LocalTensor<uint32_t> offsets = indexBuffer_.Get<uint32_t>();
        AscendC::LocalTensor<float> table = decodeTableBuffer_.Get<float>();
        AscendC::Cast(halfOffsets, raw, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(halfOffsets, halfOffsets, half(4.0), count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<int32_t> signedOffsets =
            offsets.ReinterpretCast<int32_t>();
        AscendC::Cast(signedOffsets, halfOffsets,
                      AscendC::RoundMode::CAST_RINT, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(decoded, table, offsets, 0, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float DotTile(
        const AscendC::LocalTensor<float> &activation,
        const AscendC::LocalTensor<float> &weight, uint32_t count) {
        AscendC::LocalTensor<float> product = productBuffer_.Get<float>();
        AscendC::LocalTensor<float> reduce = reduceBuffer_.Get<float>();
        AscendC::LocalTensor<float> sum = sumBuffer_.Get<float>();
        AscendC::Mul(product, activation, weight, count);
        AscendC::PipeBarrier<PIPE_V>();
        if (count == kGemvTileK) {
            // A full tile is 16 FP32 vector repeats.  Reducing 64 elements
            // per repeat produces 128 partials, then 16; one whole-reduce
            // finishes them without the generic ReduceSum scalar pipeline.
            AscendC::BlockReduceSum<float>(product, product, 16, 64, 1, 1,
                                           8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceSum<float>(product, product, 2, 64, 1, 1,
                                           8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(sum, product, 16, 1, 1, 1, 8);
        } else {
            AscendC::ReduceSum<float, true>(sum, product, reduce, count);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        return sum.GetValue(0);
    }

    __aicore__ inline void ProcessBatch(uint32_t row, uint32_t channelBegin,
                                        uint32_t batch,
                                        uint64_t outputIndex) {
        AscendC::LocalTensor<uint8_t> activationRaw =
            activationRawBuffer_.Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> weightRaw =
            weightRawBuffer_.Get<uint8_t>();
        AscendC::LocalTensor<float> activationDecoded =
            activationDecodedBuffer_.Get<float>();
        AscendC::LocalTensor<float> weightDecoded =
            weightDecodedBuffer_.Get<float>();
        float accumulators[kGemvOutputBatch] = {0.0f};

        for (uint32_t kBegin = 0; kBegin < tiling_.k;
             kBegin += kGemvTileK) {
            const uint32_t count =
                tiling_.k - kBegin < kGemvTileK ? tiling_.k - kBegin
                                                : kGemvTileK;
            DecodeTile(activationGm_,
                       static_cast<uint64_t>(row) * tiling_.k + kBegin,
                       count, activationRaw, activationDecoded);
            for (uint32_t channelBase = 0; channelBase < batch;
                 channelBase += kGemvChannelBatch) {
                const uint32_t channelsRemaining = batch - channelBase;
                const uint32_t weightBatch =
                    channelsRemaining < kGemvChannelBatch
                        ? channelsRemaining
                        : kGemvChannelBatch;
                const bool batchedWeightCopy =
                    count == kGemvTileK &&
                    (tiling_.k % kDmaBlockBytes) == 0;
                if (batchedWeightCopy) {
                    AscendC::DataCopyExtParams copyParams{
                        static_cast<uint16_t>(weightBatch), count,
                        static_cast<uint32_t>(tiling_.k - count), 0, 0};
                    AscendC::DataCopyPadExtParams<uint8_t> padParams{
                        false, 0, 0, 0};
                    AscendC::DataCopyPad(
                        weightRaw,
                        weightGm_[static_cast<uint64_t>(channelBegin +
                                                        channelBase) *
                                      tiling_.k +
                                  kBegin],
                        copyParams, padParams);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                }
                for (uint32_t decodeBase = 0; decodeBase < weightBatch;
                     decodeBase += kGemvDecodeBatch) {
                    const uint32_t decodeRemaining = weightBatch - decodeBase;
                    const uint32_t decodeRows =
                        decodeRemaining < kGemvDecodeBatch
                            ? decodeRemaining
                            : kGemvDecodeBatch;
                    if (batchedWeightCopy && decodeRows > 1) {
                        DecodeLocal(
                            weightRaw[decodeBase * kGemvTileK],
                            decodeRows * kGemvTileK, weightDecoded);
                    }
                    for (uint32_t inner = 0; inner < decodeRows; ++inner) {
                        const uint32_t channelOffset =
                            channelBase + decodeBase + inner;
                        const uint32_t channel = channelBegin + channelOffset;
                        if (batchedWeightCopy) {
                            if (decodeRows == 1) {
                                DecodeLocal(
                                    weightRaw[(decodeBase + inner) *
                                              kGemvTileK],
                                    count, weightDecoded);
                            }
                            accumulators[channelOffset] += DotTile(
                                activationDecoded,
                                weightDecoded[inner * kGemvTileK], count);
                        } else {
                            DecodeTile(
                                weightGm_,
                                static_cast<uint64_t>(channel) * tiling_.k +
                                    kBegin,
                                count, weightRaw, weightDecoded);
                            accumulators[channelOffset] += DotTile(
                                activationDecoded, weightDecoded, count);
                        }
                    }
                }
            }
        }

        const float tokenScale = tokenScaleGm_.GetValue(row);
        for (uint32_t channelOffset = 0; channelOffset < batch;
             ++channelOffset) {
            const uint32_t channel = channelBegin + channelOffset;
            float value = accumulators[channelOffset] * tokenScale *
                          channelScaleGm_.GetValue(channel);
            if (tiling_.hasBias != 0) {
                value += DecodeGemvBias(biasGm_.GetValue(channel),
                                        tiling_.biasDtype);
            }
            outputGm_.SetValue(outputIndex + channelOffset,
                               FloatToDevice<DTYPE_Y>(value));
        }
    }

    SoftHif8GemvTilingData tiling_{};
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> activationRawBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> weightRawBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> activationDecodedBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> weightDecodedBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfIndexBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> productBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> decodeTableBuffer_;
    AscendC::GlobalTensor<uint8_t> activationGm_;
    AscendC::GlobalTensor<uint8_t> weightGm_;
    AscendC::GlobalTensor<float> tokenScaleGm_;
    AscendC::GlobalTensor<float> channelScaleGm_;
    AscendC::GlobalTensor<uint16_t> biasGm_;
    AscendC::GlobalTensor<DTYPE_Y> outputGm_;
};

}  // namespace vllm_ascend_hif8

extern "C" __global__ __aicore__ void soft_hif8_gemv(
    GM_ADDR activation, GM_ADDR weight, GM_ADDR token_scale,
    GM_ADDR channel_scale, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
    GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SoftHif8GemvTilingData);
    GET_TILING_DATA(tilingData, tiling);
    vllm_ascend_hif8::SoftHif8GemvKernel kernel;
    kernel.Init(activation, weight, token_scale, channel_scale, bias, y,
                tilingData);
    kernel.Process();
}
