#include "hif8_codec.h"
#include "hif8_dynamic_quant_tiling.h"
#include "kernel_operator.h"

namespace vllm_ascend_hif8 {

constexpr uint32_t kDmaBlockBytes = 32;
constexpr uint32_t kScaleStride = 64 / sizeof(float);

template <typename InputType>
class Hif8DynamicQuantKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR data, GM_ADDR tokenScale, GM_ADDR workspace,
        const Hif8DynamicQuantTilingData &tiling) {
        tiling_ = tiling;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputType *>(x));
        dataGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(data));
        tokenScaleGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(tokenScale));
        (void)workspace;

        pipe_.InitBuffer(inputBuffer_, tiling_.k * sizeof(InputType) +
                                           kDmaBlockBytes);
        pipe_.InitBuffer(normalizedBuffer_, tiling_.k * sizeof(float));
        pipe_.InitBuffer(absBuffer_, tiling_.k * sizeof(float));
        pipe_.InitBuffer(encodedBuffer_, tiling_.k * sizeof(uint8_t) +
                                             kDmaBlockBytes);
        pipe_.InitBuffer(gatheredBuffer_, tiling_.k * sizeof(uint16_t));
        pipe_.InitBuffer(encodeTableBuffer_,
                         sizeof(kFloat32EncodeSignedU16Packed));
        pipe_.InitBuffer(scaleBuffer_, kDmaBlockBytes);
    }

    __aicore__ inline void Process() {
        const uint32_t block = AscendC::GetBlockIdx();
        const uint32_t blocks = AscendC::GetBlockNum();
        if (tiling_.k % kDmaBlockBytes != 0) {
            for (uint32_t row = block; row < tiling_.rows; row += blocks) {
                ProcessScalarRow(row);
            }
            return;
        }
        InitEncodeTable();
        for (uint32_t row = block; row < tiling_.rows; row += blocks) {
            ProcessRow(row);
        }

    }

private:
    __aicore__ inline void InitEncodeTable() {
        AscendC::LocalTensor<uint32_t> table =
            encodeTableBuffer_.Get<uint32_t>();
        for (uint32_t index = 0;
             index < sizeof(kFloat32EncodeSignedU16Packed) / sizeof(uint32_t);
             ++index) {
            table.SetValue(index, kFloat32EncodeSignedU16Packed[index]);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(2);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(2);
    }

    __aicore__ inline void ProcessScalarRow(uint32_t row) {
        const uint64_t rowOffset = static_cast<uint64_t>(row) * tiling_.k;
        float amax = 0.0f;
        for (uint32_t kk = 0; kk < tiling_.k; ++kk) {
            const float value = DeviceToFloat(xGm_.GetValue(rowOffset + kk));
            if (IsFinite(value)) {
                const float magnitude = value < 0.0f ? -value : value;
                if (magnitude > amax) {
                    amax = magnitude;
                }
            }
        }
        const float scale = amax > kScaleEpsilon ? amax : kScaleEpsilon;
        tokenScaleGm_.SetValue(row * kScaleStride, scale);
        for (uint32_t kk = 0; kk < tiling_.k; ++kk) {
            const float value = DeviceToFloat(xGm_.GetValue(rowOffset + kk));
            dataGm_.SetValue(rowOffset + kk, Encode(value / scale));
        }
    }

    __aicore__ inline void ProcessRow(uint32_t row) {
        AscendC::LocalTensor<InputType> input =
            inputBuffer_.Get<InputType>();
        AscendC::LocalTensor<float> normalized =
            normalizedBuffer_.Get<float>();
        AscendC::LocalTensor<float> absolute = absBuffer_.Get<float>();
        AscendC::LocalTensor<uint8_t> encoded =
            encodedBuffer_.Get<uint8_t>();
        AscendC::LocalTensor<int16_t> gathered =
            gatheredBuffer_.Get<int16_t>();
        AscendC::LocalTensor<int16_t> encodeTable =
            encodeTableBuffer_.Get<int16_t>();
        AscendC::LocalTensor<float> scaleLocal = scaleBuffer_.Get<float>();
        const uint64_t rowOffset = static_cast<uint64_t>(row) * tiling_.k;

        AscendC::DataCopyExtParams inputCopy{1,
                                             static_cast<uint32_t>(
                                                 tiling_.k * sizeof(InputType)),
                                             0, 0, 0};
        AscendC::DataCopyPadExtParams<InputType> inputPad{true, 0, 0, 0};
        AscendC::DataCopyPad(input, xGm_[rowOffset], inputCopy, inputPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);

        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        AscendC::Cast(normalized, input, AscendC::RoundMode::CAST_NONE,
                      tiling_.k);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs(absolute, normalized, tiling_.k);
        AscendC::PipeBarrier<PIPE_V>();
        // The basic reduction API stores max value and index in scaleLocal.
        // It may reuse the temporary tensor, which is safe because absolute
        // is not needed after the reduction.
        AscendC::ReduceMax(scaleLocal, absolute, absolute, tiling_.k);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        float amax = scaleLocal.GetValue(0);
        // ReduceMax over a row containing NaN/Inf is allowed to return a
        // non-finite value.  Preserve the format contract by ignoring those
        // values when selecting scale, while still encoding them as special
        // HiFloat8 payloads below.  Normal model activations stay entirely on
        // the vector fast path.
        const bool hasNonFinite = !IsFinite(amax);
        if (hasNonFinite) {
            amax = 0.0f;
            for (uint32_t kk = 0; kk < tiling_.k; ++kk) {
                const float value = normalized.GetValue(kk);
                if (IsFinite(value)) {
                    const float magnitude = value < 0.0f ? -value : value;
                    if (magnitude > amax) {
                        amax = magnitude;
                    }
                }
            }
        }
        const float scale = amax > kScaleEpsilon ? amax : kScaleEpsilon;

        AscendC::SetFlag<AscendC::HardEvent::S_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(1);
        AscendC::Muls(normalized, normalized, 1.0f / scale, tiling_.k);
        if (hasNonFinite) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            for (uint32_t kk = 0; kk < tiling_.k; ++kk) {
                encoded.SetValue(kk, Encode(normalized.GetValue(kk)));
            }
        } else {
            AscendC::LocalTensor<uint32_t> fp32Bits =
                normalized.ReinterpretCast<uint32_t>();
            AscendC::LocalTensor<uint32_t> encodeOffsets =
                absolute.ReinterpretCast<uint32_t>();
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftRight(encodeOffsets, fp32Bits,
                                kEncodeMantissaShift, tiling_.k);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftLeft(encodeOffsets, encodeOffsets, 1u,
                               tiling_.k);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(gathered, encodeTable, encodeOffsets, 0,
                            tiling_.k);
            AscendC::LocalTensor<half> packedHalf =
                normalizedBuffer_.Get<half>();
            AscendC::PipeBarrier<PIPE_V>();
            // C220 has no int16-to-u8 vector conversion.  Every table value
            // is an integer in [0, 255], so FP16 is an exact intermediate.
            AscendC::Cast(packedHalf, gathered,
                          AscendC::RoundMode::CAST_NONE, tiling_.k);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(encoded, packedHalf,
                          AscendC::RoundMode::CAST_NONE, tiling_.k);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        }

        scaleLocal.SetValue(0, scale);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::DataCopyExtParams outputCopy{1, tiling_.k, 0, 0, 0};
        AscendC::DataCopyPad(dataGm_[rowOffset], encoded, outputCopy);
        AscendC::DataCopyExtParams scaleCopy{1, sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(tokenScaleGm_[row * kScaleStride], scaleLocal,
                             scaleCopy);
        // Both writes above read buffers that the next row reuses on the V
        // pipe.  V_MTE3 only orders production before the DMA starts; without
        // the reverse hand-off, the following ReduceMax/Cast can overwrite
        // scaleLocal/encoded while MTE3 is still reading them.  This showed up
        // as isolated corrupt FP32 scale bytes in large batched KV writes.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    Hif8DynamicQuantTilingData tiling_{};
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<InputType> xGm_;
    AscendC::GlobalTensor<uint8_t> dataGm_;
    AscendC::GlobalTensor<float> tokenScaleGm_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> inputBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> normalizedBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> absBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> encodedBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> gatheredBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> encodeTableBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scaleBuffer_;
};

template <typename InputType>
__aicore__ inline void DynamicQuant(
    GM_ADDR x, GM_ADDR data, GM_ADDR tokenScale, GM_ADDR workspace,
    const Hif8DynamicQuantTilingData &tiling) {
    Hif8DynamicQuantKernel<InputType> kernel;
    kernel.Init(x, data, tokenScale, workspace, tiling);
    kernel.Process();
}

}  // namespace vllm_ascend_hif8

extern "C" __global__ __aicore__ void hif8_dynamic_quant(
    GM_ADDR x, GM_ADDR data, GM_ADDR token_scale, GM_ADDR workspace,
    GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(Hif8DynamicQuantTilingData);
    GET_TILING_DATA(tilingData, tiling);
    vllm_ascend_hif8::DynamicQuant<DTYPE_X>(
        x, data, token_scale, workspace, tilingData);
}
