#include "hif8_codec.h"
#include "hif8_static_quant_tiling.h"
#include "kernel_operator.h"

namespace vllm_ascend_hif8 {

constexpr uint32_t kStaticDmaBlockBytes = 32;
constexpr uint32_t kStaticScaleStride = 64 / sizeof(float);

template <typename InputType>
class Hif8StaticQuantKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR fixedScale, GM_ADDR data, GM_ADDR rowScale,
        GM_ADDR workspace, const Hif8StaticQuantTilingData &tiling) {
        tiling_ = tiling;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputType *>(x));
        fixedScaleGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(fixedScale));
        dataGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(data));
        rowScaleGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(rowScale));
        (void)workspace;

        pipe_.InitBuffer(inputBuffer_, tiling_.k * sizeof(InputType) +
                                           kStaticDmaBlockBytes);
        pipe_.InitBuffer(normalizedBuffer_, tiling_.k * sizeof(float));
        pipe_.InitBuffer(offsetBuffer_, tiling_.k * sizeof(float));
        pipe_.InitBuffer(encodedBuffer_, tiling_.k * sizeof(uint8_t) +
                                             kStaticDmaBlockBytes);
        pipe_.InitBuffer(gatheredBuffer_, tiling_.k * sizeof(uint16_t));
        pipe_.InitBuffer(encodeTableBuffer_,
                         sizeof(kFloat32StaticEncodeSignedU16Packed));
        pipe_.InitBuffer(scaleBuffer_, kStaticDmaBlockBytes);
    }

    __aicore__ inline void Process() {
        const uint32_t block = AscendC::GetBlockIdx();
        const uint32_t blocks = AscendC::GetBlockNum();
        if (tiling_.k % kStaticDmaBlockBytes != 0) {
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
             index < sizeof(kFloat32StaticEncodeSignedU16Packed) /
                         sizeof(uint32_t);
             ++index) {
            table.SetValue(index,
                           kFloat32StaticEncodeSignedU16Packed[index]);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(2);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(2);
    }

    __aicore__ inline uint8_t EncodeSaturating(float value) {
        uint8_t payload = Encode(value);
        if (IsFinite(value) &&
            (payload & kMagnitudeMask) == kPositiveInfinity) {
            payload = static_cast<uint8_t>((payload & kSignMask) | 0x6e);
        }
        return payload;
    }

    __aicore__ inline float ScaleForRow(uint32_t row) {
        return fixedScaleGm_.GetValue(row % tiling_.scaleCount);
    }

    __aicore__ inline void ProcessScalarRow(uint32_t row) {
        const uint64_t rowOffset = static_cast<uint64_t>(row) * tiling_.k;
        const float scale = ScaleForRow(row);
        for (uint32_t kk = 0; kk < tiling_.k; ++kk) {
            const float value = DeviceToFloat(xGm_.GetValue(rowOffset + kk));
            dataGm_.SetValue(rowOffset + kk,
                             EncodeSaturating(value / scale));
        }
        rowScaleGm_.SetValue(row * kStaticScaleStride, scale);
    }

    __aicore__ inline void ProcessRow(uint32_t row) {
        AscendC::LocalTensor<InputType> input = inputBuffer_.Get<InputType>();
        AscendC::LocalTensor<float> normalized =
            normalizedBuffer_.Get<float>();
        AscendC::LocalTensor<uint32_t> encodeOffsets =
            offsetBuffer_.Get<uint32_t>();
        AscendC::LocalTensor<uint8_t> encoded =
            encodedBuffer_.Get<uint8_t>();
        AscendC::LocalTensor<int16_t> gathered =
            gatheredBuffer_.Get<int16_t>();
        AscendC::LocalTensor<int16_t> encodeTable =
            encodeTableBuffer_.Get<int16_t>();
        AscendC::LocalTensor<float> scaleLocal = scaleBuffer_.Get<float>();
        const uint64_t rowOffset = static_cast<uint64_t>(row) * tiling_.k;
        const float scale = ScaleForRow(row);

        AscendC::DataCopyExtParams inputCopy{
            1, static_cast<uint32_t>(tiling_.k * sizeof(InputType)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<InputType> inputPad{true, 0, 0, 0};
        AscendC::DataCopyPad(input, xGm_[rowOffset], inputCopy, inputPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        AscendC::Cast(normalized, input, AscendC::RoundMode::CAST_NONE,
                      tiling_.k);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(normalized, normalized, 1.0f / scale, tiling_.k);
        AscendC::LocalTensor<uint32_t> fp32Bits =
            normalized.ReinterpretCast<uint32_t>();
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(encodeOffsets, fp32Bits,
                            kEncodeMantissaShift, tiling_.k);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(encodeOffsets, encodeOffsets, 1u, tiling_.k);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(gathered, encodeTable, encodeOffsets, 0, tiling_.k);
        AscendC::LocalTensor<half> packedHalf =
            normalizedBuffer_.Get<half>();
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(packedHalf, gathered, AscendC::RoundMode::CAST_NONE,
                      tiling_.k);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(encoded, packedHalf, AscendC::RoundMode::CAST_NONE,
                      tiling_.k);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        scaleLocal.SetValue(0, scale);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::DataCopyExtParams outputCopy{1, tiling_.k, 0, 0, 0};
        AscendC::DataCopyPad(dataGm_[rowOffset], encoded, outputCopy);
        AscendC::DataCopyExtParams scaleCopy{1, sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(rowScaleGm_[row * kStaticScaleStride],
                             scaleLocal, scaleCopy);
        // The next row reuses both UB sources.  Wait for MTE3 to finish
        // reading them before allowing the V pipe to overwrite either one.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    Hif8StaticQuantTilingData tiling_{};
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<InputType> xGm_;
    AscendC::GlobalTensor<float> fixedScaleGm_;
    AscendC::GlobalTensor<uint8_t> dataGm_;
    AscendC::GlobalTensor<float> rowScaleGm_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> inputBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> normalizedBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> offsetBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> encodedBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> gatheredBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> encodeTableBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scaleBuffer_;
};

template <typename InputType>
__aicore__ inline void StaticQuant(
    GM_ADDR x, GM_ADDR fixedScale, GM_ADDR data, GM_ADDR rowScale,
    GM_ADDR workspace, const Hif8StaticQuantTilingData &tiling) {
    Hif8StaticQuantKernel<InputType> kernel;
    kernel.Init(x, fixedScale, data, rowScale, workspace, tiling);
    kernel.Process();
}

}  // namespace vllm_ascend_hif8

extern "C" __global__ __aicore__ void hif8_static_quant(
    GM_ADDR x, GM_ADDR fixed_scale, GM_ADDR data, GM_ADDR row_scale,
    GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(Hif8StaticQuantTilingData);
    GET_TILING_DATA(tilingData, tiling);
    vllm_ascend_hif8::StaticQuant<DTYPE_X>(
        x, fixed_scale, data, row_scale, workspace, tilingData);
}
