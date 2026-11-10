#include "hif8_codec.h"
#include "hif8_scaled_decode_tiling.h"
#include "kernel_operator.h"

namespace vllm_ascend_hif8 {

constexpr uint32_t kDecodeTileElements = 4096;
constexpr uint32_t kCopyAlignElements = 32;

class ScaledDecodeKernel {
public:
    __aicore__ inline void Init(GM_ADDR data, GM_ADDR scale, GM_ADDR output,
                                const Hif8ScaledDecodeTilingData &tiling) {
        tiling_ = tiling;
        dataGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(data));
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scale));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output));

        const uint32_t maxTileSize = kDecodeTileElements < tiling_.k
                                      ? kDecodeTileElements
                                      : tiling_.k;
        pipe_.InitBuffer(dataBuffer_, maxTileSize * sizeof(uint8_t));
        pipe_.InitBuffer(floatBuffer_, maxTileSize * sizeof(float));
        pipe_.InitBuffer(outputBuffer_, maxTileSize * sizeof(bfloat16_t));
        pipe_.InitBuffer(halfIndexBuffer_, maxTileSize * sizeof(half));
        pipe_.InitBuffer(indexBuffer_, maxTileSize * sizeof(uint32_t));
        pipe_.InitBuffer(tableBuffer_, 256 * sizeof(float));
    }

    __aicore__ inline void Process() {
        AscendC::LocalTensor<uint8_t> dataLocal = dataBuffer_.Get<uint8_t>();
        AscendC::LocalTensor<float> floatLocal = floatBuffer_.Get<float>();
        AscendC::LocalTensor<bfloat16_t> outputLocal =
            outputBuffer_.Get<bfloat16_t>();
        auto table = tableBuffer_.Get<float>();
        for (uint32_t byte = 0; byte < 256; ++byte) {
            table.SetValue(byte, Decode(static_cast<uint8_t>(byte)));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);

        // Assign whole rows directly. DMA writes are used for aligned spans;
        // scalar tails are safe because no two cores own the same row.
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t rowsPerCore = (tiling_.rows + cores - 1) / cores;
        const uint32_t begin = AscendC::GetBlockIdx() * rowsPerCore;
        const uint32_t rawEnd = begin + rowsPerCore;
        const uint32_t end = rawEnd < tiling_.rows ? rawEnd : tiling_.rows;

        for (uint32_t row = begin; row < end; ++row) {
            DecodeRow(row, dataLocal, floatLocal, outputLocal);
        }
    }

private:
    __aicore__ inline void DecodeRow(
        uint32_t row, AscendC::LocalTensor<uint8_t> dataLocal,
        AscendC::LocalTensor<float> floatLocal,
        AscendC::LocalTensor<bfloat16_t> outputLocal) {
        const uint64_t rowOffset = static_cast<uint64_t>(row) * tiling_.k;
        const float scale = scaleGm_.GetValue(row);
        const bfloat16_t bf16Nan =
            AscendC::GetScalarBitcodeValue<uint16_t, bfloat16_t>(0x7fc0u);
        uint32_t kStart = 0;

        // DataCopy requires a block-aligned GM address. Production model K
        // dimensions satisfy this; use the scalar path for later rows of an
        // unaligned test shape rather than reading across row boundaries.
        if (rowOffset % kCopyAlignElements != 0) {
            for (; kStart < tiling_.k; ++kStart) {
                const uint8_t byte = dataGm_.GetValue(rowOffset + kStart);
                outputGm_.SetValue(
                    rowOffset + kStart,
                    byte == kNan ? bf16Nan : FloatToDevice<bfloat16_t>(Decode(byte) * scale));
            }
            return;
        }

        while (tiling_.k - kStart >= kCopyAlignElements) {
            uint32_t count = tiling_.k - kStart;
            if (count > kDecodeTileElements) {
                count = kDecodeTileElements;
            }
            count -= count % kCopyAlignElements;

            AscendC::DataCopy(dataLocal, dataGm_[rowOffset + kStart], count);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

            // Gather uses byte offsets into the exact FP32 HiFloat8 table.
            auto halfOffsets = halfIndexBuffer_.Get<half>();
            auto offsets = indexBuffer_.Get<uint32_t>();
            auto signedOffsets = offsets.ReinterpretCast<int32_t>();
            AscendC::Cast(halfOffsets, dataLocal,
                          AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(halfOffsets, halfOffsets, static_cast<half>(4.0), count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(signedOffsets, halfOffsets,
                          AscendC::RoundMode::CAST_RINT, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(floatLocal, tableBuffer_.Get<float>(), offsets, 0, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(floatLocal, floatLocal, scale, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outputLocal, floatLocal,
                          AscendC::RoundMode::CAST_RINT, count);
            // The next DMA read must not overwrite bytes still used by Cast.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopy(outputGm_[rowOffset + kStart], outputLocal, count);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            kStart += count;
        }

        for (; kStart < tiling_.k; ++kStart) {
            const uint8_t byte = dataGm_.GetValue(rowOffset + kStart);
            outputGm_.SetValue(
                rowOffset + kStart,
                byte == kNan ? bf16Nan : FloatToDevice<bfloat16_t>(Decode(byte) * scale));
        }
    }

    Hif8ScaledDecodeTilingData tiling_{};
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dataBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfIndexBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tableBuffer_;
    AscendC::GlobalTensor<uint8_t> dataGm_;
    AscendC::GlobalTensor<float> scaleGm_;
    AscendC::GlobalTensor<bfloat16_t> outputGm_;
};

}  // namespace vllm_ascend_hif8

extern "C" __global__ __aicore__ void hif8_scaled_decode(
    GM_ADDR data, GM_ADDR scale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(Hif8ScaledDecodeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    vllm_ascend_hif8::ScaledDecodeKernel kernel;
    kernel.Init(data, scale, y, tilingData);
    kernel.Process();
}
