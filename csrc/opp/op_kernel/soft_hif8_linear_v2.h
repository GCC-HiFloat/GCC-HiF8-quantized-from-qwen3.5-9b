#pragma once

#include "hif8_codec.h"
#include "kernel_operator.h"

namespace vllm_ascend_hif8 {

// Stage 2 minimal: DataCopy for batch GM->UB transfer, keep scalar decode loop.
// Eliminates per-element GetValue from GM, reduces memory latency.
constexpr uint32_t kUbTileK = 1024;

template <typename OutputType, typename TilingData>
__aicore__ inline void SoftLinearV2(GM_ADDR activation, GM_ADDR weight, GM_ADDR tokenScale,
                                    GM_ADDR channelScale, GM_ADDR bias, GM_ADDR output,
                                    const TilingData &tiling) {
    AscendC::GlobalTensor<uint8_t> activationGm;
    AscendC::GlobalTensor<uint8_t> weightGm;
    AscendC::GlobalTensor<float> tokenScaleGm;
    AscendC::GlobalTensor<float> channelScaleGm;
    AscendC::GlobalTensor<uint16_t> biasGm;
    AscendC::GlobalTensor<OutputType> outputGm;
    activationGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(activation));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(weight));
    tokenScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tokenScale));
    channelScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(channelScale));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutputType *>(output));
    if (tiling.hasBias != 0) {
        biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(bias));
    }

    // Allocate UB tile buffers for batch copy.
    AscendC::LocalTensor<uint8_t> activationTile = AscendC::LocalTensor<uint8_t>();
    AscendC::LocalTensor<uint8_t> weightTile = AscendC::LocalTensor<uint8_t>();
    activationTile.SetSize(kUbTileK);
    weightTile.SetSize(kUbTileK);

    const uint64_t outputElements = static_cast<uint64_t>(tiling.m) * tiling.n;
    constexpr uint32_t kGlobalWriteAlignBytes = 64;
    constexpr uint32_t elementsPerGroup = kGlobalWriteAlignBytes / sizeof(OutputType);
    const uint64_t groups = (outputElements + elementsPerGroup - 1) / elementsPerGroup;
    const uint64_t cores = AscendC::GetBlockNum();
    const uint64_t groupsPerCore = (groups + cores - 1) / cores;
    const uint64_t begin = AscendC::GetBlockIdx() * groupsPerCore * elementsPerGroup;
    const uint64_t rawEnd = begin + groupsPerCore * elementsPerGroup;
    const uint64_t end = rawEnd < outputElements ? rawEnd : outputElements;

    for (uint64_t outputIndex = begin; outputIndex < end; ++outputIndex) {
        const uint32_t row = outputIndex / tiling.n;
        const uint32_t channel = outputIndex % tiling.n;
        float accumulator = 0.0f;
        const uint64_t activationOffset = static_cast<uint64_t>(row) * tiling.k;
        const uint64_t weightOffset = static_cast<uint64_t>(channel) * tiling.k;

        // Process K dimension in tiles: DataCopy GM->UB, then scalar decode loop.
        for (uint32_t kStart = 0; kStart < tiling.k; kStart += kUbTileK) {
            const uint32_t tileSize = (kStart + kUbTileK <= tiling.k)
                                          ? kUbTileK
                                          : (tiling.k - kStart);

            // Batch copy tiles from GM to UB.
            // Create offset GlobalTensors by advancing base pointer
            AscendC::GlobalTensor<uint8_t> activationSrc = activationGm;
            activationSrc.SetGlobalBuffer((__gm__ uint8_t *)(activation) + activationOffset + kStart);
            AscendC::GlobalTensor<uint8_t> weightSrc = weightGm;
            weightSrc.SetGlobalBuffer((__gm__ uint8_t *)(weight) + weightOffset + kStart);

            AscendC::DataCopy(activationTile, activationSrc, tileSize);
            AscendC::DataCopy(weightTile, weightSrc, tileSize);

            // Scalar decode and accumulate from UB (faster than GM GetValue).
            for (uint32_t i = 0; i < tileSize; ++i) {
                accumulator += Decode(activationTile.GetValue(i)) * Decode(weightTile.GetValue(i));
            }
        }

        accumulator *= tokenScaleGm.GetValue(row) * channelScaleGm.GetValue(channel);
        if (tiling.hasBias != 0) {
            const uint16_t biasPayload = biasGm.GetValue(channel);
            if (tiling.biasDtype == 1) {
                const bfloat16_t biasValue =
                    AscendC::GetScalarBitcodeValue<uint16_t, bfloat16_t>(biasPayload);
                accumulator += DeviceToFloat(biasValue);
            } else {
                const half biasValue =
                    AscendC::GetScalarBitcodeValue<uint16_t, half>(biasPayload);
                accumulator += static_cast<float>(biasValue);
            }
        }
        outputGm.SetValue(outputIndex, FloatToDevice<OutputType>(accumulator));
    }
}

}  // namespace vllm_ascend_hif8
