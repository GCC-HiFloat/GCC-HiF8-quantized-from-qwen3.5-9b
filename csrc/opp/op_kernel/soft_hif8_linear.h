#pragma once

#include "hif8_codec.h"
#include "kernel_operator.h"

namespace vllm_ascend_hif8 {

// Scalar SetValue writes go through the per-core data cache, which reads,
// modifies and writes back a whole cache line. Two cores touching one line race
// and one core's half is lost, so each core owns a whole number of cache lines.
constexpr uint32_t kGlobalWriteAlignBytes = 64;

__aicore__ inline float DecodeBias(uint16_t payload, uint32_t biasDtype) {
    if (biasDtype == 1) {
        const bfloat16_t value =
            AscendC::GetScalarBitcodeValue<uint16_t, bfloat16_t>(payload);
        return DeviceToFloat(value);
    }
    const half value = AscendC::GetScalarBitcodeValue<uint16_t, half>(payload);
    return static_cast<float>(value);
}

template <typename OutputType, typename TilingData>
__aicore__ inline void SoftLinear(GM_ADDR activation, GM_ADDR weight, GM_ADDR tokenScale,
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

    const uint64_t outputElements = static_cast<uint64_t>(tiling.m) * tiling.n;
    // Split into whole groups so a group never straddles two cores; the tail
    // group is short when the element count is not a multiple of the group.
    const uint32_t elementsPerGroup = kGlobalWriteAlignBytes / sizeof(OutputType);
    const uint64_t groups = (outputElements + elementsPerGroup - 1) / elementsPerGroup;
    const uint64_t cores = AscendC::GetBlockNum();
    const uint64_t groupsPerCore = (groups + cores - 1) / cores;
    const uint64_t begin =
        AscendC::GetBlockIdx() * groupsPerCore * elementsPerGroup;
    const uint64_t rawEnd = begin + groupsPerCore * elementsPerGroup;
    const uint64_t end = rawEnd < outputElements ? rawEnd : outputElements;

    for (uint64_t outputIndex = begin; outputIndex < end; ++outputIndex) {
        const uint32_t row = outputIndex / tiling.n;
        const uint32_t channel = outputIndex % tiling.n;
        float accumulator = 0.0f;
        const uint64_t activationOffset = static_cast<uint64_t>(row) * tiling.k;
        const uint64_t weightOffset = static_cast<uint64_t>(channel) * tiling.k;
        for (uint32_t kk = 0; kk < tiling.k; ++kk) {
            accumulator += Decode(activationGm.GetValue(activationOffset + kk)) *
                           Decode(weightGm.GetValue(weightOffset + kk));
        }
        accumulator *= tokenScaleGm.GetValue(row) * channelScaleGm.GetValue(channel);
        if (tiling.hasBias != 0) {
            accumulator += DecodeBias(biasGm.GetValue(channel), tiling.biasDtype);
        }
        outputGm.SetValue(outputIndex, FloatToDevice<OutputType>(accumulator));
    }
}

}  // namespace vllm_ascend_hif8
