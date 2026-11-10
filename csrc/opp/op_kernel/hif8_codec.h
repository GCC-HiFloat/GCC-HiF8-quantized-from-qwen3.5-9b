#pragma once

#include "hif8_tables.h"
#include "kernel_operator.h"

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

#ifndef NAN
#define NAN (__builtin_nanf(""))
#endif

namespace vllm_ascend_hif8 {

constexpr uint8_t kSignMask = 0x80;
constexpr uint8_t kMagnitudeMask = 0x7f;
constexpr uint8_t kNan = kAscendNan;
constexpr uint8_t kPositiveInfinity = kAscendPositiveInfinity;
constexpr float kScaleEpsilon = 1.0e-12f;

// The encode table is indexed by the top 12 bits of the FP32 bit pattern:
// the 8 exponent bits followed by the 4 leading mantissa bits.
constexpr uint32_t kEncodeMantissaShift = 19;
constexpr uint32_t kEncodeExponentShift = 23;
constexpr uint32_t kEncodeMantissaMask = 0x007fffffu;
constexpr uint32_t kEncodeExponentMask = 0xffu;
constexpr uint32_t kEncodeSpecialExponent = 0xffu;

template <typename T>
__aicore__ inline float DeviceToFloat(T value) {
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        return AscendC::Cast(value);
    }
    return static_cast<float>(value);
}

template <typename T>
__aicore__ inline T FloatToDevice(float value) {
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        return AscendC::Cast(value);
    }
    return static_cast<T>(value);
}

__aicore__ inline float DecodeMagnitude(uint8_t code) {
    return kAscendDecodeMagnitude[code & kMagnitudeMask];
}

__aicore__ inline float Decode(uint8_t byte) {
    if (byte == kNan) {
        return NAN;
    }
    const float magnitude = kAscendDecodeMagnitude[byte & kMagnitudeMask];
    return (byte & kSignMask) != 0 ? -magnitude : magnitude;
}

__aicore__ inline uint8_t Encode(float value) {
    const uint32_t bits = AscendC::GetScalarBitcodeValue<float, uint32_t>(value);
    const uint32_t exponent = (bits >> kEncodeExponentShift) & kEncodeExponentMask;
    const uint32_t mantissa = bits & kEncodeMantissaMask;
    const uint8_t sign = static_cast<uint8_t>((bits >> 24) & kSignMask);

    if (exponent == kEncodeSpecialExponent) {
        return mantissa != 0 ? kNan : static_cast<uint8_t>(kPositiveInfinity | sign);
    }
    const uint32_t index = exponent * 16u + (mantissa >> kEncodeMantissaShift);
    const uint8_t magnitude = kFloat32EncodeMagnitude[index];
    // Zero has no signed encoding, so a rounded-to-zero magnitude drops the sign.
    return magnitude == 0 ? 0 : static_cast<uint8_t>(magnitude | sign);
}

// The FP32 index is the eight-bit exponent followed by the four leading
// mantissa bits.  This table returns the final decoded HiFloat8 value in a
// BF16 bit pattern, so the fused path can encode and decode an activation
// without scalar GetValue/SetValue calls or a second byte-address gather.
__aicore__ inline bfloat16_t EncodeDecodeBf16(float value) {
    const uint32_t bits =
        AscendC::GetScalarBitcodeValue<float, uint32_t>(value);
    // Keep the sign as bit 12 of the 13-bit index.  The generated table is
    // signed (8192 entries), unlike the older magnitude-only encode table.
    const uint32_t index = (bits >> kEncodeMantissaShift) & 0x1fffu;
    const uint32_t packed =
        kFloat32EncodeDecodeBf16U16Packed[index >> 1];
    const uint16_t payload =
        static_cast<uint16_t>((index & 1u) != 0 ? (packed >> 16) : packed);
    return AscendC::GetScalarBitcodeValue<uint16_t, bfloat16_t>(payload);
}

__aicore__ inline bool IsFinite(float value) {
    const uint32_t bits = AscendC::GetScalarBitcodeValue<float, uint32_t>(value);
    return ((bits >> kEncodeExponentShift) & kEncodeExponentMask) != kEncodeSpecialExponent;
}

}  // namespace vllm_ascend_hif8
