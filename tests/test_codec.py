import math
import unittest

import torch

from vllm_ascend_hif8.codec import (
    ASCEND_DECODE_TABLE,
    ASCEND_NAN,
    ASCEND_NINF,
    ASCEND_PINF,
    ASCEND_ZERO,
    MAX_FINITE,
    OVERFLOW_TIE,
    POSITIVE_FINITE,
    UNDERFLOW_TIE,
    decode_ascend,
    encode_ascend,
    quantize_rows,
)
from vllm_ascend_hif8.tablegen import (
    _signed_encode_lut,
    _static_signed_encode_lut,
)


class HiFloat8CodecTest(unittest.TestCase):
    def test_special_values(self):
        self.assertEqual(encode_ascend(0.0), ASCEND_ZERO)
        self.assertEqual(encode_ascend(-0.0), ASCEND_ZERO)
        self.assertEqual(encode_ascend(math.nan), ASCEND_NAN)
        self.assertEqual(encode_ascend(math.inf), ASCEND_PINF)
        self.assertEqual(encode_ascend(-math.inf), ASCEND_NINF)
        self.assertTrue(math.isnan(decode_ascend(ASCEND_NAN)))
        self.assertEqual(decode_ascend(ASCEND_PINF), math.inf)
        self.assertEqual(decode_ascend(ASCEND_NINF), -math.inf)

    def test_all_finite_codes_round_trip(self):
        for value, positive_code in POSITIVE_FINITE:
            self.assertEqual(encode_ascend(value), positive_code)
            self.assertEqual(decode_ascend(positive_code), value)
            if value != 0.0:
                negative_code = positive_code | 0x80
                self.assertEqual(encode_ascend(-value), negative_code)
                self.assertEqual(decode_ascend(negative_code), -value)

    def test_every_byte_has_defined_decode(self):
        self.assertEqual(len(ASCEND_DECODE_TABLE), 256)
        for byte in range(256):
            value = decode_ascend(byte)
            if byte == ASCEND_NAN:
                self.assertTrue(math.isnan(value))
            else:
                self.assertFalse(math.isnan(value))

    def test_round_half_away_from_zero(self):
        lower = 1.0
        upper = 1.125
        tie = (lower + upper) / 2.0
        self.assertEqual(decode_ascend(encode_ascend(tie)), upper)
        self.assertEqual(decode_ascend(encode_ascend(-tie)), -upper)

    def test_underflow_and_overflow_ties(self):
        self.assertEqual(encode_ascend(math.nextafter(UNDERFLOW_TIE, 0.0)), ASCEND_ZERO)
        self.assertEqual(decode_ascend(encode_ascend(UNDERFLOW_TIE)), 2.0**-22)
        self.assertEqual(decode_ascend(encode_ascend(math.nextafter(OVERFLOW_TIE, 0.0))), MAX_FINITE)
        self.assertEqual(encode_ascend(OVERFLOW_TIE), ASCEND_PINF)

    def test_default_scaling_maps_amax_to_one(self):
        quantized = quantize_rows([[0.0, -2.0, 1.0], [0.0, 0.0, 0.0]])
        self.assertEqual(quantized.scale[0], 2.0)
        self.assertAlmostEqual(decode_ascend(quantized.data[0][1]), -1.0)
        self.assertGreater(quantized.scale[1], 0.0)

    def test_static_encode_lut_only_saturates_finite_overflow(self):
        regular = _signed_encode_lut()
        static = _static_signed_encode_lut()
        self.assertEqual(len(static), 8192)
        for index, (original, saturated) in enumerate(zip(regular, static, strict=True)):
            exponent = (index & 0x0FFF) >> 4
            if exponent == 0xFF:
                self.assertEqual(saturated, original)
            elif (original & 0x7F) == ASCEND_PINF:
                self.assertEqual(saturated & 0x7F, 0x6E)
                self.assertEqual(saturated & 0x80, original & 0x80)
            else:
                self.assertEqual(saturated, original)


if __name__ == "__main__":
    unittest.main()
