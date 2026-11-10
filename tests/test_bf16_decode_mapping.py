import unittest

from vllm_ascend_hif8.codec import (
    ASCEND_DECODE_TABLE,
    decode_ascend_bf16_bits_arithmetic,
)
from vllm_ascend_hif8.tablegen import _float_to_bf16_bits


class HiFloat8Bf16MappingTest(unittest.TestCase):
    def test_arithmetic_mapping_matches_generated_lut_for_all_bytes(self):
        for byte, decoded in enumerate(ASCEND_DECODE_TABLE):
            with self.subTest(byte=byte):
                expected = _float_to_bf16_bits(decoded)
                actual = decode_ascend_bf16_bits_arithmetic(byte)
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
