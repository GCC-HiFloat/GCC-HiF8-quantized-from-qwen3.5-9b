import unittest

from vllm_ascend_hif8.codec import quantize_rows
from vllm_ascend_hif8.reference import linear_u8_u8


class ReferenceLinearTest(unittest.TestCase):
    def test_exact_small_matrix(self):
        activation = quantize_rows([[1.0, 0.5], [-1.0, 0.25]])
        weight = quantize_rows([[1.0, 1.0], [0.5, -1.0]])
        output = linear_u8_u8(activation, weight, bias=[0.25, -0.5])
        self.assertEqual(output, ((1.75, -0.5), (-0.5, -1.25)))

    def test_rejects_incompatible_shapes(self):
        activation = quantize_rows([[1.0, 2.0]])
        weight = quantize_rows([[1.0]])
        with self.assertRaisesRegex(ValueError, "incompatible K"):
            linear_u8_u8(activation, weight)


if __name__ == "__main__":
    unittest.main()
