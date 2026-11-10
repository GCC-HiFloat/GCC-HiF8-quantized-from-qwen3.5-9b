import os
import unittest
from unittest.mock import patch

import torch

from vllm_ascend_hif8.codec import ASCEND_DECODE_TABLE, encode_ascend
from vllm_ascend_hif8.scheme import AscendW8A8HiFloat8SoftLinearMethod


class QuantizedEmbeddingTest(unittest.TestCase):
    def _layer(self):
        values = torch.tensor(
            [
                [-1.0, -0.5, 0.0, 0.5],
                [-0.5, 0.0, 0.5, 1.0],
                [0.25, -0.25, 0.75, -0.75],
            ],
            dtype=torch.float32,
        )
        data = torch.tensor(
            [[encode_ascend(float(value)) for value in row] for row in values],
            dtype=torch.uint8,
        )
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(data, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            torch.tensor([[2.0], [3.0], [4.0]], dtype=torch.float32),
            requires_grad=False,
        )
        decode = torch.tensor(ASCEND_DECODE_TABLE, dtype=torch.float32)
        expected_weight = decode[data.long()] * layer.weight_scale.float()
        return layer, expected_weight

    def test_soft_backend_gathers_only_selected_hif8_rows(self):
        layer, expected_weight = self._layer()
        method = AscendW8A8HiFloat8SoftLinearMethod()
        with patch.dict(os.environ, {"HIF8_COMPUTE_BACKEND": "soft_hif8"}):
            method.process_weights_after_loading(layer)

        ids = torch.tensor([[2, 0], [1, 2]])
        actual = method.embedding(layer, ids)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(layer._hif8_decode_lut.numel(), 256)
        self.assertFalse(hasattr(layer, "_decoded_weight_cache"))
        torch.testing.assert_close(
            actual.float(), expected_weight[ids].to(torch.bfloat16).float(), rtol=0, atol=0
        )

    def test_soft_hif8_is_the_default_and_keeps_checkpoint_storage(self):
        layer, _ = self._layer()
        storage_pointer = layer.weight.untyped_storage().data_ptr()
        method = AscendW8A8HiFloat8SoftLinearMethod()
        with patch.dict(os.environ, {}, clear=True):
            method.process_weights_after_loading(layer)

        self.assertEqual(layer._hif8_compute_backend, "soft_hif8")
        self.assertEqual(layer.weight.dtype, torch.uint8)
        self.assertEqual(layer.weight.untyped_storage().data_ptr(), storage_pointer)
        self.assertTrue(hasattr(layer, "_hif8_decode_lut"))
        self.assertFalse(hasattr(layer, "_hif8_weight_kn"))


if __name__ == "__main__":
    unittest.main()
