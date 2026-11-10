"""Drive real exported HiFloat8 weights through the registered scheme on an NPU.

This closes the loop the unit tests leave open: the exporter's bytes, the
registered vllm-ascend linear scheme, and the native kernels all in one path,
compared against the CPU reference codec. Point ``HIF8_EXPORTED_MODEL`` at an
exported checkpoint to run it.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

RUN_NPU_TESTS = os.environ.get("HIF8_RUN_NPU_TESTS") == "1"
EXPORTED_MODEL = os.environ.get("HIF8_EXPORTED_MODEL")

try:
    import torch
    import torch_npu  # noqa: F401
except ImportError:
    torch = None

# The one-core correctness kernels are slow, so the test uses a real weight
# sliced down rather than a synthetic one: the bytes and scales stay authentic.
ROWS = 128
COLS = 512


@unittest.skipUnless(
    RUN_NPU_TESTS and EXPORTED_MODEL and torch is not None,
    "set HIF8_RUN_NPU_TESTS=1 and HIF8_EXPORTED_MODEL=<exported checkpoint> in a torch-npu environment",
)
class ExportedCheckpointOnNpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("no NPU is available")

        import vllm_ascend_hif8.plugin as plugin
        from vllm_ascend.quantization.methods import get_scheme_class

        plugin.register()
        cls.scheme_class = get_scheme_class("W8A8HIF8_SOFT", "linear")
        cls.model = Path(EXPORTED_MODEL)

    def _load_sliced_weight(self):
        """Return one real quantized weight and its scale, row-sliced.

        Slicing rows keeps every output channel paired with its own scale, so
        the operand stays a valid per-channel-quantized weight.
        """
        from safetensors import safe_open

        from vllm_ascend_hif8.exporter import METADATA_FILENAME, scale_name

        layers = json.loads((self.model / METADATA_FILENAME).read_text(encoding="utf-8"))["layers"]
        name = sorted(layers)[0]

        index_path = self.model / "model.safetensors.index.json"
        if index_path.exists():
            shard = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"][name]
        else:
            shard = "model.safetensors"

        with safe_open(self.model / shard, framework="pt", device="cpu") as handle:
            weight = handle.get_tensor(name)[:ROWS, :COLS].contiguous()
            scale = handle.get_tensor(scale_name(name))[:ROWS].contiguous()
        return name, weight, scale

    def test_registered_scheme_matches_the_cpu_oracle(self):
        from vllm_ascend_hif8.codec import ASCEND_DECODE_TABLE
        from vllm_ascend_hif8.torch_codec import quantize_tensor_rows

        name, weight, scale = self._load_sliced_weight()
        self.assertEqual(weight.dtype, torch.uint8, name)
        self.assertEqual(scale.dtype, torch.float32, name)

        method = self.scheme_class()
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight.npu(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale.npu(), requires_grad=False)
        method.process_weights_after_loading(layer)

        # The only supported runtime keeps one byte per weight and no decoded
        # BF16 copy.
        self.assertEqual(layer.weight.dtype, torch.uint8)
        self.assertEqual(layer.weight.untyped_storage().nbytes(), weight.numel())
        self.assertEqual(tuple(layer.weight_scale.shape), (ROWS,))
        self.assertFalse(hasattr(layer, "_decoded_weight_cache"))
        self.assertEqual(layer.weight_scale.dtype, torch.float32)

        decode = torch.tensor(ASCEND_DECODE_TABLE, dtype=torch.float32)
        weight_f32 = decode[weight.to(torch.int64)] * scale.reshape(-1, 1)

        generator = torch.Generator().manual_seed(0)
        # M=1 normally takes GEMV; M=2 and larger take Cube GEMM.
        for tokens in (1, 2, 7, 32):
            with self.subTest(tokens=tokens):
                source = torch.randn(tokens, COLS, generator=generator, dtype=torch.float32)
                activation = source.to(torch.bfloat16)

                actual = method.apply(layer, activation.npu(), None)
                self.assertEqual(actual.dtype, activation.dtype)
                self.assertEqual(tuple(actual.shape), (tokens, ROWS))

                quantized, token_scale = quantize_tensor_rows(activation.to(torch.float32))
                dequantized = decode[quantized.to(torch.int64)] * token_scale.reshape(-1, 1)
                expected = dequantized @ weight_f32.T

                result = actual.cpu().to(torch.float32)
                denominator = expected.abs().amax().clamp(min=1e-12)
                error = float((result - expected).abs().amax() / denominator)
                similarity = float(
                    torch.nn.functional.cosine_similarity(result.reshape(1, -1), expected.reshape(1, -1))
                )
                self.assertLess(error, 2.0e-2, f"{name} M={tokens}")
                self.assertGreater(similarity, 0.9999, f"{name} M={tokens}")


if __name__ == "__main__":
    unittest.main()
