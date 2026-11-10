import types
import unittest
from unittest.mock import patch

from vllm_ascend_hif8 import native


class NativeLoaderTest(unittest.TestCase):
    def setUp(self):
        self.configure_opp_path = patch("vllm_ascend_hif8.runtime.configure_opp_path")
        self.configure_opp_path.start()
        self.addCleanup(self.configure_opp_path.stop)
        # Clear cached handles so load_extension() exercises the mocked importer.
        native.reset_extension_cache()

    def tearDown(self):
        native.reset_extension_cache()

    def test_missing_extension_raises_without_fallback(self):
        with patch("importlib.import_module", side_effect=ModuleNotFoundError("missing")):
            with self.assertRaisesRegex(native.NativeExtensionUnavailable, "no decoded-weight or floating-point fallback"):
                native.load_extension()

    def test_missing_symbol_is_rejected(self):
        incomplete = types.SimpleNamespace(hif8_dynamic_quant=lambda value: value)
        with patch("importlib.import_module", return_value=incomplete):
            with self.assertRaisesRegex(native.NativeExtensionUnavailable, "soft_hif8_gemv"):
                native.load_extension()

    def test_valid_extension_is_cached(self):
        complete = types.SimpleNamespace(
            hif8_dynamic_quant=lambda value: value,
            soft_hif8_gemv=lambda *args: args,
            soft_hif8_gemm=lambda *args: args,
        )
        with patch("importlib.import_module", return_value=complete) as importer:
            self.assertIs(native.load_extension(), complete)
            self.assertIs(native.load_extension(), complete)
            importer.assert_called_once()


class NativeOperandNormalizationTest(unittest.TestCase):
    """The native bridge rejects offset or strided tensors, so soft_linear must
    normalize whatever ``reshape`` hands it."""

    def setUp(self):
        self.torch = __import__("torch")

    def test_offset_and_strided_views_are_materialized(self):
        torch = self.torch
        base = torch.arange(24, dtype=torch.uint8).reshape(6, 4)

        offset_view = base[2:]
        self.assertTrue(offset_view.is_contiguous())
        self.assertGreater(offset_view.storage_offset(), 0)
        normalized = native._as_native_operand(offset_view)
        self.assertEqual(normalized.storage_offset(), 0)
        self.assertTrue(normalized.is_contiguous())
        self.assertTrue(bool(torch.equal(normalized, offset_view)))

        strided_view = base[:, ::2]
        self.assertFalse(strided_view.is_contiguous())
        normalized = native._as_native_operand(strided_view)
        self.assertEqual(normalized.storage_offset(), 0)
        self.assertTrue(normalized.is_contiguous())
        self.assertTrue(bool(torch.equal(normalized, strided_view)))

    def test_clean_tensors_pass_through_without_a_copy(self):
        clean = self.torch.zeros((3, 4), dtype=self.torch.uint8)
        self.assertIs(native._as_native_operand(clean), clean)

    def test_scaled_decode_materializes_inline_sidecar_views(self):
        torch = self.torch
        seen = {}

        def fake_decode(data, scale):
            seen["data_offset"] = data.storage_offset()
            seen["data_contiguous"] = data.is_contiguous()
            seen["scale_offset"] = scale.storage_offset()
            seen["scale_contiguous"] = scale.is_contiguous()
            return torch.zeros(data.shape, dtype=torch.bfloat16)

        extension = types.SimpleNamespace(
            hif8_dynamic_quant=lambda value: value,
            hif8_scaled_decode=fake_decode,
            soft_hif8_gemv=lambda *args: args,
            soft_hif8_gemm=lambda *args: args,
        )
        # KV rows append a four-byte FP32 scale sidecar.  Removing that
        # sidecar leaves a logically dense matrix whose row stride is wider
        # than K, which the ACL bridge cannot consume directly.
        packed = torch.arange(4 * 36, dtype=torch.uint8).reshape(4, 36)
        data = packed[:, :32]
        self.assertFalse(data.is_contiguous())
        scale = torch.ones(4, dtype=torch.float32)

        with (
            patch.object(native, "load_extension", return_value=extension),
            patch.object(
                torch.ops.vllm_ascend_hif8,
                "hif8_scaled_decode",
                fake_decode,
                create=True,
            ),
        ):
            output = native.scaled_decode(data, scale)

        self.assertEqual(seen["data_offset"], 0)
        self.assertTrue(seen["data_contiguous"])
        self.assertEqual(seen["scale_offset"], 0)
        self.assertTrue(seen["scale_contiguous"])
        self.assertEqual(tuple(output.shape), (4, 32))

    def test_soft_linear_feeds_the_operator_clean_operands(self):
        torch = self.torch
        seen = {}

        def fake_gemm(activation, weight, token_scale, channel_scale, bias, output_dtype):
            seen["offsets"] = [tensor.storage_offset() for tensor in (activation, token_scale, channel_scale)]
            seen["contiguous"] = [tensor.is_contiguous() for tensor in (activation, token_scale, channel_scale)]
            return torch.zeros((activation.shape[0], weight.shape[0]), dtype=output_dtype)

        extension = types.SimpleNamespace(
            hif8_dynamic_quant=lambda value: value,
            soft_hif8_gemv=fake_gemm,
            soft_hif8_gemm=fake_gemm,
            soft_hif8_fused_gemm=fake_gemm,
        )
        # A 3D activation whose token rows start at a nonzero storage offset.
        activation = torch.zeros((5, 2, 4), dtype=torch.uint8)[1:]
        scale = torch.ones(10, dtype=torch.float32)[2:]
        weight = torch.zeros((3, 4), dtype=torch.uint8)
        weight_scale = torch.ones(3, dtype=torch.float32)

        with (
            patch.object(native, "load_extension", return_value=extension),
            patch.object(torch.ops.vllm_ascend_hif8, "soft_hif8_gemm", fake_gemm, create=True),
        ):
            output = native.soft_linear(activation, weight, scale, weight_scale, None, output_dtype=torch.float32)

        self.assertEqual(seen["offsets"], [0, 0, 0])
        self.assertTrue(all(seen["contiguous"]))
        self.assertEqual(tuple(output.shape), (4, 2, 3))

    def test_soft_linear_routes_small_rows_by_benchmarked_shape(self):
        torch = self.torch
        calls = []

        def fake_gemv(activation, weight, *args):
            calls.append("gemv")
            return torch.zeros((activation.shape[0], weight.shape[0]), dtype=torch.bfloat16)

        def fake_gemm(activation, weight, *args):
            calls.append("gemm")
            return torch.zeros((activation.shape[0], weight.shape[0]), dtype=torch.bfloat16)

        extension = types.SimpleNamespace(
            hif8_dynamic_quant=lambda value: value,
            soft_hif8_gemv=fake_gemv,
            soft_hif8_gemm=fake_gemm,
            soft_hif8_fused_gemm=fake_gemm,
        )
        scale = torch.ones(2, dtype=torch.float32)
        channel_scale = torch.ones(3, dtype=torch.float32)
        with (
            patch.object(native, "load_extension", return_value=extension),
            patch.object(torch.ops.vllm_ascend_hif8, "soft_hif8_gemv", fake_gemv, create=True),
            patch.object(torch.ops.vllm_ascend_hif8, "soft_hif8_gemm", fake_gemm, create=True),
        ):
            native.soft_linear(
                torch.zeros((1, 4), dtype=torch.uint8),
                torch.zeros((3, 4), dtype=torch.uint8),
                scale[:1], channel_scale, output_dtype=torch.bfloat16,
            )
            native.soft_linear(
                torch.zeros((2, 4), dtype=torch.uint8),
                torch.zeros((3, 4), dtype=torch.uint8),
                scale, channel_scale, output_dtype=torch.bfloat16,
            )
            native.soft_linear(
                torch.zeros((1, 4), dtype=torch.uint8),
                torch.zeros((1_536, 4), dtype=torch.uint8),
                scale[:1], torch.ones(1_536, dtype=torch.float32),
                output_dtype=torch.bfloat16,
            )
            native.soft_linear(
                torch.zeros((1, 4), dtype=torch.uint8),
                torch.zeros((1_664, 4), dtype=torch.uint8),
                scale[:1], torch.ones(1_664, dtype=torch.float32),
                output_dtype=torch.bfloat16,
            )
            native.soft_linear(
                torch.zeros((1, 8_192), dtype=torch.uint8),
                torch.zeros((512, 8_192), dtype=torch.uint8),
                scale[:1], torch.ones(512, dtype=torch.float32),
                output_dtype=torch.bfloat16,
            )
            native.soft_linear(
                torch.zeros((1, 12_288), dtype=torch.uint8),
                torch.zeros((512, 12_288), dtype=torch.uint8),
                scale[:1], torch.ones(512, dtype=torch.float32),
                output_dtype=torch.bfloat16,
            )

        self.assertEqual(
            calls, ["gemv", "gemm", "gemv", "gemm", "gemv", "gemm"]
        )

    def test_static_fused_linear_feeds_clean_fixed_scale(self):
        torch = self.torch
        seen = {}

        def fake_static_fused(
            activation,
            weight,
            fixed_scale,
            channel_scale,
            bias,
            output_dtype,
        ):
            seen["shapes"] = (
                tuple(activation.shape),
                tuple(weight.shape),
                tuple(fixed_scale.shape),
                tuple(channel_scale.shape),
            )
            seen["offsets"] = (
                activation.storage_offset(),
                fixed_scale.storage_offset(),
                channel_scale.storage_offset(),
            )
            return torch.zeros(
                (activation.shape[0], weight.shape[0]), dtype=output_dtype
            )

        extension = types.SimpleNamespace(
            hif8_dynamic_quant=lambda value: value,
            soft_hif8_gemv=lambda *args: args,
            soft_hif8_gemm=lambda *args: args,
            soft_hif8_static_fused_gemm=fake_static_fused,
        )
        activation = torch.zeros((3, 1, 4), dtype=torch.bfloat16)[1:]
        weight = torch.zeros((5, 4), dtype=torch.uint8)
        fixed_scale = torch.ones(3, dtype=torch.float32)[1:2]
        channel_scale = torch.ones(7, dtype=torch.float32)[1:6]

        with (
            patch.object(native, "load_extension", return_value=extension),
            patch.object(
                torch.ops.vllm_ascend_hif8,
                "soft_hif8_static_fused_gemm",
                fake_static_fused,
                create=True,
            ),
        ):
            output = native.soft_static_fused_linear(
                activation,
                weight,
                fixed_scale,
                channel_scale,
                output_dtype=torch.bfloat16,
            )

        self.assertEqual(seen["shapes"], ((2, 4), (5, 4), (1,), (5,)))
        self.assertEqual(seen["offsets"], (0, 0, 0))
        self.assertEqual(tuple(output.shape), (2, 1, 5))


if __name__ == "__main__":
    unittest.main()
