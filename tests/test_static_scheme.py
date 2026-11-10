from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from vllm_ascend_hif8.scheme import (
    AscendW8A8HiFloat8StaticLinearMethod,
    STATIC_QUANT_TYPE,
)


class StaticLinearSchemeTest(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        os.environ.pop("HIF8_COMPUTE_BACKEND", None)
        os.environ.pop("HIF8_STATIC_FUSED_GENERATION", None)

    def tearDown(self):
        self.environment.stop()

    @staticmethod
    def layer(input_scale: float = 0.25) -> torch.nn.Module:
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.zeros((5, 8), dtype=torch.uint8), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.ones((5, 1), dtype=torch.float32), requires_grad=False
        )
        layer.input_scale = torch.nn.Parameter(
            torch.tensor([input_scale], dtype=torch.float32),
            requires_grad=False,
        )
        return layer

    def test_declares_one_fp32_scale_independent_of_model_dtype(self):
        method = AscendW8A8HiFloat8StaticLinearMethod()
        params = method.get_pertensor_param(torch.bfloat16, layer_type="others")
        self.assertEqual(set(params), {"input_scale"})
        self.assertEqual(params["input_scale"].dtype, torch.float32)
        self.assertEqual(tuple(params["input_scale"].shape), (1,))

    def test_rejects_invalid_or_non_scalar_checkpoint_scales(self):
        method = AscendW8A8HiFloat8StaticLinearMethod()
        for value in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite and greater"):
                    method.process_weights_after_loading(self.layer(value))

        layer = self.layer()
        layer.input_scale = torch.nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=False
        )
        with self.assertRaisesRegex(ValueError, "one input_scale"):
            method.process_weights_after_loading(layer)

    def test_static_apply_defaults_to_standalone_static_quant(self):
        method = AscendW8A8HiFloat8StaticLinearMethod()
        layer = self.layer()
        method.process_weights_after_loading(layer)
        source = torch.randn((3, 8), dtype=torch.bfloat16)
        encoded = torch.zeros_like(source, dtype=torch.uint8)
        row_scale = torch.full((3,), 0.25, dtype=torch.float32)
        expected = torch.randn((3, 5), dtype=torch.bfloat16)

        with (
            patch(
                "vllm_ascend_hif8.scheme.native.static_quant",
                return_value=(encoded, row_scale),
            ) as static_quant,
            patch(
                "vllm_ascend_hif8.scheme.native.dynamic_quant"
            ) as dynamic_quant,
            patch(
                "vllm_ascend_hif8.scheme.native.soft_fused_linear"
            ) as dynamic_fused,
            patch(
                "vllm_ascend_hif8.scheme.native.soft_static_fused_linear"
            ) as static_fused,
            patch(
                "vllm_ascend_hif8.scheme.native.soft_linear",
                return_value=expected,
            ) as linear,
        ):
            actual = method.apply(layer, source, None)

        self.assertIs(actual, expected)
        static_quant.assert_called_once_with(source, layer.input_scale)
        dynamic_quant.assert_not_called()
        dynamic_fused.assert_not_called()
        static_fused.assert_not_called()
        linear.assert_called_once_with(
            encoded,
            layer.weight,
            row_scale,
            layer.weight_scale,
            None,
            output_dtype=source.dtype,
        )

    def test_static_fused_generation_is_opt_in_and_uses_checkpoint_scale(self):
        os.environ["HIF8_STATIC_FUSED_GENERATION"] = "1"
        method = AscendW8A8HiFloat8StaticLinearMethod()
        layer = self.layer()
        method.process_weights_after_loading(layer)
        source = torch.randn((2, 8), dtype=torch.bfloat16)
        expected = torch.randn((2, 5), dtype=torch.bfloat16)

        with (
            patch("vllm_ascend_hif8.scheme.native.static_quant") as static_quant,
            patch("vllm_ascend_hif8.scheme.native.dynamic_quant") as dynamic_quant,
            patch("vllm_ascend_hif8.scheme.native.soft_linear") as linear,
            patch(
                "vllm_ascend_hif8.scheme.native.soft_static_fused_linear",
                return_value=expected,
            ) as fused,
        ):
            actual = method.apply(layer, source, None)

        self.assertIs(actual, expected)
        static_quant.assert_not_called()
        dynamic_quant.assert_not_called()
        linear.assert_not_called()
        fused.assert_called_once_with(
            source,
            layer.weight,
            layer.input_scale,
            layer.weight_scale,
            None,
            output_dtype=source.dtype,
        )

    def test_static_fused_generation_keeps_prefill_on_unfused_path(self):
        os.environ["HIF8_STATIC_FUSED_GENERATION"] = "1"
        method = AscendW8A8HiFloat8StaticLinearMethod()
        layer = self.layer()
        method.process_weights_after_loading(layer)
        # The 910B static fused kernel is validated through M=4; M=5 must
        # take the ordinary static-quantized GEMM path for larger batches.
        source = torch.randn((5, 8), dtype=torch.bfloat16)
        encoded = torch.zeros_like(source, dtype=torch.uint8)
        row_scale = torch.full((5,), 0.25, dtype=torch.float32)
        expected = torch.randn((5, 5), dtype=torch.bfloat16)

        with (
            patch(
                "vllm_ascend_hif8.scheme.native.static_quant",
                return_value=(encoded, row_scale),
            ) as static_quant,
            patch(
                "vllm_ascend_hif8.scheme.native.soft_static_fused_linear"
            ) as fused,
            patch(
                "vllm_ascend_hif8.scheme.native.soft_linear",
                return_value=expected,
            ) as linear,
        ):
            actual = method.apply(layer, source, None)

        self.assertIs(actual, expected)
        fused.assert_not_called()
        static_quant.assert_called_once_with(source, layer.input_scale)
        linear.assert_called_once()

    def test_model_shapes_keep_static_two_stage_without_soc_environment(self):
        os.environ["HIF8_STATIC_FUSED_GENERATION"] = "1"
        os.environ.pop("HIF8_SOC", None)
        method = AscendW8A8HiFloat8StaticLinearMethod()
        layer = self.layer()
        layer.weight = torch.nn.Parameter(
            torch.zeros((5, 4096), dtype=torch.uint8), requires_grad=False
        )
        method.process_weights_after_loading(layer)
        for batch_size in (1, 4, 8, 16, 32):
            with self.subTest(batch_size=batch_size):
                source = torch.zeros((batch_size, 4096), dtype=torch.bfloat16)
                encoded = torch.zeros_like(source, dtype=torch.uint8)
                row_scale = torch.full((batch_size,), 0.25, dtype=torch.float32)
                expected = torch.zeros((batch_size, 5), dtype=torch.bfloat16)
                with (
                    patch("vllm_ascend_hif8.scheme.native.static_quant", return_value=(encoded, row_scale)) as quant,
                    patch("vllm_ascend_hif8.scheme.native.soft_linear", return_value=expected),
                    patch("vllm_ascend_hif8.scheme.native.soft_static_fused_linear") as fused,
                    patch("vllm_ascend_hif8.scheme.native.dynamic_quant") as dynamic,
                ):
                    self.assertIs(method.apply(layer, source), expected)
                quant.assert_called_once_with(source, layer.input_scale)
                fused.assert_not_called()
                dynamic.assert_not_called()

    def test_static_fused_kernel_preprocess_has_no_amax_or_reduction(self):
        source = (
            __import__("pathlib")
            .Path(__file__)
            .parents[1]
            .joinpath("csrc/opp/op_kernel/soft_hif8_fused_gemm.cpp")
            .read_text()
        )
        preprocess = source.split(
            "__aicore__ inline void QuantizeAndStageSharedActivation()", 1
        )[1].split("__aicore__ inline void AccumulateTile", 1)[0]
        static_branch = preprocess.split("#if HIF8_FUSED_STATIC_SCALE", 1)[1]
        static_branch = static_branch.split("#else", 1)[0]
        self.assertIn("fixedScaleGm_.GetValue(0)", static_branch)
        self.assertNotIn("ScalarActivationAmax", static_branch)
        self.assertNotIn("VectorActivationAmax", static_branch)
        self.assertNotIn("ReduceMax", static_branch)
        self.assertIn("#if !HIF8_FUSED_STATIC_SCALE", source)

    def test_rejects_unsupported_compute_backend_instead_of_bypassing_scale(self):
        os.environ["HIF8_COMPUTE_BACKEND"] = "unsupported"
        method = AscendW8A8HiFloat8StaticLinearMethod()
        with self.assertRaisesRegex(ValueError, "only 'soft_hif8' is supported"):
            method.process_weights_after_loading(self.layer())

    def test_quant_type_is_distinct_from_legacy_dynamic_format(self):
        self.assertEqual(
            AscendW8A8HiFloat8StaticLinearMethod.quant_type,
            STATIC_QUANT_TYPE,
        )
        self.assertEqual(STATIC_QUANT_TYPE, "W8A8HIF8_STATIC")


if __name__ == "__main__":
    unittest.main()
