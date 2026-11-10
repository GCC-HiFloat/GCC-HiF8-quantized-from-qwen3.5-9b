from __future__ import annotations

import math
import os
import unittest

from vllm_ascend_hif8.codec import decode_ascend, encode_ascend, quantize_rows
from vllm_ascend_hif8.reference import linear_u8_u8

RUN_NPU_TESTS = os.environ.get("HIF8_RUN_NPU_TESTS") == "1"

try:
    import torch
    import torch_npu  # noqa: F401
except ImportError:
    torch = None


@unittest.skipUnless(RUN_NPU_TESTS and torch is not None, "set HIF8_RUN_NPU_TESTS=1 in a torch-npu environment")
class NpuOperatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vllm_ascend_hif8 import native

        if not torch.npu.is_available():
            raise unittest.SkipTest("no NPU is available")
        cls.native = native
        cls.device = torch.device("npu:0")

    def test_dynamic_quant_matches_physical_bytes(self):
        rows = [
            [0.0, 1.0, -2.0, 4.0],
            [0.25, -0.5, 1.0, -4.0],
            [float("nan"), float("inf"), -float("inf"), 0.0],
            [2.0**-12, -(2.0**-11), 2.0**-10, -(2.0**-9)],
        ]
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                source = torch.tensor(rows, dtype=dtype, device=self.device)
                data, scale = self.native.dynamic_quant(source)
                torch.npu.synchronize()
                rounded_rows = source.cpu().float().tolist()
                expected = quantize_rows(rounded_rows)
                self.assertEqual(data.cpu().tolist(), [list(row) for row in expected.data])
                torch.testing.assert_close(
                    scale.cpu(),
                    torch.tensor(expected.scale, dtype=torch.float32),
                    rtol=0.0,
                    atol=0.0,
                )

    def test_static_quant_matches_saturating_reference(self):
        from vllm_ascend_hif8.calibration import static_quantize_tensor

        rows = torch.tensor(
            [
                [0.0, 1.0, -2.0, 4.0, 1.0e20],
                [0.25, -0.5, 1.0, -4.0, -1.0e20],
                [float("nan"), float("inf"), -float("inf"), 0.0, 2.0],
                [2.0**-12, -(2.0**-11), 2.0**-10, -(2.0**-9), 1.0],
            ],
            dtype=torch.bfloat16,
        ).reshape(2, 2, 5)
        fixed = torch.tensor([0.5, 2.0], dtype=torch.float32)
        expected_data, _ = static_quantize_tensor(rows, fixed, group_axis=1)
        source = rows.to(self.device)
        data, row_scale = self.native.static_quant(source, fixed.to(self.device))
        torch.npu.synchronize()
        self.assertTrue(torch.equal(data.cpu(), expected_data))
        torch.testing.assert_close(
            row_scale.cpu(),
            torch.tensor([0.5, 2.0, 0.5, 2.0], dtype=torch.float32),
            rtol=0.0,
            atol=0.0,
        )

    def test_every_decode_byte(self):
        data = torch.arange(256, dtype=torch.int32).to(torch.uint8).to(self.device).reshape(1, 256)
        scale = torch.tensor([1.5], dtype=torch.float32, device=self.device)
        output = self.native.load_extension().hif8_scaled_decode(data, scale)
        torch.npu.synchronize()
        actual = output.cpu().float().reshape(-1).tolist()
        for byte, value in enumerate(actual):
            expected = decode_ascend(byte) * 1.5
            if math.isnan(expected):
                self.assertTrue(math.isnan(value), f"byte 0x{byte:02x}")
            else:
                self.assertEqual(value, expected, f"byte 0x{byte:02x}")

    def test_scaled_decode_handles_rows_and_unaligned_k(self):
        rows = 3
        k = 37
        data = torch.arange(rows * k, dtype=torch.int32).remainder(256).to(torch.uint8)
        data = data.reshape(rows, k).to(self.device)
        scales = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32, device=self.device)
        output = self.native.load_extension().hif8_scaled_decode(data, scales)
        torch.npu.synchronize()

        expected = torch.tensor(
            [[decode_ascend(byte) * scales[row].item() for byte in data[row].cpu().tolist()] for row in range(rows)],
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(output.cpu(), expected, rtol=0.0, atol=0.0, equal_nan=True)

    def test_scaled_decode_lut_rounding_and_tile_boundaries(self):
        lut = torch.tensor([decode_ascend(byte) for byte in range(256)], dtype=torch.float32)
        generator = torch.Generator().manual_seed(42)
        scales = torch.cat((
            torch.tensor([0.0, -0.0, 1.0, -1.0, 0.1, 1.00390625,
                          2.0**-20, 2.0**20, float('inf'), float('nan')]),
            torch.rand(54, generator=generator) * 0.02,
        ))
        for k in (31, 37, 256, 4096, 4128, 4133):
            with self.subTest(k=k):
                codes = ((torch.arange(scales.numel() * k).reshape(-1, k) * 73)
                         % 256).to(torch.uint8)
                expected = (lut[codes.long()] * scales[:, None]).to(torch.bfloat16)
                actual = self.native.scaled_decode(
                    codes.to(self.device), scales.to(self.device)).cpu()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                finite = ~torch.isnan(expected)
                self.assertTrue(torch.equal(actual.view(torch.int16)[finite],
                                            expected.view(torch.int16)[finite]))

    def test_scaled_decode_long_context_repeated_without_sync(self):
        generator = torch.Generator().manual_seed(17)
        codes = torch.randint(0, 256, (4 * 4096 * 4, 256),
                              generator=generator, dtype=torch.uint8)
        scales = torch.rand(codes.shape[0], generator=generator) * 0.01 + 1e-5
        lut = torch.tensor([decode_ascend(byte) for byte in range(256)], dtype=torch.float32)
        expected = (lut[codes.long()] * scales[:, None]).to(torch.bfloat16)
        data, fixed = codes.to(self.device), scales.to(self.device)
        outputs = [self.native.scaled_decode(data, fixed) for _ in range(3)]
        for actual in outputs:
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)

    def test_scaled_decode_handles_inline_kv_scale_sidecar(self):
        from vllm_ascend_hif8.kv_cache import pack_kv_rows, unpack_kv_rows

        tokens = 3
        heads = 2
        k = 256
        data = (
            torch.arange(tokens * heads * k, dtype=torch.int32)
            .remainder(256)
            .to(torch.uint8)
            .reshape(tokens, heads, k)
            .to(self.device)
        )
        scales = torch.tensor(
            [[0.5, 0.75], [1.0, 1.25], [1.5, 2.0]],
            dtype=torch.float32,
            device=self.device,
        )
        packed = pack_kv_rows(data, scales)
        unpacked_data, unpacked_scale = unpack_kv_rows(packed, k)
        strided_data = unpacked_data.reshape(tokens * heads, k)
        self.assertFalse(strided_data.is_contiguous())
        self.assertEqual(strided_data.stride(), (k + 4, 1))

        output = self.native.scaled_decode(
            strided_data,
            unpacked_scale.reshape(-1),
        )
        torch.npu.synchronize()

        expected = torch.tensor(
            [
                [
                    decode_ascend(byte) * unpacked_scale.reshape(-1)[row].item()
                    for byte in data.reshape(-1, k)[row].cpu().tolist()
                ]
                for row in range(tokens * heads)
            ],
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(output.cpu(), expected, rtol=0.0, atol=0.0, equal_nan=True)

    def test_gemv_propagates_special_codes(self):
        one = encode_ascend(1.0)
        special_codes = [0x80, 0x6F, 0xEF]
        activation_data = torch.tensor(
            [[code, one] for code in special_codes],
            dtype=torch.uint8,
            device=self.device,
        )
        weight_data = torch.tensor(
            [[one, 0] for _ in special_codes],
            dtype=torch.uint8,
            device=self.device,
        )
        token_scale = torch.tensor([1.5, 0.5, 2.0], dtype=torch.float32, device=self.device)
        channel_scale = torch.ones(len(special_codes), dtype=torch.float32, device=self.device)

        for row, code in enumerate(special_codes):
            for output_dtype in (torch.float16, torch.bfloat16):
                with self.subTest(code=f"0x{code:02x}", output=output_dtype):
                    activation_row = activation_data[row].reshape(1, -1).clone()
                    token_scale_row = token_scale[row].reshape(1).clone()
                    output = self.native.load_extension().soft_hif8_gemv(
                        activation_row,
                        weight_data,
                        token_scale_row,
                        channel_scale,
                        None,
                        output_dtype,
                    )
                    torch.npu.synchronize()
                    actual = output.cpu().float()[0, row].item()
                    expected = decode_ascend(code) * token_scale[row].item()
                    if math.isnan(expected):
                        self.assertTrue(math.isnan(actual))
                    else:
                        self.assertEqual(actual, expected)

    def test_gemv_and_gemm_match_reference(self):
        weight_rows = [
            [1.0, -2.0, 0.5, 4.0],
            [-0.25, 1.5, -3.0, 2.0],
            [2.0, 0.0, -1.0, 0.25],
        ]
        weight = quantize_rows(weight_rows)
        weight_data = torch.tensor(weight.data, dtype=torch.uint8, device=self.device)
        weight_scale = torch.tensor(weight.scale, dtype=torch.float32, device=self.device)

        for rows in (1, 2, 3, 5):
            activation_rows = [[0.5 * (row + 1), -1.0, 2.0, -0.25 * (row + 1)] for row in range(rows)]
            activation = quantize_rows(activation_rows)
            activation_data = torch.tensor(activation.data, dtype=torch.uint8, device=self.device)
            activation_scale = torch.tensor(activation.scale, dtype=torch.float32, device=self.device)

            for bias_dtype in (None, torch.float16, torch.bfloat16):
                bias_values = None if bias_dtype is None else [0.5, -1.0, 2.0]
                bias = None if bias_dtype is None else torch.tensor(bias_values, dtype=bias_dtype, device=self.device)
                expected = linear_u8_u8(activation, weight, bias_values)
                expected_tensor = torch.tensor(expected, dtype=torch.float32)

                for output_dtype in (torch.float16, torch.bfloat16):
                    with self.subTest(rows=rows, bias=bias_dtype, output=output_dtype):
                        actual = self.native.soft_linear(
                            activation_data,
                            weight_data,
                            activation_scale,
                            weight_scale,
                            bias,
                            output_dtype=output_dtype,
                        )
                        torch.npu.synchronize()
                        tolerance = 2.0e-2 if output_dtype == torch.float16 else 8.0e-2
                        torch.testing.assert_close(
                            actual.cpu().float(),
                            expected_tensor,
                            rtol=tolerance,
                            atol=tolerance,
                        )

    def test_gemm_handles_aligned_and_unaligned_k(self):
        extension = self.native.load_extension()
        for rows, channels, k in (
            (1, 17, 37),
            (2, 17, 37),
            (3, 3, 4),
            (5, 16, 16),
            (3, 17, 37),
            # Cross the fixed 128-column Cube split.  This catches host
            # tilings whose N tile is not an integral number of Cube splits.
            (3, 256, 16),
            # 17 N tiles run on 16 Cube cores.  Every block must still reach
            # the round barrier when the final round has one real tile.
            (3, 2176, 16),
            (5, 16, 255),
        ):
            with self.subTest(rows=rows, channels=channels, k=k):
                activation = quantize_rows(
                    [[((row * 7 + col * 3) % 17 - 8) / 4.0 for col in range(k)] for row in range(rows)]
                )
                weight = quantize_rows(
                    [[((row * 5 + col * 11) % 19 - 9) / 5.0 for col in range(k)] for row in range(channels)]
                )
                activation_data = torch.tensor(activation.data, dtype=torch.uint8, device=self.device)
                weight_data = torch.tensor(weight.data, dtype=torch.uint8, device=self.device)
                activation_scale = torch.tensor(activation.scale, dtype=torch.float32, device=self.device)
                weight_scale = torch.tensor(weight.scale, dtype=torch.float32, device=self.device)
                actual = extension.soft_hif8_gemm(
                    activation_data,
                    weight_data,
                    activation_scale,
                    weight_scale,
                    None,
                    torch.bfloat16,
                )
                torch.npu.synchronize()
                expected = torch.tensor(linear_u8_u8(activation, weight, None), dtype=torch.float32)
                torch.testing.assert_close(actual.cpu().float(), expected, rtol=8.0e-2, atol=8.0e-2)

    def test_gemm_handles_exported_style_long_k(self):
        extension = self.native.load_extension()
        decode = torch.tensor(
            [decode_ascend(byte) for byte in range(256)],
            dtype=torch.float32,
        )
        generator = torch.Generator().manual_seed(17)
        for rows, channels, k in (
            (1, 128, 4096),
            (2, 128, 12288),
            (12, 128, 4096),
        ):
            with self.subTest(rows=rows, channels=channels, k=k):
                # Match exporter/runtime inputs: each row is normalized before
                # encoding, so decoded operands remain finite and near [-1, 1].
                activation_source = torch.randn(rows, k, generator=generator, dtype=torch.float32)
                weight_source = torch.randn(channels, k, generator=generator, dtype=torch.float32)
                activation = quantize_rows(activation_source.tolist())
                weight = quantize_rows(weight_source.tolist())
                activation_cpu = torch.tensor(activation.data, dtype=torch.uint8)
                weight_cpu = torch.tensor(weight.data, dtype=torch.uint8)
                activation_scale_cpu = torch.tensor(activation.scale, dtype=torch.float32)
                weight_scale_cpu = torch.tensor(weight.scale, dtype=torch.float32)
                activation_data = activation_cpu.to(self.device)
                weight_data = weight_cpu.to(self.device)
                activation_scale = activation_scale_cpu.to(self.device)
                weight_scale = weight_scale_cpu.to(self.device)

                actual = extension.soft_hif8_gemm(
                    activation_data,
                    weight_data,
                    activation_scale,
                    weight_scale,
                    None,
                    torch.bfloat16,
                )
                torch.npu.synchronize()

                decoded_activation = decode[activation_cpu.long()] * activation_scale_cpu.reshape(-1, 1)
                decoded_weight = decode[weight_cpu.long()] * weight_scale_cpu.reshape(-1, 1)
                expected = decoded_activation @ decoded_weight.T
                denominator = expected.abs().amax().clamp(min=1.0e-12)
                relative_error = (actual.cpu().float() - expected).abs().amax() / denominator
                self.assertLess(float(relative_error), 2.0e-2)
                similarity = torch.nn.functional.cosine_similarity(
                    actual.cpu().float().reshape(1, -1),
                    expected.reshape(1, -1),
                )
                self.assertGreater(float(similarity), 0.9999)

    def test_gemm_prefill_epilogue_routing_boundaries(self):
        extension = self.native.load_extension()
        decode = torch.tensor(
            [decode_ascend(byte) for byte in range(256)],
            dtype=torch.float32,
        )
        generator = torch.Generator().manual_seed(20260816)
        for rows, channels, k in (
            (63, 16, 37),
            (64, 16, 37),
            (65, 16, 37),
            (65, 17, 37),
        ):
            activation_cpu = torch.randint(0, 111, (rows, k), generator=generator, dtype=torch.uint8)
            weight_cpu = torch.randint(0, 111, (channels, k), generator=generator, dtype=torch.uint8)
            token_scale_cpu = torch.rand(rows, generator=generator, dtype=torch.float32).add_(1.0).mul_(1.0e-3)
            channel_scale_cpu = torch.rand(channels, generator=generator, dtype=torch.float32).add_(1.0).mul_(1.0e-3)
            activation = activation_cpu.to(self.device)
            weight = weight_cpu.to(self.device)
            token_scale = token_scale_cpu.to(self.device)
            channel_scale = channel_scale_cpu.to(self.device)
            expected_base = (decode[activation_cpu.long()] * token_scale_cpu[:, None]) @ (
                decode[weight_cpu.long()] * channel_scale_cpu[:, None]
            ).transpose(0, 1)

            for output_dtype in (torch.float16, torch.bfloat16):
                for bias_dtype in (None, torch.float16, torch.bfloat16):
                    bias_cpu = (
                        None
                        if bias_dtype is None
                        else torch.rand(channels, generator=generator, dtype=torch.float32).sub_(0.5).to(bias_dtype)
                    )
                    bias = None if bias_cpu is None else bias_cpu.to(self.device)
                    expected = expected_base if bias_cpu is None else expected_base + bias_cpu.float()
                    with self.subTest(
                        rows=rows,
                        channels=channels,
                        output=output_dtype,
                        bias=bias_dtype,
                    ):
                        first = extension.soft_hif8_gemm(
                            activation,
                            weight,
                            token_scale,
                            channel_scale,
                            bias,
                            output_dtype,
                        )
                        second = extension.soft_hif8_gemm(
                            activation,
                            weight,
                            token_scale,
                            channel_scale,
                            bias,
                            output_dtype,
                        )
                        torch.npu.synchronize()
                        self.assertTrue(torch.equal(first.cpu(), second.cpu()))
                        denominator = expected.abs().amax().clamp(min=1.0e-12)
                        relative_error = (first.cpu().float() - expected).abs().amax() / denominator
                        self.assertLess(float(relative_error), 2.0e-2)

    def test_gemm_activation_cache_handles_m_and_k_tails(self):
        extension = self.native.load_extension()
        generator = torch.Generator().manual_seed(29)
        rows, channels, k = 129, 1024, 4097
        activation_cpu = torch.randint(0, 111, (rows, k), generator=generator, dtype=torch.uint8)
        weight_cpu = torch.randint(0, 111, (channels, k), generator=generator, dtype=torch.uint8)
        activation = activation_cpu.to(self.device)
        weight = weight_cpu.to(self.device)
        token_scale = torch.ones(rows, dtype=torch.float32, device=self.device)
        channel_scale = torch.ones(channels, dtype=torch.float32, device=self.device)

        decode = torch.tensor(
            [decode_ascend(byte) for byte in range(256)],
            dtype=torch.bfloat16,
            device=self.device,
        )
        expected = torch.matmul(decode[activation.long()], decode[weight.long()].transpose(0, 1))
        first = extension.soft_hif8_gemm(
            activation,
            weight,
            token_scale,
            channel_scale,
            None,
            torch.bfloat16,
        )
        second = extension.soft_hif8_gemm(
            activation,
            weight,
            token_scale,
            channel_scale,
            None,
            torch.bfloat16,
        )
        torch.npu.synchronize()

        self.assertTrue(torch.equal(first.cpu(), second.cpu()))
        denominator = expected.float().abs().amax().clamp(min=1.0e-12)
        relative_error = (first.float() - expected.float()).abs().amax() / denominator
        self.assertLess(float(relative_error), 2.0e-2)

    def test_fused_gemm_matches_dynamic_quant_then_gemm(self):
        extension = self.native.load_extension()
        generator = torch.Generator().manual_seed(41)
        for rows, channels, k in (
            (1, 128, 4096),
            (2, 128, 4096),
            (1, 128, 12288),
        ):
            with self.subTest(rows=rows, channels=channels, k=k):
                activation = (
                    torch.randn(rows, k, generator=generator, dtype=torch.float32).to(torch.bfloat16).to(self.device)
                )
                # Keep this numerical GEMM test finite.  Ascend HiFP8 codes
                # 0x6f/0xef represent infinities and 0x80 is NaN; randomly
                # sampling the raw byte interval used to hit them and made
                # the reference reduction undefined (inf + -inf = NaN).
                # The dedicated special-code tests cover propagation semantics.
                weight = torch.randint(
                    0,
                    111,
                    (channels, k),
                    generator=generator,
                    dtype=torch.uint8,
                ).to(self.device)
                channel_scale = (
                    torch.rand(channels, generator=generator, dtype=torch.float32).add_(0.01).to(self.device)
                )
                bias = (
                    torch.randn(channels, generator=generator, dtype=torch.float32).to(torch.bfloat16).to(self.device)
                )

                data, token_scale = self.native.dynamic_quant(activation)
                expected = extension.soft_hif8_gemm(
                    data,
                    weight,
                    token_scale.contiguous(),
                    channel_scale,
                    bias,
                    torch.bfloat16,
                )
                actual = extension.soft_hif8_fused_gemm(
                    activation,
                    weight,
                    channel_scale,
                    bias,
                    torch.bfloat16,
                )
                torch.npu.synchronize()
                torch.testing.assert_close(
                    actual.cpu().float(),
                    expected.cpu().float(),
                    rtol=8.0e-2,
                    atol=8.0e-2,
                )

    def test_static_fused_gemm_matches_static_quant_then_gemm(self):
        extension = self.native.load_extension()
        generator = torch.Generator().manual_seed(20260821)
        for rows, channels, k, with_bias in (
            (1, 128, 2048, False),
            (2, 256, 4096, True),
            (4, 256, 4096, True),
            (1, 128, 12288, True),
        ):
            with self.subTest(rows=rows, channels=channels, k=k, bias=with_bias):
                activation = (
                    torch.randn(rows, k, generator=generator, dtype=torch.float32)
                    .mul_(3.0)
                    .to(torch.bfloat16)
                    .to(self.device)
                )
                weight = torch.randint(
                    0,
                    111,
                    (channels, k),
                    generator=generator,
                    dtype=torch.uint8,
                ).to(self.device)
                fixed_scale = torch.tensor([0.75], dtype=torch.float32, device=self.device)
                channel_scale = (
                    torch.rand(channels, generator=generator, dtype=torch.float32)
                    .add_(0.01)
                    .to(self.device)
                )
                bias = (
                    torch.randn(channels, generator=generator, dtype=torch.float32)
                    .to(torch.bfloat16)
                    .to(self.device)
                    if with_bias
                    else None
                )

                data, row_scale = self.native.static_quant(activation, fixed_scale)
                expected = extension.soft_hif8_gemm(
                    data,
                    weight,
                    row_scale.contiguous(),
                    channel_scale,
                    bias,
                    torch.bfloat16,
                )
                actual = extension.soft_hif8_static_fused_gemm(
                    activation,
                    weight,
                    fixed_scale,
                    channel_scale,
                    bias,
                    torch.bfloat16,
                )
                torch.npu.synchronize()
                torch.testing.assert_close(
                    actual.cpu().float(),
                    expected.cpu().float(),
                    rtol=8.0e-2,
                    atol=8.0e-2,
                )

    def test_static_fused_gemm_saturates_finite_outliers(self):
        extension = self.native.load_extension()
        k = 2048
        activation = torch.zeros((1, k), dtype=torch.bfloat16)
        activation[0, :6] = torch.tensor(
            [1.0e20, -1.0e20, 32768.0, -32768.0, 65536.0, -65536.0],
            dtype=torch.bfloat16,
        )
        activation = activation.to(self.device)
        weight = torch.zeros((128, k), dtype=torch.uint8, device=self.device)
        one = encode_ascend(1.0)
        for channel in range(6):
            weight[channel, channel] = one
        fixed_scale = torch.ones(1, dtype=torch.float32, device=self.device)
        channel_scale = torch.ones(128, dtype=torch.float32, device=self.device)

        data, row_scale = self.native.static_quant(activation, fixed_scale)
        expected = extension.soft_hif8_gemm(
            data,
            weight,
            row_scale.contiguous(),
            channel_scale,
            None,
            torch.bfloat16,
        )
        actual = extension.soft_hif8_static_fused_gemm(
            activation,
            weight,
            fixed_scale,
            channel_scale,
            None,
            torch.bfloat16,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(
            actual.cpu().float(),
            expected.cpu().float(),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )

    def test_static_fused_gemm_qwen35_generation_shapes(self):
        """Cover every static Linear shape used by Qwen3.5-9B at M=1."""
        extension = self.native.load_extension()
        generator = torch.Generator().manual_seed(20260827)
        for channels, k in (
            (1024, 4096),
            (4096, 4096),
            (4096, 12288),
            (8192, 4096),
            (12288, 4096),
        ):
            with self.subTest(channels=channels, k=k):
                activation = (
                    torch.randn(1, k, generator=generator, dtype=torch.float32)
                    .mul_(3.0)
                    .to(torch.bfloat16)
                    .to(self.device)
                )
                weight = torch.randint(
                    0,
                    111,
                    (channels, k),
                    generator=generator,
                    dtype=torch.uint8,
                ).to(self.device)
                fixed_scale = torch.tensor(
                    [0.75], dtype=torch.float32, device=self.device
                )
                channel_scale = torch.rand(
                    channels, generator=generator, dtype=torch.float32
                ).add_(0.01).to(self.device)

                data, row_scale = self.native.static_quant(
                    activation, fixed_scale
                )
                expected = extension.soft_hif8_gemm(
                    data,
                    weight,
                    row_scale.contiguous(),
                    channel_scale,
                    None,
                    torch.bfloat16,
                )
                actual = extension.soft_hif8_static_fused_gemm(
                    activation,
                    weight,
                    fixed_scale,
                    channel_scale,
                    None,
                    torch.bfloat16,
                )
                torch.npu.synchronize()
                actual_cpu = actual.cpu().float()
                expected_cpu = expected.cpu().float()
                close = torch.isclose(
                    actual_cpu, expected_cpu, rtol=8.0e-2, atol=8.0e-2
                ).reshape(-1)
                bad = torch.nonzero(~close).reshape(-1).tolist()
                if bad:
                    ranges = []
                    begin = previous = bad[0]
                    for index in bad[1:]:
                        if index != previous + 1:
                            ranges.append((begin, previous))
                            begin = index
                        previous = index
                    ranges.append((begin, previous))
                    print(
                        f"qwen static fused mismatch N={channels} K={k}: "
                        f"count={len(bad)}, ranges={ranges[:32]}"
                    )
                torch.testing.assert_close(
                    actual_cpu,
                    expected_cpu,
                    rtol=8.0e-2,
                    atol=8.0e-2,
                )

    def test_gemv_batched_reduce_handles_full_and_tail_tiles(self):
        extension = self.native.load_extension()
        for rows, channels, k in (
            (1, 17, 1024),
            (2, 17, 1037),
            (1, 33, 2048),
        ):
            activation = quantize_rows(
                [[((row * 7 + col * 3) % 17 - 8) / 4.0 for col in range(k)] for row in range(rows)]
            )
            weight = quantize_rows(
                [[((row * 5 + col * 11) % 19 - 9) / 5.0 for col in range(k)] for row in range(channels)]
            )
            activation_data = torch.tensor(activation.data, dtype=torch.uint8, device=self.device)
            weight_data = torch.tensor(weight.data, dtype=torch.uint8, device=self.device)
            activation_scale = torch.tensor(activation.scale, dtype=torch.float32, device=self.device)
            weight_scale = torch.tensor(weight.scale, dtype=torch.float32, device=self.device)
            bias_values = [((channel * 3) % 11 - 5) / 8.0 for channel in range(channels)]
            expected = torch.tensor(linear_u8_u8(activation, weight, bias_values), dtype=torch.float32)
            for bias_dtype in (torch.float16, torch.bfloat16):
                with self.subTest(rows=rows, channels=channels, k=k, bias=bias_dtype):
                    bias = torch.tensor(bias_values, dtype=bias_dtype, device=self.device)
                    actual = extension.soft_hif8_gemv(
                        activation_data,
                        weight_data,
                        activation_scale,
                        weight_scale,
                        bias,
                        torch.bfloat16,
                    )
                    torch.npu.synchronize()
                    torch.testing.assert_close(actual.cpu().float(), expected, rtol=8.0e-2, atol=8.0e-2)

    def test_operator_row_contracts_are_strict(self):
        extension = self.native.load_extension()
        activation = torch.zeros((3, 4), dtype=torch.uint8, device=self.device)
        weight = torch.zeros((2, 4), dtype=torch.uint8, device=self.device)
        token_scale = torch.ones(3, dtype=torch.float32, device=self.device)
        channel_scale = torch.ones(2, dtype=torch.float32, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "M <= 2"):
            extension.soft_hif8_gemv(
                activation,
                weight,
                token_scale,
                channel_scale,
                None,
                torch.float16,
            )

    def test_nonzero_storage_offset_is_rejected(self):
        extension = self.native.load_extension()
        activation_view = torch.zeros((3, 4), dtype=torch.uint8, device=self.device)[1:]
        weight = torch.zeros((2, 4), dtype=torch.uint8, device=self.device)
        token_scale = torch.ones(2, dtype=torch.float32, device=self.device)
        channel_scale = torch.ones(2, dtype=torch.float32, device=self.device)
        self.assertTrue(activation_view.is_contiguous())
        self.assertGreater(activation_view.storage_offset(), 0)
        with self.assertRaisesRegex(RuntimeError, "storage_offset == 0"):
            extension.soft_hif8_gemv(
                activation_view,
                weight,
                token_scale,
                channel_scale,
                None,
                torch.float16,
            )


if __name__ == "__main__":
    unittest.main()
