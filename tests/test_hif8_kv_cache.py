import unittest
from types import SimpleNamespace

import torch

from vllm_ascend_hif8.codec import ASCEND_NAN, ASCEND_NINF, ASCEND_PINF, encode_ascend
from vllm_ascend_hif8.kv_cache import (
    HIF8_SCALE_BYTES,
    Hif8KVCacheLayout,
    decode_kv_rows,
    gather_kv_rows,
    gather_kv_rows_concat,
    pack_kv_rows,
    paged_attention_reference,
    quantize_kv_rows,
    scatter_kv_rows,
    unpack_kv_rows,
)
from vllm_ascend_hif8.kv_runtime import _attention_sequence_plan


class Hif8KVCacheTest(unittest.TestCase):
    def test_attention_sequence_plan_builds_cumulative_dense_kv_lengths(self):
        metadata = SimpleNamespace(
            actual_seq_lengths_q=[1, 4],
            seq_lens_list=[5, 3],
            num_actual_tokens=4,
        )
        self.assertEqual(
            _attention_sequence_plan(metadata, query_rows=4),
            (4, [1, 4], [5, 8]),
        )

    def test_attention_sequence_plan_allows_trailing_padding_request(self):
        metadata = SimpleNamespace(
            actual_seq_lengths_q=[1, 8],
            seq_lens_list=[5, 1],
            num_actual_tokens=1,
        )
        self.assertEqual(
            _attention_sequence_plan(metadata, query_rows=8),
            (8, [1, 8], [5, 6]),
        )

    def test_attention_sequence_plan_rejects_malformed_real_requests(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            _attention_sequence_plan(
                SimpleNamespace(
                    actual_seq_lengths_q=[1, 2],
                    seq_lens_list=[2],
                    num_actual_tokens=2,
                ),
                query_rows=2,
            )
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            _attention_sequence_plan(
                SimpleNamespace(
                    actual_seq_lengths_q=[2, 1],
                    seq_lens_list=[2, 1],
                    num_actual_tokens=2,
                ),
                query_rows=2,
            )
        with self.assertRaisesRegex(ValueError, "cannot be shorter"):
            _attention_sequence_plan(
                SimpleNamespace(
                    actual_seq_lengths_q=[3],
                    seq_lens_list=[2],
                    num_actual_tokens=3,
                ),
                query_rows=3,
            )

    def test_layout_accounts_for_k_and_v_inline_scales(self):
        layout = Hif8KVCacheLayout(3, 8, 2, 16, 24)
        self.assertEqual(layout.key_width, 16 + HIF8_SCALE_BYTES)
        self.assertEqual(layout.value_width, 24 + HIF8_SCALE_BYTES)
        self.assertEqual(
            layout.page_size_bytes,
            8 * 2 * ((16 + HIF8_SCALE_BYTES) + (24 + HIF8_SCALE_BYTES)),
        )
        self.assertEqual(layout.key_shape(), (3, 8, 2, 20))
        self.assertEqual(layout.value_shape(), (3, 8, 2, 28))

    def test_layout_defaults_value_width_to_key_width(self):
        layout = Hif8KVCacheLayout(1, 128, 4, 128)
        self.assertEqual(layout.resolved_head_size_v, 128)
        self.assertEqual(layout.key_width, 132)
        self.assertEqual(layout.value_width, 132)
        self.assertEqual(layout.page_size_bytes, 128 * 4 * (132 + 132))

    def test_quantize_round_trip_keeps_independent_token_head_scales(self):
        values = torch.tensor(
            [
                [[-2.0, -1.0, 0.0, 1.0], [0.25, 0.5, 0.75, 1.0]],
                [[3.0, -1.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        data, scale = quantize_kv_rows(values)
        self.assertEqual(data.dtype, torch.uint8)
        self.assertEqual(tuple(scale.shape), (2, 2))
        torch.testing.assert_close(
            scale,
            torch.tensor([[2.0, 1.0], [3.0, 1.0e-12]], dtype=torch.float32),
            rtol=0,
            atol=1.0e-18,
        )
        decoded = decode_kv_rows(data, scale)
        torch.testing.assert_close(decoded, values, rtol=0.08, atol=0.08)

    def test_special_codes_survive_quantization(self):
        values = torch.tensor([[[float("nan"), float("inf"), float("-inf"), 0.0]]])
        data, scale = quantize_kv_rows(values)
        self.assertEqual(int(data[0, 0, 0]), ASCEND_NAN)
        self.assertEqual(int(data[0, 0, 1]), ASCEND_PINF)
        self.assertEqual(int(data[0, 0, 2]), ASCEND_NINF)
        self.assertTrue(torch.isnan(decode_kv_rows(data, scale)[0, 0, 0]))

    def test_pack_unpack_is_byte_exact(self):
        data = torch.tensor(
            [[
                [encode_ascend(-1.0), encode_ascend(0.5), encode_ascend(0.0), encode_ascend(1.0)],
                [encode_ascend(0.25), encode_ascend(-0.25), encode_ascend(0.75), encode_ascend(-0.75)],
            ]],
            dtype=torch.uint8,
        )
        scale = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
        packed = pack_kv_rows(data, scale)
        unpacked_data, unpacked_scale = unpack_kv_rows(packed, 4)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.shape[-1], 4 + HIF8_SCALE_BYTES)
        torch.testing.assert_close(unpacked_data, data)
        torch.testing.assert_close(unpacked_scale, scale)

    def test_slot_mapping_and_block_table_gather(self):
        cache = torch.zeros((2, 4, 1, 8), dtype=torch.uint8)
        rows = torch.arange(3 * 1 * 8, dtype=torch.uint8).reshape(3, 1, 8)
        slots = torch.tensor([0, 5, -1], dtype=torch.int32)
        scatter_kv_rows(cache, rows, slots, block_size=4)
        gathered = gather_kv_rows(
            cache,
            torch.tensor([[0, 1]], dtype=torch.int32),
            [6],
            block_size=4,
        )[0]
        torch.testing.assert_close(gathered[0], rows[0])
        torch.testing.assert_close(gathered[1], torch.zeros_like(rows[1]))
        torch.testing.assert_close(gathered[5], rows[1])
        self.assertTrue(torch.equal(gathered[2], torch.zeros_like(gathered[2])))

    def test_concat_gather_matches_request_list_byte_for_byte(self):
        cache = torch.arange(5 * 4 * 2 * 8, dtype=torch.int32).to(torch.uint8)
        cache = cache.reshape(5, 4, 2, 8)
        block_table = torch.tensor([[3, 1, -1], [4, 0, 2]], dtype=torch.int32)
        seq_lens = [5, 10]
        reference = torch.cat(
            gather_kv_rows(cache, block_table, seq_lens, block_size=4), dim=0
        )
        actual = gather_kv_rows_concat(
            cache, block_table, seq_lens, block_size=4
        )
        self.assertTrue(torch.equal(actual, reference))

    def test_concat_gather_handles_zero_length_requests(self):
        cache = torch.arange(2 * 4 * 1 * 8, dtype=torch.int32).to(torch.uint8)
        cache = cache.reshape(2, 4, 1, 8)
        block_table = torch.tensor([[-1], [1]], dtype=torch.int32)
        gathered = gather_kv_rows_concat(
            cache, block_table, [0, 3], block_size=4
        )
        torch.testing.assert_close(gathered, cache[1, :3])

    def test_reference_attention_decodes_only_gathered_rows(self):
        key = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]]
        )
        value = torch.tensor(
            [[[2.0, 0.0, 0.0, 0.0]], [[0.0, 3.0, 0.0, 0.0]], [[2.0, 0.0, 0.0, 0.0]], [[0.0, 3.0, 0.0, 0.0]]]
        )
        k_data, k_scale = quantize_kv_rows(key)
        v_data, v_scale = quantize_kv_rows(value)
        k_cache = pack_kv_rows(k_data, k_scale).reshape(1, 4, 1, 8)
        v_cache = pack_kv_rows(v_data, v_scale).reshape(1, 4, 1, 8)
        query = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        output = paged_attention_reference(
            query,
            k_cache,
            v_cache,
            torch.tensor([[0]], dtype=torch.int32),
            [4],
            block_size=4,
            scale=1.0,
        )
        self.assertEqual(tuple(output.shape), (1, 1, 4))
        self.assertGreater(float(output[0, 0, 0]), 1.0)
        self.assertLess(float(output[0, 0, 1]), 2.0)

    def test_invalid_layout_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            Hif8KVCacheLayout(1, 4, 1, 6)
        with self.assertRaises(ValueError):
            Hif8KVCacheLayout(1, 4, 1, 8, 6)


if __name__ == "__main__":
    unittest.main()
