"""Opt-in NPU regression tests for the HiFloat8 KV-cache adapter.

These tests exercise the production native quant/decode operators and Ascend
FIA while keeping the persistent K/V pages in the project's private uint8
format.  They are skipped in normal CPU CI.  Use a matching native extension
and OPP, then set ``HIF8_RUN_NPU_TESTS=1``.
"""

from __future__ import annotations

import math
import os
import unittest
from types import SimpleNamespace

RUN_NPU_TESTS = os.environ.get("HIF8_RUN_NPU_TESTS") == "1"

try:
    import torch
    import torch_npu  # noqa: F401
except ImportError:
    torch = None


@unittest.skipUnless(
    RUN_NPU_TESTS and torch is not None,
    "set HIF8_RUN_NPU_TESTS=1 in a torch-npu environment",
)
class Hif8KVOnNpuTest(unittest.TestCase):
    NUM_HEADS = 16
    NUM_KV_HEADS = 4
    HEAD_SIZE = 256
    BLOCK_SIZE = 128

    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("no NPU is available")

        os.environ["HIF8_KV_CACHE"] = "1"
        import vllm_ascend_hif8.plugin as plugin
        from vllm_ascend_hif8 import kv_runtime

        # General-plugin registration must precede the worker patch.  Importing
        # attention_v1 before adapt_patch(False) recreates DeviceOperator's
        # initialization cycle in a real vLLM worker.
        plugin.register()
        import vllm_ascend.utils as ascend_utils

        ascend_utils.adapt_patch(False)
        from vllm_ascend.attention import attention_v1

        cls.attention_v1 = attention_v1
        cls.impl_cls = attention_v1.AscendAttentionBackend.get_impl_cls()
        if not getattr(cls.impl_cls, "_hif8_kv_cache", False):
            raise AssertionError("HIF8 Ascend attention adapter was not installed")
        cls.device = torch.device("npu:0")
        cls.kv_runtime = kv_runtime
        # FIA's causal mask is a byte mask; FP16 0/1 values do not mask logits.
        cls.causal_mask = torch.triu(
            torch.ones((2048, 2048), dtype=torch.bool),
            diagonal=1,
        ).to(cls.device)

    def _make_impl(self):
        from unittest.mock import patch

        config = SimpleNamespace(
            model_config=SimpleNamespace(
                enforce_eager=True,
                hf_config=SimpleNamespace(model_type="qwen3_5"),
                hf_text_config=SimpleNamespace(model_type="qwen3_5"),
                # The production HIF8 attention adapter resolves checkpoint
                # metadata during construction.  Keep this NPU fixture on a
                # legacy/dynamic path by default, while allowing a caller to
                # point it at a real static checkpoint for an end-to-end KV
                # metadata test.
                model_weights=os.environ.get("HIF8_TEST_CHECKPOINT", os.getcwd()),
            ),
            compilation_config=SimpleNamespace(cudagraph_mode=self._graph_none()),
            scheduler_config=SimpleNamespace(async_scheduling=False),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            use_v2_model_runner=False,
            kv_transfer_config=None,
            quant_config=None,
        )
        patcher = patch.object(
            self.attention_v1,
            "get_current_vllm_config",
            return_value=config,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        from vllm_ascend.attention import utils as attention_utils

        utils_patcher = patch.object(
            attention_utils,
            "get_current_vllm_config",
            return_value=config,
        )
        utils_patcher.start()
        self.addCleanup(utils_patcher.stop)
        attention_utils.needs_layer_aware_fia_graph_replay.cache_clear()
        self.addCleanup(attention_utils.needs_layer_aware_fia_graph_replay.cache_clear)
        return self.impl_cls(
            num_heads=self.NUM_HEADS,
            head_size=self.HEAD_SIZE,
            head_size_v=self.HEAD_SIZE,
            scale=1.0 / math.sqrt(self.HEAD_SIZE),
            num_kv_heads=self.NUM_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8_per_token_head",
            logits_soft_cap=None,
            attn_type=self.attention_v1.AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
        )

    @staticmethod
    def _graph_none():
        from vllm.config.compilation import CUDAGraphMode

        return CUDAGraphMode.NONE

    def _cache_pair(self, num_blocks: int = 2):
        width = self.HEAD_SIZE + 4
        shape = (
            num_blocks,
            self.BLOCK_SIZE,
            self.NUM_KV_HEADS,
            width,
        )
        return (
            torch.zeros(shape, dtype=torch.uint8, device=self.device),
            torch.zeros(shape, dtype=torch.uint8, device=self.device),
        )

    def _metadata(
        self,
        *,
        q_cumulative,
        seq_lens,
        slot_mapping,
        block_tables,
        num_actual_tokens,
    ):
        return SimpleNamespace(
            actual_seq_lengths_q=list(q_cumulative),
            seq_lens_list=list(seq_lens),
            slot_mapping=torch.tensor(
                slot_mapping,
                dtype=torch.int32,
                device=self.device,
            ),
            block_tables=torch.tensor(
                block_tables,
                dtype=torch.int32,
                device=self.device,
            ),
            num_actual_tokens=num_actual_tokens,
            causal=True,
            attn_mask=self.causal_mask,
        )

    @staticmethod
    def _cosine(actual, expected) -> float:
        return float(
            torch.nn.functional.cosine_similarity(
                actual.float().reshape(1, -1),
                expected.float().reshape(1, -1),
            ).item()
        )

    def _bf16_fia(self, query, key, value, q_lens, kv_lens):
        output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key.contiguous(),
            value=value.contiguous(),
            input_layout="TND",
            actual_seq_lengths=list(q_lens),
            actual_seq_lengths_kv=list(kv_lens),
            num_key_value_heads=self.NUM_KV_HEADS,
            num_heads=self.NUM_HEADS,
            scale=1.0 / math.sqrt(self.HEAD_SIZE),
            sparse_mode=3,
            atten_mask=self.causal_mask,
        )
        return output.view(query.shape[0], self.NUM_HEADS, self.HEAD_SIZE)

    @staticmethod
    def _mismatch_summary(stage, actual, expected):
        actual = actual.cpu()
        expected = expected.cpu()
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            return (
                f"{stage}: metadata mismatch actual={actual.shape}/{actual.dtype} "
                f"expected={expected.shape}/{expected.dtype}"
            )
        if actual.is_floating_point():
            equal = (actual == expected) | (
                torch.isnan(actual) & torch.isnan(expected)
            )
        else:
            equal = actual == expected
        if bool(equal.all()):
            return None
        mismatch = ~equal
        coordinates = torch.nonzero(mismatch, as_tuple=False)
        examples = []
        for coordinate in coordinates[:8].tolist():
            index = tuple(coordinate)
            examples.append(
                f"{index}: actual={actual[index].item()!r}, "
                f"expected={expected[index].item()!r}"
            )
        bad_axis0 = torch.unique(coordinates[:, 0]).tolist()
        return (
            f"{stage}: {int(mismatch.sum().item())}/{mismatch.numel()} "
            f"elements differ across {len(bad_axis0)}/{actual.shape[0]} "
            f"axis-0 entries (first {bad_axis0[:16]}); "
            f"first mismatches: {'; '.join(examples)}"
        )

    def _paged_math_reference(self, query, cache, metadata):
        """Evaluate cache attention without using Ascend FIA as the oracle."""

        from vllm_ascend_hif8.kv_cache import paged_attention_reference

        q_cumulative = list(metadata.actual_seq_lengths_q)
        q_lengths = [
            end - (q_cumulative[index - 1] if index else 0)
            for index, end in enumerate(q_cumulative)
        ]
        num_tokens = q_cumulative[-1]
        key_cache, value_cache = cache
        return paged_attention_reference(
            query[:num_tokens].detach().cpu().float(),
            key_cache.detach().cpu(),
            value_cache.detach().cpu(),
            metadata.block_tables.detach().cpu(),
            metadata.seq_lens_list,
            block_size=self.BLOCK_SIZE,
            scale=1.0 / math.sqrt(self.HEAD_SIZE),
            query_lens=q_lengths,
            causal=metadata.causal,
        )

    def _dense_math_reference(
        self,
        query,
        key,
        value,
        q_lens,
        kv_lens,
        *,
        num_heads=None,
    ):
        """Independent dense TND/GQA attention with right-aligned causality."""

        query = query.detach().cpu().float()
        key = key.detach().cpu().float()
        value = value.detach().cpu().float()
        resolved_num_heads = self.NUM_HEADS if num_heads is None else num_heads
        repeat = resolved_num_heads // self.NUM_KV_HEADS
        outputs = []
        q_begin = 0
        kv_begin = 0
        for q_end, kv_end in zip(q_lens, kv_lens, strict=True):
            q = query[q_begin:q_end]
            k = key[kv_begin:kv_end].repeat_interleave(repeat, dim=1)
            v = value[kv_begin:kv_end].repeat_interleave(repeat, dim=1)
            q_length = q_end - q_begin
            kv_length = kv_end - kv_begin
            logits = torch.einsum("qhd,khd->qhk", q, k)
            logits.mul_(1.0 / math.sqrt(self.HEAD_SIZE))
            query_positions = (
                torch.arange(q_length, dtype=torch.long)
                + kv_length
                - q_length
            )
            key_positions = torch.arange(kv_length, dtype=torch.long)
            logits.masked_fill_(
                key_positions[None, None, :]
                > query_positions[:, None, None],
                float("-inf"),
            )
            probabilities = torch.softmax(logits, dim=-1)
            outputs.append(torch.einsum("qhk,khd->qhd", probabilities, v))
            q_begin = q_end
            kv_begin = kv_end
        return torch.cat(outputs, dim=0)

    def test_dense_fia_matches_independent_math_reference(self):
        generator = torch.Generator().manual_seed(2026082201)
        cases = (
            ([5], [5]),
            ([1], [7]),
            ([1, 4], [5, 10]),
            ([1, 2], [7, 15]),
        )
        for q_lens, kv_lens in cases:
            with self.subTest(q_lens=q_lens, kv_lens=kv_lens):
                query = torch.randn(
                    q_lens[-1],
                    self.NUM_HEADS,
                    self.HEAD_SIZE,
                    generator=generator,
                    dtype=torch.bfloat16,
                ).to(self.device)
                key = torch.randn(
                    kv_lens[-1],
                    self.NUM_KV_HEADS,
                    self.HEAD_SIZE,
                    generator=generator,
                    dtype=torch.bfloat16,
                ).to(self.device)
                value = torch.randn(
                    kv_lens[-1],
                    self.NUM_KV_HEADS,
                    self.HEAD_SIZE,
                    generator=generator,
                    dtype=torch.bfloat16,
                ).to(self.device)
                actual = self._bf16_fia(query, key, value, q_lens, kv_lens)
                expected = self._dense_math_reference(
                    query, key, value, q_lens, kv_lens
                )
                torch.npu.synchronize()
                self.assertGreater(self._cosine(actual.cpu(), expected), 0.999)

    def test_qwen35_inline_sidecar_data_path(self):
        from vllm_ascend_hif8 import native
        from vllm_ascend_hif8.kv_cache import pack_kv_rows, unpack_kv_rows

        generator = torch.Generator().manual_seed(20260821)
        source = torch.randn(
            6,
            self.NUM_KV_HEADS,
            self.HEAD_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(self.device)
        rows = source.reshape(-1, self.HEAD_SIZE)
        data, scale = native.dynamic_quant(rows)
        packed = pack_kv_rows(
            data.reshape(6, self.NUM_KV_HEADS, self.HEAD_SIZE),
            scale.reshape(6, self.NUM_KV_HEADS),
        )
        cache = torch.zeros(
            (2, self.BLOCK_SIZE, self.NUM_KV_HEADS, self.HEAD_SIZE + 4),
            dtype=torch.uint8,
            device=self.device,
        )
        slots = torch.tensor([0, 1, 2, 128, 129, 130], device=self.device)
        blocks = slots // self.BLOCK_SIZE
        offsets = slots % self.BLOCK_SIZE
        cache.index_put_((blocks, offsets), packed, accumulate=False)

        table = torch.tensor([[0, 1]], dtype=torch.long, device=self.device)
        gathered = cache.index_select(0, table[0]).reshape(
            -1,
            self.NUM_KV_HEADS,
            self.HEAD_SIZE + 4,
        )
        gathered = torch.cat([gathered[:3], gathered[128:131]], dim=0)
        decoded_data, decoded_scale = unpack_kv_rows(gathered, self.HEAD_SIZE)
        decoded = native.scaled_decode(
            decoded_data.reshape(-1, self.HEAD_SIZE),
            decoded_scale.reshape(-1),
        ).reshape_as(source)
        torch.npu.synchronize()

        self.assertEqual(cache.dtype, torch.uint8)
        self.assertEqual(cache.shape[-1], 260)
        self.assertGreater(self._cosine(decoded, source), 0.999)

    def test_qwen35_static_production_shape_pa_round_trip(self):
        """Cover the K=256 static-quant vector path used by Qwen3.5 KV.

        The small static-quant operator test intentionally includes special
        values, but its K=5 rows take the scalar fallback.  Production KV rows
        are [tokens, 4, 256] and use the Cast/Shift/Gather encoder.  Exercise
        that path together with the real paged-cache byte-copy primitive and
        the request-local scaled decoder, including a 128-token page boundary.
        """

        import torch_npu

        from vllm_ascend_hif8 import native
        from vllm_ascend_hif8.calibration import static_quantize_tensor
        from vllm_ascend_hif8.codec import ASCEND_DECODE_TABLE
        from vllm_ascend_hif8.kv_cache import (
            gather_kv_rows,
            pack_kv_rows,
            unpack_kv_rows,
        )

        tokens = self.BLOCK_SIZE + 9
        generator = torch.Generator().manual_seed(20260822)
        key_cpu = (
            torch.randn(
                tokens,
                self.NUM_KV_HEADS,
                self.HEAD_SIZE,
                generator=generator,
                dtype=torch.float32,
            )
            .mul_(3.0)
            .to(torch.bfloat16)
        )
        value_cpu = (
            torch.randn(
                tokens,
                self.NUM_KV_HEADS,
                self.HEAD_SIZE,
                generator=generator,
                dtype=torch.float32,
            )
            .mul_(1.5)
            .to(torch.bfloat16)
        )
        # Realistic layer-3 checkpoint scales, kept distinct across heads and
        # between K/V so an incorrect flattened-row scale mapping is visible.
        key_scale_cpu = torch.tensor(
            [0.9432383179664612, 0.7343872785568237,
             0.9391698241233826, 1.1160656213760376],
            dtype=torch.float32,
        )
        value_scale_cpu = torch.tensor(
            [0.3244977593421936, 0.416015625,
             0.5311850309371948, 0.3086029291152954],
            dtype=torch.float32,
        )
        expected_key, _ = static_quantize_tensor(
            key_cpu, key_scale_cpu, group_axis=1
        )
        expected_value, _ = static_quantize_tensor(
            value_cpu, value_scale_cpu, group_axis=1
        )

        key = key_cpu.to(self.device)
        value = value_cpu.to(self.device)
        key_scale = key_scale_cpu.to(self.device)
        value_scale = value_scale_cpu.to(self.device)
        key_data, key_rows_scale = native.static_quant(
            key.reshape(-1, self.HEAD_SIZE), key_scale
        )
        value_data, value_rows_scale = native.static_quant(
            value.reshape(-1, self.HEAD_SIZE), value_scale
        )
        expected_rows_key_scale = key_scale_cpu.repeat(tokens)
        expected_rows_value_scale = value_scale_cpu.repeat(tokens)
        expected_packed_key = pack_kv_rows(
            expected_key,
            expected_rows_key_scale.reshape(tokens, self.NUM_KV_HEADS),
        )
        expected_packed_value = pack_kv_rows(
            expected_value,
            expected_rows_value_scale.reshape(tokens, self.NUM_KV_HEADS),
        )

        packed_key = pack_kv_rows(
            key_data.reshape(tokens, self.NUM_KV_HEADS, self.HEAD_SIZE),
            key_rows_scale.reshape(tokens, self.NUM_KV_HEADS).contiguous(),
        )
        packed_value = pack_kv_rows(
            value_data.reshape(tokens, self.NUM_KV_HEADS, self.HEAD_SIZE),
            value_rows_scale.reshape(tokens, self.NUM_KV_HEADS).contiguous(),
        )
        key_cache, value_cache = self._cache_pair(num_blocks=2)
        slots = torch.arange(tokens, dtype=torch.int64, device=self.device)
        key_cache.reshape(-1, *key_cache.shape[2:]).index_copy_(0, slots, packed_key)
        value_cache.reshape(-1, *value_cache.shape[2:]).index_copy_(0, slots, packed_value)

        block_table = torch.tensor([[0, 1]], dtype=torch.int64, device=self.device)
        gathered_key = gather_kv_rows(
            key_cache, block_table, [tokens], block_size=self.BLOCK_SIZE
        )[0]
        gathered_value = gather_kv_rows(
            value_cache, block_table, [tokens], block_size=self.BLOCK_SIZE
        )[0]
        decoded_key_data, decoded_key_scale = unpack_kv_rows(
            gathered_key, self.HEAD_SIZE
        )
        decoded_value_data, decoded_value_scale = unpack_kv_rows(
            gathered_value, self.HEAD_SIZE
        )
        decoded_key = native.scaled_decode(
            decoded_key_data.reshape(-1, self.HEAD_SIZE),
            decoded_key_scale.reshape(-1),
        ).reshape_as(key)
        decoded_value = native.scaled_decode(
            decoded_value_data.reshape(-1, self.HEAD_SIZE),
            decoded_value_scale.reshape(-1),
        ).reshape_as(value)
        torch.npu.synchronize()

        decode_lut = torch.tensor(ASCEND_DECODE_TABLE, dtype=torch.float32)
        expected_decoded_key = (
            decode_lut[expected_key.to(torch.long)]
            * key_scale_cpu.reshape(1, self.NUM_KV_HEADS, 1)
        ).to(torch.bfloat16)
        expected_decoded_value = (
            decode_lut[expected_value.to(torch.long)]
            * value_scale_cpu.reshape(1, self.NUM_KV_HEADS, 1)
        ).to(torch.bfloat16)

        comparisons = (
            (
                "static_quant.key.data",
                key_data.reshape(-1, self.HEAD_SIZE),
                expected_key.reshape(-1, self.HEAD_SIZE),
            ),
            (
                "static_quant.value.data",
                value_data.reshape(-1, self.HEAD_SIZE),
                expected_value.reshape(-1, self.HEAD_SIZE),
            ),
            ("static_quant.key.scale", key_rows_scale, expected_rows_key_scale),
            ("static_quant.value.scale", value_rows_scale, expected_rows_value_scale),
            ("pack.key", packed_key, expected_packed_key),
            ("pack.value", packed_value, expected_packed_value),
            ("scatter_gather.key", gathered_key, expected_packed_key),
            ("scatter_gather.value", gathered_value, expected_packed_value),
            ("unpack.key.data", decoded_key_data, expected_key),
            ("unpack.value.data", decoded_value_data, expected_value),
            (
                "unpack.key.scale",
                decoded_key_scale,
                expected_rows_key_scale.reshape(tokens, self.NUM_KV_HEADS),
            ),
            (
                "unpack.value.scale",
                decoded_value_scale,
                expected_rows_value_scale.reshape(tokens, self.NUM_KV_HEADS),
            ),
            ("scaled_decode.key", decoded_key, expected_decoded_key),
            ("scaled_decode.value", decoded_value, expected_decoded_value),
        )
        failures = [
            summary
            for stage, actual, expected in comparisons
            if (
                summary := self._mismatch_summary(stage, actual, expected)
            ) is not None
        ]
        if failures:
            self.fail("production-shape HIF8 KV mismatch:\n" + "\n".join(failures))

    def test_qwen35_dynamic_production_shape_pa_round_trip(self):
        """Validate dynamic KV8 bytes, scales, pages, and decode at K=256."""

        import torch_npu

        from vllm_ascend_hif8 import native
        from vllm_ascend_hif8.kv_cache import (
            decode_kv_rows,
            gather_kv_rows,
            pack_kv_rows,
            quantize_kv_rows,
            unpack_kv_rows,
        )

        tokens = self.BLOCK_SIZE + 9
        generator = torch.Generator().manual_seed(2026082202)
        head_amplitudes = torch.tensor(
            [0.375, 0.875, 1.75, 3.5], dtype=torch.float32
        ).reshape(1, self.NUM_KV_HEADS, 1)
        key_cpu = (
            torch.randn(
                tokens,
                self.NUM_KV_HEADS,
                self.HEAD_SIZE,
                generator=generator,
                dtype=torch.float32,
            )
            .mul_(head_amplitudes)
            .to(torch.bfloat16)
        )
        value_cpu = (
            torch.randn(
                tokens,
                self.NUM_KV_HEADS,
                self.HEAD_SIZE,
                generator=generator,
                dtype=torch.float32,
            )
            .mul_(head_amplitudes.flip(1))
            .to(torch.bfloat16)
        )
        expected_key, expected_key_scale = quantize_kv_rows(key_cpu)
        expected_value, expected_value_scale = quantize_kv_rows(value_cpu)
        expected_packed_key = pack_kv_rows(
            expected_key, expected_key_scale
        )
        expected_packed_value = pack_kv_rows(
            expected_value, expected_value_scale
        )

        key = key_cpu.to(self.device)
        value = value_cpu.to(self.device)
        key_data, key_rows_scale = native.dynamic_quant(
            key.reshape(-1, self.HEAD_SIZE)
        )
        value_data, value_rows_scale = native.dynamic_quant(
            value.reshape(-1, self.HEAD_SIZE)
        )
        packed_key = pack_kv_rows(
            key_data.reshape(tokens, self.NUM_KV_HEADS, self.HEAD_SIZE),
            key_rows_scale.reshape(tokens, self.NUM_KV_HEADS).contiguous(),
        )
        packed_value = pack_kv_rows(
            value_data.reshape(tokens, self.NUM_KV_HEADS, self.HEAD_SIZE),
            value_rows_scale.reshape(tokens, self.NUM_KV_HEADS).contiguous(),
        )
        key_cache, value_cache = self._cache_pair(num_blocks=2)
        slots = torch.arange(tokens, dtype=torch.int64, device=self.device)
        key_cache.reshape(-1, *key_cache.shape[2:]).index_copy_(0, slots, packed_key)
        value_cache.reshape(-1, *value_cache.shape[2:]).index_copy_(0, slots, packed_value)

        block_table = torch.tensor(
            [[0, 1]], dtype=torch.int64, device=self.device
        )
        gathered_key = gather_kv_rows(
            key_cache, block_table, [tokens], block_size=self.BLOCK_SIZE
        )[0]
        gathered_value = gather_kv_rows(
            value_cache, block_table, [tokens], block_size=self.BLOCK_SIZE
        )[0]
        decoded_key_data, decoded_key_scale = unpack_kv_rows(
            gathered_key, self.HEAD_SIZE
        )
        decoded_value_data, decoded_value_scale = unpack_kv_rows(
            gathered_value, self.HEAD_SIZE
        )
        decoded_key = native.scaled_decode(
            decoded_key_data.reshape(-1, self.HEAD_SIZE),
            decoded_key_scale.reshape(-1),
        ).reshape_as(key)
        decoded_value = native.scaled_decode(
            decoded_value_data.reshape(-1, self.HEAD_SIZE),
            decoded_value_scale.reshape(-1),
        ).reshape_as(value)
        torch.npu.synchronize()
        expected_decoded_key = decode_kv_rows(
            expected_key, expected_key_scale
        ).to(torch.bfloat16)
        expected_decoded_value = decode_kv_rows(
            expected_value, expected_value_scale
        ).to(torch.bfloat16)

        comparisons = (
            (
                "dynamic_quant.key.data",
                key_data.reshape(-1, self.HEAD_SIZE),
                expected_key.reshape(-1, self.HEAD_SIZE),
            ),
            (
                "dynamic_quant.value.data",
                value_data.reshape(-1, self.HEAD_SIZE),
                expected_value.reshape(-1, self.HEAD_SIZE),
            ),
            (
                "dynamic_quant.key.scale",
                key_rows_scale,
                expected_key_scale.reshape(-1),
            ),
            (
                "dynamic_quant.value.scale",
                value_rows_scale,
                expected_value_scale.reshape(-1),
            ),
            ("pack.key", packed_key, expected_packed_key),
            ("pack.value", packed_value, expected_packed_value),
            ("scatter_gather.key", gathered_key, expected_packed_key),
            ("scatter_gather.value", gathered_value, expected_packed_value),
            ("unpack.key.data", decoded_key_data, expected_key),
            ("unpack.value.data", decoded_value_data, expected_value),
            ("unpack.key.scale", decoded_key_scale, expected_key_scale),
            ("unpack.value.scale", decoded_value_scale, expected_value_scale),
            ("scaled_decode.key", decoded_key, expected_decoded_key),
            ("scaled_decode.value", decoded_value, expected_decoded_value),
        )
        failures = [
            summary
            for stage, actual, expected in comparisons
            if (
                summary := self._mismatch_summary(stage, actual, expected)
            ) is not None
        ]
        if failures:
            self.fail("production-shape HIF8 KV mismatch:\n" + "\n".join(failures))

    def test_qwen35_dynamic_nonzero_logical_blocks_round_trip(self):
        """Cover the virtual manager-block layout used by batched Qwen3.5.

        A packed HIF8 manager block contains 2048 tokens, while Ascend dense
        attention exposes 128-token kernel pages.  Consequently the first
        request in each newly allocated manager block starts at kernel block
        IDs 0, 16, 32, and 48.  Reproduce that exact four-request layout and
        validate every byte and scale after the real NPU scatter transport.
        """

        import torch_npu

        from vllm_ascend_hif8 import native
        from vllm_ascend_hif8.kv_cache import (
            decode_kv_rows,
            gather_kv_rows,
            pack_kv_rows,
            quantize_kv_rows,
            unpack_kv_rows,
        )

        request_count = 4
        tokens_per_request = 89
        first_block_ids = [0, 16, 32, 48]
        slot_bases = [block_id * self.BLOCK_SIZE for block_id in first_block_ids]
        total_tokens = request_count * tokens_per_request
        generator = torch.Generator().manual_seed(2026082203)
        request_amplitudes = torch.tensor(
            [0.25, 0.75, 1.5, 4.0], dtype=torch.float32
        ).reshape(request_count, 1, 1, 1)
        head_amplitudes = torch.tensor(
            [0.5, 1.0, 2.0, 3.0], dtype=torch.float32
        ).reshape(1, 1, self.NUM_KV_HEADS, 1)
        key_cpu = (
            torch.randn(
                request_count,
                tokens_per_request,
                self.NUM_KV_HEADS,
                self.HEAD_SIZE,
                generator=generator,
                dtype=torch.float32,
            )
            .mul_(request_amplitudes)
            .mul_(head_amplitudes)
            .reshape(total_tokens, self.NUM_KV_HEADS, self.HEAD_SIZE)
            .to(torch.bfloat16)
        )
        value_cpu = (
            torch.randn(
                request_count,
                tokens_per_request,
                self.NUM_KV_HEADS,
                self.HEAD_SIZE,
                generator=generator,
                dtype=torch.float32,
            )
            .mul_(request_amplitudes.flip(0))
            .mul_(head_amplitudes.flip(2))
            .reshape(total_tokens, self.NUM_KV_HEADS, self.HEAD_SIZE)
            .to(torch.bfloat16)
        )
        expected_key, expected_key_scale = quantize_kv_rows(key_cpu)
        expected_value, expected_value_scale = quantize_kv_rows(value_cpu)
        expected_packed_key = pack_kv_rows(expected_key, expected_key_scale)
        expected_packed_value = pack_kv_rows(expected_value, expected_value_scale)

        key = key_cpu.to(self.device)
        value = value_cpu.to(self.device)
        key_data, key_scale = native.dynamic_quant(
            key.reshape(-1, self.HEAD_SIZE)
        )
        value_data, value_scale = native.dynamic_quant(
            value.reshape(-1, self.HEAD_SIZE)
        )
        packed_key = pack_kv_rows(
            key_data.reshape_as(key),
            key_scale.reshape(total_tokens, self.NUM_KV_HEADS).contiguous(),
        )
        packed_value = pack_kv_rows(
            value_data.reshape_as(value),
            value_scale.reshape(total_tokens, self.NUM_KV_HEADS).contiguous(),
        )

        key_cache, value_cache = self._cache_pair(num_blocks=64)
        slots = torch.tensor(
            [
                base + offset
                for base in slot_bases
                for offset in range(tokens_per_request)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        key_cache.reshape(-1, *key_cache.shape[2:]).index_copy_(0, slots, packed_key)
        value_cache.reshape(-1, *value_cache.shape[2:]).index_copy_(0, slots, packed_value)

        block_table = torch.tensor(
            [[block_id] for block_id in first_block_ids],
            dtype=torch.int64,
            device=self.device,
        )
        seq_lens = [tokens_per_request] * request_count
        gathered_key = torch.cat(
            gather_kv_rows(
                key_cache, block_table, seq_lens, block_size=self.BLOCK_SIZE
            ),
            dim=0,
        )
        gathered_value = torch.cat(
            gather_kv_rows(
                value_cache, block_table, seq_lens, block_size=self.BLOCK_SIZE
            ),
            dim=0,
        )
        unpacked_key, unpacked_key_scale = unpack_kv_rows(
            gathered_key, self.HEAD_SIZE
        )
        unpacked_value, unpacked_value_scale = unpack_kv_rows(
            gathered_value, self.HEAD_SIZE
        )
        decoded_key = native.scaled_decode(
            unpacked_key.reshape(-1, self.HEAD_SIZE),
            unpacked_key_scale.reshape(-1),
        ).reshape_as(key)
        decoded_value = native.scaled_decode(
            unpacked_value.reshape(-1, self.HEAD_SIZE),
            unpacked_value_scale.reshape(-1),
        ).reshape_as(value)
        torch.npu.synchronize()

        expected_decoded_key = decode_kv_rows(
            expected_key, expected_key_scale
        ).to(torch.bfloat16)
        expected_decoded_value = decode_kv_rows(
            expected_value, expected_value_scale
        ).to(torch.bfloat16)
        comparisons = (
            ("dynamic_quant.key.data", key_data.reshape_as(key), expected_key),
            ("dynamic_quant.value.data", value_data.reshape_as(value), expected_value),
            (
                "dynamic_quant.key.scale",
                key_scale.reshape(total_tokens, self.NUM_KV_HEADS),
                expected_key_scale,
            ),
            (
                "dynamic_quant.value.scale",
                value_scale.reshape(total_tokens, self.NUM_KV_HEADS),
                expected_value_scale,
            ),
            ("pack.key", packed_key, expected_packed_key),
            ("pack.value", packed_value, expected_packed_value),
            ("scatter_gather.key", gathered_key, expected_packed_key),
            ("scatter_gather.value", gathered_value, expected_packed_value),
            ("unpack.key.data", unpacked_key, expected_key),
            ("unpack.value.data", unpacked_value, expected_value),
            ("unpack.key.scale", unpacked_key_scale, expected_key_scale),
            ("unpack.value.scale", unpacked_value_scale, expected_value_scale),
            ("scaled_decode.key", decoded_key, expected_decoded_key),
            ("scaled_decode.value", decoded_value, expected_decoded_value),
        )
        failures = [
            summary
            for stage, actual, expected in comparisons
            if (
                summary := self._mismatch_summary(stage, actual, expected)
            ) is not None
        ]
        if failures:
            self.fail(
                "nonzero-logical-block HIF8 KV mismatch:\n" + "\n".join(failures)
            )

    def test_qwen35_production_shape_multi_request_fia(self):
        """Validate the exact first-decode TND shape used by Qwen3.5-9B."""

        production_num_heads = 32
        q_lens = [1, 2, 3, 4]
        kv_lens = [90, 180, 270, 360]
        generator = torch.Generator().manual_seed(2026082204)
        query = torch.randn(
            q_lens[-1],
            production_num_heads,
            self.HEAD_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(self.device)
        key = torch.randn(
            kv_lens[-1],
            self.NUM_KV_HEADS,
            self.HEAD_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(self.device)
        value = torch.randn(
            kv_lens[-1],
            self.NUM_KV_HEADS,
            self.HEAD_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(self.device)
        actual, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key.contiguous(),
            value=value.contiguous(),
            input_layout="TND",
            actual_seq_lengths=q_lens,
            actual_seq_lengths_kv=kv_lens,
            num_key_value_heads=self.NUM_KV_HEADS,
            num_heads=production_num_heads,
            scale=1.0 / math.sqrt(self.HEAD_SIZE),
            sparse_mode=3,
            atten_mask=self.causal_mask,
        )
        actual = actual.view(
            q_lens[-1], production_num_heads, self.HEAD_SIZE
        )
        expected = self._dense_math_reference(
            query,
            key,
            value,
            q_lens,
            kv_lens,
            num_heads=production_num_heads,
        )
        torch.npu.synchronize()

        q_begin = 0
        for request, q_end in enumerate(q_lens):
            with self.subTest(request=request):
                request_output = actual[q_begin:q_end]
                request_expected = expected[q_begin:q_end]
                self.assertTrue(bool(torch.isfinite(request_output).all()))
                self.assertGreater(
                    self._cosine(request_output.cpu(), request_expected), 0.999
                )
            q_begin = q_end

    def test_single_request_prefill_then_decode(self):
        generator = torch.Generator().manual_seed(17)
        impl = self._make_impl()
        cache = self._cache_pair()
        layer = SimpleNamespace(head_size_v=self.HEAD_SIZE)

        query = torch.randn(
            6,
            self.NUM_HEADS,
            self.HEAD_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(self.device)
        key = torch.randn(
            6,
            self.NUM_KV_HEADS,
            self.HEAD_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(self.device)
        value = torch.randn(
            6,
            self.NUM_KV_HEADS,
            self.HEAD_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(self.device)
        metadata = self._metadata(
            q_cumulative=[6],
            seq_lens=[6],
            slot_mapping=range(6),
            block_tables=[[0, 1]],
            num_actual_tokens=6,
        )
        actual = torch.empty_like(query)
        impl.forward(layer, query, key, value, cache, metadata, actual)
        expected = self._bf16_fia(query, key, value, [6], [6])
        expected_math = self._paged_math_reference(query, cache, metadata)

        decode_query = torch.randn(
            1,
            self.NUM_HEADS,
            self.HEAD_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(self.device)
        decode_key = torch.randn(
            1,
            self.NUM_KV_HEADS,
            self.HEAD_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(self.device)
        decode_value = torch.randn(
            1,
            self.NUM_KV_HEADS,
            self.HEAD_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(self.device)
        decode_metadata = self._metadata(
            q_cumulative=[1],
            seq_lens=[7],
            slot_mapping=[6],
            block_tables=[[0, 1]],
            num_actual_tokens=1,
        )
        decode_actual = torch.empty_like(decode_query)
        impl.forward(
            layer,
            decode_query,
            decode_key,
            decode_value,
            cache,
            decode_metadata,
            decode_actual,
        )
        expected_decode = self._bf16_fia(
            decode_query,
            torch.cat([key, decode_key], dim=0),
            torch.cat([value, decode_value], dim=0),
            [1],
            [7],
        )
        expected_decode_math = self._paged_math_reference(
            decode_query, cache, decode_metadata
        )
        torch.npu.synchronize()

        self.assertEqual(cache[0].dtype, torch.uint8)
        self.assertEqual(cache[0].shape[-1], 260)
        self.assertGreater(self._cosine(actual, expected), 0.999)
        self.assertGreater(self._cosine(decode_actual, expected_decode), 0.999)
        self.assertGreater(self._cosine(actual.cpu(), expected_math), 0.999)
        self.assertGreater(
            self._cosine(decode_actual.cpu(), expected_decode_math), 0.999
        )

    def test_mixed_decode_and_chunked_prefill_batch(self):
        generator = torch.Generator().manual_seed(29)
        impl = self._make_impl()
        cache = self._cache_pair()
        layer = SimpleNamespace(head_size_v=self.HEAD_SIZE)

        history_k = torch.randn(
            6,
            self.NUM_KV_HEADS,
            self.HEAD_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(self.device)
        history_v = torch.randn(
            6,
            self.NUM_KV_HEADS,
            self.HEAD_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(self.device)
        impl._write_hif8_cache(
            history_k,
            history_v,
            cache,
            torch.tensor([0, 1, 2, 3, 128, 129], device=self.device),
            6,
        )

        query = torch.randn(
            4,
            self.NUM_HEADS,
            self.HEAD_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(self.device)
        current_k = torch.randn(
            4,
            self.NUM_KV_HEADS,
            self.HEAD_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(self.device)
        current_v = torch.randn(
            4,
            self.NUM_KV_HEADS,
            self.HEAD_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(self.device)
        metadata = self._metadata(
            q_cumulative=[1, 4],
            seq_lens=[5, 5],
            slot_mapping=[4, 130, 131, 132],
            block_tables=[[0, 0], [1, 0]],
            num_actual_tokens=4,
        )
        actual = torch.empty_like(query)
        impl.forward(layer, query, current_k, current_v, cache, metadata, actual)

        dense_key = torch.cat(
            [history_k[:4], current_k[:1], history_k[4:], current_k[1:]],
            dim=0,
        )
        dense_value = torch.cat(
            [history_v[:4], current_v[:1], history_v[4:], current_v[1:]],
            dim=0,
        )
        expected = self._bf16_fia(query, dense_key, dense_value, [1, 4], [5, 10])
        expected_math = self._paged_math_reference(query, cache, metadata)
        torch.npu.synchronize()

        self.assertGreater(self._cosine(actual, expected), 0.999)
        self.assertGreater(self._cosine(actual[:1], expected[:1]), 0.999)
        self.assertGreater(self._cosine(actual[1:], expected[1:]), 0.999)
        self.assertGreater(self._cosine(actual.cpu(), expected_math), 0.999)

    def test_multi_request_prefill_then_repeated_decode_without_step_sync(self):
        generator = torch.Generator().manual_seed(20260822)
        impl = self._make_impl()
        cache = self._cache_pair()
        layer = SimpleNamespace(head_size_v=self.HEAD_SIZE)

        def randn(rows, heads):
            return torch.randn(
                rows,
                heads,
                self.HEAD_SIZE,
                generator=generator,
                dtype=torch.bfloat16,
            ).to(self.device)

        actual_outputs = []
        expected_outputs = []

        # Step 0: request 0 enters alone through the regular prefill path.
        query = randn(5, self.NUM_HEADS)
        key = randn(5, self.NUM_KV_HEADS)
        value = randn(5, self.NUM_KV_HEADS)
        metadata = self._metadata(
            q_cumulative=[5],
            seq_lens=[5],
            slot_mapping=range(5),
            block_tables=[[0, 0]],
            num_actual_tokens=5,
        )
        actual = torch.empty_like(query)
        impl.forward(layer, query, key, value, cache, metadata, actual)
        expected = self._bf16_fia(query, key, value, [5], [5])
        actual_outputs.append(actual)
        expected_outputs.append(expected)
        history_k = [key]
        history_v = [value]

        # Step 1: request 0 decodes one token while request 1 is prefilling.
        query = randn(7, self.NUM_HEADS)
        key = randn(7, self.NUM_KV_HEADS)
        value = randn(7, self.NUM_KV_HEADS)
        metadata = self._metadata(
            q_cumulative=[1, 7],
            seq_lens=[6, 6],
            slot_mapping=[5, *range(128, 134)],
            block_tables=[[0, 0], [1, 0]],
            num_actual_tokens=7,
        )
        actual = torch.empty_like(query)
        impl.forward(layer, query, key, value, cache, metadata, actual)
        history_k = [torch.cat([history_k[0], key[:1]]), key[1:]]
        history_v = [torch.cat([history_v[0], value[:1]]), value[1:]]
        dense_key = torch.cat(history_k)
        dense_value = torch.cat(history_v)
        expected = self._bf16_fia(query, dense_key, dense_value, [1, 7], [6, 12])
        actual_outputs.append(actual)
        expected_outputs.append(expected)

        # Steps 2+: both requests decode together. Do not synchronize between
        # steps: this catches cache-write/decode lifetime bugs hidden by item(),
        # tolist(), or explicit stream synchronization in diagnostic code.
        for offset in range(3):
            query = randn(2, self.NUM_HEADS)
            key = randn(2, self.NUM_KV_HEADS)
            value = randn(2, self.NUM_KV_HEADS)
            sequence_length = 7 + offset
            metadata = self._metadata(
                q_cumulative=[1, 2],
                seq_lens=[sequence_length, sequence_length],
                slot_mapping=[6 + offset, 134 + offset],
                block_tables=[[0, 0], [1, 0]],
                num_actual_tokens=2,
            )
            actual = torch.empty_like(query)
            impl.forward(layer, query, key, value, cache, metadata, actual)
            history_k = [
                torch.cat([history_k[0], key[:1]]),
                torch.cat([history_k[1], key[1:]]),
            ]
            history_v = [
                torch.cat([history_v[0], value[:1]]),
                torch.cat([history_v[1], value[1:]]),
            ]
            dense_key = torch.cat(history_k)
            dense_value = torch.cat(history_v)
            expected = self._bf16_fia(
                query,
                dense_key,
                dense_value,
                [1, 2],
                [sequence_length, 2 * sequence_length],
            )
            actual_outputs.append(actual)
            expected_outputs.append(expected)

        # Preserve the no-sync property of every update above, then validate
        # the final two-request decode against cache bytes plus an independent
        # QK/softmax/PV implementation after the lifetime regression is over.
        last_math = self._paged_math_reference(query, cache, metadata)

        torch.npu.synchronize()
        for step, (actual, expected) in enumerate(
            zip(actual_outputs, expected_outputs, strict=True)
        ):
            with self.subTest(step=step):
                self.assertTrue(bool(torch.isfinite(actual).all()))
                self.assertGreater(self._cosine(actual, expected), 0.999)
        self.assertGreater(self._cosine(actual_outputs[-1].cpu(), last_math), 0.999)


if __name__ == "__main__":
    unittest.main()
