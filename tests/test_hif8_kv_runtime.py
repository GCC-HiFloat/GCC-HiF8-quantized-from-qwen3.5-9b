from __future__ import annotations

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.config.cache import CacheConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    MLAAttentionSpec,
    MambaSpec,
    SinkFullAttentionSpec,
    SlidingWindowSpec,
)
from vllm_ascend_hif8.kv_cache import Hif8KVCacheLayout
from vllm_ascend_hif8 import kv_runtime


class Hif8KVRuntimeTest(unittest.TestCase):
    def setUp(self):
        kv_runtime.reset_hif8_kv_cache_adapter_for_tests()
        self.environment = patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        os.environ.pop(kv_runtime.HIF8_KV_CACHE_ENV, None)

    def tearDown(self):
        kv_runtime.reset_hif8_kv_cache_adapter_for_tests()
        self.environment.stop()

    @staticmethod
    def _logical_spec(
        *,
        block_size: int = 128,
        head_size: int = 256,
        head_size_v: int = 256,
    ) -> FullAttentionSpec:
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=4,
            head_size=head_size,
            head_size_v=head_size_v,
            dtype=torch.uint8,
            kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD,
        )

    @staticmethod
    def _runtime_config(
        *,
        enforce_eager: bool = True,
        cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        use_v2_model_runner: bool = False,
        dcp: int = 1,
        pcp: int = 1,
        async_scheduling: bool = False,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=enforce_eager),
            compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
            scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=dcp,
                prefill_context_parallel_size=pcp,
            ),
            use_v2_model_runner=use_v2_model_runner,
        )

    def test_flag_off_leaves_cache_config_constructor_unchanged(self):
        original = CacheConfig.__init__
        kv_runtime.install_hif8_kv_cache_adapter()
        self.assertIs(CacheConfig.__init__, original)
        self.assertEqual(CacheConfig(cache_dtype="auto").cache_dtype, "auto")

    def test_flag_on_normalizes_auto_and_positional_cache_dtype(self):
        os.environ[kv_runtime.HIF8_KV_CACHE_ENV] = "1"
        kv_runtime.install_hif8_kv_cache_adapter()

        self.assertEqual(CacheConfig(cache_dtype="auto").cache_dtype, "fp8_per_token_head")
        positional = CacheConfig(128, None, 0.8, "auto")
        self.assertEqual(positional.block_size, 128)
        self.assertEqual(positional.cache_dtype, "fp8_per_token_head")
        explicit = CacheConfig(cache_dtype="fp8_per_token_head")
        self.assertEqual(explicit.cache_dtype, "fp8_per_token_head")

    def test_flag_on_rejects_other_public_cache_dtypes(self):
        os.environ[kv_runtime.HIF8_KV_CACHE_ENV] = "1"
        kv_runtime.install_hif8_kv_cache_adapter()
        with self.assertRaisesRegex(ValueError, "requires --kv-cache-dtype"):
            CacheConfig(cache_dtype="bfloat16")

    def test_physical_spec_matches_inline_sidecar_layout(self):
        physical = kv_runtime._physical_attention_spec(self._logical_spec())
        layout = Hif8KVCacheLayout(1, 128, 4, 256)

        self.assertIs(type(physical), FullAttentionSpec)
        self.assertEqual(physical.head_size, 260)
        self.assertEqual(physical.head_size_v, 260)
        self.assertEqual(physical.dtype, torch.uint8)
        self.assertEqual(physical.kv_quant_mode, KVQuantMode.NONE)
        self.assertEqual(physical.page_size_bytes, 266_240)
        self.assertEqual(physical.page_size_bytes, layout.page_size_bytes)
        self.assertEqual(physical._hif8_logical_head_size, 256)
        self.assertEqual(physical._hif8_logical_head_size_v, 256)
        self.assertIs(kv_runtime._physical_attention_spec(physical), physical)

    def test_physical_spec_supports_different_k_and_v_widths(self):
        physical = kv_runtime._physical_attention_spec(
            self._logical_spec(head_size=256, head_size_v=128)
        )
        self.assertEqual((physical.head_size, physical.head_size_v), (260, 132))
        self.assertEqual(physical.page_size_bytes, 128 * 4 * (260 + 132))

    def test_hybrid_layout_separates_mamba_and_hif8_payloads(self):
        physical = kv_runtime._physical_attention_spec(
            self._logical_spec(block_size=2048)
        )
        # Production Qwen3.5 TP=1 shapes: 49,152-byte BF16 conv state and a
        # 2,097,152-byte FP32 recurrent state per manager block.
        mamba = MambaSpec(
            shapes=((3, 8192), (32, 128, 128)),
            dtypes=(torch.bfloat16, torch.float32),
            block_size=2048,
            # The stock fp8_per_token_head planner reserves BF16 K/V plus the
            # conv prefix before the HIF8 adapter adds inline scales.
            page_size_padded=4_243_456,
        )
        planned = kv_runtime._plan_separated_hybrid_cache_specs(
            {"linear_attn": mamba, "self_attn": physical}
        )

        attention_payload = getattr(physical, "real_page_size_bytes", None)
        if attention_payload is None:
            attention_payload = getattr(physical, "unpadded_page_size_bytes")
        mamba_payload = 49_152 + 2_097_152
        common_page = attention_payload + mamba_payload
        self.assertEqual(attention_payload, 4_259_840)
        self.assertEqual(mamba_payload, 2_146_304)
        self.assertEqual(common_page, 6_406_144)
        for spec in planned.values():
            self.assertEqual(spec.page_size_bytes, common_page)
            self.assertTrue(spec._hif8_hybrid_separated)
            self.assertEqual(
                spec._hif8_hybrid_attention_payload_bytes,
                attention_payload,
            )
            self.assertEqual(
                spec._hif8_hybrid_mamba_payload_bytes,
                mamba_payload,
            )

        # Model the runner's exact physical policy: recurrent states begin at
        # the slab base and attention K/V occupy the tail.  Every manager block
        # must remain disjoint, including a non-zero global block ID.
        num_blocks = 7
        slab = torch.zeros(num_blocks * common_page, dtype=torch.uint8)
        recurrent = slab[: num_blocks * mamba_payload].view(
            num_blocks, mamba_payload
        )
        attention = slab[
            num_blocks * mamba_payload :
        ].view(num_blocks, attention_payload)
        attention[5].fill_(0xA5)
        snapshot = attention.clone()
        recurrent[4].fill_(0x5A)
        recurrent[5].fill_(0x3C)
        self.assertTrue(torch.equal(attention, snapshot))
        self.assertGreaterEqual(
            attention.data_ptr(),
            recurrent.data_ptr() + recurrent.numel(),
        )

    def test_hybrid_layout_is_noop_without_both_cache_types(self):
        physical = kv_runtime._physical_attention_spec(self._logical_spec())
        attention_only = {"self_attn": physical}
        self.assertIs(
            kv_runtime._plan_separated_hybrid_cache_specs(attention_only),
            attention_only,
        )
        marker = {"other": object()}
        self.assertIs(
            kv_runtime._plan_separated_hybrid_cache_specs(marker), marker
        )

    def test_attention_layer_index_accepts_real_qwen35_prefixes(self):
        for prefix, expected in (
            ("language_model.model.layers.3.self_attn.attn", 3),
            ("model.layers.31.self_attn.attn", 31),
            ("layers.7.self_attn.attn", 7),
        ):
            self.assertEqual(kv_runtime._attention_layer_index(prefix), expected)
        for invalid in ("self_attn.attn", "layers.3.layers.7.attn"):
            with self.assertRaisesRegex(ValueError, "one decoder layer index"):
                kv_runtime._attention_layer_index(invalid)

    def test_static_kv_metadata_is_loaded_and_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = {
                "format": "W8A8HIF8_STATIC",
                "kv_cache": {
                    "mode": "static_per_layer_per_head",
                    "layers": {
                        "3": {
                            "key_scales": [0.1, 0.2],
                            "value_scales": [0.3, 0.4],
                        }
                    },
                },
            }
            (root / kv_runtime.HIF8_METADATA_FILENAME).write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            actual = kv_runtime._load_static_kv_scale_values(str(root))
            self.assertEqual(actual, {3: ((0.1, 0.2), (0.3, 0.4))})

            kv_runtime._load_static_kv_scale_values.cache_clear()
            metadata["kv_cache"]["layers"]["3"]["key_scales"][0] = 0.0
            (root / kv_runtime.HIF8_METADATA_FILENAME).write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "invalid key_scales"):
                kv_runtime._load_static_kv_scale_values(str(root))

    def test_static_checkpoint_never_silently_uses_dynamic_kv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / kv_runtime.HIF8_METADATA_FILENAME).write_text(
                json.dumps({"format": "W8A8HIF8_STATIC"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "missing kv_cache metadata"):
                kv_runtime._load_static_kv_scale_values(str(root))

    def test_legacy_checkpoint_without_static_metadata_keeps_dynamic_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertIsNone(kv_runtime._load_static_kv_scale_values(str(root)))
            kv_runtime._load_static_kv_scale_values.cache_clear()
            (root / kv_runtime.HIF8_METADATA_FILENAME).write_text(
                json.dumps({"format": "W8A8HIF8_SOFT"}), encoding="utf-8"
            )
            self.assertIsNone(kv_runtime._load_static_kv_scale_values(str(root)))

    def test_non_cache_specs_pass_through_but_special_attention_is_rejected(self):
        marker = object()
        self.assertIs(kv_runtime._physical_attention_spec(marker), marker)
        unsupported = (
            MLAAttentionSpec(
                block_size=128,
                num_kv_heads=1,
                head_size=256,
                dtype=torch.uint8,
            ),
            SinkFullAttentionSpec(
                block_size=128,
                num_kv_heads=4,
                head_size=256,
                dtype=torch.uint8,
                sink_len=4,
            ),
            SlidingWindowSpec(
                block_size=128,
                num_kv_heads=4,
                head_size=256,
                dtype=torch.uint8,
                sliding_window=4096,
            ),
        )
        for spec in unsupported:
            with self.subTest(spec=type(spec).__name__):
                with self.assertRaisesRegex(NotImplementedError, type(spec).__name__):
                    kv_runtime._physical_attention_spec(spec)

    def test_cache_binding_accepts_pair_and_equal_width_combined_tensor(self):
        key = torch.empty((2, 128, 4, 260), dtype=torch.uint8)
        value = torch.empty((2, 128, 4, 260), dtype=torch.uint8)
        bound_key, bound_value = kv_runtime._bind_hif8_cache_tensors(
            (key, value),
            logical_head_size=256,
            logical_head_size_v=256,
        )
        self.assertIs(bound_key, key)
        self.assertIs(bound_value, value)

        combined = torch.empty((2, 2, 128, 4, 260), dtype=torch.uint8)
        bound_key, bound_value = kv_runtime._bind_hif8_cache_tensors(
            combined,
            logical_head_size=256,
            logical_head_size_v=256,
        )
        self.assertEqual(bound_key.data_ptr(), combined[0].data_ptr())
        self.assertEqual(bound_value.data_ptr(), combined[1].data_ptr())

    def test_cache_binding_rejects_wrong_contracts(self):
        key = torch.empty((2, 128, 4, 260), dtype=torch.uint8)
        value = torch.empty((2, 128, 4, 132), dtype=torch.uint8)
        kv_runtime._bind_hif8_cache_tensors(
            [key, value],
            logical_head_size=256,
            logical_head_size_v=128,
        )
        with self.assertRaisesRegex(ValueError, "exactly one K/V cache pair"):
            kv_runtime._bind_hif8_cache_tensors(
                [key, value, value],
                logical_head_size=256,
                logical_head_size_v=128,
            )
        with self.assertRaisesRegex(ValueError, "value cache width"):
            kv_runtime._bind_hif8_cache_tensors(
                [key, key],
                logical_head_size=256,
                logical_head_size_v=128,
            )
        with self.assertRaisesRegex(TypeError, "4-D uint8"):
            kv_runtime._bind_hif8_cache_tensors(
                [key.to(torch.float16), value],
                logical_head_size=256,
                logical_head_size_v=128,
            )

    def test_causal_attention_refuses_missing_ascend_mask_before_fia(self):
        from unittest.mock import patch

        # Extracting the nested implementation class requires the real Ascend
        # backend, so use the installed adapter and a minimal eager config.
        os.environ[kv_runtime.HIF8_KV_CACHE_ENV] = "1"
        kv_runtime.install_hif8_kv_cache_adapter()
        with patch.object(kv_runtime, "_validate_hif8_runtime_config"):
            import vllm_ascend.utils as ascend_utils

            ascend_utils.adapt_patch(False)
            from vllm_ascend.attention import attention_v1

            impl_cls = attention_v1.AscendAttentionBackend.get_impl_cls()
        minimal = object.__new__(impl_cls)
        minimal.head_size = 4
        minimal.head_size_v = 4
        minimal.num_heads = 1
        minimal.num_kv_heads = 1
        minimal.scale = 0.5
        metadata = SimpleNamespace(
            causal=True,
            attn_mask=None,
            actual_seq_lengths_q=[1],
            seq_lens_list=[1],
            num_actual_tokens=1,
            block_tables=None,
        )
        query = torch.zeros((1, 1, 4), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "requires.*attention mask"):
            impl_cls._forward_hif8_attention(
                minimal,
                query,
                query,
                query,
                metadata,
                torch.empty_like(query),
            )

    def test_runtime_kv_quantization_routes_static_and_legacy_paths(self):
        from unittest.mock import patch

        os.environ[kv_runtime.HIF8_KV_CACHE_ENV] = "1"
        kv_runtime.install_hif8_kv_cache_adapter()
        with patch.object(kv_runtime, "_validate_hif8_runtime_config"):
            import vllm_ascend.utils as ascend_utils

            ascend_utils.adapt_patch(False)
            from vllm_ascend.attention import attention_v1

            impl_cls = attention_v1.AscendAttentionBackend.get_impl_cls()

        value = torch.randn((2, 2, 4), dtype=torch.bfloat16)
        data = torch.zeros((4, 4), dtype=torch.uint8)
        scale = torch.ones(4, dtype=torch.float32)
        fixed = torch.tensor([0.1, 0.2], dtype=torch.float32)
        minimal = object.__new__(impl_cls)

        with (
            patch(
                "vllm_ascend_hif8.native.static_quant",
                return_value=(data, scale),
            ) as static_quant,
            patch(
                "vllm_ascend_hif8.native.dynamic_quant",
                return_value=(data, scale),
            ) as dynamic_quant,
        ):
            packed = impl_cls._quantize_kv(minimal, value, fixed_scale=fixed)
            self.assertEqual(tuple(packed.shape), (2, 2, 8))
            static_quant.assert_called_once()
            self.assertIs(static_quant.call_args.args[1], fixed)
            dynamic_quant.assert_not_called()

        with (
            patch(
                "vllm_ascend_hif8.native.static_quant",
                return_value=(data, scale),
            ) as static_quant,
            patch(
                "vllm_ascend_hif8.native.dynamic_quant",
                return_value=(data, scale),
            ) as dynamic_quant,
        ):
            impl_cls._quantize_kv(minimal, value)
            dynamic_quant.assert_called_once()
            static_quant.assert_not_called()

    def test_runtime_config_requires_eager_or_graph_none(self):
        kv_runtime._validate_hif8_runtime_config(self._runtime_config())
        kv_runtime._validate_hif8_runtime_config(
            self._runtime_config(enforce_eager=False, cudagraph_mode=CUDAGraphMode.NONE)
        )
        with self.assertRaisesRegex(NotImplementedError, "--enforce-eager"):
            kv_runtime._validate_hif8_runtime_config(
                self._runtime_config(
                    enforce_eager=False,
                    cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                )
            )
        with self.assertRaisesRegex(NotImplementedError, "runner V1"):
            kv_runtime._validate_hif8_runtime_config(
                self._runtime_config(use_v2_model_runner=True)
            )
        with self.assertRaisesRegex(NotImplementedError, "no-async-scheduling"):
            kv_runtime._validate_hif8_runtime_config(
                self._runtime_config(async_scheduling=True)
            )
        for field in ("dcp", "pcp"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(NotImplementedError, "DCP/PCP"):
                    kv_runtime._validate_hif8_runtime_config(
                        self._runtime_config(**{field: 2})
                    )

    def test_deferred_patch_does_not_import_attention_v1_until_worker_patch(self):
        os.environ[kv_runtime.HIF8_KV_CACHE_ENV] = "1"
        sys.modules.pop("vllm_ascend.attention.attention_v1", None)
        with patch.object(kv_runtime, "_install_ascend_attention_patch") as install:
            kv_runtime.install_hif8_kv_cache_adapter()
            self.assertNotIn("vllm_ascend.attention.attention_v1", sys.modules)

            import vllm_ascend.utils as ascend_utils

            ascend_utils.adapt_patch(is_global_patch=True)
            install.assert_not_called()
            ascend_utils.adapt_patch(is_global_patch=False)
            install.assert_called_once_with()
            ascend_utils.adapt_patch(is_global_patch=False)
            self.assertEqual(install.call_count, 2)

    def test_install_is_idempotent(self):
        os.environ[kv_runtime.HIF8_KV_CACHE_ENV] = "1"
        kv_runtime.install_hif8_kv_cache_adapter()
        cache_init = CacheConfig.__init__

        import vllm_ascend.utils as ascend_utils

        deferred_patch = ascend_utils.adapt_patch
        kv_runtime.install_hif8_kv_cache_adapter()
        self.assertIs(CacheConfig.__init__, cache_init)
        self.assertIs(ascend_utils.adapt_patch, deferred_patch)


if __name__ == "__main__":
    unittest.main()
