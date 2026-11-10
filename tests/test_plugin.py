import sys
import types
import unittest
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class PluginRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.saved = {name: sys.modules.get(name) for name in (
            "torch",
            "vllm_ascend",
            "vllm_ascend.quantization",
            "vllm_ascend.quantization.methods",
            "vllm_ascend.quantization.methods.base",
            "vllm_ascend.quantization.methods.registry",
            "vllm_ascend.quantization.method_adapters",
            "vllm_ascend.ops",
            "vllm_ascend.ops.vocab_parallel_embedding",
            "vllm",
            "vllm.config",
        )}
        # Another test may already have imported the scheme against the real
        # vllm_ascend. ``from . import scheme`` resolves through the parent
        # package attribute, so clearing sys.modules alone is not enough: the
        # stale module would be handed back and the fake registry below would
        # never see a registration.
        import vllm_ascend_hif8

        self.package = vllm_ascend_hif8
        self.saved_scheme = sys.modules.pop("vllm_ascend_hif8.scheme", None)
        self.saved_attribute = getattr(vllm_ascend_hif8, "scheme", None)
        if self.saved_attribute is not None:
            delattr(vllm_ascend_hif8, "scheme")

    def tearDown(self):
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.modules.pop("vllm_ascend_hif8.scheme", None)
        if self.saved_scheme is not None:
            sys.modules["vllm_ascend_hif8.scheme"] = self.saved_scheme
        if self.saved_attribute is not None:
            self.package.scheme = self.saved_attribute
        else:
            # The fake-registry import left a stub bound to the parent package.
            if hasattr(self.package, "scheme"):
                delattr(self.package, "scheme")
        import vllm_ascend_hif8.plugin as plugin
        plugin._registered = False

    def test_register_is_idempotent_and_registers_linear_scheme(self):
        registry = {}

        class FakeTensor:
            pass

        class FakeDType:
            pass

        class FakeModule:
            pass

        class FakeTorch(types.ModuleType):
            uint8 = "uint8"
            float32 = "float32"
            dtype = FakeDType
            Tensor = FakeTensor
            nn = types.SimpleNamespace(Module=FakeModule)

            @staticmethod
            def empty(*shape, dtype=None):
                return types.SimpleNamespace(shape=shape, dtype=dtype)

        def register_scheme(quant_type, layer_type):
            def decorator(cls):
                registry[(quant_type, layer_type)] = cls
                return cls
            return decorator

        class FakeEmbeddingMethod:
            def __init__(self, quant_method):
                self.quant_method = quant_method

        class FakeVocabParallelEmbedding:
            def __init__(
                self,
                num_embeddings,
                embedding_dim,
                params_dtype=None,
                org_num_embeddings=None,
                padding_size=64,
                quant_config=None,
                prefix="",
            ):
                self.init_args = (
                    num_embeddings,
                    embedding_dim,
                    params_dtype,
                    org_num_embeddings,
                    padding_size,
                    quant_config,
                    prefix,
                )
                self.init_kwargs = {
                    "quant_config": quant_config,
                    "prefix": prefix,
                }

        modules = {
            "torch": FakeTorch("torch"),
            "vllm_ascend": types.ModuleType("vllm_ascend"),
            "vllm_ascend.quantization": types.ModuleType("vllm_ascend.quantization"),
            "vllm_ascend.quantization.methods": types.ModuleType("vllm_ascend.quantization.methods"),
            "vllm_ascend.quantization.methods.base": types.ModuleType("vllm_ascend.quantization.methods.base"),
            "vllm_ascend.quantization.methods.registry": types.ModuleType("vllm_ascend.quantization.methods.registry"),
            "vllm_ascend.quantization.method_adapters": types.ModuleType(
                "vllm_ascend.quantization.method_adapters"
            ),
            "vllm_ascend.ops": types.ModuleType("vllm_ascend.ops"),
            "vllm_ascend.ops.vocab_parallel_embedding": types.ModuleType(
                "vllm_ascend.ops.vocab_parallel_embedding"
            ),
            "vllm": types.ModuleType("vllm"),
            "vllm.config": types.ModuleType("vllm.config"),
        }
        modules["vllm_ascend"].__path__ = []
        modules["vllm_ascend.quantization"].__path__ = []
        modules["vllm_ascend.quantization.methods"].__path__ = []
        modules["vllm_ascend.ops"].__path__ = []
        modules["vllm"].__path__ = []
        modules["vllm_ascend.quantization.methods.base"].AscendLinearScheme = object
        modules["vllm_ascend.quantization.methods.registry"].register_scheme = register_scheme
        modules["vllm_ascend.quantization.method_adapters"].AscendEmbeddingMethod = FakeEmbeddingMethod
        modules["vllm_ascend.ops.vocab_parallel_embedding"].AscendVocabParallelEmbedding = (
            FakeVocabParallelEmbedding
        )
        config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(model_type="qwen3_5")
            ),
            quant_config=types.SimpleNamespace(
                quant_description={
                    "language_model.model.embed_tokens.weight": "W8A8HIF8_SOFT"
                }
            ),
        )
        modules["vllm.config"].get_current_vllm_config = lambda: config
        sys.modules.update(modules)

        import vllm_ascend_hif8.plugin as plugin
        plugin._registered = False
        plugin.register()
        registered_class = registry[("W8A8HIF8_SOFT", "linear")]
        static_registered_class = registry[("W8A8HIF8_STATIC", "linear")]
        plugin.register()
        self.assertIs(registry[("W8A8HIF8_SOFT", "linear")], registered_class)
        self.assertIs(
            registry[("W8A8HIF8_STATIC", "linear")],
            static_registered_class,
        )
        self.assertIsNot(registered_class, static_registered_class)
        delegate = types.SimpleNamespace(embedding=lambda layer, ids: (layer, ids))
        wrapper = FakeEmbeddingMethod(delegate)
        self.assertEqual(wrapper.embedding("layer", "ids"), ("layer", "ids"))
        embedding = FakeVocabParallelEmbedding(32, 8)
        self.assertIs(embedding.init_kwargs["quant_config"], config.quant_config)
        self.assertEqual(
            embedding.init_kwargs["prefix"],
            "language_model.model.embed_tokens",
        )
        lm_head_quant = object()
        lm_head = FakeVocabParallelEmbedding(32, 8, None, None, 64, lm_head_quant, "lm_head")
        self.assertEqual(lm_head.init_args[-2:], (lm_head_quant, "lm_head"))
        self.assertIs(lm_head.init_kwargs["quant_config"], lm_head_quant)
        self.assertEqual(lm_head.init_kwargs["prefix"], "lm_head")


class StaticKVEngineCoreActivationTest(unittest.TestCase):
    def setUp(self):
        import vllm_ascend_hif8.plugin as plugin

        self.plugin = plugin
        self.saved_flag = os.environ.get(plugin._HIF8_KV_CACHE_ENV)
        os.environ.pop(plugin._HIF8_KV_CACHE_ENV, None)

    def tearDown(self):
        if self.saved_flag is None:
            os.environ.pop(self.plugin._HIF8_KV_CACHE_ENV, None)
        else:
            os.environ[self.plugin._HIF8_KV_CACHE_ENV] = self.saved_flag

    @staticmethod
    def _config(path: Path):
        return SimpleNamespace(
            model_config=SimpleNamespace(
                model_weights="",
                model=str(path),
            )
        )

    def test_static_metadata_enables_child_process_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / self.plugin._HIF8_METADATA_FILENAME).write_text(
                json.dumps(
                    {
                        "format": self.plugin._STATIC_QUANT_TYPE,
                        "kv_cache": {"mode": "static_per_layer_per_head"},
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                self.plugin._enable_static_kv_from_config(self._config(checkpoint))
            )
            self.assertEqual(
                os.environ[self.plugin._HIF8_KV_CACHE_ENV],
                "1",
            )

    def test_bf16_and_legacy_checkpoints_do_not_enable_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            self.assertFalse(
                self.plugin._enable_static_kv_from_config(self._config(checkpoint))
            )
            (checkpoint / self.plugin._HIF8_METADATA_FILENAME).write_text(
                json.dumps({"format": "W8A8HIF8_SOFT"}),
                encoding="utf-8",
            )
            self.assertFalse(
                self.plugin._enable_static_kv_from_config(self._config(checkpoint))
            )

    def test_explicit_flag_is_preserved_without_checkpoint_inspection(self):
        os.environ[self.plugin._HIF8_KV_CACHE_ENV] = "1"
        with patch.object(self.plugin, "_checkpoint_path_from_config") as resolve:
            self.assertTrue(self.plugin._enable_static_kv_from_config(None))
            resolve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
