"""vLLM general-plugin entry point."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

_registered = False
_QUANT_TYPE = "W8A8HIF8_SOFT"
_STATIC_QUANT_TYPE = "W8A8HIF8_STATIC"
_HIF8_KV_CACHE_ENV = "HIF8_KV_CACHE"
_HIF8_METADATA_FILENAME = "hif8_metadata.json"
_ENGINE_CORE_WRAPPER_MARKER = "_hif8_static_kv_engine_core_wrapper"
_ADAPT_PATCH_WRAPPER_MARKER = "_hif8_static_kv_adapt_patch_wrapper"
_original_engine_core_run: Any = None
_original_adapt_patch: Any = None
# Use vLLM's process-aware log handler for worker messages.
_logger = logging.getLogger("vllm")


def _checkpoint_path_from_config(vllm_config: Any) -> Path | None:
    """Resolve a local model directory from a serialized VllmConfig.

    Spawned workers receive the config even when environment variables are
    filtered by the launcher. Remote model IDs are resolved by vLLM.
    """

    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return None
    for attribute in ("model_weights", "model"):
        value = getattr(model_config, attribute, None)
        if not value or not isinstance(value, (str, os.PathLike)):
            continue
        try:
            candidate = Path(value).expanduser()
        except (TypeError, ValueError, OSError):
            continue
        if candidate.is_dir():
            return candidate
    return None


def _is_static_hif8_checkpoint(checkpoint: Path) -> bool:
    """Return whether ``checkpoint`` declares static W8A8KV8 metadata.

    Scale validation is handled by ``kv_runtime._load_static_kv_scale_values``.
    """

    metadata_path = checkpoint / _HIF8_METADATA_FILENAME
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "cannot inspect HiFloat8 checkpoint metadata in "
            f"{metadata_path}"
        ) from error
    return isinstance(metadata, dict) and metadata.get("format") == _STATIC_QUANT_TYPE


def _enable_static_kv_from_config(vllm_config: Any) -> bool:
    """Enable static KV in an EngineCore process from checkpoint metadata."""

    # Allow BF16 KV for weight/activation-only accuracy diagnostics.
    if os.environ.get("HIF8_DISABLE_KV_CACHE") == "1":
        return False
    if os.environ.get(_HIF8_KV_CACHE_ENV) == "1":
        return True
    checkpoint = _checkpoint_path_from_config(vllm_config)
    if checkpoint is None or not _is_static_hif8_checkpoint(checkpoint):
        return False
    os.environ[_HIF8_KV_CACHE_ENV] = "1"
    _logger.info(
        "HIF8 EngineCore enabled static KV from checkpoint metadata "
        "(pid=%d, checkpoint=%s)",
        os.getpid(),
        checkpoint,
    )
    return True


def _hif8_engine_core_run(*args: Any, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs: Any):
    """Spawn-safe V1 EngineCore entry point.

    Resolve the original entry point in both parent and spawned workers.
    """

    try:
        from vllm.v1.engine.core import EngineCoreProc
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "HIF8 EngineCore wrapper was invoked without vLLM V1 available"
        ) from error

    current = EngineCoreProc.run_engine_core
    if current is _hif8_engine_core_run:
        current = _original_engine_core_run
    if current is None:
        raise RuntimeError("HIF8 EngineCore wrapper has no upstream entry point")
    _enable_static_kv_from_config(kwargs.get("vllm_config"))
    return current(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)


def _install_engine_core_static_kv_wrapper() -> None:
    """Wrap the V1 spawn entry point exactly once per interpreter."""

    global _original_engine_core_run
    try:
        from vllm.v1.engine.core import EngineCoreProc
    except Exception:
        # V0 and tooling processes may expose incomplete vLLM/torch modules.
        return

    current = EngineCoreProc.run_engine_core
    if getattr(current, _ENGINE_CORE_WRAPPER_MARKER, False):
        return
    _original_engine_core_run = current
    setattr(_hif8_engine_core_run, _ENGINE_CORE_WRAPPER_MARKER, True)
    EngineCoreProc.run_engine_core = staticmethod(_hif8_engine_core_run)


def _install_adapt_patch_hook() -> None:
    """Re-apply the EngineCore wrapper after vLLM-Ascend global patches.

    Platform setup can replace ``run_engine_core`` after plugin registration.
    Reinstall the wrapper after each ``adapt_patch`` call.
    """

    global _original_adapt_patch
    try:
        import vllm_ascend.utils as ascend_utils
    except Exception:
        return

    current = ascend_utils.adapt_patch
    if getattr(current, _ADAPT_PATCH_WRAPPER_MARKER, False):
        return
    _original_adapt_patch = current

    def adapt_patch(is_global_patch: bool = False, *args: Any, **kwargs: Any):
        result = current(is_global_patch=is_global_patch, *args, **kwargs)
        _install_engine_core_static_kv_wrapper()
        return result

    setattr(adapt_patch, _ADAPT_PATCH_WRAPPER_MARKER, True)
    ascend_utils.adapt_patch = adapt_patch


def _install_embedding_adapter() -> None:
    """Delegate embedding calls to the selected Ascend scheme.

    vLLM checks for a class-level ``embedding`` override while constructing a
    VocabParallelEmbedding. AscendEmbeddingMethod currently only inherits the
    linear wrapper, even when the selected scheme supports embedding.
    """
    from vllm_ascend.quantization.method_adapters import AscendEmbeddingMethod

    if "embedding" in AscendEmbeddingMethod.__dict__:
        return

    def embedding(self, layer, input_ids):
        method = getattr(self.quant_method, "embedding", None)
        if method is None:
            raise NotImplementedError(
                f"{type(self.quant_method).__name__} does not implement quantized embedding"
            )
        return method(layer, input_ids)

    AscendEmbeddingMethod.embedding = embedding


def _install_qwen35_embedding_constructor_adapter() -> None:
    """Supply quantization arguments omitted by Qwen3.5's embedding call."""
    import inspect

    from vllm_ascend.ops.vocab_parallel_embedding import AscendVocabParallelEmbedding

    original_init = AscendVocabParallelEmbedding.__init__
    if getattr(original_init, "_hif8_qwen35_adapter", False):
        return

    def init(self, *args, **kwargs):
        bound = inspect.signature(original_init).bind_partial(self, *args, **kwargs)
        quant_config_missing = "quant_config" not in bound.arguments
        prefix_missing = "prefix" not in bound.arguments
        if quant_config_missing and prefix_missing:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
            model_type = vllm_config.model_config.hf_config.model_type
            candidate = vllm_config.quant_config
            description = getattr(candidate, "quant_description", {})
            for weight_name in (
                "language_model.model.embed_tokens.weight",
                "model.embed_tokens.weight",
            ):
                if model_type in {"qwen3_5", "qwen3_5_moe"} and description.get(weight_name) == _QUANT_TYPE:
                    kwargs["quant_config"] = candidate
                    kwargs["prefix"] = weight_name[: -len(".weight")]
                    break
        original_init(self, *args, **kwargs)

    init._hif8_qwen35_adapter = True
    AscendVocabParallelEmbedding.__init__ = init


def register() -> None:
    """Import the scheme once in every vLLM process.

    Native operators are loaded lazily when the HIF8 scheme is selected.
    """
    global _registered
    # Retry hooks when V1 is imported after initial plugin registration.
    _install_engine_core_static_kv_wrapper()
    _install_adapt_patch_hook()
    if _registered:
        return
    from .runtime import configure_opp_path

    configure_opp_path(required=False)
    # Install before the process manager captures the spawn target.
    if __import__("os").environ.get("HIF8_KV_CACHE") == "1":
        from .kv_runtime import install_hif8_kv_cache_adapter

        install_hif8_kv_cache_adapter()
        _logger.info(
            "Installed HiFloat8 KV-cache adapter (pid=%d)",
            os.getpid(),
        )
    # Register fake tensor implementations for torch.compile in every process
    try:
        from . import fake_tensors
        fake_tensors.register_fake_tensor_impls()
    except Exception:
        pass  # Fake tensors are optional for eager execution.
    from . import scheme  # noqa: F401
    _install_embedding_adapter()
    _install_qwen35_embedding_constructor_adapter()

    _registered = True
