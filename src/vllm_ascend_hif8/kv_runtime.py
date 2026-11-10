"""HiFloat8 KV-cache adapter for static W8A8KV8 deployment on vLLM-Ascend.

Stores packed uint8 KV pages and decodes request-local BF16 tensors for
attention. The plugin installs the adapter when HIF8_KV_CACHE is enabled.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

HIF8_KV_CACHE_ENV = "HIF8_KV_CACHE"
HIF8_KV_CACHE_DTYPE = "fp8_per_token_head"
HIF8_SCALE_BYTES = 4
HIF8_METADATA_FILENAME = "hif8_metadata.json"
HIF8_KV_DEBUG_LAYER_ENV = "HIF8_KV_DEBUG_LAYER"
HIF8_KV_DEBUG_STEPS_ENV = "HIF8_KV_DEBUG_STEPS"

_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?=\.|$)")

_installed = False
_ascend_patch_installed = False
_originals: dict[str, Any] = {}
# Use vLLM's process-aware log handler for worker messages.
_logger = logging.getLogger("vllm")
_logged_static_checkpoints: set[str] = set()
_logged_quant_paths: set[tuple[int, str]] = set()
_logged_hybrid_layouts: set[tuple[int, int, int]] = set()


def _optional_nonnegative_integer_env(name: str) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a non-negative integer, got {raw_value!r}") from error
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {raw_value!r}")
    return value


def _positive_integer_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}")
    return value


def _debug_tensor_summary(tensor: Any) -> dict[str, Any]:
    """Synchronously summarize a tensor for an explicitly enabled debug run."""

    import torch

    if not isinstance(tensor, torch.Tensor):
        return {"type": type(tensor).__name__, "value": repr(tensor)}
    result: dict[str, Any] = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "device": str(tensor.device),
        "contiguous": tensor.is_contiguous(),
        "stride": list(tensor.stride()),
        "numel": tensor.numel(),
        **_debug_tensor_layout(tensor),
    }
    if tensor.numel() == 0:
        return result

    # The CPU copy synchronizes the NPU stage and can affect race diagnostics.
    values = tensor.detach().to(dtype=torch.float32, device="cpu")
    finite = torch.isfinite(values)
    finite_count = int(finite.sum().item())
    result.update(
        finite=finite_count,
        nan=int(torch.isnan(values).sum().item()),
        posinf=int(torch.isposinf(values).sum().item()),
        neginf=int(torch.isneginf(values).sum().item()),
    )
    if finite_count:
        finite_values = values[finite]
        result.update(
            min=float(finite_values.min().item()),
            max=float(finite_values.max().item()),
            norm=float(torch.linalg.vector_norm(finite_values).item()),
        )
    return result


def _debug_tensor_layout(tensor: Any) -> dict[str, Any]:
    """Describe the exact storage interval addressed by one tensor view.

    This helper intentionally performs no device copy.  It is used only by the
    opt-in KV diagnostic and makes aliasing between attention pages and hybrid
    GDN state views visible in one log without dereferencing their contents.
    """

    import torch

    if not isinstance(tensor, torch.Tensor):
        return {"type": type(tensor).__name__}
    storage = tensor.untyped_storage()
    element_size = int(tensor.element_size())
    storage_data_ptr = int(storage.data_ptr())
    data_ptr = int(tensor.data_ptr())
    storage_nbytes = int(storage.nbytes())
    if tensor.numel() == 0:
        view_byte_end = data_ptr
    else:
        # NPU tensor views in this path use non-negative strides.  Compute the
        # half-open byte interval covering every addressed element; it remains
        # useful for as_strided block-major views with padding between pages.
        max_element_offset = sum(
            (int(size) - 1) * int(stride)
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        )
        view_byte_end = data_ptr + (max_element_offset + 1) * element_size
    return {
        "data_ptr": data_ptr,
        "storage_data_ptr": storage_data_ptr,
        "storage_offset": int(tensor.storage_offset()),
        "storage_nbytes": storage_nbytes,
        "storage_byte_end": storage_data_ptr + storage_nbytes,
        "element_size": element_size,
        "view_byte_end": view_byte_end,
    }


def _debug_cache_binding_layout(value: Any) -> Any:
    """Recursively describe a runner KV binding without touching device data."""

    import torch

    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "stride": list(value.stride()),
            **_debug_tensor_layout(value),
        }
    if isinstance(value, (tuple, list)):
        return [_debug_cache_binding_layout(item) for item in value]
    return {"type": type(value).__name__, "value": repr(value)}


def _debug_hif8_data_summary(data: Any) -> dict[str, Any]:
    """Report emitted special HiFloat8 codes in addition to byte statistics."""

    import torch

    result = _debug_tensor_summary(data)
    if not isinstance(data, torch.Tensor) or data.numel() == 0:
        return result
    values = data.detach().to(dtype=torch.uint8, device="cpu")
    result.update(
        nan_code=int((values == 0x80).sum().item()),
        posinf_code=int((values == 0x6F).sum().item()),
        neginf_code=int((values == 0xEF).sum().item()),
    )
    return result


def hif8_kv_cache_enabled() -> bool:
    """Return whether the physical HiFloat8 KV cache is enabled."""

    return os.environ.get(HIF8_KV_CACHE_ENV) == "1"


def _attention_layer_index(layer_name: str) -> int:
    """Extract a decoder index from a real vLLM Attention prefix."""

    matches = _LAYER_INDEX_PATTERN.findall(str(layer_name))
    if len(matches) != 1:
        raise ValueError(
            "cannot resolve one decoder layer index from HIF8 attention "
            f"prefix {layer_name!r}"
        )
    return int(matches[0])


def _model_checkpoint_path(vllm_config: Any) -> Path:
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        raise ValueError("HIF8 runtime has no model_config")
    model_weights = getattr(model_config, "model_weights", "")
    model = model_weights or getattr(model_config, "model", "")
    if not model:
        raise ValueError("HIF8 runtime cannot resolve the checkpoint path")
    return Path(model)


@functools.lru_cache(maxsize=8)
def _load_static_kv_scale_values(
    checkpoint: str,
) -> dict[int, tuple[tuple[float, ...], tuple[float, ...]]] | None:
    """Load and validate static K/V scales without importing torch.

    Legacy dynamic checkpoints either have no metadata or no ``kv_cache``
    object and return ``None``.  A checkpoint that declares static HiFloat8 is
    fail-closed: malformed/missing scale metadata may never silently fall back
    to runtime reductions.
    """

    directory = Path(checkpoint)
    metadata_path = directory / HIF8_METADATA_FILENAME
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid HIF8 metadata object: {metadata_path}")
    static_checkpoint = metadata.get("format") == "W8A8HIF8_STATIC"
    kv_cache = metadata.get("kv_cache")
    if kv_cache is None:
        if static_checkpoint:
            raise ValueError(
                f"static HIF8 checkpoint is missing kv_cache metadata: {metadata_path}"
            )
        return None
    if not isinstance(kv_cache, dict) or kv_cache.get("mode") != "static_per_layer_per_head":
        raise ValueError(f"unsupported HIF8 kv_cache metadata in {metadata_path}")
    layers = kv_cache.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"HIF8 kv_cache.layers is empty in {metadata_path}")

    output: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {}
    for raw_index, entry in layers.items():
        try:
            layer_index = int(raw_index)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid static KV layer index {raw_index!r}") from error
        if str(layer_index) != str(raw_index) or layer_index < 0:
            raise ValueError(f"invalid static KV layer index {raw_index!r}")
        if not isinstance(entry, dict):
            raise ValueError(f"static KV layer {layer_index} is not an object")
        values: list[tuple[float, ...]] = []
        for kind in ("key_scales", "value_scales"):
            raw_scales = entry.get(kind)
            if not isinstance(raw_scales, list) or not raw_scales:
                raise ValueError(
                    f"static KV layer {layer_index} has no {kind}"
                )
            scales = tuple(float(value) for value in raw_scales)
            if any(scale <= 0.0 or not math.isfinite(scale) for scale in scales):
                raise ValueError(
                    f"static KV layer {layer_index} has invalid {kind}"
                )
            values.append(scales)
        if len(values[0]) != len(values[1]):
            raise ValueError(
                f"static KV layer {layer_index} has different K/V head counts"
            )
        output[layer_index] = (values[0], values[1])
    return output


def _validate_hif8_runtime_config(vllm_config: Any) -> None:
    """Reject runtime modes that can bypass the correctness adapter.

    The current implementation gathers and decodes request-local cache rows
    into newly allocated BF16 tensors.  Those tensors are deliberately
    temporary, but that also means their addresses and shapes are not stable
    enough for ACL graph capture.  Refuse graph execution up front instead of
    letting Ascend's ``full_graph_fia`` reinterpret the private HIF8 bytes as a
    standard cache format.
    """

    from vllm.config.compilation import CUDAGraphMode

    if getattr(vllm_config, "use_v2_model_runner", False):
        raise NotImplementedError(
            "HIF8 KV cache currently supports vLLM-Ascend model runner V1 only"
        )

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if bool(getattr(scheduler_config, "async_scheduling", False)):
        raise NotImplementedError(
            "HIF8 KV cache correctness mode does not support vLLM asynchronous "
            "scheduling; pass --no-async-scheduling or "
            "async_scheduling=False"
        )

    parallel_config = vllm_config.parallel_config
    if (
        parallel_config.decode_context_parallel_size != 1
        or getattr(parallel_config, "prefill_context_parallel_size", 1) != 1
    ):
        raise NotImplementedError(
            "HIF8 KV cache does not yet support DCP/PCP context parallelism"
        )

    enforce_eager = bool(vllm_config.model_config.enforce_eager)
    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    if not enforce_eager and cudagraph_mode != CUDAGraphMode.NONE:
        raise NotImplementedError(
            "HIF8 KV cache correctness mode does not support ACL graph capture; "
            "start vLLM with --enforce-eager or set cudagraph_mode=NONE"
        )


def _physical_attention_spec(spec: Any) -> Any:
    """Return a cache spec whose dimensions describe the packed byte rows.

    vLLM's per-token-head cache accounting already budgets two FP32 scales per
    token.  We replace that logical quantized spec with an ordinary uint8
    spec whose K and V dimensions explicitly include the four inline scale
    bytes.  This makes the existing Ascend runner split and reshape the raw
    allocation correctly, while the attention implementation still receives
    the original logical ``head_size`` in its constructor.
    """

    import torch
    from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec

    if not isinstance(spec, AttentionSpec):
        return spec
    if getattr(spec, "_hif8_physical", False):
        return spec

    # Several cache formats (MLA, sinks, R-SWA, hidden-state caches) inherit
    # FullAttentionSpec but have different storage or attention semantics.  A
    # strict type check prevents their bytes from being silently reinterpreted
    # as ordinary dense K/V pages.
    if type(spec) is not FullAttentionSpec:
        raise NotImplementedError(
            "HIF8 KV cache currently supports ordinary dense FullAttentionSpec "
            "only; "
            f"got {type(spec).__name__}"
        )
    if spec.sliding_window is not None or spec.attention_chunk_size is not None:
        raise NotImplementedError(
            "HIF8 KV cache does not yet support sliding-window or chunk-local "
            "attention"
        )

    logical_head_size_v = int(spec.head_size_v)

    # All dense Qwen attention head dimensions are byte-aligned.  Refuse an
    # invalid layout early instead of allowing a misaligned float32 sidecar.
    if spec.head_size % HIF8_SCALE_BYTES or logical_head_size_v % HIF8_SCALE_BYTES:
        raise ValueError(
            "HIF8 KV cache requires K/V head dimensions divisible by four; "
            f"got head_size={spec.head_size}, head_size_v={logical_head_size_v}"
        )

    # AttentionSpec is frozen.  ``dataclasses.replace`` changes only the
    # physical cache contract while retaining ordinary full-attention fields.
    from dataclasses import replace

    physical = replace(
        spec,
        head_size=spec.head_size + HIF8_SCALE_BYTES,
        head_size_v=logical_head_size_v + HIF8_SCALE_BYTES,
        dtype=torch.uint8,
        # The sidecar is represented in the physical dimensions above, so do
        # not add vLLM's separate per-token-head scale budget a second time.
        kv_quant_mode=type(spec.kv_quant_mode).NONE,
    )
    object.__setattr__(physical, "_hif8_physical", True)
    object.__setattr__(physical, "_hif8_logical_head_size", spec.head_size)
    object.__setattr__(physical, "_hif8_logical_head_size_v", logical_head_size_v)
    return physical


def _mamba_payload_bytes(spec: Any) -> int:
    """Return the real recurrent-state bytes, excluding page padding."""

    from vllm.utils.torch_utils import get_dtype_size
    from vllm.v1.kv_cache_interface import MambaSpec

    if not isinstance(spec, MambaSpec):
        raise TypeError(f"expected MambaSpec, got {type(spec).__name__}")
    if len(spec.shapes) != len(spec.dtypes):
        raise ValueError(
            "HIF8 hybrid layout requires one dtype for every Mamba state shape"
        )
    payload = sum(
        math.prod(shape) * get_dtype_size(dtype)
        for shape, dtype in zip(spec.shapes, spec.dtypes, strict=True)
    )
    if payload <= 0:
        raise ValueError("HIF8 hybrid layout requires non-empty Mamba state")
    return payload


def _replace_hybrid_page_size(spec: Any, page_size: int, **metadata: int) -> Any:
    """Replace a frozen cache spec while preserving HIF8 runtime markers."""

    from dataclasses import replace

    updated = replace(spec, page_size_padded=page_size)
    for attribute in (
        "_hif8_physical",
        "_hif8_logical_head_size",
        "_hif8_logical_head_size_v",
    ):
        if hasattr(spec, attribute):
            object.__setattr__(updated, attribute, getattr(spec, attribute))
    object.__setattr__(updated, "_hif8_hybrid_separated", True)
    for name, value in metadata.items():
        object.__setattr__(updated, name, value)
    return updated


def _plan_separated_hybrid_cache_specs(
    kv_cache_specs: dict[str, Any],
) -> dict[str, Any]:
    """Give hybrid HiFP8 attention and recurrent state disjoint storage.

    Ascend's hybrid V1 runner normally overlays a Mamba/GDN SSM block with an
    attention K block.  That is valid only when both blocks have exactly the
    same byte width.  Inline FP32 HiFP8 scales make K larger, while C220's
    recurrent GDN operator computes state addresses as if its block dimension
    were tightly packed and therefore cannot consume a padded ``as_strided``
    view.

    Reserve one common manager page as::

        [all recurrent state payload][all HiFP8 K/V payload]

    The existing Ascend reshape code already puts recurrent states at the
    beginning of a shared raw tensor and K/V at its end.  Increasing both
    specs' planned page size is therefore enough to make those regions
    disjoint, while also making vLLM account for the extra memory when it
    determines the number of blocks.
    """

    from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

    hif8_attention = [
        spec
        for spec in kv_cache_specs.values()
        if isinstance(spec, AttentionSpec)
        and bool(getattr(spec, "_hif8_physical", False))
    ]
    mamba = [
        spec for spec in kv_cache_specs.values() if isinstance(spec, MambaSpec)
    ]
    if not hif8_attention or not mamba:
        return kv_cache_specs

    # ``AttentionSpec`` exposes the unpadded physical size as
    # ``real_page_size_bytes``; older vLLM revisions called this field
    # ``unpadded_page_size_bytes``.  Use the canonical property and retain a
    # compatibility fallback for revisions that still provide the old name.
    attention_payload = max(
        int(
            getattr(
                spec,
                "real_page_size_bytes",
                getattr(spec, "unpadded_page_size_bytes", spec.page_size_bytes),
            )
        )
        for spec in hif8_attention
    )
    mamba_payload = max(_mamba_payload_bytes(spec) for spec in mamba)
    existing_page = max(
        int(spec.page_size_bytes) for spec in (*hif8_attention, *mamba)
    )
    common_page = max(existing_page, attention_payload + mamba_payload)

    metadata = {
        "_hif8_hybrid_attention_payload_bytes": attention_payload,
        "_hif8_hybrid_mamba_payload_bytes": mamba_payload,
        "_hif8_hybrid_common_page_bytes": common_page,
    }
    planned = dict(kv_cache_specs)
    for layer_name, spec in kv_cache_specs.items():
        if (
            isinstance(spec, MambaSpec)
            or isinstance(spec, AttentionSpec)
            and bool(getattr(spec, "_hif8_physical", False))
        ):
            planned[layer_name] = _replace_hybrid_page_size(
                spec,
                common_page,
                **metadata,
            )
    return planned


def physical_kv_cache_width(logical_head_size: int) -> int:
    """Return the byte width of one packed HIF8 KV row."""

    if logical_head_size <= 0 or logical_head_size % HIF8_SCALE_BYTES:
        raise ValueError(
            "HIF8 KV logical head size must be a positive multiple of four, "
            f"got {logical_head_size}"
        )
    return logical_head_size + HIF8_SCALE_BYTES


def _validate_cache_tensor(
    cache: Any,
    *,
    logical_head_size: int,
    name: str,
) -> None:
    """Validate one physical dense cache tensor without importing Ascend."""

    import torch

    if (
        not isinstance(cache, torch.Tensor)
        or cache.dtype != torch.uint8
        or cache.ndim != 4
    ):
        raise TypeError(
            f"HIF8 {name} cache must be a 4-D uint8 tensor, got "
            f"dtype={getattr(cache, 'dtype', None)}, "
            f"shape={tuple(getattr(cache, 'shape', ()))}"
        )
    expected_width = physical_kv_cache_width(logical_head_size)
    if cache.shape[-1] != expected_width:
        raise ValueError(
            f"HIF8 {name} cache width must be {expected_width}, got "
            f"{cache.shape[-1]}"
        )


def _bind_hif8_cache_tensors(
    kv_cache: Any,
    *,
    logical_head_size: int,
    logical_head_size_v: int,
) -> tuple[Any, Any]:
    """Resolve and validate the two supported Ascend cache bindings."""

    import torch

    if isinstance(kv_cache, torch.Tensor):
        if kv_cache.ndim != 5 or kv_cache.shape[0] != 2:
            raise ValueError(
                "HIF8 combined cache must have shape [2, blocks, block_size, "
                "heads, width]"
            )
        key_cache, value_cache = kv_cache[0], kv_cache[1]
    elif isinstance(kv_cache, (tuple, list)) and len(kv_cache) == 2:
        key_cache, value_cache = kv_cache
    else:
        raise ValueError("HIF8 dense attention requires exactly one K/V cache pair")

    _validate_cache_tensor(
        key_cache,
        logical_head_size=logical_head_size,
        name="key",
    )
    _validate_cache_tensor(
        value_cache,
        logical_head_size=logical_head_size_v,
        name="value",
    )
    if key_cache.shape[:3] != value_cache.shape[:3]:
        raise ValueError(
            "HIF8 K/V cache block, token, and head dimensions must match, got "
            f"{tuple(key_cache.shape)} and {tuple(value_cache.shape)}"
        )
    return key_cache, value_cache


def _integer_lengths(values: Any, *, name: str) -> list[int]:
    """Materialize CPU/NPU/list sequence lengths as validated Python ints."""

    if values is None:
        raise ValueError(f"HIF8 attention metadata is missing {name}")
    if hasattr(values, "tolist"):
        values = values.tolist()
    lengths = [int(value) for value in values]
    if not lengths:
        raise ValueError(f"HIF8 attention metadata has empty {name}")
    if any(length < 0 for length in lengths):
        raise ValueError(f"HIF8 attention metadata has negative {name}: {lengths}")
    return lengths


def _attention_sequence_plan(attn_metadata: Any, query_rows: int) -> tuple[int, list[int], list[int]]:
    """Resolve the TND cumulative Q/KV lengths used by the BF16 FIA call.

    Ascend metadata stores query lengths cumulatively (``[q0, q0+q1, ...]``)
    but stores post-update KV lengths per request.  Padded graph requests are
    represented in both lists, so keeping them in this plan preserves FIA's
    batch-size contract while cache writes remain limited by
    ``num_actual_tokens``.
    """

    q_cumulative = _integer_lengths(
        attn_metadata.actual_seq_lengths_q,
        name="actual_seq_lengths_q",
    )
    previous = 0
    for current in q_cumulative:
        if current < previous:
            raise ValueError(
                "HIF8 actual_seq_lengths_q must be cumulative and "
                f"non-decreasing, got {q_cumulative}"
            )
        previous = current
    num_tokens = q_cumulative[-1]
    if num_tokens > query_rows:
        raise ValueError(
            "HIF8 actual_seq_lengths_q exceeds the query buffer: "
            f"{num_tokens} > {query_rows}"
        )

    kv_lengths = _integer_lengths(attn_metadata.seq_lens_list, name="seq_lens_list")
    if len(kv_lengths) != len(q_cumulative):
        raise ValueError(
            "HIF8 TND metadata must describe the same number of Q and KV "
            f"requests, got {len(q_cumulative)} and {len(kv_lengths)}"
        )
    kv_cumulative: list[int] = []
    total = 0
    q_begin = 0
    num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", num_tokens))
    for q_end, kv_length in zip(q_cumulative, kv_lengths, strict=True):
        # Graph/SP padding may append one dummy request whose Q range consists
        # entirely of padding while the builder intentionally gives it a
        # one-token KV row.  Only enforce the post-update relationship for
        # requests that contain real input tokens.
        contains_real_tokens = q_begin < num_actual_tokens
        if contains_real_tokens and kv_length < q_end - q_begin:
            raise ValueError(
                "HIF8 post-update KV length cannot be shorter than its query "
                f"length, got q={q_end - q_begin}, kv={kv_length}"
            )
        total += kv_length
        kv_cumulative.append(total)
        q_begin = q_end
    return num_tokens, q_cumulative, kv_cumulative


def _install_cache_dtype_patch() -> None:
    from vllm.config.cache import CacheConfig

    if getattr(CacheConfig.__init__, "_hif8_kv_adapter", False):
        return

    original_init = CacheConfig.__init__
    constructor_signature = inspect.signature(CacheConfig)

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        # CacheConfig is constructed after general plugins are loaded.  The
        # explicit environment flag therefore provides a stable, process-wide
        # opt-in without changing vLLM's CLI or the default ``auto`` behavior.
        if not hif8_kv_cache_enabled():
            return original_init(self, *args, **kwargs)

        # Pydantic dataclasses expose a useful generated class signature while
        # their generic ``__init__(*args, **kwargs)`` does not preserve the same
        # positional ordering internally. Bind once and invoke the generated
        # initializer with normalized keyword arguments.
        bound = constructor_signature.bind(*args, **kwargs)
        requested = bound.arguments.get("cache_dtype", "auto")
        if requested not in ("auto", HIF8_KV_CACHE_DTYPE):
            raise ValueError(
                "HIF8_KV_CACHE=1 requires --kv-cache-dtype=fp8_per_token_head "
                f"or auto, got {requested!r}"
            )
        bound.arguments["cache_dtype"] = HIF8_KV_CACHE_DTYPE
        original_init(self, **bound.arguments)

    init._hif8_kv_adapter = True
    _originals["CacheConfig.__init__"] = original_init
    CacheConfig.__init__ = init


def _install_attention_spec_patch() -> None:
    from vllm.model_executor.layers.attention import Attention

    original = Attention.get_kv_cache_spec
    if getattr(original, "_hif8_kv_adapter", False):
        return

    @functools.wraps(original)
    def get_kv_cache_spec(self, vllm_config):
        spec = original(self, vllm_config)
        return _physical_attention_spec(spec) if hif8_kv_cache_enabled() else spec

    get_kv_cache_spec._hif8_kv_adapter = True
    _originals["Attention.get_kv_cache_spec"] = original
    Attention.get_kv_cache_spec = get_kv_cache_spec


def _install_deferred_ascend_patch() -> None:
    """Install the backend hook after vllm-ascend finishes worker imports.

    General plugins may run while ``vllm_ascend.device.device_op`` is only
    partially initialized.  Importing ``attention_v1`` at that point creates a
    DeviceOperator cycle.  Every NPU worker calls ``adapt_patch(False)`` before
    it constructs or loads a model, so wrapping that call gives us a safe and
    deterministic installation point.
    """

    import vllm_ascend.utils as ascend_utils

    current = ascend_utils.adapt_patch
    if getattr(current, "_hif8_kv_deferred_adapter", False):
        return

    @functools.wraps(current)
    def adapt_patch(is_global_patch: bool = False):
        result = current(is_global_patch=is_global_patch)
        if not is_global_patch and hif8_kv_cache_enabled():
            _install_ascend_attention_patch()
        return result

    adapt_patch._hif8_kv_deferred_adapter = True
    _originals["vllm_ascend.utils.adapt_patch"] = current
    ascend_utils.adapt_patch = adapt_patch


def _install_ascend_runner_layout_patch() -> None:
    """Make Ascend V1 account for separated hybrid recurrent/HiFP8 bytes."""

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    current = NPUModelRunner.get_kv_cache_spec
    if getattr(current, "_hif8_hybrid_layout_adapter", False):
        return

    @functools.wraps(current)
    def get_kv_cache_spec(self):
        specs = current(self)
        if not hif8_kv_cache_enabled():
            return specs
        planned = _plan_separated_hybrid_cache_specs(specs)
        if planned is specs:
            return specs

        first = next(
            spec
            for spec in planned.values()
            if bool(getattr(spec, "_hif8_hybrid_separated", False))
        )
        layout = (
            int(first._hif8_hybrid_attention_payload_bytes),
            int(first._hif8_hybrid_mamba_payload_bytes),
            int(first._hif8_hybrid_common_page_bytes),
        )
        if layout not in _logged_hybrid_layouts:
            _logged_hybrid_layouts.add(layout)
            _logger.info(
                "Separated hybrid HiFloat8 KV and recurrent state storage: "
                "attention=%d bytes, recurrent=%d bytes, common page=%d bytes",
                *layout,
            )
        return planned

    get_kv_cache_spec._hif8_hybrid_layout_adapter = True
    _originals["NPUModelRunner.get_kv_cache_spec"] = current
    NPUModelRunner.get_kv_cache_spec = get_kv_cache_spec


def _install_ascend_attention_patch() -> None:
    global _ascend_patch_installed
    if _ascend_patch_installed:
        return

    _install_ascend_runner_layout_patch()

    from vllm_ascend.attention import attention_v1

    backend = attention_v1.AscendAttentionBackend
    supported = list(getattr(backend, "supported_kv_cache_dtypes", ()))
    if HIF8_KV_CACHE_DTYPE not in supported:
        backend.supported_kv_cache_dtypes = [*supported, HIF8_KV_CACHE_DTYPE]

    original_get_impl_cls = backend.get_impl_cls
    if getattr(original_get_impl_cls, "_hif8_kv_adapter", False):
        _ascend_patch_installed = True
        return

    impl_base = attention_v1.AscendAttentionBackendImpl

    class Hif8AscendAttentionBackendImpl(impl_base):
        """Ascend attention implementation consuming packed HIF8 KV pages."""

        _hif8_kv_cache = True

        def __init__(self, *args, **kwargs):
            head_size_v = kwargs.get("head_size_v")
            super().__init__(*args, **kwargs)
            _validate_hif8_runtime_config(self.vllm_config)
            if self.attn_type != attention_v1.AttentionType.DECODER:
                raise NotImplementedError(
                    "HIF8 KV cache currently supports decoder self-attention only; "
                    f"got attention type {self.attn_type!r}"
                )
            if self.sliding_window is not None:
                raise NotImplementedError(
                    "HIF8 KV cache does not yet support sliding-window attention"
                )
            if self.sinks is not None:
                raise NotImplementedError(
                    "HIF8 KV cache does not yet support attention sinks"
                )
            self.head_size_v = int(
                self.head_size if head_size_v is None else head_size_v
            )
            checkpoint = _model_checkpoint_path(self.vllm_config)
            resolved_checkpoint = str(checkpoint.resolve())
            static_values = _load_static_kv_scale_values(resolved_checkpoint)
            self._hif8_static_kv_values = static_values
            self._hif8_key_scale = None
            self._hif8_value_scale = None
            self._hif8_scale_layer_index = None
            self._hif8_debug_layer = _optional_nonnegative_integer_env(
                HIF8_KV_DEBUG_LAYER_ENV
            )
            self._hif8_debug_steps = _positive_integer_env(
                HIF8_KV_DEBUG_STEPS_ENV, 3
            )
            self._hif8_debug_call_index = 0
            self._hif8_debug_context: dict[str, Any] | None = None
            self._hif8_debug_page_snapshots: dict[
                tuple[str, int], Any
            ] = {}
            if (
                static_values is not None
                and resolved_checkpoint not in _logged_static_checkpoints
            ):
                _logged_static_checkpoints.add(resolved_checkpoint)
                _logger.info(
                    "Loaded checkpoint-static HiFloat8 KV scales for %d "
                    "attention layers (per-layer/per-head, K/V separate); "
                    "KV writes bypass dynamic Abs/ReduceMax",
                    len(static_values),
                )

        def _begin_hif8_debug(self, layer, attn_metadata) -> None:
            if self._hif8_debug_layer is None:
                self._hif8_debug_context = None
                return
            layer_index = _attention_layer_index(layer.layer_name)
            call_index = self._hif8_debug_call_index
            self._hif8_debug_call_index += 1
            if (
                layer_index != self._hif8_debug_layer
                or call_index >= self._hif8_debug_steps
            ):
                self._hif8_debug_context = None
                return
            self._hif8_debug_context = {
                "pid": os.getpid(),
                "layer": layer_index,
                "layer_name": layer.layer_name,
                "step": call_index,
                "attn_state": str(getattr(attn_metadata, "attn_state", None)),
            }

        def _end_hif8_debug(self) -> None:
            self._hif8_debug_context = None

        def _hif8_debug(self, stage: str, **fields: Any) -> None:
            context = self._hif8_debug_context
            if context is None:
                return
            payload = {**context, "stage": stage, **fields}
            _logger.warning(
                "HIF8_KV_DEBUG=%s",
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
            )

        @staticmethod
        def _debug_values(values: Any, limit: int = 128) -> Any:
            if values is None:
                return None
            if hasattr(values, "detach"):
                values = values.detach().reshape(-1)[:limit].to("cpu").tolist()
            elif hasattr(values, "tolist"):
                values = values.tolist()
            if isinstance(values, (tuple, list)):
                return [int(value) for value in values[:limit]]
            return values

        @staticmethod
        def _debug_request_summaries(tensor, cumulative_lengths) -> list[dict[str, Any]]:
            summaries: list[dict[str, Any]] = []
            begin = 0
            for request, end in enumerate(cumulative_lengths):
                summaries.append(
                    {
                        "request": request,
                        "rows": [begin, int(end)],
                        "tensor": _debug_tensor_summary(tensor[begin:int(end)]),
                    }
                )
                begin = int(end)
            return summaries

        def _debug_hif8_page_rows(
            self,
            cache,
            block_ids,
            *,
            logical_head_size: int,
            kind: str,
            boundary: str,
        ) -> list[dict[str, Any]]:
            """Snapshot selected physical pages and identify changed rows."""

            import torch

            if self._hif8_debug_context is None:
                return []
            from .kv_cache import unpack_kv_rows

            unique_blocks = sorted({int(block) for block in block_ids})
            pages: list[dict[str, Any]] = []
            for block in unique_blocks:
                if block < 0 or block >= int(cache.shape[0]):
                    pages.append({"block": block, "invalid": True})
                    continue
                page = cache[block]
                # The CPU clone is the intentional completion boundary for
                # this opt-in diagnostic.  Keeping snapshots on CPU prevents
                # later NPU writes from changing the reference underneath us.
                page_cpu = page.detach().to(device="cpu", dtype=torch.uint8).clone()
                snapshot_key = (kind, block)
                previous = self._hif8_debug_page_snapshots.get(snapshot_key)
                changed_rows: list[int] = []
                changed_bytes = 0
                if previous is not None and tuple(previous.shape) == tuple(page_cpu.shape):
                    changed = page_cpu != previous
                    changed_bytes = int(changed.sum().item())
                    changed_rows = (
                        changed.reshape(page_cpu.shape[0], -1)
                        .any(dim=1)
                        .nonzero(as_tuple=False)
                        .reshape(-1)
                        .tolist()
                    )
                self._hif8_debug_page_snapshots[snapshot_key] = page_cpu

                data, scale = unpack_kv_rows(page, logical_head_size)
                data_cpu = data.detach().to(device="cpu", dtype=torch.uint8)
                scale_cpu = scale.detach().to(device="cpu", dtype=torch.float32)
                special_by_row = (
                    (data_cpu == 0x80)
                    | (data_cpu == 0x6F)
                    | (data_cpu == 0xEF)
                ).reshape(data_cpu.shape[0], -1).sum(dim=1)
                invalid_scale_by_row = (
                    ~torch.isfinite(scale_cpu) | (scale_cpu <= 0)
                ).reshape(scale_cpu.shape[0], -1).sum(dim=1)
                special_rows = [
                    {"row": int(row), "count": int(special_by_row[row].item())}
                    for row in (special_by_row > 0).nonzero(as_tuple=False).reshape(-1)
                ]
                invalid_scale_rows = [
                    {
                        "row": int(row),
                        "count": int(invalid_scale_by_row[row].item()),
                    }
                    for row in (invalid_scale_by_row > 0)
                    .nonzero(as_tuple=False)
                    .reshape(-1)
                ]
                pages.append(
                    {
                        "block": block,
                        "boundary": boundary,
                        "page_layout": _debug_cache_binding_layout(page),
                        "changed_since_previous": previous is not None,
                        "changed_bytes": changed_bytes,
                        "changed_rows": [int(row) for row in changed_rows],
                        "special_rows": special_rows,
                        "invalid_scale_rows": invalid_scale_rows,
                        "data": _debug_hif8_data_summary(data),
                        "scale": _debug_tensor_summary(scale),
                    }
                )
            return pages

        def _debug_cache_pages(
            self,
            boundary: str,
            block_tables=None,
            extra_blocks=(),
            seq_lens=None,
        ) -> None:
            import torch

            if self._hif8_debug_context is None:
                return
            if self.key_cache is None or self.value_cache is None:
                self._hif8_debug("cache_pages", boundary=boundary, bound=False)
                return
            block_ids = [int(block) for block in extra_blocks]
            if block_tables is not None:
                tables_cpu = block_tables.detach().to("cpu")
                if seq_lens is None:
                    selected = tables_cpu.reshape(-1)
                else:
                    selected_rows = []
                    block_size = int(self.key_cache.shape[1])
                    for request, seq_len in enumerate(seq_lens):
                        block_count = (int(seq_len) + block_size - 1) // block_size
                        selected_rows.append(tables_cpu[request, :block_count])
                    selected = (
                        torch.cat(selected_rows)
                        if selected_rows
                        else tables_cpu.new_empty((0,))
                    )
                block_ids.extend(
                    int(block) for block in selected.reshape(-1).tolist()
                    if int(block) >= 0
                )
            self._hif8_debug(
                "cache_pages",
                boundary=boundary,
                cache_binding={
                    "key": _debug_cache_binding_layout(self.key_cache),
                    "value": _debug_cache_binding_layout(self.value_cache),
                },
                key_pages=self._debug_hif8_page_rows(
                    self.key_cache,
                    block_ids,
                    logical_head_size=self.head_size,
                    kind="key",
                    boundary=boundary,
                ),
                value_pages=self._debug_hif8_page_rows(
                    self.value_cache,
                    block_ids,
                    logical_head_size=self.head_size_v,
                    kind="value",
                    boundary=boundary,
                ),
            )

        def _debug_all_cache_bindings(self, boundary: str) -> None:
            """Audit every attention/GDN cache view for shared storage ranges."""

            if self._hif8_debug_context is None:
                return
            compilation_config = getattr(self.vllm_config, "compilation_config", None)
            forward_context = getattr(
                compilation_config, "static_forward_context", None
            )
            if not isinstance(forward_context, dict):
                self._hif8_debug(
                    "all_cache_bindings",
                    boundary=boundary,
                    available=False,
                )
                return
            bindings = []
            for name, module in forward_context.items():
                cache = getattr(module, "kv_cache", None)
                if cache is None:
                    continue
                bindings.append(
                    {
                        "name": str(name),
                        "module": type(module).__name__,
                        "cache": _debug_cache_binding_layout(cache),
                    }
                )
            self._hif8_debug(
                "all_cache_bindings",
                boundary=boundary,
                available=True,
                bindings=bindings,
            )

        def _debug_attention_inputs(
            self,
            query,
            key,
            value,
            attn_metadata,
        ) -> None:
            if self._hif8_debug_context is None:
                return
            query_rows = int(query.shape[0])
            num_tokens, q_lens, kv_lens = _attention_sequence_plan(
                attn_metadata, query_rows
            )
            seq_lens = _integer_lengths(
                attn_metadata.seq_lens_list, name="seq_lens_list"
            )
            metadata = {
                "actual_seq_lengths_q": q_lens,
                "seq_lens_list": seq_lens,
                "actual_seq_lengths_kv": kv_lens,
                "num_tokens": num_tokens,
                "num_actual_tokens": int(attn_metadata.num_actual_tokens),
                "num_decode_tokens": int(
                    getattr(attn_metadata, "num_decode_tokens", 0)
                ),
                "num_decodes": int(getattr(attn_metadata, "num_decodes", 0)),
                "num_prefills": int(getattr(attn_metadata, "num_prefills", 0)),
                "causal": bool(attn_metadata.causal),
                "slot_mapping": self._debug_values(attn_metadata.slot_mapping),
                "block_tables": (
                    None
                    if attn_metadata.block_tables is None
                    else {
                        "summary": _debug_tensor_summary(attn_metadata.block_tables),
                        "values": self._debug_values(attn_metadata.block_tables),
                    }
                ),
            }
            tensors: dict[str, Any] = {
                "query": _debug_tensor_summary(query),
                "query_requests": self._debug_request_summaries(
                    query[:num_tokens], q_lens
                ),
            }
            if key is not None:
                tensors["key"] = _debug_tensor_summary(key)
                tensors["key_requests"] = self._debug_request_summaries(
                    key[:num_tokens], q_lens
                )
            if value is not None:
                tensors["value"] = _debug_tensor_summary(value)
                tensors["value_requests"] = self._debug_request_summaries(
                    value[:num_tokens], q_lens
                )
            self._hif8_debug("attention_input", metadata=metadata, tensors=tensors)

        def _prepare_static_kv_scales(self, layer, device):
            values = self._hif8_static_kv_values
            if values is None:
                return None
            layer_index = _attention_layer_index(layer.layer_name)
            if layer_index not in values:
                raise ValueError(
                    f"static HIF8 checkpoint has no KV scales for attention "
                    f"layer {layer_index} ({layer.layer_name})"
                )
            key_values, value_values = values[layer_index]
            if len(key_values) != self.num_kv_heads:
                raise ValueError(
                    f"static HIF8 KV layer {layer_index} has "
                    f"{len(key_values)} scales for {self.num_kv_heads} runtime heads"
                )
            if (
                self._hif8_key_scale is None
                or self._hif8_key_scale.device != device
                or self._hif8_scale_layer_index != layer_index
            ):
                import torch

                self._hif8_key_scale = torch.tensor(
                    key_values, dtype=torch.float32, device=device
                ).contiguous()
                self._hif8_value_scale = torch.tensor(
                    value_values, dtype=torch.float32, device=device
                ).contiguous()
                self._hif8_scale_layer_index = layer_index
            return self._hif8_key_scale, self._hif8_value_scale

        def _quantize_kv(self, value, fixed_scale=None):
            from . import native
            from .kv_cache import pack_kv_rows

            if value is None:
                return None
            if value.ndim != 3:
                raise ValueError(f"expected K/V [tokens, heads, dim], got {value.shape}")
            tokens, heads, dim = value.shape
            rows = value.reshape(tokens * heads, dim)
            if fixed_scale is None:
                route = "dynamic_quant"
                data, scale = native.dynamic_quant(rows)
            else:
                route = "static_quant"
                data, scale = native.static_quant(rows, fixed_scale)
            route_key = (os.getpid(), route)
            if route_key not in _logged_quant_paths:
                _logged_quant_paths.add(route_key)
                _logger.info(
                    "HiFloat8 KV writes use %s (pid=%d); %s",
                    route,
                    os.getpid(),
                    (
                        "checkpoint-fixed scales bypass dynamic Abs/ReduceMax"
                        if fixed_scale is not None
                        else "runtime per-token-head reductions remain enabled"
                    ),
                )
            data = data.reshape(tokens, heads, dim)
            scale = scale.reshape(tokens, heads).contiguous()
            return pack_kv_rows(data, scale)

        @staticmethod
        def _decode_packed_rows(packed, logical_head_size):
            from . import native
            from .kv_cache import unpack_kv_rows

            data, scale = unpack_kv_rows(packed, logical_head_size)
            decoded = native.scaled_decode(
                data.reshape(-1, logical_head_size),
                scale.reshape(-1),
            )
            return decoded.reshape(*data.shape)

        def _bind_cache(self, kv_cache):
            key_cache, value_cache = _bind_hif8_cache_tensors(
                kv_cache,
                logical_head_size=self.head_size,
                logical_head_size_v=self.head_size_v,
            )
            self.key_cache, self.value_cache = key_cache, value_cache
            return key_cache, value_cache

        def _write_hif8_cache(
            self,
            key,
            value,
            kv_cache,
            slot_mapping,
            num_actual_tokens,
            *,
            layer=None,
        ):
            import torch
            import torch_npu

            if key is None or value is None:
                raise ValueError("HIF8 KV cache update requires both key and value")
            key_cache, value_cache = self._bind_cache(kv_cache)
            if key.shape[-1] != self.head_size or value.shape[-1] != self.head_size_v:
                raise ValueError(
                    "HIF8 KV input dimensions do not match the attention layer: "
                    f"K={key.shape[-1]}/{self.head_size}, "
                    f"V={value.shape[-1]}/{self.head_size_v}"
                )
            if (
                num_actual_tokens < 0
                or num_actual_tokens > key.shape[0]
                or num_actual_tokens > value.shape[0]
            ):
                raise ValueError(
                    f"invalid num_actual_tokens={num_actual_tokens} for "
                    f"K/V rows {key.shape[0]}/{value.shape[0]}"
                )
            block_size = int(key_cache.shape[1])
            static_scales = (
                None
                if self._hif8_static_kv_values is None
                else self._prepare_static_kv_scales(layer, key.device)
            )
            key_scale = None if static_scales is None else static_scales[0]
            value_scale = None if static_scales is None else static_scales[1]
            rows_k = self._quantize_kv(
                key[:num_actual_tokens], fixed_scale=key_scale
            )
            rows_v = self._quantize_kv(
                value[:num_actual_tokens], fixed_scale=value_scale
            )
            if self._hif8_debug_context is not None:
                from .kv_cache import unpack_kv_rows

                key_data, key_row_scale = unpack_kv_rows(rows_k, self.head_size)
                value_data, value_row_scale = unpack_kv_rows(
                    rows_v, self.head_size_v
                )
                self._hif8_debug(
                    "quantized_rows",
                    route=("dynamic" if static_scales is None else "static"),
                    key={
                        "packed": _debug_tensor_summary(rows_k),
                        "data": _debug_hif8_data_summary(key_data),
                        "scale": _debug_tensor_summary(key_row_scale),
                        "scale_positive": int(
                            (key_row_scale.detach().to("cpu") > 0).sum().item()
                        ),
                    },
                    value={
                        "packed": _debug_tensor_summary(rows_v),
                        "data": _debug_hif8_data_summary(value_data),
                        "scale": _debug_tensor_summary(value_row_scale),
                        "scale_positive": int(
                            (value_row_scale.detach().to("cpu") > 0).sum().item()
                        ),
                    },
                )
            slots = slot_mapping[:num_actual_tokens].to(
                device=key_cache.device, dtype=torch.long
            )
            valid = slots >= 0
            if bool(valid.any()):
                valid_slots = slots[valid]
                blocks = valid_slots // block_size
                if bool((blocks < 0).any()) or bool(
                    (blocks >= key_cache.shape[0]).any()
                ):
                    raise IndexError("HIF8 KV slot mapping references an invalid block")
                self._debug_cache_pages(
                    "before_scatter",
                    extra_blocks=self._debug_values(blocks),
                )
                # Keep the cache as physical HiFloat8 bytes end to end.  The
                # native paged-cache primitive only accepts signed storage on
                # this CANN release, so use an indexed copy that preserves the
                # uint8 representation and slot mapping without reinterpretation.
                flat_key_cache = key_cache.reshape(-1, *key_cache.shape[2:])
                flat_value_cache = value_cache.reshape(-1, *value_cache.shape[2:])
                flat_key_cache.index_copy_(0, valid_slots.contiguous(), rows_k[valid].contiguous())
                flat_value_cache.index_copy_(0, valid_slots.contiguous(), rows_v[valid].contiguous())
                if self._hif8_debug_context is not None:
                    offsets = valid_slots % block_size
                    cached_key = key_cache[blocks, offsets]
                    cached_value = value_cache[blocks, offsets]
                    self._hif8_debug(
                        "cache_write",
                        block_size=block_size,
                        slots=self._debug_values(valid_slots),
                        blocks=self._debug_values(blocks),
                        offsets=self._debug_values(offsets),
                        cached_key=_debug_tensor_summary(cached_key),
                        cached_value=_debug_tensor_summary(cached_value),
                        key_byte_match=bool(
                            (cached_key == rows_k[valid]).all().item()
                        ),
                        value_byte_match=bool(
                            (cached_value == rows_v[valid]).all().item()
                        ),
                    )
                    self._debug_cache_pages(
                        "after_scatter",
                        extra_blocks=self._debug_values(blocks),
                    )

        def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
            self.head_size_v = int(getattr(layer, "head_size_v", self.head_size))
            if self.attn_type in (
                attention_v1.AttentionType.ENCODER,
                attention_v1.AttentionType.ENCODER_ONLY,
            ):
                return
            self._write_hif8_cache(
                key,
                value,
                kv_cache,
                slot_mapping,
                int(key.shape[0]),
                layer=layer,
            )
            attention_v1.notify_kv_cache_written()

        def reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
            if kv_cache is not None:
                self._bind_cache(kv_cache)
            if self.kv_sharing_target_layer_name is None:
                if attn_metadata is None:
                    return query, key, value, output
                self._write_hif8_cache(
                    key,
                    value,
                    (self.key_cache, self.value_cache),
                    attn_metadata.slot_mapping,
                    int(attn_metadata.num_actual_tokens),
                    layer=getattr(self, "_hif8_current_layer", None),
                )
                attention_v1.notify_kv_cache_written()
            return query, key, value, output

        def _gather_dense_hif8(self, block_tables, seq_lens):
            import torch

            from .kv_cache import gather_kv_rows_concat

            if self.key_cache is None or self.value_cache is None:
                raise RuntimeError("HIF8 KV cache is not bound to the attention implementation")
            request_count = len(seq_lens)
            if block_tables.ndim != 2 or block_tables.shape[0] < request_count:
                raise ValueError(
                    "HIF8 block table has fewer rows than attention requests: "
                    f"{tuple(block_tables.shape)} for {request_count} requests"
                )
            # Some runner paths retain extra capacity rows.  They are not FIA
            # requests and must not create corresponding gathered sequences.
            block_tables = block_tables[:request_count]
            self._debug_cache_pages(
                "before_gather",
                block_tables,
                seq_lens=seq_lens,
            )
            block_size = int(self.key_cache.shape[1])
            key_packed = gather_kv_rows_concat(
                self.key_cache,
                block_tables,
                seq_lens,
                block_size=block_size,
                validate=False,
            )
            value_packed = gather_kv_rows_concat(
                self.value_cache,
                block_tables,
                seq_lens,
                block_size=block_size,
                validate=False,
            )
            logical_k = int(self.head_size)
            logical_v = int(self.head_size_v)
            keys = self._decode_packed_rows(key_packed, logical_k)
            values = self._decode_packed_rows(value_packed, logical_v)
            self._debug_cache_pages(
                "after_decode",
                block_tables,
                seq_lens=seq_lens,
            )
            if self._hif8_debug_context is not None:
                from .kv_cache import unpack_kv_rows

                key_rows = key_packed.split(seq_lens, dim=0)
                value_rows = value_packed.split(seq_lens, dim=0)
                decoded_keys = keys.split(seq_lens, dim=0)
                decoded_values = values.split(seq_lens, dim=0)
                requests: list[dict[str, Any]] = []
                for request, (
                    key_packed,
                    value_packed,
                    decoded_key,
                    decoded_value,
                ) in enumerate(
                    zip(
                        key_rows,
                        value_rows,
                        decoded_keys,
                        decoded_values,
                        strict=True,
                    )
                ):
                    key_data, key_scale = unpack_kv_rows(key_packed, logical_k)
                    value_data, value_scale = unpack_kv_rows(
                        value_packed, logical_v
                    )
                    requests.append(
                        {
                            "request": request,
                            "seq_len": int(seq_lens[request]),
                            "blocks": self._debug_values(block_tables[request]),
                            "key_packed": _debug_tensor_summary(key_packed),
                            "value_packed": _debug_tensor_summary(value_packed),
                            "key_data": _debug_hif8_data_summary(key_data),
                            "value_data": _debug_hif8_data_summary(value_data),
                            "key_scale": _debug_tensor_summary(key_scale),
                            "value_scale": _debug_tensor_summary(value_scale),
                            "decoded_key": _debug_tensor_summary(decoded_key),
                            "decoded_value": _debug_tensor_summary(decoded_value),
                        }
                    )
                self._hif8_debug("cache_gather_decode", requests=requests)
            if keys.shape[0] == 0:
                empty = self.key_cache.new_empty((0, self.num_kv_heads, logical_k))
                value_empty = self.value_cache.new_empty(
                    (0, self.num_kv_heads, logical_v)
                )
                return empty, value_empty
            return keys, values

        def _forward_hif8_attention(self, query, key, value, attn_metadata, output):
            import torch_npu

            if attn_metadata is None:
                return output.fill_(0)
            if attn_metadata.causal and attn_metadata.attn_mask is None:
                raise ValueError(
                    "HIF8 causal attention requires the Ascend split-fuse "
                    "attention mask; refusing to run sparse_mode=3 without it"
                )

            num_tokens, q_lens, kv_lens = _attention_sequence_plan(
                attn_metadata,
                int(query.shape[0]),
            )

            # Cache writes happen before this method.  When no block table is
            # available (encoder-only/prefill without a cache), use current K/V
            # directly; decoder attention always follows the paged branch.
            seq_lens = None
            if attn_metadata.block_tables is None:
                if key is None or value is None:
                    raise ValueError(
                        "HIF8 attention without a block table requires current K/V"
                    )
                dense_key = key
                dense_value = value
                # This branch has no historical cache, so every request's KV
                # length equals its query length.
                kv_lens = q_lens
            else:
                seq_lens = _integer_lengths(
                    attn_metadata.seq_lens_list,
                    name="seq_lens_list",
                )
                dense_key, dense_value = self._gather_dense_hif8(
                    attn_metadata.block_tables,
                    seq_lens,
                )
            q = query[:num_tokens]
            if self._hif8_debug_context is not None:
                self._hif8_debug(
                    "fia_input",
                    q_lens=q_lens,
                    kv_lens=kv_lens,
                    query=_debug_tensor_summary(q),
                    query_requests=self._debug_request_summaries(q, q_lens),
                    dense_key=_debug_tensor_summary(dense_key),
                    dense_key_requests=self._debug_request_summaries(
                        dense_key, kv_lens
                    ),
                    dense_value=_debug_tensor_summary(dense_value),
                    dense_value_requests=self._debug_request_summaries(
                        dense_value, kv_lens
                    ),
                )
            kwargs = dict(
                query=q,
                key=dense_key.contiguous(),
                value=dense_value.contiguous(),
                input_layout="TND",
                actual_seq_lengths=q_lens,
                actual_seq_lengths_kv=kv_lens,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3 if attn_metadata.causal else 0,
            )
            if attn_metadata.attn_mask is not None:
                kwargs["atten_mask"] = attn_metadata.attn_mask
            attn_output, _ = torch_npu.npu_fused_infer_attention_score(**kwargs)
            if seq_lens is not None:
                self._debug_cache_pages(
                    "after_fia",
                    attn_metadata.block_tables,
                    seq_lens=seq_lens,
                )
            if self._hif8_debug_context is not None:
                shaped_output = attn_output.view(
                    num_tokens, self.num_heads, self.head_size_v
                )
                self._hif8_debug(
                    "fia_output",
                    output=_debug_tensor_summary(shaped_output),
                    requests=self._debug_request_summaries(
                        shaped_output, q_lens
                    ),
                )
            output[:num_tokens] = attn_output.view(
                num_tokens, self.num_heads, self.head_size_v
            )
            if num_tokens < output.shape[0]:
                output[num_tokens:].zero_()
            return output

        def forward_paged_attention(self, query, attn_metadata, output=None):
            return self._forward_hif8_attention(
                query, None, None, attn_metadata, output
            )

        def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
            del kv_cache
            return self._forward_hif8_attention(query, key, value, attn_metadata, output)

        def forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output=None,
            output_scale=None,
            output_block_scale=None,
        ):
            if output is None:
                raise ValueError("HIF8 attention requires an output buffer")
            if output_scale is not None or output_block_scale is not None:
                raise NotImplementedError("HIF8 KV cache does not support output quantization")
            if attn_metadata is None:
                return output.fill_(0)
            self.head_size_v = int(getattr(layer, "head_size_v", self.head_size))
            self._hif8_current_layer = layer
            self._begin_hif8_debug(layer, attn_metadata)
            try:
                self._debug_attention_inputs(query, key, value, attn_metadata)
                if kv_cache is not None:
                    self._bind_cache(kv_cache)
                if (
                    self._hif8_debug_context is not None
                    and self._hif8_debug_context["step"] == 0
                ):
                    self._debug_all_cache_bindings("after_cache_bind")
                debug_seq_lens = None
                if (
                    self._hif8_debug_context is not None
                    and getattr(attn_metadata, "block_tables", None) is not None
                ):
                    debug_seq_lens = _integer_lengths(
                        attn_metadata.seq_lens_list,
                        name="seq_lens_list",
                    )
                self._debug_cache_pages(
                    "attention_entry",
                    getattr(attn_metadata, "block_tables", None),
                    seq_lens=debug_seq_lens,
                )
                # Shared KV layers consume the producer's cache.  They must not
                # overwrite it, but they still need the HIF8-aware read path.
                if (
                    key is not None
                    and value is not None
                    and self.kv_sharing_target_layer_name is None
                ):
                    self.reshape_and_cache(
                        query, key, value, kv_cache, attn_metadata, output
                    )
                result = self._forward_hif8_attention(
                    query, key, value, attn_metadata, output
                )
                if self._hif8_debug_context is not None:
                    self._hif8_debug(
                        "attention_output_buffer",
                        output=_debug_tensor_summary(result),
                    )
                return result
            finally:
                self._end_hif8_debug()

    @staticmethod
    def get_impl_cls():
        # HIF8 is dense-attention-only for this first runtime adapter.  DCP,
        # MLA, SFA and GDN remain on their original implementations.
        if os.environ.get(HIF8_KV_CACHE_ENV) == "1":
            return Hif8AscendAttentionBackendImpl
        return original_get_impl_cls()

    get_impl_cls._hif8_kv_adapter = True
    _originals["AscendAttentionBackend.get_impl_cls"] = original_get_impl_cls
    _originals["AscendAttentionBackend.supported_kv_cache_dtypes"] = supported
    backend.get_impl_cls = staticmethod(get_impl_cls)
    _ascend_patch_installed = True


def install_hif8_kv_cache_adapter() -> None:
    """Install the opt-in cache/runtime hooks exactly once per process."""

    global _installed
    if _installed:
        return
    if not hif8_kv_cache_enabled():
        return

    _install_cache_dtype_patch()
    _install_attention_spec_patch()
    _install_deferred_ascend_patch()
    _installed = True


def reset_hif8_kv_cache_adapter_for_tests() -> None:
    """Best-effort reset hook for isolated plugin tests."""

    global _installed, _ascend_patch_installed
    for key, original in list(_originals.items()):
        if key == "CacheConfig.__init__":
            from vllm.config.cache import CacheConfig

            CacheConfig.__init__ = original
        elif key == "Attention.get_kv_cache_spec":
            from vllm.model_executor.layers.attention import Attention

            Attention.get_kv_cache_spec = original
        elif key == "AscendAttentionBackend.get_impl_cls":
            from vllm_ascend.attention import attention_v1

            attention_v1.AscendAttentionBackend.get_impl_cls = staticmethod(original)
        elif key == "AscendAttentionBackend.supported_kv_cache_dtypes":
            from vllm_ascend.attention import attention_v1

            attention_v1.AscendAttentionBackend.supported_kv_cache_dtypes = original
        elif key == "vllm_ascend.utils.adapt_patch":
            import vllm_ascend.utils as ascend_utils

            ascend_utils.adapt_patch = original
        elif key == "NPUModelRunner.get_kv_cache_spec":
            from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

            NPUModelRunner.get_kv_cache_spec = original
    _originals.clear()
    _load_static_kv_scale_values.cache_clear()
    _logged_static_checkpoints.clear()
    _logged_quant_paths.clear()
    _logged_hybrid_layouts.clear()
    _installed = False
    _ascend_patch_installed = False


__all__ = [
    "HIF8_KV_CACHE_DTYPE",
    "HIF8_KV_CACHE_ENV",
    "HIF8_METADATA_FILENAME",
    "_attention_layer_index",
    "_attention_sequence_plan",
    "_bind_hif8_cache_tensors",
    "_mamba_payload_bytes",
    "_physical_attention_spec",
    "_plan_separated_hybrid_cache_specs",
    "_load_static_kv_scale_values",
    "_validate_hif8_runtime_config",
    "hif8_kv_cache_enabled",
    "install_hif8_kv_cache_adapter",
    "physical_kv_cache_width",
]
