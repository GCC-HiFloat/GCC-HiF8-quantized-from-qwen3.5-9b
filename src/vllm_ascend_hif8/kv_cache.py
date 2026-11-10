"""HiFloat8 KV layout and reference operations.

K and V are separate uint8 tensors with shape
``[num_blocks, block_size, num_kv_heads, head_size + 4]``.
Each row stores HiFloat8 bytes followed by a little-endian FP32 scale.
Static mode repeats the checkpoint's per-layer/per-head scale for each token.
The inline sidecar supports byte-exact block copy and swap.

Reference operations run on CPU without vLLM or CANN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .codec import ASCEND_DECODE_TABLE
from .torch_codec import encode_tensor

HIF8_SCALE_BYTES = 4
HIF8_CACHE_DTYPE = "fp8_per_token_head"


@dataclass(frozen=True)
class Hif8KVCacheLayout:
    """Physical layout of one dense HIF8 K/V cache pair."""

    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_size: int
    head_size_v: int | None = None

    def __post_init__(self) -> None:
        resolved_head_size_v = (
            self.head_size if self.head_size_v is None else self.head_size_v
        )
        values = (
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_size,
            resolved_head_size_v,
        )
        if any(not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError(f"HIF8 KV dimensions must be positive integers, got {values}")
        if self.head_size % HIF8_SCALE_BYTES:
            raise ValueError(
                "HIF8 KV head_size must be divisible by four so the inline FP32 "
                f"scale is aligned, got {self.head_size}"
            )
        if resolved_head_size_v % HIF8_SCALE_BYTES:
            raise ValueError(
                "HIF8 KV head_size_v must be divisible by four so the inline FP32 "
                f"scale is aligned, got {resolved_head_size_v}"
            )

    @property
    def resolved_head_size_v(self) -> int:
        return self.head_size if self.head_size_v is None else self.head_size_v

    @property
    def key_width(self) -> int:
        return self.head_size + HIF8_SCALE_BYTES

    @property
    def value_width(self) -> int:
        return self.resolved_head_size_v + HIF8_SCALE_BYTES

    @property
    def page_size_bytes(self) -> int:
        return self.block_size * self.num_kv_heads * (self.key_width + self.value_width)

    @property
    def key_page_size_bytes(self) -> int:
        return self.block_size * self.num_kv_heads * self.key_width

    @property
    def value_page_size_bytes(self) -> int:
        return self.block_size * self.num_kv_heads * self.value_width

    def key_shape(self) -> tuple[int, int, int, int]:
        return (self.num_blocks, self.block_size, self.num_kv_heads, self.key_width)

    def value_shape(self) -> tuple[int, int, int, int]:
        return (self.num_blocks, self.block_size, self.num_kv_heads, self.value_width)


def _check_rows(x: torch.Tensor, name: str) -> tuple[int, int, int]:
    if not isinstance(x, torch.Tensor) or x.ndim != 3:
        raise TypeError(f"{name} must be a tensor with shape [tokens, heads, dim], got {type(x)!r}")
    if x.shape[0] <= 0 or x.shape[1] <= 0 or x.shape[2] <= 0:
        raise ValueError(f"{name} dimensions must be positive, got {tuple(x.shape)}")
    return tuple(int(value) for value in x.shape)  # type: ignore[return-value]


def quantize_kv_rows(
    x: torch.Tensor,
    *,
    scale: torch.Tensor | None = None,
    amax_clip: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[tokens, heads, dim]`` rows into physical HIF8 bytes.

    The scale shape is ``[tokens, heads]`` and is always FP32.  Scales are
    independent for K and V; callers must not share one scale between them.
    ``scale`` may be supplied to reproduce a previously established cache
    quantization contract, but it must still have one positive finite value per
    token/head row.
    """

    tokens, heads, dim = _check_rows(x, "x")
    values = x.to(torch.float32)
    if scale is None:
        finite = torch.where(torch.isfinite(values), values.abs(), torch.zeros_like(values))
        row_scale = finite.amax(dim=-1)
        if amax_clip is not None:
            if amax_clip <= 0 or not torch.isfinite(torch.tensor(amax_clip)):
                raise ValueError("amax_clip must be finite and greater than zero")
            row_scale = row_scale.clamp(max=float(amax_clip))
        row_scale = row_scale.clamp_min(1.0e-12)
    else:
        row_scale = scale.to(device=x.device, dtype=torch.float32)
        if tuple(row_scale.shape) != (tokens, heads):
            raise ValueError(
                f"scale must have shape {(tokens, heads)}, got {tuple(row_scale.shape)}"
            )
        if bool((~torch.isfinite(row_scale) | (row_scale <= 0)).any()):
            raise ValueError("all HIF8 KV scales must be finite and greater than zero")

    data = encode_tensor(values / row_scale.unsqueeze(-1))
    return data.reshape(tokens, heads, dim), row_scale.reshape(tokens, heads)


def decode_kv_rows(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode HIF8 rows and apply their FP32 scales."""

    if not isinstance(data, torch.Tensor) or data.dtype != torch.uint8 or data.ndim != 3:
        raise TypeError(f"data must be a 3-D uint8 tensor, got {getattr(data, 'dtype', None)}")
    if not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32 or scale.ndim != 2:
        raise TypeError(f"scale must be a 2-D float32 tensor, got {getattr(scale, 'dtype', None)}")
    if tuple(scale.shape) != tuple(data.shape[:2]):
        raise ValueError(f"scale shape {tuple(scale.shape)} does not match data {tuple(data.shape)}")
    lut = torch.as_tensor(ASCEND_DECODE_TABLE, dtype=torch.float32, device=data.device)
    decoded = lut[data.to(torch.long)]
    return decoded * scale.unsqueeze(-1)


def pack_kv_rows(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Pack HIF8 rows and FP32 scales into an inline-sidecar byte tensor."""

    if data.dtype != torch.uint8 or data.ndim != 3:
        raise TypeError("data must be a 3-D torch.uint8 tensor")
    if scale.dtype != torch.float32 or scale.shape != data.shape[:2]:
        raise TypeError("scale must be float32 with shape [tokens, heads]")
    if data.shape[-1] % HIF8_SCALE_BYTES:
        raise ValueError("head dimension must be divisible by four for inline scales")
    packed = torch.empty(
        (*data.shape[:-1], data.shape[-1] + HIF8_SCALE_BYTES),
        dtype=torch.uint8,
        device=data.device,
    )
    packed[..., :-HIF8_SCALE_BYTES].copy_(data)
    # The scale view is exactly four bytes per row/head.  ``contiguous`` is
    # important for backends that reject a byte view with unusual strides.
    scale_bytes = scale.contiguous().view(torch.uint8).reshape(*scale.shape, HIF8_SCALE_BYTES)
    packed[..., -HIF8_SCALE_BYTES:].copy_(scale_bytes)
    return packed


def unpack_kv_rows(packed: torch.Tensor, head_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an inline-sidecar tensor into HIF8 data and FP32 scales."""

    if packed.dtype != torch.uint8 or packed.ndim != 3:
        raise TypeError("packed must be a 3-D torch.uint8 tensor")
    expected = head_size + HIF8_SCALE_BYTES
    if packed.shape[-1] != expected:
        raise ValueError(f"packed last dimension must be {expected}, got {packed.shape[-1]}")
    data = packed[..., :head_size]
    scale_bytes = packed[..., head_size:].contiguous()
    scale = scale_bytes.view(torch.float32).reshape(*packed.shape[:2])
    return data, scale


def scatter_kv_rows(
    cache: torch.Tensor,
    rows: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    block_size: int,
) -> None:
    """Reference slot-mapping scatter into a paged packed cache.

    This function is intentionally simple and deterministic.  The NPU adapter
    uses the same contract but may replace the byte copy with
    ``npu_scatter_pa_kv_cache`` when the platform supports it.
    """

    if cache.dtype != torch.uint8 or cache.ndim != 4:
        raise TypeError("cache must be a 4-D uint8 tensor")
    if rows.dtype != torch.uint8 or rows.ndim != 3:
        raise TypeError("rows must be a 3-D uint8 tensor")
    if slot_mapping.ndim != 1 or slot_mapping.numel() != rows.shape[0]:
        raise ValueError("slot_mapping must contain one slot per row")
    if cache.shape[1] != block_size or cache.shape[2:] != rows.shape[1:]:
        raise ValueError(
            f"cache shape {tuple(cache.shape)} is incompatible with rows {tuple(rows.shape)} "
            f"and block_size={block_size}"
        )
    slots = slot_mapping.to(device=cache.device, dtype=torch.long)
    valid = slots >= 0
    if bool(valid.any()):
        valid_slots = slots[valid]
        if bool((valid_slots // block_size >= cache.shape[0]).any()):
            raise IndexError("slot_mapping references a block outside the cache")
        cache[valid_slots // block_size, valid_slots % block_size] = rows[valid]


def gather_kv_rows(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: Sequence[int] | torch.Tensor,
    *,
    block_size: int,
) -> list[torch.Tensor]:
    """Gather valid packed rows for each request from a paged cache."""

    if cache.dtype != torch.uint8 or cache.ndim != 4:
        raise TypeError("cache must be a 4-D uint8 tensor")
    if block_table.ndim != 2:
        raise ValueError("block_table must have shape [batch, max_blocks]")
    lengths = [int(value) for value in (seq_lens.tolist() if isinstance(seq_lens, torch.Tensor) else seq_lens)]
    if len(lengths) != block_table.shape[0]:
        raise ValueError("one sequence length is required for each block-table row")
    gathered: list[torch.Tensor] = []
    for batch, length in enumerate(lengths):
        if length < 0:
            raise ValueError("sequence lengths must be non-negative")
        blocks = (length + block_size - 1) // block_size
        if blocks > block_table.shape[1]:
            raise ValueError("block_table has too few entries for a sequence length")
        if length == 0:
            gathered.append(cache.new_empty((0, *cache.shape[2:])))
            continue
        block_ids = block_table[batch, :blocks].to(device=cache.device, dtype=torch.long)
        if bool((block_ids < 0).any()) or bool((block_ids >= cache.shape[0]).any()):
            raise IndexError("block_table references a block outside the cache")
        rows = cache.index_select(0, block_ids).reshape(-1, *cache.shape[2:])
        gathered.append(rows[:length])
    return gathered


def gather_kv_rows_concat(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: Sequence[int] | torch.Tensor,
    *,
    block_size: int,
    validate: bool = True,
) -> torch.Tensor:
    """Gather the scheduler batch with one cache index operation.

    Rows stay grouped by request and then by token, so splitting the output
    by ``seq_lens`` is byte-identical to :func:`gather_kv_rows`. Production
    callers may skip redundant device range checks to avoid a hot-path host
    synchronization after vLLM has validated its block table.
    """

    if cache.dtype != torch.uint8 or cache.ndim != 4:
        raise TypeError("cache must be a 4-D uint8 tensor")
    if block_table.ndim != 2:
        raise ValueError("block_table must have shape [batch, max_blocks]")
    lengths = [
        int(value)
        for value in (
            seq_lens.tolist() if isinstance(seq_lens, torch.Tensor) else seq_lens
        )
    ]
    if len(lengths) != block_table.shape[0]:
        raise ValueError("one sequence length is required for each block-table row")
    if any(length < 0 for length in lengths):
        raise ValueError("sequence lengths must be non-negative")
    if not lengths or max(lengths) == 0:
        return cache.new_empty((0, *cache.shape[2:]))

    max_length = max(lengths)
    required_blocks = (max_length + block_size - 1) // block_size
    if required_blocks > block_table.shape[1]:
        raise ValueError("block_table has too few entries for a sequence length")

    device = cache.device
    lengths_device = torch.as_tensor(lengths, device=device, dtype=torch.long)
    positions = torch.arange(max_length, device=device, dtype=torch.long)
    valid = positions.unsqueeze(0) < lengths_device.unsqueeze(1)
    block_columns = torch.div(positions, block_size, rounding_mode="floor")
    request_blocks = block_table[:, :required_blocks].to(
        device=device, dtype=torch.long
    )
    block_ids = request_blocks[:, block_columns]
    selected_blocks = block_ids[valid]
    if validate and (
        bool((selected_blocks < 0).any())
        or bool((selected_blocks >= cache.shape[0]).any())
    ):
        raise IndexError("block_table references a block outside the cache")
    offsets = (positions % block_size).unsqueeze(0).expand_as(block_ids)[valid]
    slots = selected_blocks * block_size + offsets
    flat_cache = cache.reshape(-1, *cache.shape[2:])
    return flat_cache.index_select(0, slots.contiguous())


def paged_attention_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: Sequence[int] | torch.Tensor,
    *,
    block_size: int,
    scale: float,
    query_lens: Sequence[int] | torch.Tensor | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Small, device-agnostic HIF8 paged-attention oracle.

    It decodes only the gathered request-local rows.  The implementation is
    intentionally not the production performance path; it provides an exact
    contract for native/tile-local kernels and is useful on CPU CI.
    """

    if query.ndim != 3:
        raise ValueError("query must have shape [tokens, num_heads, head_size]")
    if key_cache.shape[2] != value_cache.shape[2]:
        raise ValueError("K/V cache head counts must match")
    batch = block_table.shape[0]
    lengths = [int(value) for value in (seq_lens.tolist() if isinstance(seq_lens, torch.Tensor) else seq_lens)]
    if len(lengths) != batch:
        raise ValueError("one sequence length is required per request")
    if query_lens is None:
        q_lengths = [1] * batch
    else:
        q_lengths = [int(value) for value in (query_lens.tolist() if isinstance(query_lens, torch.Tensor) else query_lens)]
    if len(q_lengths) != batch or sum(q_lengths) != query.shape[0]:
        raise ValueError("query_lens must describe every query row")

    key_rows = gather_kv_rows(key_cache, block_table, lengths, block_size=block_size)
    value_rows = gather_kv_rows(value_cache, block_table, lengths, block_size=block_size)
    outputs: list[torch.Tensor] = []
    q_start = 0
    for request, q_len in enumerate(q_lengths):
        k_data, k_scale = unpack_kv_rows(key_rows[request], key_cache.shape[-1] - HIF8_SCALE_BYTES)
        v_data, v_scale = unpack_kv_rows(value_rows[request], value_cache.shape[-1] - HIF8_SCALE_BYTES)
        keys = decode_kv_rows(k_data, k_scale).to(query.dtype)
        values = decode_kv_rows(v_data, v_scale).to(query.dtype)
        q = query[q_start : q_start + q_len]
        q_start += q_len
        if query.shape[1] % keys.shape[1] != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        repeat = query.shape[1] // keys.shape[1]
        keys = keys.repeat_interleave(repeat, dim=1)
        values = values.repeat_interleave(repeat, dim=1)
        logits = torch.einsum("qhd,khd->qhk", q, keys) * float(scale)
        if causal:
            # Query rows represent the final q_len positions of this request.
            positions = torch.arange(q_len, device=query.device) + lengths[request] - q_len
            key_positions = torch.arange(lengths[request], device=query.device)
            logits = logits.masked_fill(key_positions[None, None, :] > positions[:, None, None], float("-inf"))
        probs = torch.softmax(logits.float(), dim=-1).to(query.dtype)
        outputs.append(torch.einsum("qhk,khd->qhd", probs, values))
    return torch.cat(outputs, dim=0) if outputs else query.new_empty(query.shape)


__all__ = [
    "HIF8_CACHE_DTYPE",
    "HIF8_SCALE_BYTES",
    "Hif8KVCacheLayout",
    "decode_kv_rows",
    "gather_kv_rows",
    "gather_kv_rows_concat",
    "pack_kv_rows",
    "paged_attention_reference",
    "quantize_kv_rows",
    "scatter_kv_rows",
    "unpack_kv_rows",
]
