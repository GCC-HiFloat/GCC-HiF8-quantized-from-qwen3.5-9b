#!/usr/bin/env python3
"""Compose a projection-scoped HIF8 checkpoint from verified checkpoints.

The base checkpoint supplies every tensor by default.  Matching quantized
weights and their per-output-channel FP32 scales are copied byte-for-byte from
the donor checkpoint.  No tensor is decoded or re-quantized, so a projection
ablation can reuse a fully verified activation-MSE donor without repeatedly
encoding the embedding and all unrelated amax weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TensorPayload:
    path: Path
    offset: int
    size: int
    dtype: str
    shape: tuple[int, ...]


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _index(directory: Path) -> dict[str, str]:
    value = _load_json(directory / "model.safetensors.index.json")
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"invalid safetensors index in {directory}")
    return {str(name): str(shard) for name, shard in weight_map.items()}


def _tensor_payload(directory: Path, index: dict[str, str], name: str) -> TensorPayload:
    try:
        path = directory / index[name]
    except KeyError as exc:
        raise KeyError(f"{name} is absent from the index in {directory}") from exc
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors header: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = json.loads(stream.read(header_length))
    entry = header.get(name)
    if not isinstance(entry, dict):
        raise KeyError(f"{name} is absent from {path}")
    offsets = entry.get("data_offsets")
    shape = entry.get("shape")
    dtype = entry.get("dtype")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not isinstance(shape, list)
        or not isinstance(dtype, str)
    ):
        raise ValueError(f"invalid tensor metadata for {name} in {path}")
    begin, end = (int(value) for value in offsets)
    if begin < 0 or end <= begin:
        raise ValueError(f"invalid tensor offsets for {name} in {path}")
    return TensorPayload(
        path=path,
        offset=8 + header_length + begin,
        size=end - begin,
        dtype=dtype,
        shape=tuple(int(value) for value in shape),
    )


def _copy_payload(source: TensorPayload, destination: TensorPayload) -> None:
    if (source.size, source.dtype, source.shape) != (
        destination.size,
        destination.dtype,
        destination.shape,
    ):
        raise ValueError(
            "source/destination tensor contracts differ: "
            f"{source} != {destination}"
        )
    source_fd = os.open(source.path, os.O_RDONLY)
    destination_fd = os.open(destination.path, os.O_WRONLY)
    try:
        copied = 0
        chunk_size = 16 * 1024 * 1024
        while copied < source.size:
            size = min(chunk_size, source.size - copied)
            data = os.pread(source_fd, size, source.offset + copied)
            if len(data) != size:
                raise OSError(f"short read from {source.path}")
            written = 0
            while written < size:
                count = os.pwrite(
                    destination_fd,
                    data[written:],
                    destination.offset + copied + written,
                )
                if count <= 0:
                    raise OSError(f"short write to {destination.path}")
                written += count
            copied += size
    finally:
        os.close(source_fd)
        os.close(destination_fd)


def _write_payload(destination: TensorPayload, data: bytes, *, relative_offset: int = 0) -> None:
    if relative_offset < 0 or relative_offset + len(data) > destination.size:
        raise ValueError(f"write exceeds tensor payload: {destination}")
    descriptor = os.open(destination.path, os.O_WRONLY)
    try:
        written = 0
        while written < len(data):
            count = os.pwrite(
                descriptor,
                data[written:],
                destination.offset + relative_offset + written,
            )
            if count <= 0:
                raise OSError(f"short write to {destination.path}")
            written += count
    finally:
        os.close(descriptor)


def _weight_scale_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"not a weight tensor: {weight_name}")
    return weight_name + "_scale"


def _reencode_blended_weight(
    *,
    weight_name: str,
    base: Path,
    base_index: dict[str, str],
    donor: Path,
    donor_index: dict[str, str],
    float_source: Path,
    float_index: dict[str, str],
    destination: Path,
    destination_index: dict[str, str],
    blend_alpha: float,
) -> None:
    import torch
    from safetensors import safe_open

    from vllm_ascend_hif8.codec import MAX_FINITE
    from vllm_ascend_hif8.torch_codec import encode_tensor

    scale_tensor_name = _weight_scale_name(weight_name)

    def tensor(directory: Path, index: dict[str, str], name: str) -> "torch.Tensor":
        with safe_open(
            directory / index[name], framework="pt", device="cpu"
        ) as handle:
            return handle.get_tensor(name)

    source_weight = tensor(float_source, float_index, weight_name)
    base_scale = tensor(base, base_index, scale_tensor_name).to(torch.float32).reshape(-1)
    donor_scale = tensor(donor, donor_index, scale_tensor_name).to(torch.float32).reshape(-1)
    if source_weight.ndim != 2 or source_weight.shape[0] != base_scale.numel():
        raise ValueError(f"source weight/scale shape mismatch for {weight_name}")
    if base_scale.shape != donor_scale.shape:
        raise ValueError(f"base/donor scale shape mismatch for {weight_name}")
    blended_scale = torch.exp(
        (1.0 - blend_alpha) * torch.log(base_scale)
        + blend_alpha * torch.log(donor_scale)
    ).contiguous()

    weight_payload = _tensor_payload(destination, destination_index, weight_name)
    if weight_payload.dtype != "U8" or weight_payload.shape != tuple(source_weight.shape):
        raise ValueError(f"destination weight contract is not U8 for {weight_name}")
    columns = source_weight.shape[1]
    rows_per_chunk = max(1, (4 * 1024 * 1024) // columns)
    bytes_per_row = columns
    for begin in range(0, source_weight.shape[0], rows_per_chunk):
        end = min(begin + rows_per_chunk, source_weight.shape[0])
        normalized = (
            source_weight[begin:end].to(torch.float32)
            / blended_scale[begin:end, None]
        ).clamp(min=-MAX_FINITE, max=MAX_FINITE)
        encoded = encode_tensor(normalized).contiguous()
        _write_payload(
            weight_payload,
            encoded.numpy().tobytes(),
            relative_offset=begin * bytes_per_row,
        )

    scale_payload = _tensor_payload(
        destination, destination_index, scale_tensor_name
    )
    scale_bytes = blended_scale.reshape(scale_payload.shape).numpy().tobytes()
    _write_payload(scale_payload, scale_bytes)


def compose(
    base: Path,
    donor: Path,
    destination: Path,
    name_regex: str,
    *,
    blend_alpha: float = 1.0,
    float_source: Path | None = None,
) -> int:
    base = base.resolve()
    donor = donor.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    if not math.isfinite(blend_alpha) or not 0.0 <= blend_alpha <= 1.0:
        raise ValueError("blend_alpha must be finite and in [0,1]")
    if blend_alpha != 1.0 and float_source is None:
        raise ValueError("float_source is required when blend_alpha is not 1")
    try:
        pattern = re.compile(name_regex)
    except re.error as exc:
        raise ValueError(f"invalid name regex: {exc}") from exc

    base_metadata = _load_json(base / "hif8_metadata.json")
    donor_metadata = _load_json(donor / "hif8_metadata.json")
    for label, metadata in (("base", base_metadata), ("donor", donor_metadata)):
        if metadata.get("format") != "W8A8HIF8_STATIC":
            raise ValueError(f"{label} is not a W8A8HIF8_STATIC checkpoint")
        if metadata.get("byte_layout") != "ascend_hifloat8":
            raise ValueError(f"{label} does not use physical Ascend HiFloat8 bytes")
    if base_metadata.get("quantized_layer_count") != donor_metadata.get(
        "quantized_layer_count"
    ):
        raise ValueError("base and donor quantized-layer counts differ")

    base_layers = base_metadata.get("layers")
    donor_layers = donor_metadata.get("layers")
    if not isinstance(base_layers, dict) or not isinstance(donor_layers, dict):
        raise ValueError("base and donor metadata must contain layer maps")
    if set(base_layers) != set(donor_layers):
        raise ValueError("base and donor layer sets differ")
    selected = sorted(
        name
        for name, entry in donor_layers.items()
        if pattern.search(name)
        and isinstance(entry, dict)
        and entry.get("weight_scale_mode") == "activation_mse"
    )
    if not selected:
        raise ValueError("name regex selected no activation-MSE donor weights")

    base_index = _index(base)
    donor_index = _index(donor)
    if set(base_index) != set(donor_index):
        raise ValueError("base and donor tensor sets differ")

    shutil.copytree(base, destination, copy_function=shutil.copy2)
    destination_index = _index(destination)
    float_index = _index(float_source.resolve()) if float_source is not None else None
    for weight_name in selected:
        if blend_alpha == 1.0:
            for tensor_name in (weight_name, _weight_scale_name(weight_name)):
                _copy_payload(
                    _tensor_payload(donor, donor_index, tensor_name),
                    _tensor_payload(destination, destination_index, tensor_name),
                )
        else:
            assert float_source is not None and float_index is not None
            _reencode_blended_weight(
                weight_name=weight_name,
                base=base,
                base_index=base_index,
                donor=donor,
                donor_index=donor_index,
                float_source=float_source.resolve(),
                float_index=float_index,
                destination=destination,
                destination_index=destination_index,
                blend_alpha=blend_alpha,
            )

    output_metadata = base_metadata
    for name, entry in output_metadata["layers"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"invalid base layer metadata for {name}")
        # Older amax checkpoints predate per-layer mode provenance.  Their
        # physical row-scale construction is amax, so make that contract
        # explicit in every composed checkpoint.
        entry.setdefault("weight_scale_mode", "amax")
    for name in selected:
        entry = dict(donor_layers[name])
        if blend_alpha == 1.0:
            # A byte-for-byte payload copy must preserve the donor's effective
            # scale provenance.  ``blend_alpha`` describes this composition
            # operation; it does not turn an already blended donor weight back
            # into a full activation-MSE (alpha=1) weight.
            inherited_alpha = entry.get(
                "weight_scale_blend_alpha",
                donor_metadata.get("weight_scale_blend_alpha", 1.0),
            )
            if (
                not isinstance(inherited_alpha, (int, float))
                or not math.isfinite(float(inherited_alpha))
            ):
                inherited_alpha = 1.0
            entry["weight_scale_blend_alpha"] = float(inherited_alpha)
        else:
            entry["weight_scale_blend_alpha"] = blend_alpha
        output_metadata["layers"][name] = entry
    output_metadata["weight_scale_mode"] = "activation_mse"
    output_metadata["weight_scale_scope"] = donor_metadata.get("weight_scale_scope")
    output_metadata["weight_scale_factor"] = donor_metadata.get(
        "weight_scale_factor", 1.0
    )
    layer_blend_alphas = sorted(
        {
            float(entry.get("weight_scale_blend_alpha", 1.0))
            for entry in output_metadata["layers"].values()
            if entry.get("weight_scale_mode") == "activation_mse"
        }
    )
    output_metadata["weight_scale_blend_alpha"] = (
        layer_blend_alphas[0] if len(layer_blend_alphas) == 1 else "mixed"
    )
    output_metadata["weight_scale_blend_alphas"] = layer_blend_alphas
    output_metadata["weight_scale_name_regex"] = name_regex
    output_metadata["weight_activation_calibration"] = donor_metadata.get(
        "weight_activation_calibration"
    )
    output_metadata["weight_scale_composition"] = {
        "method": (
            "verified_tensor_payload_copy"
            if blend_alpha == 1.0
            else "fixed_geometric_scale_reencode"
        ),
        "base": str(base),
        "donor": str(donor),
        "selected_count": len(selected),
        "selected_weights": selected,
        "blend_alpha": blend_alpha,
        "float_source": str(float_source.resolve()) if float_source else None,
    }
    metadata_path = destination / "hif8_metadata.json"
    temporary = metadata_path.with_name(metadata_path.name + ".tmp")
    temporary.write_text(
        json.dumps(output_metadata, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(metadata_path)
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("donor", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--name-regex", required=True)
    parser.add_argument("--blend-alpha", type=float, default=1.0)
    parser.add_argument("--float-source", type=Path)
    args = parser.parse_args()
    count = compose(
        args.base,
        args.donor,
        args.destination,
        args.name_regex,
        blend_alpha=args.blend_alpha,
        float_source=args.float_source,
    )
    print(f"composed {args.destination} with {count} donor weights")


if __name__ == "__main__":
    main()
