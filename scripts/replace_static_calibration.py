#!/usr/bin/env python3
"""Replace only static activation/KV scales in an exported HIF8 checkpoint.

Preserve U8 weights and per-channel weight scales. Patch FP32 ``input_scale``
payloads in place after dtype/shape checks, then replace metadata atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path


def _index(directory: Path) -> dict[str, str]:
    path = directory / "model.safetensors.index.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    return {str(name): str(shard) for name, shard in value["weight_map"].items()}


def _tensor_header(path: Path, name: str) -> tuple[int, int]:
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors header: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = json.loads(stream.read(header_length))
    entry = header.get(name)
    if not isinstance(entry, dict):
        raise KeyError(f"{name} is absent from {path}")
    if entry.get("dtype") != "F32" or entry.get("shape") != [1]:
        raise ValueError(
            f"{name} must be one F32 value, got "
            f"dtype={entry.get('dtype')}, shape={entry.get('shape')}"
        )
    offsets = entry.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"invalid data_offsets for {name} in {path}")
    begin, end = (int(value) for value in offsets)
    if end - begin != 4:
        raise ValueError(f"{name} payload is not four bytes")
    return 8 + header_length + begin, 4


def replace(reference: Path, destination: Path) -> int:
    reference_meta = json.loads(
        (reference / "hif8_metadata.json").read_text(encoding="utf-8")
    )
    destination_path = destination / "hif8_metadata.json"
    destination_meta = json.loads(destination_path.read_text(encoding="utf-8"))
    if reference_meta.get("format") != "W8A8HIF8_STATIC" or destination_meta.get(
        "format"
    ) != "W8A8HIF8_STATIC":
        raise ValueError("both checkpoints must use W8A8HIF8_STATIC")
    reference_identity = reference_meta.get("calibration", {}).get("model", {}).get(
        "identity_sha256"
    )
    destination_identity = destination_meta.get("calibration", {}).get(
        "model", {}
    ).get("identity_sha256")
    if reference_identity != destination_identity:
        raise ValueError("calibration model identities differ")

    reference_map = _index(reference)
    destination_map = _index(destination)
    scale_names = sorted(
        name for name in reference_map if name.endswith(".input_scale")
    )
    if not scale_names or set(scale_names) != {
        name for name in destination_map if name.endswith(".input_scale")
    }:
        raise ValueError("input_scale tensor sets differ")

    from safetensors import safe_open

    values: dict[str, float] = {}
    for name in scale_names:
        with safe_open(
            reference / reference_map[name], framework="pt", device="cpu"
        ) as handle:
            tensor = handle.get_tensor(name)
        value = float(tensor.item())
        if not value > 0.0:
            raise ValueError(f"invalid reference scale for {name}: {value!r}")
        values[name] = value

    for name, value in values.items():
        shard = destination / destination_map[name]
        offset, size = _tensor_header(shard, name)
        descriptor = os.open(shard, os.O_WRONLY)
        try:
            written = os.pwrite(descriptor, struct.pack("<f", value), offset)
            if written != size:
                raise OSError(f"short write while patching {name}")
        finally:
            os.close(descriptor)

    destination_meta["calibration"] = reference_meta["calibration"]
    destination_meta["kv_cache"] = reference_meta["kv_cache"]
    destination_meta["static_activation_scale_count"] = len(scale_names)
    temporary = destination_path.with_name(destination_path.name + ".tmp")
    temporary.write_text(
        json.dumps(destination_meta, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n",
        encoding="ascii",
    )
    temporary.replace(destination_path)
    print(
        f"replaced {len(scale_names)} activation scales and "
        f"{len(destination_meta['kv_cache']['layers'])} KV layer scale sets"
    )
    return len(scale_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    replace(args.reference, args.destination)


if __name__ == "__main__":
    main()
