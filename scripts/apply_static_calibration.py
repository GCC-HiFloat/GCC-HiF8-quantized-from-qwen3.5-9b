#!/usr/bin/env python3
"""Apply one complete static calibration to an existing HIF8 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

from vllm_ascend_hif8.calibration import validate_calibration_artifact


def _index(directory: Path) -> dict[str, str]:
    value = json.loads(
        (directory / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    return {str(name): str(shard) for name, shard in value["weight_map"].items()}


def _tensor_offset(path: Path, name: str) -> int:
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
        raise ValueError(f"{name} is not a scalar F32 input scale")
    begin, end = (int(value) for value in entry["data_offsets"])
    if end - begin != 4:
        raise ValueError(f"{name} payload is not four bytes")
    return 8 + header_length + begin


def apply(calibration_path: Path, destination: Path) -> int:
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    validate_calibration_artifact(calibration)
    metadata_path = destination / "hif8_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "W8A8HIF8_STATIC":
        raise ValueError("destination is not a W8A8HIF8_STATIC checkpoint")
    if metadata.get("calibration", {}).get("model", {}).get(
        "identity_sha256"
    ) != calibration.get("model", {}).get("identity_sha256"):
        raise ValueError("calibration and checkpoint model identities differ")

    weight_map = _index(destination)
    scale_names = {
        f"{prefix}.input_scale": float(entry["scale"])
        for prefix, entry in calibration["linear_groups"].items()
    }
    actual_names = {name for name in weight_map if name.endswith(".input_scale")}
    if set(scale_names) != actual_names:
        raise ValueError("calibration and checkpoint input_scale sets differ")

    writes = []
    for name, value in sorted(scale_names.items()):
        if not value > 0.0:
            raise ValueError(f"invalid scale for {name}: {value!r}")
        shard = destination / weight_map[name]
        writes.append((shard, _tensor_offset(shard, name), value))
    for shard, offset, value in writes:
        descriptor = os.open(shard, os.O_WRONLY)
        try:
            if os.pwrite(descriptor, struct.pack("<f", value), offset) != 4:
                raise OSError(f"short write while patching {shard}")
        finally:
            os.close(descriptor)

    kv_layers = calibration["kv_cache"]["layers"]
    metadata["calibration"] = calibration
    metadata["kv_cache"] = {
        "mode": "static_per_layer_per_head",
        "layers": {
            layer: {
                "key_scales": list(entry["key"]["scales"]),
                "value_scales": list(entry["value"]["scales"]),
            }
            for layer, entry in sorted(kv_layers.items())
        },
    }
    metadata["static_activation_scale_count"] = len(scale_names)
    temporary = metadata_path.with_name(metadata_path.name + ".tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(metadata_path)
    return len(scale_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    count = apply(args.calibration, args.checkpoint)
    print(f"applied {count} static Linear scales from {args.calibration}")


if __name__ == "__main__":
    main()
