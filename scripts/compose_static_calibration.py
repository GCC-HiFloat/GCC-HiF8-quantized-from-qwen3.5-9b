#!/usr/bin/env python3
"""Compose Linear and KV sections from two complete static calibrations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from vllm_ascend_hif8.calibration import (
    file_sha256,
    validate_calibration_artifact,
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"calibration artifact is not an object: {path}")
    validate_calibration_artifact(value)
    return value


def compose(linear_path: Path, kv_path: Path) -> dict:
    linear = _load(linear_path)
    kv = _load(kv_path)
    linear_identity = linear.get("model", {}).get("identity_sha256")
    kv_identity = kv.get("model", {}).get("identity_sha256")
    if not isinstance(linear_identity, dict) or linear_identity != kv_identity:
        raise ValueError("Linear and KV calibration model identities differ")

    result = copy.deepcopy(linear)
    result["kv_cache"] = copy.deepcopy(kv["kv_cache"])
    result.setdefault("provenance", {})["composition"] = {
        "linear_source": str(linear_path.resolve()),
        "linear_source_sha256": file_sha256(linear_path),
        "kv_source": str(kv_path.resolve()),
        "kv_source_sha256": file_sha256(kv_path),
        "kv_source_dataset": copy.deepcopy(kv.get("dataset")),
        "kv_source_sampling": copy.deepcopy(kv.get("sampling")),
        "kv_source_search": copy.deepcopy(kv.get("search")),
    }
    validate_calibration_artifact(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("linear", type=Path, help="artifact supplying Linear scales")
    parser.add_argument("kv", type=Path, help="artifact supplying the complete KV section")
    parser.add_argument("output", type=Path, help="new composed artifact")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = compose(args.linear, args.kv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
