#!/usr/bin/env python3
"""Verify an exported HiFloat8 checkpoint against its float source.

Checks that every quantized weight round-trips through the reference codec
exactly (byte-for-byte), that scales are finite and positive, and reports the
relative dequantization error per layer so a bad export is visible before
serving. Run after ``python -m vllm_ascend_hif8.exporter SRC DST``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path


def _load_shard_map(directory: Path) -> dict[str, str]:
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        return dict(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"])

    from safetensors import safe_open

    single = directory / "model.safetensors"
    with safe_open(single, framework="pt", device="cpu") as handle:
        return {name: single.name for name in handle.keys()}


def verify(
    source_dir: Path,
    output_dir: Path,
    *,
    tolerance: float,
    contract_only: bool = False,
) -> int:
    import torch
    from safetensors import safe_open

    from vllm_ascend_hif8.codec import ASCEND_DECODE_TABLE
    from vllm_ascend_hif8.exporter import (
        DYNAMIC_QUANT_TYPE,
        METADATA_FILENAME,
        QUANT_DESCRIPTION_FILENAME,
        STATIC_QUANT_TYPE,
        _load_config,
        _validate_calibration_for_checkpoint,
        build_quant_description,
        input_scale_name,
        is_qwen35_target_weight,
        scale_name,
    )
    from vllm_ascend_hif8.torch_codec import encode_tensor

    decode_table = torch.tensor(ASCEND_DECODE_TABLE, dtype=torch.float32)

    metadata = json.loads((output_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    expected = set(metadata["layers"])
    source_map = _load_shard_map(source_dir)
    output_map = _load_shard_map(output_dir)

    # Validate Linear scale coverage and KV source identity before weight bytes.
    static_contract: dict[str, object] = {
        "format": metadata.get("format"),
        "validated": False,
    }
    if metadata.get("format") == STATIC_QUANT_TYPE:
        config = _load_config(source_dir)
        calibration = metadata.get("calibration")
        if not isinstance(calibration, dict):
            print("FAIL: static checkpoint has no embedded calibration artifact")
            return 1

        static_scales = _validate_calibration_for_checkpoint(
            calibration,
            input_dir=source_dir,
            config=config,
            tensor_names=set(source_map),
            shard_names=set(source_map.values()),
        )
        expected_input_scales = {
            input_scale_name(prefix) for prefix in static_scales
        }
        actual_input_scales = {
            name for name in output_map if name.endswith(".input_scale")
        }
        if actual_input_scales != expected_input_scales:
            print(
                "FAIL: static input_scale tensors do not exactly match the "
                "calibrated Linear groups"
            )
            print(
                "  missing:",
                sorted(expected_input_scales - actual_input_scales)[:5],
            )
            print(
                "  unexpected:",
                sorted(actual_input_scales - expected_input_scales)[:5],
            )
            return 1

        # Load scale tensors without materializing the weight tensors.
        input_scale_values: dict[str, float] = {}
        for name in sorted(expected_input_scales):
            shard = output_dir / output_map[name]
            with safe_open(shard, framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(name)
            if tensor.dtype != torch.float32 or tuple(tensor.shape) != (1,):
                print(
                    f"FAIL {name}: expected one FP32 value, got "
                    f"dtype={tensor.dtype}, shape={tuple(tensor.shape)}"
                )
                return 1
            value = float(tensor.item())
            prefix = name[: -len(".input_scale")]
            expected_value = float(static_scales[prefix])
            if (
                not math.isfinite(value)
                or value <= 0.0
                or not math.isclose(value, expected_value, rel_tol=0.0, abs_tol=1.0e-7)
            ):
                print(
                    f"FAIL {name}: checkpoint={value!r}, "
                    f"calibration={expected_value!r}"
                )
                return 1
            input_scale_values[name] = value

        quant_description_path = output_dir / QUANT_DESCRIPTION_FILENAME
        if not quant_description_path.is_file():
            print(f"FAIL: missing {quant_description_path}")
            return 1
        quant_description = json.loads(
            quant_description_path.read_text(encoding="utf-8")
        )
        expected_description = build_quant_description(
            tuple(source_map), static_activation=True
        )
        if quant_description != expected_description:
            print("FAIL: quant_model_description.json does not match static contract")
            only_expected = sorted(set(expected_description) - set(quant_description))
            only_actual = sorted(set(quant_description) - set(expected_description))
            mismatched = sorted(
                name
                for name in set(expected_description) & set(quant_description)
                if expected_description[name] != quant_description[name]
            )
            print("  missing:", only_expected[:5])
            print("  unexpected:", only_actual[:5])
            print("  mismatched:", mismatched[:5])
            return 1

        if metadata.get("activation_quantization") != "static_per_linear_input_tensor":
            print("FAIL: static checkpoint has the wrong activation_quantization metadata")
            return 1
        if metadata.get("linear_quant_type") != STATIC_QUANT_TYPE:
            print("FAIL: static checkpoint has the wrong linear_quant_type metadata")
            return 1
        if metadata.get("embedding_quant_type") != DYNAMIC_QUANT_TYPE:
            print("FAIL: embedding compatibility quantization type is missing")
            return 1
        if metadata.get("static_activation_scale_count") != len(static_scales):
            print(
                "FAIL: static_activation_scale_count disagrees with the "
                f"calibration ({metadata.get('static_activation_scale_count')} "
                f"vs {len(static_scales)})"
            )
            return 1

        kv_metadata = metadata.get("kv_cache")
        expected_kv = calibration.get("kv_cache", {}).get("layers")
        if not isinstance(kv_metadata, dict) or not isinstance(expected_kv, dict):
            print("FAIL: static checkpoint is missing KV metadata")
            return 1
        if kv_metadata.get("mode") != "static_per_layer_per_head":
            print("FAIL: static checkpoint has the wrong KV cache mode")
            return 1
        actual_kv = kv_metadata.get("layers")
        if not isinstance(actual_kv, dict) or set(actual_kv) != set(expected_kv):
            print("FAIL: checkpoint KV layer set differs from calibration")
            return 1
        for layer, expected_entry in expected_kv.items():
            actual_entry = actual_kv[layer]
            for kind, metadata_key in (("key", "key_scales"), ("value", "value_scales")):
                expected_values = [float(value) for value in expected_entry[kind]["scales"]]
                actual_values = actual_entry.get(metadata_key)
                if not isinstance(actual_values, list) or len(actual_values) != len(expected_values):
                    print(f"FAIL: KV layer {layer} {kind} scale count differs from calibration")
                    return 1
                if any(
                    not math.isfinite(float(actual))
                    or float(actual) <= 0.0
                    or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-7)
                    for actual, expected in zip(actual_values, expected_values, strict=True)
                ):
                    print(f"FAIL: KV layer {layer} {kind} scales differ from calibration")
                    return 1

        static_contract.update(
            {
                "validated": True,
                "linear_scale_count": len(input_scale_values),
                "kv_layer_count": len(actual_kv),
                "quant_description_counts": dict(Counter(quant_description.values())),
                "calibration_schema_version": calibration.get("schema_version"),
                "calibration_dataset": calibration.get("dataset", {}).get("repository"),
                "calibration_sampling": calibration.get("sampling"),
            }
        )
    elif metadata.get("format") != DYNAMIC_QUANT_TYPE:
        print(f"FAIL: unsupported checkpoint format {metadata.get('format')!r}")
        return 1

    selected = {name for name in source_map if is_qwen35_target_weight(name)}
    if selected != expected:
        print(f"FAIL: metadata lists {len(expected)} layers but selection yields {len(selected)}")
        print(f"  only in metadata: {sorted(expected - selected)[:5]}")
        print(f"  only in source:   {sorted(selected - expected)[:5]}")
        return 1

    missing_scales = sorted(name for name in expected if scale_name(name) not in output_map)
    if missing_scales:
        print(f"FAIL: {len(missing_scales)} scale tensors missing from the export, e.g. {missing_scales[:3]}")
        return 1

    if contract_only:
        print("static contract:", json.dumps(static_contract, ensure_ascii=True, sort_keys=True))
        print(
            "OK: checkpoint metadata, activation scales, KV scales, and tensor "
            "sets satisfy the static contract (weight re-encoding skipped)"
        )
        return 0

    failures = 0
    worst = (0.0, "")
    try:
        verify_chunk_elements = int(os.environ.get("HIF8_VERIFY_CHUNK_ELEMENTS", str(32 * 1024 * 1024)))
    except ValueError as exc:
        raise ValueError("HIF8_VERIFY_CHUNK_ELEMENTS must be a positive integer") from exc
    if verify_chunk_elements < 1:
        raise ValueError("HIF8_VERIFY_CHUNK_ELEMENTS must be a positive integer")
    for shard in sorted(set(source_map.values())):
        members = [name for name, target in source_map.items() if target == shard]
        if not any(name in expected for name in members):
            continue
        # Compare tensors shard by shard to bound memory usage.
        with safe_open(source_dir / shard, framework="pt", device="cpu") as source_handle, safe_open(
            output_dir / shard, framework="pt", device="cpu"
        ) as output_handle:
            for name in sorted(name for name in members if name in expected):
                weight = output_handle.get_tensor(name)
                scale = output_handle.get_tensor(scale_name(name)).reshape(-1)
                reference_bf16 = source_handle.get_tensor(name)

                if weight.dtype != torch.uint8:
                    print(f"FAIL {name}: stored dtype is {weight.dtype}, expected torch.uint8")
                    failures += 1
                    continue
                if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
                    print(f"FAIL {name}: scale has nonfinite or nonpositive entries")
                    failures += 1
                    continue

                rows_per_chunk = max(1, verify_chunk_elements // reference_bf16.shape[-1])
                denominator = reference_bf16.abs().amax().to(torch.float32).clamp(min=1e-12)
                tensor_error = 0.0
                mismatch_count = 0
                for begin in range(0, reference_bf16.shape[0], rows_per_chunk):
                    end = min(begin + rows_per_chunk, reference_bf16.shape[0])
                    reference = reference_bf16[begin:end].to(torch.float32)
                    weight_chunk = weight[begin:end]
                    scale_chunk = scale[begin:end]
                    recomputed = encode_tensor(reference / scale_chunk.unsqueeze(-1))
                    mismatch_count += int((weight_chunk != recomputed).sum())
                    restored = decode_table[weight_chunk.to(torch.int64)] * scale_chunk.unsqueeze(-1)
                    chunk_error = float((restored - reference).abs().amax() / denominator)
                    tensor_error = max(tensor_error, chunk_error)
                    del reference, weight_chunk, scale_chunk, recomputed, restored
                if mismatch_count:
                    print(f"FAIL {name}: {mismatch_count} bytes differ from a fresh encode")
                    failures += 1
                    continue

                if tensor_error > worst[0]:
                    worst = (tensor_error, name)
                if tensor_error > tolerance:
                    print(f"FAIL {name}: relative dequantization error {tensor_error:.4f} exceeds {tolerance}")
                    failures += 1
                del weight, scale, reference_bf16

    print(f"checked {len(expected)} quantized weights; worst relative error {worst[0]:.4f} ({worst[1]})")
    print("static contract:", json.dumps(static_contract, ensure_ascii=True, sort_keys=True))
    if failures:
        print(f"FAILED: {failures} weight(s) did not verify")
        return 1
    print("OK: every quantized weight matches a fresh reference encode")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="original float checkpoint directory")
    parser.add_argument("output", type=Path, help="exported HiFloat8 checkpoint directory")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.125,
        help="max allowed dequantization error relative to the tensor amax (default: 0.125)",
    )
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help=(
            "validate metadata, static activation/KV scales, and tensor sets "
            "without re-encoding every quantized weight"
        ),
    )
    args = parser.parse_args(argv)
    return verify(
        args.source,
        args.output,
        tolerance=args.tolerance,
        contract_only=args.contract_only,
    )


if __name__ == "__main__":
    sys.exit(main())
