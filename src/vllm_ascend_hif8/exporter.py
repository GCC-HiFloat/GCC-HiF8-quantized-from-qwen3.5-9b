"""Stream a Qwen3.5 checkpoint into Ascend-layout HiFloat8 U8 weights."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

DYNAMIC_QUANT_TYPE = "W8A8HIF8_SOFT"
STATIC_QUANT_TYPE = "W8A8HIF8_STATIC"
# Alias for the dynamic checkpoint format.
QUANT_TYPE = DYNAMIC_QUANT_TYPE
QUANT_DESCRIPTION_FILENAME = "quant_model_description.json"
METADATA_FILENAME = "hif8_metadata.json"

# Match Qwen3.5 text decoder weights with or without the language_model prefix.
# MTP and vision weights are outside the text-only TP=1 quantization scope.
_ATTENTION_LEAVES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
_MLP_LEAVES = frozenset({"gate_proj", "up_proj", "down_proj"})
_GDN_LEAVES = frozenset({"in_proj_qkv", "in_proj_z", "out_proj"})
_TARGET_MODULE_LEAVES = {
    "self_attn": _ATTENTION_LEAVES,
    "mlp": _MLP_LEAVES,
    "linear_attn": _GDN_LEAVES,
}
_TARGET_PATTERN = re.compile(
    r"^model\.(?:language_model\.)?layers\.\d+\.(self_attn|mlp|linear_attn)\.([A-Za-z0-9_]+)\.weight$"
)
_EMBEDDING_WEIGHTS = frozenset(
    {
        "model.language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
    }
)
_WEIGHT_SCALE_SCOPE_FAMILIES = frozenset(
    {"mlp", "self_attn", "linear_attn", "embedding"}
)

_FULL_ATTENTION = "full_attention"
_LINEAR_ATTENTION = "linear_attention"

_COPY_EXCLUDES = frozenset(
    {
        QUANT_DESCRIPTION_FILENAME,
        METADATA_FILENAME,
        "model.safetensors.index.json",
    }
)


def is_qwen35_target_weight(name: str) -> bool:
    """Return whether a checkpoint tensor belongs to the serving W8 scope."""
    if name in _EMBEDDING_WEIGHTS:
        return True
    match = _TARGET_PATTERN.match(name)
    if match is None:
        return False
    module, leaf = match.groups()
    return leaf in _TARGET_MODULE_LEAVES[module]


def scale_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"not a weight tensor name: {weight_name}")
    return f"{weight_name[: -len('.weight')]}.weight_scale"


def _canonical_activation_group(weight_name: str) -> str | None:
    """Map one component weight to the runtime Linear input it consumes."""

    if weight_name in _EMBEDDING_WEIGHTS:
        return None
    match = _TARGET_PATTERN.match(weight_name)
    if match is None:
        raise ValueError(f"not a Qwen3.5 target weight: {weight_name}")
    module, leaf = match.groups()
    canonical_leaf = {
        ("self_attn", "k_proj"): "q_proj",
        ("self_attn", "v_proj"): "q_proj",
        ("mlp", "up_proj"): "gate_proj",
        ("linear_attn", "in_proj_z"): "in_proj_qkv",
    }.get((module, leaf), leaf)
    prefix = weight_name[: -len(".weight")].rsplit(".", 1)[0]
    return f"{prefix}.{canonical_leaf}"


def input_scale_name(checkpoint_prefix: str) -> str:
    """Return the canonical static activation-scale tensor name."""

    if checkpoint_prefix.endswith(".weight"):
        checkpoint_prefix = checkpoint_prefix[: -len(".weight")]
    return f"{checkpoint_prefix}.input_scale"


def _weight_scale_scope_members(scope: str) -> frozenset[str]:
    """Parse ``all`` or a ``+``-joined set of checkpoint module families."""

    if scope == "all":
        return _WEIGHT_SCALE_SCOPE_FAMILIES
    members = scope.split("+")
    if (
        not members
        or any(not member for member in members)
        or len(set(members)) != len(members)
        or not set(members).issubset(_WEIGHT_SCALE_SCOPE_FAMILIES)
    ):
        raise ValueError(
            "weight_scale_scope must be 'all' or a '+'-joined subset of "
            f"{sorted(_WEIGHT_SCALE_SCOPE_FAMILIES)}, got {scope!r}"
        )
    return frozenset(members)


def build_quant_description(
    tensor_names: list[str] | tuple[str, ...],
    *,
    static_activation: bool = False,
) -> dict[str, str]:
    """Build the ModelSlim layer map used by vllm-ascend scheme selection."""

    description: dict[str, str] = {}
    for name in tensor_names:
        if not name.endswith(".weight"):
            continue
        if not is_qwen35_target_weight(name):
            description[name] = "FLOAT"
        elif static_activation and name not in _EMBEDDING_WEIGHTS:
            description[name] = STATIC_QUANT_TYPE
        else:
            # The embedding method gathers weight rows without quantizing inputs.
            description[name] = DYNAMIC_QUANT_TYPE
    return description


def _tensor_nbytes(tensor: "torch.Tensor") -> int:
    return tensor.numel() * tensor.element_size()


def _quantize_tensor_rows_chunked(
    tensor: "torch.Tensor",
    *,
    scale_factor: float,
    scale_mode: str = "amax",
    channel_mean_square: "torch.Tensor | None" = None,
    activation_mse_blend_alpha: float = 1.0,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Quantize a 2-D weight without materializing full-size FP32 temporaries.

    Independent row scales allow chunking without changing quantized values.
    """
    import os

    import torch

    from .torch_codec import (
        quantize_tensor_rows,
        quantize_tensor_rows_activation_mse,
        quantize_tensor_rows_mse,
    )

    if scale_mode not in {"amax", "mse", "activation_mse"}:
        raise ValueError(f"unsupported weight scale mode: {scale_mode!r}")
    if scale_mode == "activation_mse" and channel_mean_square is None:
        raise ValueError("activation_mse weight scaling requires channel energy")

    default_chunk_elements = 4 * 1024 * 1024
    raw = os.environ.get("HIF8_EXPORT_CHUNK_ELEMENTS", str(default_chunk_elements))
    try:
        chunk_elements = int(raw)
    except ValueError as exc:
        raise ValueError("HIF8_EXPORT_CHUNK_ELEMENTS must be a positive integer") from exc
    if chunk_elements < 1:
        raise ValueError("HIF8_EXPORT_CHUNK_ELEMENTS must be a positive integer")
    rows_per_chunk = max(1, chunk_elements // tensor.shape[-1])
    encoded = torch.empty(tensor.shape, dtype=torch.uint8, device="cpu")
    scales = torch.empty(tensor.shape[0], dtype=torch.float32, device="cpu")
    for begin in range(0, tensor.shape[0], rows_per_chunk):
        end = min(begin + rows_per_chunk, tensor.shape[0])
        if scale_mode == "activation_mse":
            data_chunk, scale_chunk = quantize_tensor_rows_activation_mse(
                tensor[begin:end],
                channel_mean_square,
                blend_alpha=activation_mse_blend_alpha,
            )
        elif scale_mode == "mse":
            data_chunk, scale_chunk = quantize_tensor_rows_mse(tensor[begin:end])
        else:
            data_chunk, scale_chunk = quantize_tensor_rows(
                tensor[begin:end], scale_factor=scale_factor
            )
        encoded[begin:end].copy_(data_chunk.to(device="cpu", dtype=torch.uint8))
        scales[begin:end].copy_(scale_chunk.to(device="cpu", dtype=torch.float32))
        del data_chunk, scale_chunk
    return encoded.contiguous(), scales.contiguous()


def _quantize_shard(
    state: dict[str, "torch.Tensor"],
    *,
    static_scales: dict[str, float] | None = None,
    weight_scale_factor: float = 1.0,
    weight_scale_scope: str = "all",
    weight_scale_mode: str = "amax",
    activation_importance: dict[str, "torch.Tensor"] | None = None,
    weight_scale_blend_alpha: float = 1.0,
    weight_scale_name_pattern: "re.Pattern[str] | None" = None,
) -> tuple[dict[str, "torch.Tensor"], dict[str, Any], set[str]]:
    import torch

    output: dict[str, torch.Tensor] = {}
    layers: dict[str, Any] = {}
    written_input_scales: set[str] = set()
    scope_members = _weight_scale_scope_members(weight_scale_scope)
    for name, tensor in state.items():
        if not is_qwen35_target_weight(name):
            output[name] = tensor
            continue
        if tensor.ndim != 2:
            raise ValueError(f"target linear weight must be 2D: {name} has shape {tuple(tensor.shape)}")
        if not tensor.is_floating_point():
            raise ValueError(f"target linear weight must be floating point: {name} has dtype {tensor.dtype}")

        # Fold the selected factor into the stored FP32 row scale.
        family_in_scope = (
            ("mlp" in scope_members and ".mlp." in name)
            or ("self_attn" in scope_members and ".self_attn." in name)
            or ("linear_attn" in scope_members and ".linear_attn." in name)
            or ("embedding" in scope_members and name in _EMBEDDING_WEIGHTS)
        )
        in_scope = family_in_scope and (
            weight_scale_name_pattern is None
            or weight_scale_name_pattern.search(name) is not None
        )
        tensor_scale_factor = weight_scale_factor if in_scope else 1.0
        activation_group = _canonical_activation_group(name)
        # Embeddings have no activation group; use amax for activation_mse.
        tensor_scale_mode = (
            weight_scale_mode
            if in_scope
            and not (
                weight_scale_mode == "activation_mse"
                and activation_group is None
            )
            else "amax"
        )
        channel_mean_square = None
        if tensor_scale_mode == "activation_mse":
            if activation_importance is None or activation_group not in activation_importance:
                raise ValueError(
                    "activation calibration is missing channel energy for "
                    f"{activation_group}"
                )
            channel_mean_square = activation_importance[activation_group]
        data, scale = _quantize_tensor_rows_chunked(
            tensor,
            scale_factor=tensor_scale_factor,
            scale_mode=tensor_scale_mode,
            channel_mean_square=channel_mean_square,
            activation_mse_blend_alpha=weight_scale_blend_alpha,
        )
        output[name] = data
        output[scale_name(name)] = scale.view(-1, 1)
        checkpoint_prefix = name[: -len(".weight")]
        if static_scales is not None and checkpoint_prefix in static_scales:
            scale_tensor_name = input_scale_name(checkpoint_prefix)
            if scale_tensor_name in state or scale_tensor_name in output:
                raise ValueError(
                    f"source checkpoint already contains {scale_tensor_name}"
                )
            output[scale_tensor_name] = torch.tensor(
                [static_scales[checkpoint_prefix]], dtype=torch.float32
            )
            written_input_scales.add(checkpoint_prefix)
        layers[name] = {
            "shape": list(tensor.shape),
            "stored_dtype": "uint8",
            "encoding": "ascend_hifloat8",
            "scale": "per_output_channel_fp32",
            "activation": (
                "static_per_tensor_fp32"
                if checkpoint_prefix in (static_scales or {})
                else "none" if name in _EMBEDDING_WEIGHTS else "dynamic_per_token"
            ),
            "weight_scale_mode": tensor_scale_mode,
        }
    return output, layers, written_input_scales


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the decoder config for both multimodal and text-only layouts."""
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else config


def expected_target_count(config: dict[str, Any]) -> int:
    """Derive how many linear weights the configured decoder must contribute.

    Every layer contributes three MLP projections. A ``full_attention`` layer
    adds q/k/v/o; a ``linear_attention`` layer adds the three quantized GDN
    projections (``in_proj_a`` and ``in_proj_b`` stay floating point).
    """
    layer_types = _layer_types(config)
    layer_count = len(layer_types)

    attention_projections = sum(
        len(_ATTENTION_LEAVES) if entry == _FULL_ATTENTION else len(_GDN_LEAVES) for entry in layer_types
    )
    # Store the token embedding in the same HiF8 representation as the decoder
    # projections.  Embedding decode touches only selected rows at runtime, so
    # this avoids materialising a full-vocabulary floating-point copy.
    return attention_projections + layer_count * len(_MLP_LEAVES) + 1


def _load_config(input_dir: Path) -> dict[str, Any]:
    config_path = input_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"no config.json in {input_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"invalid config.json: {config_path}")
    return config


def _load_index(input_dir: Path) -> tuple[dict[str, str], dict[str, Any] | None]:
    index_path = input_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid safetensors index: {index_path}")
        return {str(key): str(value) for key, value in weight_map.items()}, index

    single = input_dir / "model.safetensors"
    if single.exists():
        from safetensors import safe_open

        with safe_open(single, framework="pt", device="cpu") as handle:
            return {name: single.name for name in handle.keys()}, None
    raise FileNotFoundError(f"no model.safetensors or model.safetensors.index.json in {input_dir}")


def _layer_types(config: dict[str, Any]) -> tuple[str, ...]:
    text = _text_config(config)
    layer_count = text.get("num_hidden_layers")
    if not isinstance(layer_count, int) or layer_count <= 0:
        raise ValueError(f"config has no usable num_hidden_layers: {layer_count!r}")
    values = text.get("layer_types")
    if values is None:
        interval = text.get("full_attention_interval")
        if isinstance(interval, int) and interval > 0:
            values = [
                _FULL_ATTENTION
                if (index + 1) % interval == 0
                else _LINEAR_ATTENTION
                for index in range(layer_count)
            ]
        else:
            values = [_FULL_ATTENTION] * layer_count
    if not isinstance(values, list) or len(values) != layer_count:
        raise ValueError(
            f"config layer_types must list {layer_count} entries, got {values!r}"
        )
    layer_types = tuple(str(value) for value in values)
    unknown = sorted(set(layer_types) - {_FULL_ATTENTION, _LINEAR_ATTENTION})
    if unknown:
        raise ValueError(f"unsupported layer types in config: {unknown}")
    return layer_types


def _checkpoint_layer_prefix(tensor_names: set[str]) -> str:
    if any(name.startswith("model.language_model.layers.") for name in tensor_names):
        return "model.language_model.layers"
    if any(name.startswith("model.layers.") for name in tensor_names):
        return "model.layers"
    raise ValueError("checkpoint has no recognized Qwen3.5 decoder layer prefix")


def _required_identity_files(
    input_dir: Path,
    shard_names: set[str],
) -> set[str]:
    required = {"config.json", *shard_names}
    if (input_dir / "model.safetensors.index.json").is_file():
        required.add("model.safetensors.index.json")
    return required


def _validate_calibration_for_checkpoint(
    calibration: dict[str, Any],
    *,
    input_dir: Path,
    config: dict[str, Any],
    tensor_names: set[str],
    shard_names: set[str],
) -> dict[str, float]:
    """Validate provenance/completeness and return canonical Linear scales."""

    from .calibration import (
        qwen35_full_attention_layers,
        qwen35_linear_group_specs,
        validate_calibration_artifact,
    )

    validate_calibration_artifact(calibration)
    model_record = calibration.get("model")
    if not isinstance(model_record, dict):
        raise ValueError("calibration artifact has no model provenance")
    artifact_model_type = model_record.get("model_type")
    source_model_type = config.get("model_type")
    if artifact_model_type != source_model_type:
        raise ValueError(
            "calibration model_type does not match source checkpoint: "
            f"{artifact_model_type!r} != {source_model_type!r}"
        )

    identity = model_record.get("identity_sha256")
    if not isinstance(identity, dict):
        raise ValueError("calibration artifact has no model.identity_sha256")
    required_identity = _required_identity_files(input_dir, shard_names)
    missing_identity = sorted(required_identity - set(identity))
    if missing_identity:
        raise ValueError(
            "calibration artifact does not identify every source model file; "
            f"missing hashes for {missing_identity}"
        )
    from .calibration import file_sha256

    for name in sorted(required_identity):
        expected = identity[name]
        actual = file_sha256(input_dir / name)
        if expected != actual:
            raise ValueError(
                f"calibration model hash mismatch for {name}: "
                f"artifact={expected!r}, source={actual!r}"
            )

    checkpoint_prefix = _checkpoint_layer_prefix(tensor_names)
    specs = qwen35_linear_group_specs(
        _layer_types(config), checkpoint_prefix=checkpoint_prefix
    )
    expected_prefixes = {spec.canonical_checkpoint_prefix for spec in specs}
    artifact_groups = calibration["linear_groups"]
    actual_prefixes = set(artifact_groups)
    if actual_prefixes != expected_prefixes:
        raise ValueError(
            "calibration Linear groups do not exactly cover the checkpoint: "
            f"missing={sorted(expected_prefixes - actual_prefixes)[:5]}, "
            f"unexpected={sorted(actual_prefixes - expected_prefixes)[:5]}"
        )
    scales = {
        prefix: float(artifact_groups[prefix]["scale"])
        for prefix in expected_prefixes
    }
    if any(scale <= 0.0 or not math.isfinite(scale) for scale in scales.values()):
        raise ValueError("calibration artifact contains an invalid Linear scale")
    missing_weights = sorted(f"{prefix}.weight" for prefix in scales if f"{prefix}.weight" not in tensor_names)
    if missing_weights:
        raise ValueError(
            f"canonical static scale weights are absent from checkpoint: {missing_weights[:5]}"
        )

    text = _text_config(config)
    num_kv_heads = text.get("num_key_value_heads")
    if not isinstance(num_kv_heads, int) or num_kv_heads <= 0:
        raise ValueError(f"config has invalid num_key_value_heads={num_kv_heads!r}")
    expected_kv_layers = {
        str(index) for index in qwen35_full_attention_layers(_layer_types(config))
    }
    kv_layers = calibration["kv_cache"]["layers"]
    if set(kv_layers) != expected_kv_layers:
        raise ValueError(
            "calibration KV layers do not exactly cover full attention: "
            f"expected={sorted(expected_kv_layers)}, got={sorted(kv_layers)}"
        )
    for layer_index in sorted(expected_kv_layers):
        for kind in ("key", "value"):
            values = kv_layers[layer_index][kind]["scales"]
            if len(values) != num_kv_heads:
                raise ValueError(
                    f"KV layer {layer_index} {kind} has {len(values)} scales, "
                    f"expected {num_kv_heads}"
                )
    return scales


def _load_calibration(
    calibration: str | Path | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if calibration is None:
        return None
    if isinstance(calibration, dict):
        return calibration
    path = Path(calibration)
    if not path.is_file():
        raise FileNotFoundError(f"calibration artifact does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"calibration artifact is not a JSON object: {path}")
    return value


def _prepare_output(input_dir: Path, output_dir: Path) -> None:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("input and output directories must be different")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _copy_model_files(input_dir: Path, output_dir: Path, shard_names: set[str]) -> None:
    for source in input_dir.iterdir():
        if source.name in shard_names or source.name in _COPY_EXCLUDES or source.suffix == ".safetensors":
            continue
        destination = output_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _write_config(config: dict[str, Any], output_dir: Path) -> None:
    """Copy config.json through without inventing a quantization_config.

    vllm-ascend detects a ModelSlim checkpoint by the presence of
    ``quant_model_description.json``, and ``AscendModelSlimConfig`` only reads
    that file while its ``quant_description`` is still empty. Writing a
    ``quantization_config`` section here would make vLLM build the config from
    config.json instead, leaving ``quant_description`` populated with the stub
    and permanently skipping the real per-layer map.
    """
    existing = config.get("quantization_config")
    if existing:
        raise ValueError(f"input checkpoint already has quantization_config={existing!r}")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def quantize_checkpoint(
    input_dir: Path,
    output_dir: Path,
    *,
    calibration: str | Path | dict[str, Any] | None = None,
    weight_scale_factor: float = 1.0,
    weight_scale_scope: str = "all",
    weight_scale_mode: str = "amax",
    weight_activation_calibration: str | Path | dict[str, Any] | None = None,
    weight_scale_blend_alpha: float = 1.0,
    weight_scale_name_regex: str | None = None,
) -> dict[str, Any]:
    """Quantize target Qwen3.5 linear weights while preserving shard names."""
    import torch
    from safetensors.torch import load_file, save_file

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not math.isfinite(weight_scale_factor) or weight_scale_factor <= 0.0:
        raise ValueError("weight_scale_factor must be finite and greater than zero")
    if weight_scale_mode not in {"amax", "mse", "activation_mse"}:
        raise ValueError(
            "weight_scale_mode must be 'amax', 'mse', or 'activation_mse'"
        )
    if weight_scale_mode == "activation_mse" and weight_activation_calibration is None:
        raise ValueError(
            "activation_mse requires weight_activation_calibration"
        )
    if weight_scale_mode != "activation_mse" and weight_activation_calibration is not None:
        raise ValueError(
            "weight_activation_calibration is only valid with activation_mse"
        )
    if not math.isfinite(weight_scale_blend_alpha) or not 0.0 <= weight_scale_blend_alpha <= 1.0:
        raise ValueError("weight_scale_blend_alpha must be finite and in [0,1]")
    if weight_scale_mode != "activation_mse" and weight_scale_blend_alpha != 1.0:
        raise ValueError(
            "weight_scale_blend_alpha is only meaningful with activation_mse"
        )
    try:
        weight_scale_name_pattern = (
            re.compile(weight_scale_name_regex) if weight_scale_name_regex else None
        )
    except re.error as exc:
        raise ValueError(f"invalid weight_scale_name_regex: {exc}") from exc
    _weight_scale_scope_members(weight_scale_scope)
    config = _load_config(input_dir)
    expected_count = expected_target_count(config)
    _prepare_output(input_dir, output_dir)
    weight_map, source_index = _load_index(input_dir)
    shard_names = set(weight_map.values())
    calibration_record = _load_calibration(calibration)
    weight_activation_record = _load_calibration(weight_activation_calibration)
    static_scales = None
    if calibration_record is not None:
        static_scales = _validate_calibration_for_checkpoint(
            calibration_record,
            input_dir=input_dir,
            config=config,
            tensor_names=set(weight_map),
            shard_names=shard_names,
        )
    activation_importance = None
    activation_provenance = None
    if weight_activation_record is not None:
        _validate_calibration_for_checkpoint(
            weight_activation_record,
            input_dir=input_dir,
            config=config,
            tensor_names=set(weight_map),
            shard_names=shard_names,
        )
        activation_importance = {}
        for prefix, entry in weight_activation_record["linear_groups"].items():
            observer = entry.get("observer", {})
            values = observer.get("channel_mean_square")
            if not isinstance(values, list) or not values:
                raise ValueError(
                    "weight activation calibration has no channel_mean_square "
                    f"for {prefix}"
                )
            tensor = torch.tensor(values, dtype=torch.float32)
            if bool((~torch.isfinite(tensor) | (tensor < 0)).any()) or not bool(
                tensor.sum() > 0
            ):
                raise ValueError(
                    f"invalid channel_mean_square values for {prefix}"
                )
            activation_importance[prefix] = tensor
        activation_provenance = {
            "group_count": len(activation_importance),
            "dataset": weight_activation_record.get("dataset"),
            "sampling": weight_activation_record.get("sampling"),
            "search": weight_activation_record.get("search"),
            "runtime_scaling": "fixed_checkpoint_scale",
        }
        if not isinstance(weight_activation_calibration, dict):
            from .calibration import file_sha256

            activation_path = Path(weight_activation_calibration)
            activation_provenance.update(
                {
                    "path": str(activation_path.resolve()),
                    "sha256": file_sha256(activation_path),
                }
            )
    _copy_model_files(input_dir, output_dir, shard_names)
    _write_config(config, output_dir)

    output_weight_map = dict(weight_map)
    all_layers: dict[str, Any] = {}
    written_input_scales: set[str] = set()
    total_size = 0
    for shard_name in sorted(shard_names):
        source = input_dir / shard_name
        state = load_file(source, device="cpu")
        quantized, layers, shard_input_scales = _quantize_shard(
            state,
            static_scales=static_scales,
            weight_scale_factor=weight_scale_factor,
            weight_scale_scope=weight_scale_scope,
            weight_scale_mode=weight_scale_mode,
            activation_importance=activation_importance,
            weight_scale_blend_alpha=weight_scale_blend_alpha,
            weight_scale_name_pattern=weight_scale_name_pattern,
        )
        for name in layers:
            output_weight_map[scale_name(name)] = shard_name
        for checkpoint_prefix in shard_input_scales:
            output_weight_map[input_scale_name(checkpoint_prefix)] = shard_name
        written_input_scales.update(shard_input_scales)
        all_layers.update(layers)
        total_size += sum(_tensor_nbytes(tensor) for tensor in quantized.values())
        save_file(quantized, output_dir / shard_name)
        del state, quantized
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()

    # Qwen3.5-9B contributes 201 component weights: 8 full-attention layers x 4,
    # 24 GDN layers x 3, 32 MLPs x 3, and the token embedding. vLLM packs the
    # decoder components into fused runtime linears while loading.
    if len(all_layers) != expected_count:
        raise ValueError(
            f"expected {expected_count} Qwen3.5 checkpoint linear weights, found {len(all_layers)}; "
            "refusing to emit a partial checkpoint"
        )
    if static_scales is not None and written_input_scales != set(static_scales):
        raise ValueError(
            "failed to write every calibrated Linear input scale: "
            f"missing={sorted(set(static_scales) - written_input_scales)[:5]}, "
            f"unexpected={sorted(written_input_scales - set(static_scales))[:5]}"
        )

    static_activation = calibration_record is not None
    quant_description = build_quant_description(
        tuple(weight_map), static_activation=static_activation
    )
    (output_dir / QUANT_DESCRIPTION_FILENAME).write_text(
        json.dumps(quant_description, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    checkpoint_format = (
        STATIC_QUANT_TYPE if static_activation else DYNAMIC_QUANT_TYPE
    )
    metadata: dict[str, Any] = {
        "format": checkpoint_format,
        "byte_layout": "ascend_hifloat8",
        "activation_quantization": (
            "static_per_linear_input_tensor"
            if static_activation
            else "dynamic_per_token"
        ),
        "weight_quantization": "per_output_channel",
        "weight_scale_factor": float(weight_scale_factor),
        "weight_scale_scope": weight_scale_scope,
        "weight_scale_mode": weight_scale_mode,
        "weight_scale_blend_alpha": float(weight_scale_blend_alpha),
        "weight_scale_name_regex": weight_scale_name_regex,
        "output_dtype": "bfloat16",
        "quantized_layer_count": len(all_layers),
        "layers": all_layers,
    }
    if activation_provenance is not None:
        metadata["weight_activation_calibration"] = activation_provenance
    if calibration_record is not None:
        kv_layers = calibration_record["kv_cache"]["layers"]
        metadata.update(
            {
                "static_activation_scale_count": len(static_scales or {}),
                "embedding_quant_type": DYNAMIC_QUANT_TYPE,
                "linear_quant_type": STATIC_QUANT_TYPE,
                "kv_cache": {
                    "mode": "static_per_layer_per_head",
                    "layers": {
                        layer: {
                            "key_scales": list(entry["key"]["scales"]),
                            "value_scales": list(entry["value"]["scales"]),
                        }
                        for layer, entry in sorted(kv_layers.items())
                    },
                },
                # Preserve the complete calibration protocol, provenance,
                # observations and search metrics in the checkpoint itself.
                "calibration": calibration_record,
            }
        )
    (output_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )

    if source_index is not None:
        output_index = dict(source_index)
        output_index["metadata"] = dict(output_index.get("metadata", {}), total_size=total_size)
        output_index["weight_map"] = output_weight_map
        (output_dir / "model.safetensors.index.json").write_text(
            json.dumps(output_index, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="ascii",
        )
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="source BF16/FP16 Qwen3.5 checkpoint")
    parser.add_argument("output", type=Path, help="new HiFloat8 checkpoint directory")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=(
            "WikiText-2 calibration artifact; when supplied, emit a true "
            "W8A8HIF8_STATIC checkpoint instead of the legacy dynamic format"
        ),
    )
    parser.add_argument(
        "--weight-scale-factor",
        type=float,
        default=1.0,
        help=(
            "multiply the per-output-channel amax row scale by this factor; "
            "values below one trade rare outlier clipping for lower bulk HIF8 "
            "error (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--weight-scale-scope",
        default="all",
        help=(
            "limit --weight-scale-factor to 'all' or a '+'-joined subset of "
            "mlp, self_attn, linear_attn, and embedding"
        ),
    )
    parser.add_argument(
        "--weight-scale-mode",
        choices=("amax", "mse", "activation_mse"),
        default="amax",
        help=(
            "select per-row weight scale construction: 'amax' preserves the "
            "original exporter, 'mse' minimizes weight reconstruction error, "
            "and 'activation_mse' minimizes diagonal expected Linear output "
            "error using a separate calibration artifact"
        ),
    )
    parser.add_argument(
        "--weight-activation-calibration",
        type=Path,
        default=None,
        help=(
            "calibration artifact containing per-channel activation mean "
            "squares; required only by --weight-scale-mode activation_mse"
        ),
    )
    parser.add_argument(
        "--weight-scale-blend-alpha",
        type=float,
        default=1.0,
        help=(
            "geometrically blend activation-MSE row scales with amax: 0 is "
            "amax and 1 is the full searched scale (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--weight-scale-name-regex",
        default=None,
        help=(
            "optional regular expression restricting which in-scope weight "
            "names use the selected scale mode"
        ),
    )
    args = parser.parse_args(argv)
    metadata = quantize_checkpoint(
        args.input,
        args.output,
        calibration=args.calibration,
        weight_scale_factor=args.weight_scale_factor,
        weight_scale_scope=args.weight_scale_scope,
        weight_scale_mode=args.weight_scale_mode,
        weight_activation_calibration=args.weight_activation_calibration,
        weight_scale_blend_alpha=args.weight_scale_blend_alpha,
        weight_scale_name_regex=args.weight_scale_name_regex,
    )
    print(f"quantized {metadata['quantized_layer_count']} linear weights into {args.output}")


if __name__ == "__main__":
    main()
