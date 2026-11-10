"""WikiText-2 static-scale calibration for dense Qwen3.5 models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .calibration import (
    CALIBRATION_FORMAT,
    CALIBRATION_SCHEMA_VERSION,
    DeviceSampleObserver,
    OutputAwareDeviceSampleObserver,
    ScaleSearchConfig,
    file_sha256,
    qwen35_full_attention_layers,
    qwen35_linear_group_specs,
    sample_token_windows,
    search_exact_output_aware_static_scale,
    search_output_aware_static_scale,
    search_static_scale,
    validate_calibration_artifact,
)

if TYPE_CHECKING:
    import torch


WIKITEXT_REPOSITORY = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_SPLIT = "train"
DEFAULT_SAMPLE_COUNT = 128
DEFAULT_SEQUENCE_LENGTH = 2048
DEFAULT_SEED = 42
DEFAULT_OBSERVER_CAPACITY = 65_536
DEFAULT_SAMPLES_PER_FORWARD = 512
DEFAULT_OUTPUT_AWARE_ROWS = 64
DEFAULT_OUTPUT_AWARE_ROWS_PER_FORWARD = 4


def resolve_wikitext2_parquet(
    *,
    hf_home: str | Path | None = None,
    revision: str | None = None,
) -> tuple[Path, str]:
    """Resolve the immutable cached WikiText-2 train parquet without network I/O."""

    cache_root = Path(
        hf_home if hf_home is not None else os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    repository = cache_root / "hub" / "datasets--Salesforce--wikitext"
    if revision is None:
        ref = repository / "refs" / "main"
        if not ref.is_file():
            raise FileNotFoundError(
                f"WikiText-2 cache ref is missing: {ref}; download {WIKITEXT_REPOSITORY}/{WIKITEXT_CONFIG} first"
            )
        revision = ref.read_text(encoding="utf-8").strip()
    parquet = repository / "snapshots" / revision / WIKITEXT_CONFIG / f"{WIKITEXT_SPLIT}-00000-of-00001.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(f"cached WikiText-2 parquet is missing: {parquet}")
    return parquet, revision


def load_wikitext2_train(parquet: str | Path):
    """Load the resolved local parquet without network access."""

    from datasets import load_dataset

    return load_dataset(
        "parquet",
        data_files={WIKITEXT_SPLIT: str(Path(parquet))},
        split=WIKITEXT_SPLIT,
    )


def tokenize_wikitext2(dataset, tokenizer) -> "torch.Tensor":
    """Build the standard GPTQ-style concatenated WikiText-2 token stream."""

    import torch

    texts = dataset["text"]
    try:
        row_count = len(texts)
    except TypeError as error:
        raise ValueError("WikiText-2 text column is not sized") from error
    if row_count == 0:
        raise ValueError("WikiText-2 train split has no text rows")
    # datasets>=4 returns a lazy ``Column`` rather than a concrete list.  It
    # is still a deterministic iterable over the resolved local parquet.
    joined = "\n\n".join(str(text) for text in texts)
    encoded = tokenizer(joined, add_special_tokens=False, return_tensors="pt")
    token_ids = encoded.get("input_ids")
    if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2 or token_ids.shape[0] != 1:
        raise RuntimeError(f"tokenizer returned an unexpected input_ids value: {type(token_ids)!r}")
    return token_ids.reshape(-1).to(device="cpu", dtype=torch.long)


class ObservedDynamicCacheMixin:
    """Mixin whose ``update`` observes RoPE-applied K/V before cache storage."""

    _hif8_key_observers: dict[int, DeviceSampleObserver]
    _hif8_value_observers: dict[int, DeviceSampleObserver]

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        key_observer = self._hif8_key_observers.get(int(layer_idx))
        value_observer = self._hif8_value_observers.get(int(layer_idx))
        if key_observer is not None:
            key_observer.observe(key_states)
        if value_observer is not None:
            value_observer.observe(value_states)
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)


def make_observed_dynamic_cache(
    config,
    key_observers: dict[int, DeviceSampleObserver],
    value_observers: dict[int, DeviceSampleObserver],
):
    """Construct a fresh hybrid Qwen3.5 cache with K/V observers attached."""

    from transformers.cache_utils import DynamicCache

    class ObservedDynamicCache(ObservedDynamicCacheMixin, DynamicCache):
        pass

    cache = ObservedDynamicCache(config=config)
    cache._hif8_key_observers = key_observers
    cache._hif8_value_observers = value_observers
    return cache


def _model_text_config(config) -> Any:
    text_config = getattr(config, "text_config", None)
    return config if text_config is None else text_config


def _config_layer_types(config) -> tuple[str, ...]:
    text = _model_text_config(config)
    layer_types = getattr(text, "layer_types", None)
    if not isinstance(layer_types, list) or not layer_types:
        raise ValueError(f"Qwen3.5 config has no usable layer_types: {layer_types!r}")
    return tuple(str(value) for value in layer_types)


def _config_num_kv_heads(config) -> int:
    value = getattr(_model_text_config(config), "num_key_value_heads", None)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Qwen3.5 config has invalid num_key_value_heads={value!r}")
    return value


def _input_tensor(args: tuple[Any, ...]) -> "torch.Tensor":
    import torch

    if not args or not isinstance(args[0], torch.Tensor):
        raise RuntimeError("calibrated Linear did not receive its activation as the first positional argument")
    return args[0]


def resolve_runtime_layer_prefix(model, layer_count: int) -> str:
    """Resolve the text decoder path for text-only and multimodal HF classes."""

    if not isinstance(layer_count, int) or layer_count <= 0:
        raise ValueError(f"invalid decoder layer count: {layer_count!r}")
    modules = dict(model.named_modules())
    candidates = (
        "model.layers",
        "model.language_model.layers",
        "language_model.layers",
    )
    matches = [
        prefix
        for prefix in candidates
        if prefix in modules
        and all(f"{prefix}.{index}" in modules for index in range(layer_count))
    ]
    if len(matches) != 1:
        raise ValueError(
            "could not resolve one complete Qwen3.5 text decoder module path; "
            f"candidates found: {matches}"
        )
    return matches[0]


def register_linear_observers(
    model,
    specs,
    *,
    capacity: int,
    samples_per_forward: int,
    seed: int,
    output_aware: bool = False,
) -> tuple[
    dict[str, DeviceSampleObserver | OutputAwareDeviceSampleObserver], list[Any]
]:
    """Attach one pre-hook to every independently scaled runtime Linear."""

    modules = dict(model.named_modules())
    observers: dict[
        str, DeviceSampleObserver | OutputAwareDeviceSampleObserver
    ] = {}
    handles = []
    missing = [spec.hook_module for spec in specs if spec.hook_module not in modules]
    if missing:
        raise ValueError(f"model is missing {len(missing)} calibration modules; first entries: {missing[:5]}")
    for index, spec in enumerate(specs):
        observer_type = (
            OutputAwareDeviceSampleObserver if output_aware else DeviceSampleObserver
        )
        observer = observer_type(
            capacity=capacity,
            samples_per_observation=samples_per_forward,
            seed=seed + index * 17,
        )
        observers[spec.canonical_checkpoint_prefix] = observer

        def hook(_module, args, *, current=observer):
            current.observe(_input_tensor(args))

        handles.append(modules[spec.hook_module].register_forward_pre_hook(hook))
    return observers, handles


def make_kv_observers(
    layer_indices: tuple[int, ...],
    *,
    num_kv_heads: int,
    capacity: int,
    samples_per_forward: int,
    seed: int,
) -> tuple[dict[int, DeviceSampleObserver], dict[int, DeviceSampleObserver]]:
    key: dict[int, DeviceSampleObserver] = {}
    value: dict[int, DeviceSampleObserver] = {}
    for ordinal, layer_index in enumerate(layer_indices):
        common = dict(
            capacity=capacity,
            samples_per_observation=samples_per_forward,
            group_count=num_kv_heads,
            group_axis=1,
        )
        key[layer_index] = DeviceSampleObserver(**common, seed=seed + ordinal * 31)
        value[layer_index] = DeviceSampleObserver(**common, seed=seed + ordinal * 31 + 1)
    return key, value


def collect_observations(
    model,
    windows: "torch.Tensor",
    *,
    device: "torch.device",
    model_config,
    key_observers: dict[int, DeviceSampleObserver],
    value_observers: dict[int, DeviceSampleObserver],
    batch_size: int,
) -> None:
    """Run all calibration windows through the unmodified BF16 model."""

    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    total = int(windows.shape[0])
    with torch.inference_mode():
        for begin in range(0, total, batch_size):
            end = min(begin + batch_size, total)
            input_ids = windows[begin:end].to(device=device, non_blocking=False)
            attention_mask = torch.ones_like(input_ids)
            cache = make_observed_dynamic_cache(model_config, key_observers, value_observers)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            del output, cache, attention_mask, input_ids
            print(f"calibration forward {end}/{total}", flush=True)
    if device.type == "npu":
        torch.npu.synchronize(device)


def _search_observer(
    observer: DeviceSampleObserver, config: ScaleSearchConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = observer.cpu_samples()
    results = [search_static_scale(samples[index], config).to_dict() for index in range(samples.shape[0])]
    return results, observer.summary()


def _linear_group_channel_importance(modules, spec) -> "torch.Tensor":
    """Sum downstream squared weights for one fused runtime Linear input."""

    import torch

    importance = None
    for member in spec.members:
        module = modules.get(member)
        weight = getattr(module, "weight", None) if module is not None else None
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError(
                f"output-aware calibration requires a 2-D weight for {member}"
            )
        current = weight.detach().to(torch.float32).square().sum(dim=0)
        if importance is None:
            importance = current
        else:
            if current.shape != importance.shape:
                raise ValueError(
                    f"fused Linear members for {spec.hook_module} have "
                    "different input widths"
                )
            importance.add_(current)
    if importance is None:
        raise AssertionError(f"Linear group {spec.hook_module} has no members")
    return importance.cpu()


def _linear_group_weights(modules, spec) -> tuple["torch.Tensor", ...]:
    """Return all BF16 component weights sharing one fused runtime input."""

    import torch

    weights = []
    for member in spec.members:
        module = modules.get(member)
        weight = getattr(module, "weight", None) if module is not None else None
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError(
                f"exact-output calibration requires a 2-D weight for {member}"
            )
        weights.append(weight.detach())
    return tuple(weights)


def build_calibration_artifact(
    *,
    model_dir: Path,
    model_config,
    tokenizer,
    dataset,
    dataset_parquet: Path,
    dataset_revision: str,
    windows: "torch.Tensor",
    window_starts: tuple[int, ...],
    linear_specs,
    linear_observers: dict[
        str, DeviceSampleObserver | OutputAwareDeviceSampleObserver
    ],
    key_observers: dict[int, DeviceSampleObserver],
    value_observers: dict[int, DeviceSampleObserver],
    search_config: ScaleSearchConfig,
    seed: int,
    device: "torch.device",
    model=None,
    output_aware_linear: bool = False,
    exact_output_linear: bool = False,
    reference_linear_scales: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Search scales and emit the complete reproducibility record."""

    import torch
    import transformers

    modules = dict(model.named_modules()) if output_aware_linear and model is not None else None
    if output_aware_linear and modules is None:
        raise ValueError("output-aware calibration requires the loaded BF16 model")

    linear_groups: dict[str, Any] = {}
    for linear_index, spec in enumerate(linear_specs, start=1):
        observer = linear_observers[spec.canonical_checkpoint_prefix]
        if output_aware_linear:
            if not isinstance(observer, OutputAwareDeviceSampleObserver):
                raise TypeError(
                    "output-aware calibration requires complete-row observers"
                )
            importance = _linear_group_channel_importance(modules, spec)
            diagonal_result = search_output_aware_static_scale(
                observer.cpu_samples(), importance, search_config
            )
            if exact_output_linear:
                if reference_linear_scales is None or spec.canonical_checkpoint_prefix not in reference_linear_scales:
                    raise ValueError(
                        "exact-output calibration requires a reference scale "
                        f"for {spec.canonical_checkpoint_prefix}"
                    )
                exact_result = search_exact_output_aware_static_scale(
                    observer.device_samples(),
                    _linear_group_weights(modules, spec),
                    reference_scale=reference_linear_scales[
                        spec.canonical_checkpoint_prefix
                    ],
                    diagonal_hint_scale=diagonal_result.scale,
                    config=search_config,
                ).to_dict()
                exact_result["diagonal_hint_search_result"] = (
                    diagonal_result.to_dict()
                )
                result = exact_result
            else:
                result = diagonal_result.to_dict()
            results = [result]
            summary = observer.summary()
            print(
                "output-aware scale search "
                f"{linear_index}/{len(linear_specs)} "
                f"{spec.canonical_checkpoint_prefix}: "
                f"scale={result['scale']:.9g}, "
                f"weighted_nmse={result['output_weighted_normalized_mse']:.6g}, "
                f"improvement={result['relative_output_error_improvement']:.3%}",
                flush=True,
            )
        else:
            if not isinstance(observer, DeviceSampleObserver):
                raise TypeError("input-MSE calibration requires scalar observers")
            results, summary = _search_observer(observer, search_config)
        if len(results) != 1:
            raise AssertionError("a Linear observer must produce exactly one scale")
        linear_groups[spec.canonical_checkpoint_prefix] = {
            "hook_module": spec.hook_module,
            "shared_runtime_members": list(spec.members),
            "checkpoint_input_scale": f"{spec.canonical_checkpoint_prefix}.input_scale",
            "scale": results[0]["scale"],
            "search_result": results[0],
            "observer": summary,
        }

    kv_layers: dict[str, Any] = {}
    for layer_index in sorted(key_observers):
        key_results, key_summary = _search_observer(key_observers[layer_index], search_config)
        value_results, value_summary = _search_observer(value_observers[layer_index], search_config)
        kv_layers[str(layer_index)] = {
            "num_kv_heads": len(key_results),
            "key": {
                "scales": [result["scale"] for result in key_results],
                "per_head_search_results": key_results,
                "observer": key_summary,
            },
            "value": {
                "scales": [result["scale"] for result in value_results],
                "per_head_search_results": value_results,
                "observer": value_summary,
            },
        }

    tokenizer_files = [
        name
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
        if (model_dir / name).is_file()
    ]
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid safetensors index: {index_path}")
        weight_files = sorted({str(value) for value in weight_map.values()})
        model_identity_files = [
            "config.json",
            "model.safetensors.index.json",
            *weight_files,
        ]
    elif (model_dir / "model.safetensors").is_file():
        model_identity_files = ["config.json", "model.safetensors"]
    else:
        raise FileNotFoundError(f"no safetensors checkpoint in {model_dir}")
    window_bytes = windows.to(torch.int64).contiguous().numpy().tobytes()
    search_record = search_config.to_dict()
    search_record.update(
        {
            "linear_objective": (
                "exact_fused_linear_output_mse"
                if exact_output_linear
                else (
                    "diagonal_linear_output_mse"
                    if output_aware_linear
                    else search_config.objective
                )
            ),
            "kv_objective": search_config.objective,
            "runtime_scaling": "fixed_checkpoint_scale",
        }
    )
    artifact = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "format": CALIBRATION_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_dir.resolve()),
            "model_type": getattr(model_config, "model_type", None),
            "torch_dtype": str(getattr(model_config, "torch_dtype", "bfloat16")),
            "identity_sha256": {name: file_sha256(model_dir / name) for name in model_identity_files},
            "tokenizer_sha256": {name: file_sha256(model_dir / name) for name in tokenizer_files},
            "tokenizer_class": type(tokenizer).__name__,
        },
        "dataset": {
            "repository": WIKITEXT_REPOSITORY,
            "config": WIKITEXT_CONFIG,
            "split": WIKITEXT_SPLIT,
            "snapshot_revision": dataset_revision,
            "parquet_path": str(dataset_parquet.resolve()),
            "parquet_sha256": file_sha256(dataset_parquet),
            "loader_fingerprint": getattr(dataset, "_fingerprint", None),
            "row_count": int(dataset.num_rows),
            "text_joiner": "\\n\\n",
            "add_special_tokens": False,
            # Filled by the caller, which owns the complete concatenated token
            # stream in addition to these sampled windows.
            "token_count": 0,
        },
        "sampling": {
            "method": "unique_uniform_windows_from_concatenated_token_stream",
            "seed": seed,
            "sample_count": int(windows.shape[0]),
            "sequence_length": int(windows.shape[1]),
            "window_starts": list(window_starts),
            "windows_sha256_int64_le": hashlib.sha256(window_bytes).hexdigest(),
        },
        "search": search_record,
        "collection": {
            "device": str(device),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "observer_capacity": next(iter(linear_observers.values())).capacity,
            "samples_per_forward": next(iter(linear_observers.values())).samples_per_observation,
        },
        "linear_groups": linear_groups,
        "kv_cache": {
            "quantized_scope": "full_attention_only",
            "scale_granularity": "static_per_layer_per_kv_head_separate_k_v",
            "linear_attention_recurrent_state": "floating_point",
            "layers": kv_layers,
        },
    }
    return artifact


def _set_dataset_token_count(artifact: dict[str, Any], token_count: int) -> None:
    # Hash the full token stream separately from the sampled windows.
    artifact["dataset"]["token_count"] = int(token_count)


def calibrate_qwen35(
    model_dir: str | Path,
    output_path: str | Path,
    *,
    dataset_parquet: str | Path | None = None,
    dataset_revision: str | None = None,
    hf_home: str | Path | None = None,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    seed: int = DEFAULT_SEED,
    batch_size: int = 1,
    observer_capacity: int = DEFAULT_OBSERVER_CAPACITY,
    samples_per_forward: int = DEFAULT_SAMPLES_PER_FORWARD,
    search_config: ScaleSearchConfig | None = None,
    device: str = "npu:0",
    output_aware_linear: bool = False,
    exact_output_linear: bool = False,
    reference_linear_calibration: str | Path | None = None,
    output_aware_rows: int = DEFAULT_OUTPUT_AWARE_ROWS,
    output_aware_rows_per_forward: int = DEFAULT_OUTPUT_AWARE_ROWS_PER_FORWARD,
) -> dict[str, Any]:
    """Run WikiText-2 calibration and atomically write its JSON artifact."""

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(model_dir)
    output_path = Path(output_path)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite calibration artifact: {output_path}")
    if exact_output_linear:
        output_aware_linear = True
    reference_record = None
    reference_linear_scales = None
    if reference_linear_calibration is not None:
        reference_path = Path(reference_linear_calibration)
        reference_record = json.loads(reference_path.read_text(encoding="utf-8"))
        validate_calibration_artifact(reference_record)
        reference_linear_scales = {
            name: float(entry["scale"])
            for name, entry in reference_record["linear_groups"].items()
        }
    if exact_output_linear and reference_linear_scales is None:
        raise ValueError(
            "exact-output calibration requires --reference-linear-calibration"
        )
    if dataset_parquet is None:
        resolved_parquet, resolved_revision = resolve_wikitext2_parquet(
            hf_home=hf_home,
            revision=dataset_revision,
        )
    else:
        resolved_parquet = Path(dataset_parquet)
        if not resolved_parquet.is_file():
            raise FileNotFoundError(f"dataset parquet does not exist: {resolved_parquet}")
        resolved_revision = "local" if dataset_revision is None else dataset_revision

    dataset = load_wikitext2_train(resolved_parquet)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    token_stream = tokenize_wikitext2(dataset, tokenizer)
    windows, starts = sample_token_windows(
        token_stream,
        sample_count=sample_count,
        sequence_length=sequence_length,
        seed=seed,
    )

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    layer_types = _config_layer_types(config)
    full_attention_layers = qwen35_full_attention_layers(layer_types)
    # Register PrivateUse1 as ``npu`` before parsing the device string when
    # backend autoload is deliberately disabled by the serving environment.
    if str(device).split(":", 1)[0] == "npu":
        import torch_npu  # noqa: F401

    target_device = torch.device(device)
    if target_device.type == "npu":

        if not torch.npu.is_available():
            raise RuntimeError("torch-npu reports no available NPU")
        torch.npu.set_device(target_device)

    print(f"loading BF16 calibration model from {model_dir} onto {target_device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="eager",
    ).to(target_device)
    runtime_prefix = resolve_runtime_layer_prefix(model, len(layer_types))
    checkpoint_prefix = (
        "model.language_model.layers"
        if any(
            name.startswith("model.language_model.layers.")
            for name in json.loads(
                (model_dir / "model.safetensors.index.json").read_text(
                    encoding="utf-8"
                )
            ).get("weight_map", {})
        )
        else "model.layers"
    )
    linear_specs = qwen35_linear_group_specs(
        layer_types,
        runtime_prefix=runtime_prefix,
        checkpoint_prefix=checkpoint_prefix,
    )
    if len(linear_specs) != len(layer_types) * 4:
        raise AssertionError(
            "Qwen3.5 must contribute four independently scaled runtime Linears per layer"
        )
    linear_observers, hook_handles = register_linear_observers(
        model,
        linear_specs,
        capacity=(output_aware_rows if output_aware_linear else observer_capacity),
        samples_per_forward=(
            output_aware_rows_per_forward
            if output_aware_linear
            else samples_per_forward
        ),
        seed=seed,
        output_aware=output_aware_linear,
    )
    key_observers, value_observers = make_kv_observers(
        full_attention_layers,
        num_kv_heads=_config_num_kv_heads(config),
        capacity=observer_capacity,
        samples_per_forward=samples_per_forward,
        seed=seed,
    )
    try:
        collect_observations(
            model,
            windows,
            device=target_device,
            model_config=config,
            key_observers=key_observers,
            value_observers=value_observers,
            batch_size=batch_size,
        )
    finally:
        for handle in hook_handles:
            handle.remove()

    search_config = ScaleSearchConfig() if search_config is None else search_config
    artifact = build_calibration_artifact(
        model_dir=model_dir,
        model_config=config,
        tokenizer=tokenizer,
        dataset=dataset,
        dataset_parquet=resolved_parquet,
        dataset_revision=resolved_revision,
        windows=windows,
        window_starts=starts,
        linear_specs=linear_specs,
        linear_observers=linear_observers,
        key_observers=key_observers,
        value_observers=value_observers,
        search_config=search_config,
        seed=seed,
        device=target_device,
        model=model,
        output_aware_linear=output_aware_linear,
        exact_output_linear=exact_output_linear,
        reference_linear_scales=reference_linear_scales,
    )
    if reference_record is not None and artifact["model"][
        "identity_sha256"
    ] != reference_record.get("model", {}).get("identity_sha256"):
        raise ValueError("reference calibration model identity differs")
    _set_dataset_token_count(artifact, token_stream.numel())
    validate_calibration_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(output_path)
    print(f"wrote static calibration artifact to {output_path}", flush=True)
    return artifact


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="local BF16 Qwen3.5 checkpoint")
    parser.add_argument("output", type=Path, help="new calibration artifact JSON")
    parser.add_argument("--hf-home", type=Path, default=None)
    parser.add_argument("--dataset-parquet", type=Path, default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--observer-capacity", type=int, default=DEFAULT_OBSERVER_CAPACITY)
    parser.add_argument("--samples-per-forward", type=int, default=DEFAULT_SAMPLES_PER_FORWARD)
    parser.add_argument(
        "--linear-objective",
        choices=("input_mse", "output_mse", "exact_output_mse"),
        default="input_mse",
        help=(
            "select ordinary input reconstruction or a fixed-scale diagonal "
            "Linear-output error objective"
        ),
    )
    parser.add_argument(
        "--output-aware-rows", type=int, default=DEFAULT_OUTPUT_AWARE_ROWS
    )
    parser.add_argument(
        "--output-aware-rows-per-forward",
        type=int,
        default=DEFAULT_OUTPUT_AWARE_ROWS_PER_FORWARD,
    )
    parser.add_argument("--reference-linear-calibration", type=Path, default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--objective",
        choices=("mse", "cosine", "mse_cosine"),
        default="mse",
    )
    args = parser.parse_args(argv)
    calibrate_qwen35(
        args.model,
        args.output,
        dataset_parquet=args.dataset_parquet,
        dataset_revision=args.dataset_revision,
        hf_home=args.hf_home,
        sample_count=args.samples,
        sequence_length=args.sequence_length,
        seed=args.seed,
        batch_size=args.batch_size,
        observer_capacity=args.observer_capacity,
        samples_per_forward=args.samples_per_forward,
        search_config=ScaleSearchConfig(objective=args.objective),
        device=args.device,
        output_aware_linear=args.linear_objective != "input_mse",
        exact_output_linear=args.linear_objective == "exact_output_mse",
        reference_linear_calibration=args.reference_linear_calibration,
        output_aware_rows=args.output_aware_rows,
        output_aware_rows_per_forward=args.output_aware_rows_per_forward,
    )


if __name__ == "__main__":
    main()
