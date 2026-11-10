"""Static-scale calibration primitives for Ascend HiFloat8.

The runtime format is not an integer quantizer.  In particular, dividing an
``amax`` by 127 is not meaningful for HiFloat8: its finite values form a
non-uniform floating-point codebook.  This module therefore evaluates candidate
scales against the exact codebook used by :mod:`vllm_ascend_hif8.codec`.

The public functions in this file are intentionally device agnostic.  The
calibration command can collect samples on an NPU, while the scale search and
unit tests run on CPU using exactly the same numerical contract.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

from .codec import ASCEND_DECODE_TABLE, DEFAULT_SCALE_EPS, MAX_FINITE, POSITIVE_VALUES

if TYPE_CHECKING:
    import torch


CALIBRATION_FORMAT = "W8A8HIF8_STATIC_CALIBRATION"
CALIBRATION_SCHEMA_VERSION = 1
STATIC_QUANT_TYPE = "W8A8HIF8_STATIC"


@dataclass(frozen=True)
class LinearGroupSpec:
    """One runtime Linear input and the checkpoint components sharing it."""

    hook_module: str
    members: tuple[str, ...]
    canonical_checkpoint_prefix: str


def qwen35_linear_group_specs(
    layer_types: Sequence[str],
    *,
    runtime_prefix: str = "model.layers",
    checkpoint_prefix: str = "model.language_model.layers",
) -> tuple[LinearGroupSpec, ...]:
    """Return all independently calibrated Qwen3.5 Linear input groups.

    Qwen3.5 fuses q/k/v, gate/up and GDN qkv/z while loading into vLLM.  Every
    member of such a fused projection consumes the same source tensor, so one
    canonical FP32 scale is both numerically correct and compatible with the
    upstream ``PerTensorScaleParameter`` loader.
    """

    specs: list[LinearGroupSpec] = []
    for layer_index, layer_type in enumerate(layer_types):
        runtime = f"{runtime_prefix}.{layer_index}"
        checkpoint = f"{checkpoint_prefix}.{layer_index}"
        if layer_type == "full_attention":
            specs.extend(
                (
                    LinearGroupSpec(
                        hook_module=f"{runtime}.self_attn.q_proj",
                        members=(
                            f"{runtime}.self_attn.q_proj",
                            f"{runtime}.self_attn.k_proj",
                            f"{runtime}.self_attn.v_proj",
                        ),
                        canonical_checkpoint_prefix=f"{checkpoint}.self_attn.q_proj",
                    ),
                    LinearGroupSpec(
                        hook_module=f"{runtime}.self_attn.o_proj",
                        members=(f"{runtime}.self_attn.o_proj",),
                        canonical_checkpoint_prefix=f"{checkpoint}.self_attn.o_proj",
                    ),
                )
            )
        elif layer_type == "linear_attention":
            specs.extend(
                (
                    LinearGroupSpec(
                        hook_module=f"{runtime}.linear_attn.in_proj_qkv",
                        members=(
                            f"{runtime}.linear_attn.in_proj_qkv",
                            f"{runtime}.linear_attn.in_proj_z",
                        ),
                        canonical_checkpoint_prefix=f"{checkpoint}.linear_attn.in_proj_qkv",
                    ),
                    LinearGroupSpec(
                        hook_module=f"{runtime}.linear_attn.out_proj",
                        members=(f"{runtime}.linear_attn.out_proj",),
                        canonical_checkpoint_prefix=f"{checkpoint}.linear_attn.out_proj",
                    ),
                )
            )
        else:
            raise ValueError(f"unsupported Qwen3.5 layer type at index {layer_index}: {layer_type!r}")

        specs.extend(
            (
                LinearGroupSpec(
                    hook_module=f"{runtime}.mlp.gate_proj",
                    members=(
                        f"{runtime}.mlp.gate_proj",
                        f"{runtime}.mlp.up_proj",
                    ),
                    canonical_checkpoint_prefix=f"{checkpoint}.mlp.gate_proj",
                ),
                LinearGroupSpec(
                    hook_module=f"{runtime}.mlp.down_proj",
                    members=(f"{runtime}.mlp.down_proj",),
                    canonical_checkpoint_prefix=f"{checkpoint}.mlp.down_proj",
                ),
            )
        )
    return tuple(specs)


def qwen35_full_attention_layers(layer_types: Sequence[str]) -> tuple[int, ...]:
    """Return the layer indices whose RoPE-applied K/V must be calibrated."""

    unknown = sorted(set(layer_types) - {"full_attention", "linear_attention"})
    if unknown:
        raise ValueError(f"unsupported Qwen3.5 layer types: {unknown}")
    return tuple(index for index, layer_type in enumerate(layer_types) if layer_type == "full_attention")


def file_sha256(path: str | Path) -> str:
    """Hash a local reproducibility input without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def saturating_encode_tensor(x: "torch.Tensor") -> "torch.Tensor":
    """Encode with finite overflow saturated to the largest finite HIF8 code.

    The base codec correctly maps values above the format overflow midpoint to
    infinity.  That behavior is desirable for genuine ``+/-Inf`` inputs, but a
    calibrated quantizer must clip ordinary finite outliers instead of turning
    them into non-finite model activations.
    """

    import torch

    values = x.to(torch.float32)
    clipped = torch.where(
        torch.isfinite(values),
        values.clamp(min=-MAX_FINITE, max=MAX_FINITE),
        values,
    )
    from .torch_codec import encode_tensor

    return encode_tensor(clipped)


def _scale_view(x: "torch.Tensor", scale: "torch.Tensor", group_axis: int | None) -> "torch.Tensor":
    import torch

    if not isinstance(scale, torch.Tensor):
        raise TypeError(f"scale must be a torch.Tensor, got {type(scale)!r}")
    fixed = scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if fixed.numel() == 0 or bool((~torch.isfinite(fixed) | (fixed <= 0)).any()):
        raise ValueError("all static HiFloat8 scales must be finite and greater than zero")
    if group_axis is None:
        if fixed.numel() != 1:
            raise ValueError(f"per-tensor static quantization requires one scale, got {fixed.numel()}")
        return fixed.reshape((1,) * x.ndim)

    axis = group_axis % x.ndim
    if fixed.numel() != x.shape[axis]:
        raise ValueError(f"scale count {fixed.numel()} does not match dimension {axis} of shape {tuple(x.shape)}")
    shape = [1] * x.ndim
    shape[axis] = fixed.numel()
    return fixed.reshape(shape)


def static_quantize_tensor(
    x: "torch.Tensor",
    scale: "torch.Tensor",
    *,
    group_axis: int | None = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Quantize using a fixed per-tensor or per-group FP32 scale.

    Returns physical HiFloat8 bytes and the canonical (unexpanded) FP32 scale.
    Runtime kernels may expand that scale to their row-sidecar layout without
    changing its value.
    """

    import torch

    if not isinstance(x, torch.Tensor) or x.ndim == 0:
        raise TypeError("x must be a tensor with at least one dimension")
    divisor = _scale_view(x, scale, group_axis)
    fixed = scale.to(device=x.device, dtype=torch.float32).reshape(-1).contiguous()
    return saturating_encode_tensor(x.to(torch.float32) / divisor), fixed


def static_fake_quantize(
    x: "torch.Tensor",
    scale: "torch.Tensor",
    *,
    group_axis: int | None = None,
) -> "torch.Tensor":
    """Apply the exact static HIF8 quantize/dequantize contract."""

    import torch

    data, fixed = static_quantize_tensor(x, scale, group_axis=group_axis)
    lut = torch.tensor(ASCEND_DECODE_TABLE, dtype=torch.float32, device=x.device)
    decoded = lut[data.to(torch.long)]
    return decoded * _scale_view(x, fixed, group_axis)


@dataclass(frozen=True)
class ScaleSearchConfig:
    """Candidate search settings persisted in the calibration artifact."""

    percentiles: tuple[float, ...] = (0.99, 0.995, 0.999, 0.9995, 0.9999, 1.0)
    normalized_targets: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
    refinement_steps: int = 17
    refinement_octaves: float = 0.5
    candidate_batch_size: int = 16
    objective: Literal["mse", "cosine", "mse_cosine"] = "mse"
    eps: float = DEFAULT_SCALE_EPS

    def __post_init__(self) -> None:
        if not self.percentiles or any(not 0.0 < value <= 1.0 for value in self.percentiles):
            raise ValueError("percentiles must be in (0, 1]")
        if not self.normalized_targets or any(
            value <= 0.0 or not math.isfinite(value) for value in self.normalized_targets
        ):
            raise ValueError("normalized_targets must be finite and greater than zero")
        if self.refinement_steps < 1:
            raise ValueError("refinement_steps must be positive")
        if self.refinement_octaves < 0.0 or not math.isfinite(self.refinement_octaves):
            raise ValueError("refinement_octaves must be finite and non-negative")
        if self.candidate_batch_size < 1:
            raise ValueError("candidate_batch_size must be positive")
        if self.objective not in {"mse", "cosine", "mse_cosine"}:
            raise ValueError(f"unsupported scale-search objective: {self.objective!r}")
        if self.eps <= 0.0 or not math.isfinite(self.eps):
            raise ValueError("eps must be finite and greater than zero")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScaleSearchResult:
    scale: float
    mse: float
    normalized_mse: float
    cosine_similarity: float
    clipping_rate: float
    max_abs: float
    sample_count: int
    nonfinite_count: int
    candidate_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutputAwareScaleSearchResult:
    """Fixed-scale result selected by a diagonal Linear-output objective.

    ``channel_importance`` is the sum of squared downstream Linear weights for
    each input channel.  Weighting activation reconstruction error by that
    vector is the diagonal of the exact output-error quadratic form.  It keeps
    calibration bounded while distinguishing channels that have very
    different effects on the Linear output.
    """

    scale: float
    mse: float
    normalized_mse: float
    cosine_similarity: float
    output_weighted_mse: float
    output_weighted_normalized_mse: float
    baseline_scale: float
    baseline_output_weighted_normalized_mse: float
    relative_output_error_improvement: float
    clipping_rate: float
    max_abs: float
    sample_count: int
    nonfinite_count: int
    candidate_count: int
    channel_count: int
    channel_importance_min: float
    channel_importance_max: float
    channel_importance_mean: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExactOutputScaleSearchResult:
    """Fixed activation scale selected from exact fused-Linear outputs."""

    scale: float
    output_weighted_mse: float
    output_weighted_normalized_mse: float
    reference_scale: float
    reference_output_weighted_normalized_mse: float
    relative_output_error_improvement: float
    diagonal_hint_scale: float
    clipping_rate: float
    max_abs: float
    sample_count: int
    nonfinite_count: int
    candidate_count: int
    channel_count: int
    output_channel_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_scales(samples: "torch.Tensor", config: ScaleSearchConfig) -> "torch.Tensor":
    import torch

    magnitudes = samples.abs()
    nonzero = magnitudes[magnitudes > 0]
    if nonzero.numel() == 0:
        return torch.tensor([config.eps], dtype=torch.float64)
    quantiles = torch.tensor(config.percentiles, dtype=torch.float64)
    thresholds = torch.quantile(nonzero.to(torch.float64), quantiles)
    targets = torch.tensor(config.normalized_targets, dtype=torch.float64)
    candidates = (thresholds[:, None] / targets[None, :]).reshape(-1)
    candidates = candidates[torch.isfinite(candidates) & (candidates > 0)].clamp_min(config.eps)
    return torch.unique(candidates, sorted=True)


def _evaluate_candidates(
    samples: "torch.Tensor",
    candidates: "torch.Tensor",
    config: ScaleSearchConfig,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Return MSE, normalized MSE and cosine similarity per candidate."""

    import torch

    values = samples.to(torch.float32).reshape(1, -1)
    signal_power = values.square().mean().clamp_min(torch.finfo(torch.float32).tiny)
    codebook = torch.tensor(POSITIVE_VALUES, dtype=torch.float32)
    all_mse: list[torch.Tensor] = []
    all_nmse: list[torch.Tensor] = []
    all_cosine: list[torch.Tensor] = []
    for begin in range(0, candidates.numel(), config.candidate_batch_size):
        scale = candidates[begin : begin + config.candidate_batch_size].to(torch.float32).reshape(-1, 1)
        normalized = values / scale
        magnitude = normalized.abs().clamp(max=MAX_FINITE).contiguous()
        upper = torch.searchsorted(codebook, magnitude)
        upper = upper.clamp(max=codebook.numel() - 1)
        lower = (upper - 1).clamp_min(0)
        lower_value = codebook[lower]
        upper_value = codebook[upper]
        # The physical format rounds midpoint ties away from zero.
        choose_upper = (upper_value - magnitude) <= (magnitude - lower_value)
        quantized_magnitude = torch.where(choose_upper, upper_value, lower_value)
        reconstructed = torch.copysign(quantized_magnitude, normalized) * scale
        error = reconstructed - values
        mse = error.square().mean(dim=1)
        dot = (reconstructed * values).sum(dim=1)
        denominator = torch.sqrt(reconstructed.square().sum(dim=1) * values.square().sum()).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        cosine = dot / denominator
        all_mse.append(mse)
        all_nmse.append(mse / signal_power)
        all_cosine.append(cosine)
    return torch.cat(all_mse), torch.cat(all_nmse), torch.cat(all_cosine)


def _evaluate_output_aware_candidates(
    samples: "torch.Tensor",
    valid: "torch.Tensor",
    channel_importance: "torch.Tensor",
    candidates: "torch.Tensor",
    config: ScaleSearchConfig,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Return diagonal Linear-output MSE and normalized MSE per scale."""

    import torch

    values = samples.to(torch.float32)
    validity = valid.to(torch.float32)
    importance = channel_importance.to(torch.float32).reshape(1, -1)
    effective_importance = validity * importance
    denominator = effective_importance.sum().clamp_min(
        torch.finfo(torch.float32).tiny
    )
    signal_power = (values.square() * effective_importance).sum().clamp_min(
        torch.finfo(torch.float32).tiny
    )
    codebook = torch.tensor(POSITIVE_VALUES, dtype=torch.float32)
    all_mse: list[torch.Tensor] = []
    all_nmse: list[torch.Tensor] = []
    for begin in range(0, candidates.numel(), config.candidate_batch_size):
        scale = candidates[begin : begin + config.candidate_batch_size].to(
            torch.float32
        ).reshape(-1, 1, 1)
        normalized = values.unsqueeze(0) / scale
        magnitude = normalized.abs().clamp(max=MAX_FINITE).contiguous()
        upper = torch.searchsorted(codebook, magnitude)
        upper = upper.clamp(max=codebook.numel() - 1)
        lower = (upper - 1).clamp_min(0)
        lower_value = codebook[lower]
        upper_value = codebook[upper]
        choose_upper = (upper_value - magnitude) <= (magnitude - lower_value)
        quantized_magnitude = torch.where(choose_upper, upper_value, lower_value)
        reconstructed = torch.copysign(quantized_magnitude, normalized) * scale
        error = reconstructed - values.unsqueeze(0)
        weighted_sum = (
            error.square() * effective_importance.unsqueeze(0)
        ).sum(dim=(1, 2))
        all_mse.append(weighted_sum / denominator)
        all_nmse.append(weighted_sum / signal_power)
    return torch.cat(all_mse), torch.cat(all_nmse)


def _best_candidate_index(
    mse: "torch.Tensor",
    normalized_mse: "torch.Tensor",
    cosine: "torch.Tensor",
    objective: str,
) -> int:
    if objective == "mse":
        score = mse
    elif objective == "cosine":
        score = 1.0 - cosine
    else:
        score = normalized_mse + (1.0 - cosine)
    return int(score.argmin().item())


def search_static_scale(
    samples: "torch.Tensor",
    config: ScaleSearchConfig | None = None,
) -> ScaleSearchResult:
    """Find a fixed scale by evaluating the real finite HIF8 codebook.

    Non-finite source values are excluded from scale selection and counted in
    the result.  Runtime static quantization still preserves their dedicated
    NaN/Inf codes.  Candidate selection is followed by a local logarithmic
    refinement so the result is not restricted to the initial percentile grid.
    """

    import torch

    config = ScaleSearchConfig() if config is None else config
    flattened = torch.as_tensor(samples, dtype=torch.float32, device="cpu").reshape(-1)
    finite_mask = torch.isfinite(flattened)
    nonfinite_count = int((~finite_mask).sum().item())
    finite = flattened[finite_mask]
    if finite.numel() == 0:
        return ScaleSearchResult(
            scale=float(config.eps),
            mse=0.0,
            normalized_mse=0.0,
            cosine_similarity=1.0,
            clipping_rate=0.0,
            max_abs=0.0,
            sample_count=0,
            nonfinite_count=nonfinite_count,
            candidate_count=1,
        )

    initial = _candidate_scales(finite, config)
    mse, nmse, cosine = _evaluate_candidates(finite, initial, config)
    initial_best = _best_candidate_index(mse, nmse, cosine, config.objective)
    best_scale = initial[initial_best]

    if config.refinement_steps > 1 and config.refinement_octaves > 0.0:
        offsets = torch.linspace(
            -config.refinement_octaves,
            config.refinement_octaves,
            config.refinement_steps,
            dtype=torch.float64,
        )
        refinement = best_scale * torch.pow(2.0, offsets)
        candidates = torch.unique(torch.cat((initial, refinement.clamp_min(config.eps))), sorted=True)
        mse, nmse, cosine = _evaluate_candidates(finite, candidates, config)
    else:
        candidates = initial
    best = _best_candidate_index(mse, nmse, cosine, config.objective)
    # Checkpoint/runtime contract is FP32 even though candidate generation uses
    # float64 to avoid accidental de-duplication around close percentiles.
    selected = float(candidates[best].to(torch.float32).item())
    clipping = float((finite.abs() > selected * MAX_FINITE).to(torch.float32).mean().item())
    return ScaleSearchResult(
        scale=selected,
        mse=float(mse[best].item()),
        normalized_mse=float(nmse[best].item()),
        cosine_similarity=float(cosine[best].item()),
        clipping_rate=clipping,
        max_abs=float(finite.abs().max().item()),
        sample_count=int(finite.numel()),
        nonfinite_count=nonfinite_count,
        candidate_count=int(candidates.numel()),
    )


def search_output_aware_static_scale(
    samples: "torch.Tensor",
    channel_importance: "torch.Tensor",
    config: ScaleSearchConfig | None = None,
) -> OutputAwareScaleSearchResult:
    """Search one fixed activation scale using downstream output sensitivity.

    The exact Linear output error is ``trace(E W.T W E.T)`` for activation
    reconstruction error ``E``.  Materialising the full input-channel Hessian
    for every Qwen3.5 projection is unnecessarily large, so this search uses
    its diagonal, ``sum_output(W ** 2)``.  The selected scale is still a single
    FP32 checkpoint value and does not introduce any runtime adaptation.
    """

    import torch

    config = ScaleSearchConfig() if config is None else config
    matrix = torch.as_tensor(samples, dtype=torch.float32, device="cpu")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(
            "output-aware samples must be a non-empty [rows, channels] tensor"
        )
    importance = torch.as_tensor(
        channel_importance, dtype=torch.float32, device="cpu"
    ).reshape(-1)
    if importance.numel() != matrix.shape[1]:
        raise ValueError(
            f"channel importance has {importance.numel()} values for "
            f"{matrix.shape[1]} activation channels"
        )
    if bool((~torch.isfinite(importance) | (importance < 0)).any()) or not bool(
        (importance > 0).any()
    ):
        raise ValueError(
            "channel importance must be finite, non-negative, and not all zero"
        )

    valid = torch.isfinite(matrix)
    nonfinite_count = int((~valid).sum().item())
    safe = torch.where(valid, matrix, torch.zeros_like(matrix))
    finite = matrix[valid]
    if finite.numel() == 0:
        raise ValueError("output-aware samples contain no finite values")

    # Include the input-MSE optimum even when output-aware refinement moves away.
    baseline = search_static_scale(finite, config)
    initial = _candidate_scales(finite, config)
    output_mse, output_nmse = _evaluate_output_aware_candidates(
        safe, valid, importance, initial, config
    )
    initial_best = int(output_nmse.argmin().item())

    candidate_parts = [
        initial,
        torch.tensor([baseline.scale], dtype=torch.float64),
    ]
    if config.refinement_steps > 1 and config.refinement_octaves > 0.0:
        offsets = torch.linspace(
            -config.refinement_octaves,
            config.refinement_octaves,
            config.refinement_steps,
            dtype=torch.float64,
        )
        multipliers = torch.pow(2.0, offsets)
        candidate_parts.extend(
            (
                initial[initial_best] * multipliers,
                torch.tensor(baseline.scale, dtype=torch.float64) * multipliers,
            )
        )
    candidates = torch.unique(
        torch.cat(candidate_parts).clamp_min(config.eps), sorted=True
    )
    output_mse, output_nmse = _evaluate_output_aware_candidates(
        safe, valid, importance, candidates, config
    )
    best = int(output_nmse.argmin().item())
    selected = float(candidates[best].to(torch.float32).item())

    mse, nmse, cosine = _evaluate_candidates(finite, candidates, config)
    baseline_candidate = torch.tensor([baseline.scale], dtype=torch.float64)
    _, baseline_output_nmse = _evaluate_output_aware_candidates(
        safe, valid, importance, baseline_candidate, config
    )
    baseline_nmse = float(baseline_output_nmse[0].item())
    selected_nmse = float(output_nmse[best].item())
    relative_improvement = (
        0.0
        if baseline_nmse <= 0.0
        else (baseline_nmse - selected_nmse) / baseline_nmse
    )
    clipping = float(
        (finite.abs() > selected * MAX_FINITE).to(torch.float32).mean().item()
    )
    return OutputAwareScaleSearchResult(
        scale=selected,
        mse=float(mse[best].item()),
        normalized_mse=float(nmse[best].item()),
        cosine_similarity=float(cosine[best].item()),
        output_weighted_mse=float(output_mse[best].item()),
        output_weighted_normalized_mse=selected_nmse,
        baseline_scale=float(baseline.scale),
        baseline_output_weighted_normalized_mse=baseline_nmse,
        relative_output_error_improvement=float(relative_improvement),
        clipping_rate=clipping,
        max_abs=float(finite.abs().max().item()),
        sample_count=int(finite.numel()),
        nonfinite_count=nonfinite_count,
        candidate_count=int(candidates.numel()),
        channel_count=int(matrix.shape[1]),
        channel_importance_min=float(importance.min().item()),
        channel_importance_max=float(importance.max().item()),
        channel_importance_mean=float(importance.mean().item()),
    )


def search_exact_output_aware_static_scale(
    samples: "torch.Tensor",
    weights: Sequence["torch.Tensor"],
    *,
    reference_scale: float,
    diagonal_hint_scale: float,
    config: ScaleSearchConfig | None = None,
) -> ExactOutputScaleSearchResult:
    """Search a fixed scale by measuring the exact fused Linear output error.

    Candidate activation errors are multiplied by every BF16 component weight
    that shares the runtime fused Linear input.  The persisted result remains
    one static FP32 scale; this exact objective is used only during offline
    calibration.
    """

    import torch
    import torch.nn.functional as F

    config = ScaleSearchConfig() if config is None else config
    matrix = torch.as_tensor(samples)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("exact-output samples must be non-empty [rows, channels]")
    if not weights:
        raise ValueError("exact-output search requires at least one Linear weight")
    if not math.isfinite(reference_scale) or reference_scale <= 0.0:
        raise ValueError("reference_scale must be finite and positive")
    if not math.isfinite(diagonal_hint_scale) or diagonal_hint_scale <= 0.0:
        raise ValueError("diagonal_hint_scale must be finite and positive")
    for weight in weights:
        if (
            not isinstance(weight, torch.Tensor)
            or weight.ndim != 2
            or weight.shape[1] != matrix.shape[1]
            or weight.device != matrix.device
        ):
            raise ValueError(
                "every exact-output weight must be a 2-D tensor on the sample "
                "device with matching input channels"
            )

    valid = torch.isfinite(matrix)
    nonfinite_count = int((~valid).sum().item())
    safe = torch.where(valid, matrix, torch.zeros_like(matrix)).to(torch.float32)
    finite = safe[valid]
    if finite.numel() == 0:
        raise ValueError("exact-output samples contain no finite values")

    offsets = torch.linspace(
        -config.refinement_octaves,
        config.refinement_octaves,
        config.refinement_steps,
        dtype=torch.float64,
    )
    multipliers = torch.pow(2.0, offsets)
    candidates = torch.unique(
        torch.cat(
            (
                torch.tensor([reference_scale], dtype=torch.float64) * multipliers,
                torch.tensor([diagonal_hint_scale], dtype=torch.float64)
                * multipliers,
                torch.tensor(
                    [reference_scale, diagonal_hint_scale], dtype=torch.float64
                ),
            )
        ).clamp_min(config.eps),
        sorted=True,
    )

    output_channel_count = sum(int(weight.shape[0]) for weight in weights)
    output_element_count = matrix.shape[0] * output_channel_count
    weight_dtype = weights[0].dtype
    quantization_source = safe.cpu() if safe.device.type != "cpu" else safe
    output_signal = torch.zeros((), dtype=torch.float32, device=matrix.device)
    with torch.inference_mode():
        source = safe.to(weight_dtype)
        for weight in weights:
            output_signal.add_(F.linear(source, weight).to(torch.float32).square().sum())
        output_signal.clamp_min_(torch.finfo(torch.float32).tiny)

        error_powers: list[torch.Tensor] = []
        for candidate in candidates.tolist():
            reconstructed = static_fake_quantize(
                quantization_source,
                torch.tensor(
                    [candidate],
                    dtype=torch.float32,
                    device=quantization_source.device,
                ),
            ).to(matrix.device)
            activation_error = (reconstructed - safe).to(weight_dtype)
            power = torch.zeros((), dtype=torch.float32, device=matrix.device)
            for weight in weights:
                power.add_(
                    F.linear(activation_error, weight)
                    .to(torch.float32)
                    .square()
                    .sum()
                )
            error_powers.append(power)
        powers = torch.stack(error_powers).cpu()
        signal = float(output_signal.cpu().item())

    normalized = powers / signal
    best = int(normalized.argmin().item())
    selected = float(candidates[best].to(torch.float32).item())
    reference_index = int(
        (candidates - float(reference_scale)).abs().argmin().item()
    )
    reference_nmse = float(normalized[reference_index].item())
    selected_nmse = float(normalized[best].item())
    improvement = (
        0.0
        if reference_nmse <= 0.0
        else (reference_nmse - selected_nmse) / reference_nmse
    )
    clipping = float(
        (finite.abs() > selected * MAX_FINITE).to(torch.float32).mean().item()
    )
    return ExactOutputScaleSearchResult(
        scale=selected,
        output_weighted_mse=float(powers[best].item() / output_element_count),
        output_weighted_normalized_mse=selected_nmse,
        reference_scale=float(reference_scale),
        reference_output_weighted_normalized_mse=reference_nmse,
        relative_output_error_improvement=float(improvement),
        diagonal_hint_scale=float(diagonal_hint_scale),
        clipping_rate=clipping,
        max_abs=float(finite.abs().max().item()),
        sample_count=int(finite.numel()),
        nonfinite_count=nonfinite_count,
        candidate_count=int(candidates.numel()),
        channel_count=int(matrix.shape[1]),
        output_channel_count=output_channel_count,
    )


class DeviceSampleObserver:
    """Bounded, synchronization-free sample collector for hooks.

    Samples stay on the source device until :meth:`cpu_samples` is called.
    Each observation contributes an evenly spread deterministic slice.  The
    phase changes per call, which avoids repeatedly selecting the same hidden
    dimensions while requiring no host-to-device random-index transfer.
    """

    def __init__(
        self,
        *,
        capacity: int = 65_536,
        samples_per_observation: int = 512,
        group_count: int = 1,
        group_axis: int | None = None,
        seed: int = 42,
    ) -> None:
        if capacity <= 0 or samples_per_observation <= 0:
            raise ValueError("capacity and samples_per_observation must be positive")
        if group_count <= 0:
            raise ValueError("group_count must be positive")
        if group_axis is None and group_count != 1:
            raise ValueError("group_axis is required when group_count is greater than one")
        self.capacity = int(capacity)
        self.samples_per_observation = int(samples_per_observation)
        self.group_count = int(group_count)
        self.group_axis = group_axis
        self.seed = int(seed)
        self.observation_count = 0
        self.total_values_per_group = 0
        self._filled = 0
        self._samples: torch.Tensor | None = None
        self._sampled_nonfinite: torch.Tensor | None = None
        self._max_abs: torch.Tensor | None = None

    @staticmethod
    def _coprime_stride(length: int, requested: int) -> int:
        stride = max(1, length // max(1, requested))
        while math.gcd(stride, length) != 1:
            stride += 1
        return stride

    def observe(self, value: "torch.Tensor") -> None:
        import torch

        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            raise TypeError("observed value must be a tensor with at least one dimension")
        detached = value.detach()
        if self.group_axis is None:
            grouped = detached.reshape(1, -1)
        else:
            axis = self.group_axis % detached.ndim
            if detached.shape[axis] != self.group_count:
                raise ValueError(
                    f"observer expected {self.group_count} groups on axis {axis}, got shape {tuple(detached.shape)}"
                )
            grouped = detached.movedim(axis, 0).reshape(self.group_count, -1)
        group_width = int(grouped.shape[1])
        if group_width <= 0:
            raise ValueError("cannot calibrate an empty tensor")

        if self._samples is None:
            self._samples = torch.empty(
                (self.group_count, self.capacity),
                dtype=torch.float32,
                device=detached.device,
            )
            self._sampled_nonfinite = torch.zeros(self.group_count, dtype=torch.int64, device=detached.device)
            self._max_abs = torch.zeros(self.group_count, dtype=torch.float32, device=detached.device)
        elif self._samples.device != detached.device:
            raise ValueError(f"observer device changed from {self._samples.device} to {detached.device}")

        room = self.capacity - self._filled
        take = min(room, self.samples_per_observation, group_width)
        if take > 0:
            stride = self._coprime_stride(group_width, take)
            phase = (self.seed + self.observation_count * 1_000_003) % group_width
            indices = (phase + torch.arange(take, device=detached.device) * stride) % group_width
            selected = grouped.index_select(1, indices).to(torch.float32)
            finite = torch.isfinite(selected)
            self._samples[:, self._filled : self._filled + take].copy_(
                torch.where(finite, selected, torch.zeros_like(selected))
            )
            self._sampled_nonfinite.add_((~finite).sum(dim=1))
            self._filled += take

        # Exact per-group extrema are inexpensive relative to the calibration
        # forward and catch rare channels that a bounded reservoir can miss.
        finite_grouped = torch.where(
            torch.isfinite(grouped), grouped, torch.zeros((), dtype=grouped.dtype, device=grouped.device)
        )
        current_max = finite_grouped.abs().amax(dim=1).to(torch.float32)
        self._max_abs.copy_(torch.maximum(self._max_abs, current_max))
        self.total_values_per_group += group_width
        self.observation_count += 1

    @property
    def sample_count(self) -> int:
        return self._filled

    def cpu_samples(self) -> "torch.Tensor":
        if self._samples is None:
            raise RuntimeError("observer has not received any tensors")
        return self._samples[:, : self._filled].cpu()

    def summary(self) -> dict[str, Any]:
        if self._max_abs is None or self._sampled_nonfinite is None:
            raise RuntimeError("observer has not received any tensors")
        return {
            "observation_count": self.observation_count,
            "total_values_per_group": self.total_values_per_group,
            "sample_count_per_group": self._filled,
            "sampled_nonfinite_per_group": [int(value) for value in self._sampled_nonfinite.cpu().tolist()],
            "max_abs_per_group": [float(value) for value in self._max_abs.cpu().tolist()],
        }


class OutputAwareDeviceSampleObserver:
    """Bounded collector that preserves complete Linear activation rows.

    Scalar reservoirs are sufficient for an input-MSE objective but discard
    the channel identity needed by a Linear-output objective.  This observer
    samples deterministic complete rows and keeps them on the source device
    until calibration search begins.
    """

    def __init__(
        self,
        *,
        capacity: int = 64,
        samples_per_observation: int = 4,
        seed: int = 42,
    ) -> None:
        if capacity <= 0 or samples_per_observation <= 0:
            raise ValueError("capacity and samples_per_observation must be positive")
        self.capacity = int(capacity)
        self.samples_per_observation = int(samples_per_observation)
        self.seed = int(seed)
        self.observation_count = 0
        self.total_rows = 0
        self._filled = 0
        self._width: int | None = None
        self._samples: torch.Tensor | None = None
        self._sampled_nonfinite: torch.Tensor | None = None
        self._max_abs: torch.Tensor | None = None

    def observe(self, value: "torch.Tensor") -> None:
        import torch

        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            raise TypeError("observed value must be a tensor with at least one dimension")
        detached = value.detach()
        width = int(detached.shape[-1])
        if width <= 0:
            raise ValueError("cannot calibrate a Linear with no input channels")
        rows = detached.reshape(-1, width)
        row_count = int(rows.shape[0])
        if row_count <= 0:
            raise ValueError("cannot calibrate an empty activation tensor")

        if self._samples is None:
            self._width = width
            self._samples = torch.empty(
                (self.capacity, width),
                dtype=torch.float32,
                device=detached.device,
            )
            self._sampled_nonfinite = torch.zeros(
                (), dtype=torch.int64, device=detached.device
            )
            self._max_abs = torch.zeros((), dtype=torch.float32, device=detached.device)
        elif self._width != width:
            raise ValueError(
                f"observer input width changed from {self._width} to {width}"
            )
        elif self._samples.device != detached.device:
            raise ValueError(
                f"observer device changed from {self._samples.device} to {detached.device}"
            )

        room = self.capacity - self._filled
        take = min(room, self.samples_per_observation, row_count)
        if take > 0:
            stride = DeviceSampleObserver._coprime_stride(row_count, take)
            phase = (self.seed + self.observation_count * 1_000_003) % row_count
            indices = (
                phase + torch.arange(take, device=detached.device) * stride
            ) % row_count
            selected = rows.index_select(0, indices).to(torch.float32)
            finite = torch.isfinite(selected)
            self._samples[self._filled : self._filled + take].copy_(
                torch.where(finite, selected, torch.zeros_like(selected))
            )
            self._sampled_nonfinite.add_((~finite).sum())
            self._filled += take

        finite_rows = torch.where(
            torch.isfinite(rows),
            rows,
            torch.zeros((), dtype=rows.dtype, device=rows.device),
        )
        self._max_abs.copy_(
            torch.maximum(self._max_abs, finite_rows.abs().amax().to(torch.float32))
        )
        self.total_rows += row_count
        self.observation_count += 1

    @property
    def sample_count(self) -> int:
        return self._filled

    def cpu_samples(self) -> "torch.Tensor":
        if self._samples is None:
            raise RuntimeError("observer has not received any tensors")
        return self._samples[: self._filled].cpu()

    def device_samples(self) -> "torch.Tensor":
        if self._samples is None:
            raise RuntimeError("observer has not received any tensors")
        return self._samples[: self._filled]

    def summary(self) -> dict[str, Any]:
        import torch

        if (
            self._max_abs is None
            or self._sampled_nonfinite is None
            or self._width is None
        ):
            raise RuntimeError("observer has not received any tensors")
        channel_mean_square = (
            self._samples[: self._filled]
            .to(torch.float32)
            .square()
            .mean(dim=0)
            .cpu()
            .tolist()
        )
        return {
            "observation_count": self.observation_count,
            "total_rows": self.total_rows,
            "sample_count_rows": self._filled,
            "input_channel_count": self._width,
            "sampled_nonfinite": int(self._sampled_nonfinite.cpu().item()),
            "max_abs": float(self._max_abs.cpu().item()),
            # Diagonal activation covariance weights the exporter's per-row MSE.
            "channel_mean_square": [float(value) for value in channel_mean_square],
        }


def sample_token_windows(
    token_ids: "torch.Tensor",
    *,
    sample_count: int,
    sequence_length: int,
    seed: int,
) -> tuple["torch.Tensor", tuple[int, ...]]:
    """Sample GPTQ-style fixed windows from one concatenated token stream."""

    import torch

    tokens = torch.as_tensor(token_ids, dtype=torch.long, device="cpu").reshape(-1)
    if sample_count <= 0 or sequence_length <= 0:
        raise ValueError("sample_count and sequence_length must be positive")
    possible_starts = tokens.numel() - sequence_length + 1
    if possible_starts <= 0:
        raise ValueError(f"token stream of length {tokens.numel()} is shorter than sequence_length={sequence_length}")
    if sample_count > possible_starts:
        raise ValueError(f"cannot draw {sample_count} unique windows from {possible_starts} positions")
    starts = tuple(random.Random(seed).sample(range(possible_starts), sample_count))
    windows = torch.stack([tokens[start : start + sequence_length] for start in starts])
    return windows, starts


def validate_calibration_artifact(artifact: dict[str, Any]) -> None:
    """Reject incomplete artifacts before checkpoint export or runtime use."""

    if artifact.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported calibration schema_version={artifact.get('schema_version')!r}; "
            f"expected {CALIBRATION_SCHEMA_VERSION}"
        )
    if artifact.get("format") != CALIBRATION_FORMAT:
        raise ValueError(f"unexpected calibration format: {artifact.get('format')!r}")
    linear_groups = artifact.get("linear_groups")
    kv_cache = artifact.get("kv_cache")
    if not isinstance(linear_groups, dict) or not linear_groups:
        raise ValueError("calibration artifact has no linear_groups")
    if not isinstance(kv_cache, dict) or not isinstance(kv_cache.get("layers"), dict):
        raise ValueError("calibration artifact has no kv_cache.layers")
    for name, entry in linear_groups.items():
        scale = entry.get("scale") if isinstance(entry, dict) else None
        if not isinstance(scale, (int, float)) or scale <= 0 or not math.isfinite(scale):
            raise ValueError(f"linear group {name!r} has invalid scale {scale!r}")
    for layer, entry in kv_cache["layers"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"KV layer {layer!r} is not an object")
        for kind in ("key", "value"):
            scales = entry.get(kind, {}).get("scales") if isinstance(entry.get(kind), dict) else None
            if not isinstance(scales, list) or not scales:
                raise ValueError(f"KV layer {layer!r} has no {kind} scales")
            if any(not isinstance(value, (int, float)) or value <= 0 or not math.isfinite(value) for value in scales):
                raise ValueError(f"KV layer {layer!r} has invalid {kind} scales")


__all__ = [
    "CALIBRATION_FORMAT",
    "CALIBRATION_SCHEMA_VERSION",
    "STATIC_QUANT_TYPE",
    "DeviceSampleObserver",
    "ExactOutputScaleSearchResult",
    "LinearGroupSpec",
    "OutputAwareDeviceSampleObserver",
    "OutputAwareScaleSearchResult",
    "ScaleSearchConfig",
    "ScaleSearchResult",
    "file_sha256",
    "qwen35_full_attention_layers",
    "qwen35_linear_group_specs",
    "sample_token_windows",
    "saturating_encode_tensor",
    "search_output_aware_static_scale",
    "search_exact_output_aware_static_scale",
    "search_static_scale",
    "static_fake_quantize",
    "static_quantize_tensor",
    "validate_calibration_artifact",
]
