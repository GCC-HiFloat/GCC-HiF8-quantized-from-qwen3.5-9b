#!/usr/bin/env python3
"""Reproducible single-device vLLM benchmark for BF16 and HiFloat8 models."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--prompt", default="Explain in one paragraph why the sky appears blue.")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="number of identical prompts submitted in one generate call",
    )
    parser.add_argument(
        "--batch-sizes",
        default=None,
        help="comma-separated request sizes to run on one engine (dynamic-M sweep)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="vLLM scheduler max_num_seqs (defaults to batch size)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="vLLM scheduler max_num_batched_tokens",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--kv-cache-mib", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vary-prompts",
        action="store_true",
        help=(
            "append a stable request identifier to every prompt; this makes "
            "cross-batch output comparisons expose request-mixing/GDN issues"
        ),
    )
    parser.add_argument(
        "--no-async-scheduling",
        action="store_true",
        help="disable vLLM asynchronous scheduling for correctness isolation",
    )
    parser.add_argument(
        "--enable-compile",
        action="store_true",
        help="allow vLLM/Ascend compilation and graph execution instead of forcing eager mode",
    )
    parser.add_argument(
        "--diagnostic-noop-gdn",
        action="store_true",
        help=(
            "replace Qwen GDN recurrent attention cores with no-ops inside "
            "the worker; performance diagnosis only, outputs are invalid"
        ),
    )
    return parser.parse_args()


def memory_stats(torch) -> dict[str, float]:
    mib = 1024 * 1024
    return {
        "allocated_mib": torch.npu.memory_allocated() / mib,
        "reserved_mib": torch.npu.memory_reserved() / mib,
        "max_allocated_mib": torch.npu.max_memory_allocated() / mib,
        "max_reserved_mib": torch.npu.max_memory_reserved() / mib,
    }


def audit_model(model) -> dict[str, object]:
    """Run inside the engine worker and return storage/dtype invariants."""
    import torch

    backend_modules = []
    decoded_cache_names = []
    hif8_kv_layers = []
    recurrent_cache_layers = []
    gdn_runtime_layers = []
    for name, module in model.named_modules():
        backend = getattr(module, "_hif8_compute_backend", None)
        if backend is not None:
            backend_modules.append(
                {
                    "name": name,
                    "backend": backend,
                    "scale_dtype": str(module.weight_scale.dtype),
                }
            )
        for attribute in vars(module):
            if "decoded" in attribute.lower() and "weight" in attribute.lower():
                decoded_cache_names.append(f"{name}.{attribute}")

        impl = getattr(module, "impl", None)
        kv_cache = getattr(module, "kv_cache", None)
        if getattr(impl, "_hif8_kv_cache", False):
            if isinstance(kv_cache, torch.Tensor):
                cache_tensors = list(kv_cache.unbind(0)) if kv_cache.ndim == 5 else []
                binding = "combined"
            elif isinstance(kv_cache, (tuple, list)):
                cache_tensors = [tensor for tensor in kv_cache if isinstance(tensor, torch.Tensor)]
                binding = "pair"
            else:
                cache_tensors = []
                binding = type(kv_cache).__name__
            persistent_float_cache_attrs = []
            for attribute, value in vars(impl).items():
                if (
                    "cache" in attribute.lower()
                    and isinstance(value, torch.Tensor)
                    and value.numel() > 0
                    and value.dtype in (torch.float16, torch.bfloat16, torch.float32)
                ):
                    persistent_float_cache_attrs.append(attribute)
            hif8_kv_layers.append(
                {
                    "name": name,
                    "impl": type(impl).__name__,
                    "binding": binding,
                    "cache_dtype": getattr(module, "kv_cache_dtype", None),
                    "tensor_dtypes": [str(tensor.dtype) for tensor in cache_tensors],
                    "tensor_shapes": [list(tensor.shape) for tensor in cache_tensors],
                    "tensor_strides": [list(tensor.stride()) for tensor in cache_tensors],
                    "storage_bytes": [tensor.untyped_storage().nbytes() for tensor in cache_tensors],
                    "persistent_float_cache_attrs": persistent_float_cache_attrs,
                }
            )
        elif isinstance(kv_cache, (tuple, list)) and any(
            isinstance(tensor, torch.Tensor) and tensor.numel() > 0
            for tensor in kv_cache
        ):
            cache_tensors = [
                tensor
                for tensor in kv_cache
                if isinstance(tensor, torch.Tensor) and tensor.numel() > 0
            ]
            recurrent_cache_layers.append(
                {
                    "name": name,
                    "impl": type(impl).__name__ if impl is not None else None,
                    "tensor_dtypes": [str(tensor.dtype) for tensor in cache_tensors],
                    "tensor_shapes": [list(tensor.shape) for tensor in cache_tensors],
                }
            )

        if "QwenGatedDeltaNetAttention" in type(module).__name__:
            forward_core = getattr(module, "_forward_core", None)
            gdn_runtime_layers.append(
                {
                    "name": name,
                    "class": f"{type(module).__module__}.{type(module).__name__}",
                    "forward_core_module": getattr(forward_core, "__module__", None),
                    "forward_core_qualname": getattr(forward_core, "__qualname__", None),
                }
            )

    storage_bytes_by_dtype: dict[str, int] = {}
    seen_storages: set[tuple[str, int]] = set()
    largest_storages: list[dict[str, object]] = []
    named_tensors = (*model.named_parameters(), *model.named_buffers())
    for name, tensor in named_tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key in seen_storages:
            continue
        seen_storages.add(key)
        dtype = str(tensor.dtype)
        storage_bytes_by_dtype[dtype] = storage_bytes_by_dtype.get(dtype, 0) + storage.nbytes()
        largest_storages.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": list(tensor.shape),
                "storage_bytes": storage.nbytes(),
            }
        )

    largest_storages.sort(key=lambda item: int(item["storage_bytes"]), reverse=True)

    mib = 1024 * 1024
    worker_memory = {
        "allocated_mib": torch.npu.memory_allocated() / mib,
        "reserved_mib": torch.npu.memory_reserved() / mib,
        "max_allocated_mib": torch.npu.max_memory_allocated() / mib,
        "max_reserved_mib": torch.npu.max_memory_reserved() / mib,
    }
    return {
        "hif8_linear_count": len(backend_modules),
        "hif8_bf16_scale_count": sum(item["scale_dtype"] == "torch.bfloat16" for item in backend_modules),
        "hif8_backends": sorted({item["backend"] for item in backend_modules}),
        "decoded_weight_cache_names": decoded_cache_names,
        "hif8_kv_layer_count": len(hif8_kv_layers),
        "hif8_kv_layers": hif8_kv_layers,
        "recurrent_cache_layer_count": len(recurrent_cache_layers),
        "recurrent_cache_layers": recurrent_cache_layers,
        "gdn_runtime_layers": gdn_runtime_layers,
        "ascend_gdn_ops_registered": {
            "causal_conv1d": hasattr(
                torch.ops._C_ascend, "npu_causal_conv1d_custom"
            ),
            "recurrent_gated_delta_rule": hasattr(
                torch.ops._C_ascend, "npu_recurrent_gated_delta_rule"
            ),
        },
        "resident_storage_bytes_by_dtype": storage_bytes_by_dtype,
        "resident_storage_total_bytes": sum(storage_bytes_by_dtype.values()),
        "largest_resident_storages": largest_storages[:20],
        "worker_memory": worker_memory,
    }


def _noop_gdn_core(self, *args, **kwargs) -> None:
    """Diagnostic replacement for Qwen's stateful GDN attention core."""

    del self, args, kwargs


def install_noop_gdn(model) -> dict[str, object]:
    """Install a worker-local GDN no-op to isolate non-GEMM decode cost."""

    from types import MethodType

    patched = []
    for name, module in model.named_modules():
        if (
            "QwenGatedDeltaNetAttention" in type(module).__name__
            and callable(getattr(module, "_forward_core", None))
        ):
            module._forward_core = MethodType(_noop_gdn_core, module)
            patched.append(name)
    return {"patched_count": len(patched), "patched_modules": patched}


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    batch_sizes = [args.batch_size]
    if args.batch_sizes:
        try:
            batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value]
        except ValueError as exc:
            raise ValueError("--batch-sizes must be comma-separated positive integers") from exc
        if not batch_sizes or any(value < 1 for value in batch_sizes):
            raise ValueError("--batch-sizes must contain positive integers")
    started = time.perf_counter()

    # vLLM's library entry point defaults to fork. Ascend's runtime cannot be
    # reinitialized in a forked child after torch-npu has touched the device.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # apply_model serializes this script's local, read-only audit callable to
    # the engine worker. No untrusted payload crosses this process boundary.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    import torch
    import torch_npu  # noqa: F401 - registers the NPU backend
    from vllm import LLM, SamplingParams

    load_started = time.perf_counter()
    llm_kwargs = dict(
        model=str(args.model.resolve()),
        tensor_parallel_size=1,
        dtype="bfloat16",
        enforce_eager=not args.enable_compile,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_mib * 1024 * 1024,
        seed=args.seed,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs or max(batch_sizes),
    )
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.no_async_scheduling:
        llm_kwargs["async_scheduling"] = False
    llm = LLM(**llm_kwargs)
    torch.npu.synchronize()
    load_seconds = time.perf_counter() - load_started
    gdn_diagnostic = (
        llm.apply_model(install_noop_gdn)[0]
        if args.diagnostic_noop_gdn
        else None
    )
    model_audit_after_load = llm.apply_model(audit_model)[0]

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    runs: list[dict[str, object]] = []
    for batch_size in batch_sizes:
        for index in range(args.warmup + args.repeats):
            torch.npu.synchronize()
            run_started = time.perf_counter()
            prompts = (
                [f"{args.prompt}\nRequest identifier: {request_index}." for request_index in range(batch_size)]
                if args.vary_prompts
                else [args.prompt] * batch_size
            )
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            torch.npu.synchronize()
            elapsed = time.perf_counter() - run_started
            generated = sum(len(output.outputs[0].token_ids) for output in outputs)
            if index >= args.warmup:
                runs.append(
                    {
                        "seconds": elapsed,
                        "generated_tokens": generated,
                        "batch_size": batch_size,
                        "tokens_per_second": generated / elapsed if elapsed > 0 else 0.0,
                        "text": outputs[0].outputs[0].text,
                        "output_token_ids": [
                            list(output.outputs[0].token_ids) for output in outputs
                        ],
                    }
                )

    total_tokens = sum(int(run["generated_tokens"]) for run in runs)
    total_seconds = sum(float(run["seconds"]) for run in runs)
    result = {
        "model": str(args.model.resolve()),
        "compute_backend": os.environ.get("HIF8_COMPUTE_BACKEND", "default_bf16"),
        "prompt": args.prompt,
        "vary_prompts": args.vary_prompts,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "batch_sizes": batch_sizes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "load_seconds": load_seconds,
        "diagnostic_noop_gdn": gdn_diagnostic,
        "model_audit_after_load": model_audit_after_load,
        "model_audit_after_runs": llm.apply_model(audit_model)[0],
        "aggregate_tokens_per_second": total_tokens / total_seconds,
        "runs": runs,
        "wall_seconds": time.perf_counter() - started,
    }
    print("HIF8_BENCHMARK_JSON=" + json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
