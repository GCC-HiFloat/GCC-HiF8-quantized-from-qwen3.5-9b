#!/usr/bin/env python3
"""Measure paired WikiText-2 next-token likelihood on BF16 and static HiFloat8.

The script intentionally uses vLLM's prompt-logprob API rather than a
different HF forward implementation.  Both checkpoints therefore go through
the same tokenizer, scheduler, attention implementation, and scorer.  The
target token's logprob is included alongside the requested top-k entries by
vLLM, so ``prompt_logprobs=1`` is sufficient without materializing a full
``[tokens, vocab]`` tensor on the host.

Run this file once per checkpoint and pass the two JSON reports to
``--compare`` for a deterministic paired summary.  The model checkpoint is
never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_PARQUET = (
    "/public-flash/lvyufeng/.cache/huggingface/hub/datasets--Salesforce--wikitext"
    "/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1"
    "/validation-00000-of-00001.parquet"
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parquet", type=Path, default=Path(DEFAULT_PARQUET))
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--starts-from",
        type=Path,
        default=None,
        help=(
            "reuse the sampling.starts array from an existing report JSON; "
            "this makes a BF16/candidate run exactly paired with that report"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--kv-cache-mib", type=int, default=512)
    parser.add_argument("--no-kv8", action="store_true")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="interpret the two positional paths as reference and candidate reports",
    )
    parser.add_argument("--no-async-scheduling", action="store_true")
    return parser.parse_args()


def _load_windows(
    parquet: Path,
    tokenizer: Any,
    *,
    count: int,
    length: int,
    seed: int,
    starts: list[int] | None = None,
):
    import torch
    from datasets import load_dataset

    if not parquet.is_file():
        raise FileNotFoundError(f"WikiText-2 parquet does not exist: {parquet}")
    dataset = load_dataset("parquet", data_files={"validation": str(parquet)}, split="validation")
    text = "\n\n".join(str(value) for value in dataset["text"])
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].reshape(-1)
    possible = int(encoded.numel()) - length + 1
    if possible <= 0:
        raise ValueError(f"validation token stream has {encoded.numel()} tokens, shorter than {length}")
    if count <= 0 or count > possible:
        raise ValueError(f"samples must be in [1, {possible}], got {count}")
    if starts is None:
        import random

        starts = random.Random(seed).sample(range(possible), count)
    else:
        starts = [int(start) for start in starts]
        if not starts:
            raise ValueError("the paired report contains no window starts")
        if any(start < 0 or start >= possible for start in starts):
            raise ValueError(
                f"paired window starts must be in [0, {possible}), got {starts[:3]}"
            )
        count = len(starts)
    windows = [encoded[start : start + length].tolist() for start in starts]
    return windows, starts, int(encoded.numel()), hashlib.sha256(encoded.numpy().tobytes()).hexdigest()


def _extract_target_logprob(position: Any, target_id: int) -> tuple[float, int]:
    """Return target logprob and greedy token from a vLLM prompt position."""

    if position is None:
        raise ValueError("vLLM returned no prompt logprob position")
    values = position.items() if hasattr(position, "items") else []
    target = None
    best_token = None
    best_logprob = -math.inf
    for token_id, record in values:
        value = float(record.logprob)
        token_id = int(token_id)
        if value > best_logprob:
            best_logprob = value
            best_token = token_id
        if token_id == target_id:
            target = value
    if target is None:
        raise ValueError(
            f"vLLM prompt_logprobs did not include target token {target_id}; "
            "increase --prompt-logprobs if the backend no longer includes the target"
        )
    assert best_token is not None
    return target, best_token


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    metadata_path = args.model / "hif8_metadata.json"
    is_static_checkpoint = False
    if metadata_path.is_file():
        try:
            is_static_checkpoint = (
                json.loads(metadata_path.read_text(encoding="utf-8")).get("format")
                == "W8A8HIF8_STATIC"
            )
        except (OSError, json.JSONDecodeError):
            is_static_checkpoint = False
    if not args.no_kv8 and (is_static_checkpoint or args.model.name.endswith("W8A8KV8")):
        os.environ["HIF8_KV_CACHE"] = "1"
    else:
        os.environ.pop("HIF8_KV_CACHE", None)

    import torch
    import torch_npu  # noqa: F401
    from vllm import LLM, SamplingParams, TokensPrompt

    started = time.perf_counter()
    llm_kwargs = dict(
        model=str(args.model.resolve()),
        tensor_parallel_size=1,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_mib * 1024 * 1024,
        seed=0,
        trust_remote_code=True,
    )
    if args.no_async_scheduling:
        llm_kwargs["async_scheduling"] = False
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    paired_starts = None
    if args.starts_from is not None:
        source = json.loads(args.starts_from.read_text(encoding="utf-8"))
        try:
            paired_starts = source["sampling"]["starts"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"{args.starts_from} does not contain sampling.starts"
            ) from error
    windows, starts, token_count, stream_hash = _load_windows(
        args.parquet,
        tokenizer,
        count=args.samples,
        length=args.sequence_length,
        seed=args.seed,
        starts=paired_starts,
    )

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
        detokenize=False,
        skip_special_tokens=False,
    )
    nlls: list[float] = []
    top1_matches: list[bool] = []
    token_count_scored = 0
    window_reports: list[dict[str, Any]] = []
    for begin in range(0, len(windows), args.batch_size):
        batch = windows[begin : begin + args.batch_size]
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=window) for window in batch],
            sampling,
            use_tqdm=False,
        )
        torch.npu.synchronize()
        for offset, (window, output) in enumerate(zip(batch, outputs, strict=True)):
            prompt_logprobs = output.prompt_logprobs
            if prompt_logprobs is None or len(prompt_logprobs) < len(window):
                raise RuntimeError("vLLM did not return one prompt logprob entry per input token")
            values: list[float] = []
            matches = 0
            # Position zero has no preceding context and is intentionally not
            # scored.  At position i, the model predicts window[i].
            for index in range(1, len(window)):
                target_logprob, top_token = _extract_target_logprob(
                    prompt_logprobs[index], int(window[index])
                )
                values.append(target_logprob)
                matches += int(top_token == int(window[index]))
            nll = -sum(values)
            nlls.extend(-value for value in values)
            top1_matches.extend(
                [
                    _extract_target_logprob(prompt_logprobs[index], int(window[index]))[1]
                    == int(window[index])
                    for index in range(1, len(window))
                ]
            )
            token_count_scored += len(values)
            window_reports.append(
                {
                    "index": begin + offset,
                    "start": starts[begin + offset],
                    "tokens_scored": len(values),
                    "nll": nll,
                    "mean_nll": nll / len(values),
                    "top1_accuracy": matches / len(values),
                }
            )

    mean_nll = sum(nlls) / token_count_scored
    report = {
        "model": str(args.model.resolve()),
        "format": "static_w8a8kv8" if is_static_checkpoint and os.environ.get("HIF8_KV_CACHE") == "1" else "bf16_or_w8a8",
        "dataset": {
            "parquet": str(args.parquet.resolve()),
            "sha256": hashlib.sha256(args.parquet.read_bytes()).hexdigest(),
            "token_count": token_count,
            "token_stream_sha256": stream_hash,
        },
        "sampling": {
            "count": args.samples,
            "sequence_length": args.sequence_length,
            "seed": args.seed,
            "starts": starts,
        },
        "scored_tokens": token_count_scored,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "top1_accuracy": sum(top1_matches) / len(top1_matches),
        "window_reports": window_reports,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    return report


def compare(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if reference["dataset"] != candidate["dataset"] or reference["sampling"] != candidate["sampling"]:
        raise ValueError("reference and candidate do not use identical dataset/sampling protocols")
    mean_nll_delta = float(candidate["mean_nll"]) - float(reference["mean_nll"])
    ppl_ratio = float(candidate["perplexity"]) / float(reference["perplexity"])
    return {
        "reference_model": reference["model"],
        "candidate_model": candidate["model"],
        "scored_tokens": reference["scored_tokens"],
        "reference_mean_nll": reference["mean_nll"],
        "candidate_mean_nll": candidate["mean_nll"],
        "mean_nll_delta": mean_nll_delta,
        "reference_perplexity": reference["perplexity"],
        "candidate_perplexity": candidate["perplexity"],
        "perplexity_ratio": ppl_ratio,
        "reference_top1_accuracy": reference["top1_accuracy"],
        "candidate_top1_accuracy": candidate["top1_accuracy"],
        "top1_accuracy_delta": candidate["top1_accuracy"] - reference["top1_accuracy"],
        "window_mean_nll_deltas": [
            cand["mean_nll"] - ref["mean_nll"]
            for ref, cand in zip(reference["window_reports"], candidate["window_reports"], strict=True)
        ],
    }


def main() -> None:
    args = _args()
    if args.compare:
        result = compare(args.model, args.output)
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return
    report = run_report(args)
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
