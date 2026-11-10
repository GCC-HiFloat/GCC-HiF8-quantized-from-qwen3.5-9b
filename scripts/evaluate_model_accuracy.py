#!/usr/bin/env python3
"""Run a deterministic, multi-domain smoke evaluation with one vLLM model."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


CASES = (
    ("zh_factual", "只输出答案，不要解释：中国的首都是哪座城市？", "北京"),
    ("en_factual", "Answer with only the city name: What is the capital of France?", "Paris"),
    ("arithmetic", "只输出最终数字，不要解释：(37 × 24) - 19 等于多少？", "869"),
    (
        "reasoning",
        "Answer only Yes or No. All bloops are razzies, and all razzies are lazzies. "
        "Must every bloop be a lazzy?",
        "Yes",
    ),
    ("zh_to_en", "Translate into concise English only: 人工智能可以帮助医生更早发现疾病。", None),
    ("en_to_zh", "只输出中文翻译：The experiment was repeated three times to verify the result。", None),
    (
        "code",
        "Write only a Python function named is_even(n) that returns whether integer n is even. "
        "Do not add Markdown fences or explanation.",
        None,
    ),
    (
        "summarization",
        "Summarize in one short English sentence: Water evaporates from oceans, condenses into "
        "clouds, and returns to Earth as rain or snow, continually circulating through the environment.",
        None,
    ),
    ("zh_science", "只输出化学式：水的化学式是什么？", "H2O"),
    ("en_science", "Answer only with the planet name: Which planet is known as the Red Planet?", "Mars"),
    ("zh_history", "只输出年份：中华人民共和国成立于哪一年？", "1949"),
    ("en_history", "Answer only with the year: In what year did World War II end?", "1945"),
    ("arithmetic_add", "只输出数字：487 + 326 等于多少？", "813"),
    ("arithmetic_div", "Answer with only the number: 144 divided by 12 equals what?", "12"),
    ("arithmetic_percent", "只输出数字：200 的 15% 是多少？", "30"),
    ("algebra", "Answer with only the number: If 3x + 5 = 20, what is x?", "5"),
    (
        "logic_no",
        "Answer only Yes or No. Some cats are black. Does it follow that every cat is black?",
        "No",
    ),
    (
        "logic_order",
        "Answer only with the name. Amy is taller than Bob, and Bob is taller than Chen. "
        "Who is tallest?",
        "Amy",
    ),
    ("sequence", "只输出下一个数字：2, 4, 8, 16, ?", "32"),
    ("unit", "Answer only with the number: How many centimeters are in one meter?", "100"),
    ("uppercase", "Output only the uppercase form of: Ascend", "ASCEND"),
    ("json", '只输出合法 JSON，不要 Markdown：键名为 "ok"，布尔值为真。', '{"ok":true}'),
    ("zh_antonym", "只输出反义词：‘增加’的反义词是什么？", "减少"),
    ("en_synonym", "Answer with only one word: Give a synonym for rapid.", None),
)


def normalize_answer(value: str) -> str:
    subscript_digits = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    return "".join(value.strip().translate(subscript_digits).lower().split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--kv-cache-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-async-scheduling",
        action="store_true",
        help="disable vLLM asynchronous scheduling for correctness isolation",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="run only the named case (repeatable; default: run all cases)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    available_case_ids = {case_id for case_id, _, _ in CASES}
    unknown_case_ids = sorted(set(args.case_id) - available_case_ids)
    if unknown_case_ids:
        raise ValueError(
            f"unknown case IDs: {unknown_case_ids}; available IDs: "
            f"{sorted(available_case_ids)}"
        )
    selected_cases = tuple(
        case for case in CASES if not args.case_id or case[0] in args.case_id
    )

    import torch
    import torch_npu  # noqa: F401 - registers the NPU backend
    from vllm import LLM, SamplingParams

    started = time.perf_counter()
    llm_kwargs = dict(
        model=str(args.model.resolve()),
        tensor_parallel_size=1,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_mib * 1024 * 1024,
        seed=args.seed,
        trust_remote_code=True,
    )
    if args.no_async_scheduling:
        llm_kwargs["async_scheduling"] = False
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for _, prompt, _ in selected_cases
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    torch.npu.synchronize()

    results = []
    for (case_id, source, expected), output in zip(
        selected_cases, outputs, strict=True
    ):
        completion = output.outputs[0]
        results.append(
            {
                "id": case_id,
                "source": source,
                "expected": expected,
                "matches_expected": expected is None
                or normalize_answer(completion.text) == normalize_answer(expected),
                "token_ids": list(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "text": completion.text,
            }
        )

    report = {
        "model": str(args.model.resolve()),
        "compute_backend": os.environ.get("HIF8_COMPUTE_BACKEND", "soft_hif8"),
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "elapsed_seconds": time.perf_counter() - started,
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("HIF8_ACCURACY_JSON=" + json.dumps(report, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
