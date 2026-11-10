#!/usr/bin/env python3
"""Compare deterministic generation reports from evaluate_model_accuracy.py."""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for lhs, rhs in zip(left, right):
        if lhs != rhs:
            break
        count += 1
    return count


def normalize_answer(value: str) -> str:
    subscript_digits = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    return "".join(value.strip().translate(subscript_digits).lower().split())


def main() -> None:
    args = parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    reference_cases = {case["id"]: case for case in reference["cases"]}
    candidate_cases = {case["id"]: case for case in candidate["cases"]}
    if reference_cases.keys() != candidate_cases.keys():
        raise SystemExit("reference and candidate case IDs differ")

    comparisons = []
    total_equal_positions = 0
    total_shared_positions = 0
    exact_cases = 0
    scored_cases = 0
    reference_correct = 0
    candidate_correct = 0
    for case_id, ref in reference_cases.items():
        cand = candidate_cases[case_id]
        ref_ids = ref["token_ids"]
        cand_ids = cand["token_ids"]
        shared = min(len(ref_ids), len(cand_ids))
        equal_positions = sum(lhs == rhs for lhs, rhs in zip(ref_ids, cand_ids))
        exact = ref_ids == cand_ids
        exact_cases += exact
        expected = ref.get("expected")
        if expected is not None:
            scored_cases += 1
            reference_correct += normalize_answer(ref["text"]) == normalize_answer(expected)
            candidate_correct += normalize_answer(cand["text"]) == normalize_answer(expected)
        total_equal_positions += equal_positions
        total_shared_positions += shared
        comparisons.append(
            {
                "id": case_id,
                "reference_tokens": len(ref_ids),
                "candidate_tokens": len(cand_ids),
                "exact_token_sequence": exact,
                "common_prefix_tokens": common_prefix_length(ref_ids, cand_ids),
                "aligned_token_agreement": equal_positions / shared if shared else 1.0,
                "token_sequence_similarity": SequenceMatcher(None, ref_ids, cand_ids).ratio(),
                "text_similarity": SequenceMatcher(None, ref["text"], cand["text"]).ratio(),
                "reference_text": ref["text"],
                "candidate_text": cand["text"],
                "expected": expected,
                "reference_correct": expected is None
                or normalize_answer(ref["text"]) == normalize_answer(expected),
                "candidate_correct": expected is None
                or normalize_answer(cand["text"]) == normalize_answer(expected),
            }
        )

    report = {
        "reference_model": reference["model"],
        "candidate_model": candidate["model"],
        "case_count": len(comparisons),
        "exact_case_count": exact_cases,
        "exact_case_rate": exact_cases / len(comparisons),
        "scored_case_count": scored_cases,
        "reference_accuracy": reference_correct / scored_cases,
        "candidate_accuracy": candidate_correct / scored_cases,
        "micro_aligned_token_agreement": total_equal_positions / total_shared_positions,
        "mean_token_sequence_similarity": sum(item["token_sequence_similarity"] for item in comparisons)
        / len(comparisons),
        "mean_text_similarity": sum(item["text_similarity"] for item in comparisons) / len(comparisons),
        "cases": comparisons,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
