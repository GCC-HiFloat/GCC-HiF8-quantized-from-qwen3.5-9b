#!/usr/bin/env python3
"""Gate output-aware Linear scales against a production calibration."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import struct
from pathlib import Path

from vllm_ascend_hif8.calibration import file_sha256, validate_calibration_artifact


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"calibration artifact is not an object: {path}")
    validate_calibration_artifact(value)
    return value


def _family(name: str) -> str:
    if ".linear_attn." in name:
        return "linear_attn"
    if ".self_attn." in name:
        return "self_attn"
    if ".mlp." in name:
        return "mlp"
    raise ValueError(f"unknown Linear family: {name}")


def _float32(value: float) -> float:
    """Return the exact value that the checkpoint F32 scalar will store."""

    return struct.unpack("<f", struct.pack("<f", value))[0]


def select(
    base_path: Path,
    candidate_path: Path,
    *,
    minimum_improvement: float,
    alpha: float,
    scope: frozenset[str],
    name_pattern: re.Pattern[str] | None = None,
) -> dict:
    base = _load(base_path)
    candidate = _load(candidate_path)
    if base.get("model", {}).get("identity_sha256") != candidate.get(
        "model", {}
    ).get("identity_sha256"):
        raise ValueError("base and candidate calibration model identities differ")
    if set(base["linear_groups"]) != set(candidate["linear_groups"]):
        raise ValueError("base and candidate Linear groups differ")

    result = copy.deepcopy(base)
    selected_count = 0
    for name, entry in result["linear_groups"].items():
        candidate_entry = candidate["linear_groups"][name]
        search = candidate_entry.get("search_result", {})
        improvement = float(search.get("relative_output_error_improvement", 0.0))
        in_scope = _family(name) in scope
        name_matches = name_pattern is None or name_pattern.search(name) is not None
        selected = in_scope and name_matches and improvement >= minimum_improvement
        base_scale = float(entry["scale"])
        candidate_scale = float(candidate_entry["scale"])
        if selected:
            entry["scale"] = _float32(
                math.exp(
                    (1.0 - alpha) * math.log(base_scale)
                    + alpha * math.log(candidate_scale)
                )
            )
            selected_count += 1
        entry["output_aware_selection"] = {
            "selected": selected,
            "family": _family(name),
            "base_scale": base_scale,
            "candidate_scale": candidate_scale,
            "candidate_relative_output_error_improvement": improvement,
            "minimum_improvement": minimum_improvement,
            "alpha": alpha,
            "name_pattern": None if name_pattern is None else name_pattern.pattern,
            "candidate_search_result": copy.deepcopy(search),
        }

    result["search"] = copy.deepcopy(candidate.get("search", {}))
    candidate_objective = str(
        result["search"].get("linear_objective", "linear_output_mse")
    )
    result["search"].update(
        {
            "linear_objective": f"gated_{candidate_objective}",
            "output_aware_minimum_relative_improvement": minimum_improvement,
            "output_aware_blend_alpha": alpha,
            "output_aware_scope": sorted(scope),
            "output_aware_name_pattern": (
                None if name_pattern is None else name_pattern.pattern
            ),
            "output_aware_selected_count": selected_count,
            "runtime_scaling": "fixed_checkpoint_scale",
        }
    )
    result.setdefault("provenance", {})["output_aware_selection"] = {
        "base": str(base_path.resolve()),
        "base_sha256": file_sha256(base_path),
        "candidate": str(candidate_path.resolve()),
        "candidate_sha256": file_sha256(candidate_path),
        "minimum_improvement": minimum_improvement,
        "alpha": alpha,
        "scope": sorted(scope),
        "name_pattern": None if name_pattern is None else name_pattern.pattern,
        "selected_count": selected_count,
    }
    validate_calibration_artifact(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-improvement", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--scope", default="linear_attn+self_attn+mlp", help="+ separated families"
    )
    parser.add_argument(
        "--name-regex",
        help="optional regular expression that selected Linear group names must match",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not math.isfinite(args.minimum_improvement) or args.minimum_improvement < 0:
        raise ValueError("--minimum-improvement must be finite and non-negative")
    if not math.isfinite(args.alpha) or not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be finite and in [0,1]")
    scope = frozenset(args.scope.split("+"))
    allowed = {"linear_attn", "self_attn", "mlp"}
    if not scope or not scope.issubset(allowed):
        raise ValueError(f"--scope must contain only {sorted(allowed)}")
    name_pattern = re.compile(args.name_regex) if args.name_regex else None
    result = select(
        args.base,
        args.candidate,
        minimum_improvement=args.minimum_improvement,
        alpha=args.alpha,
        scope=scope,
        name_pattern=name_pattern,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(args.output)
    print(
        f"wrote {args.output} with "
        f"{result['search']['output_aware_selected_count']} selected Linear scales"
    )


if __name__ == "__main__":
    main()
