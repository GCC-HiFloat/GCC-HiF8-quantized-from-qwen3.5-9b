#!/usr/bin/env python3
"""Create a conservative geometric blend of two static calibration artifacts."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--alpha", type=float, required=True, help="candidate weight in [0,1]")
    ap.add_argument("--linear", action="store_true", help="blend Linear activation scales")
    ap.add_argument("--kv", action="store_true", help="blend KV scales")
    args = ap.parse_args()
    if not 0.0 <= args.alpha <= 1.0 or not math.isfinite(args.alpha):
        raise ValueError("--alpha must be finite and in [0,1]")
    if not args.linear and not args.kv:
        raise ValueError("select --linear and/or --kv")
    base = json.loads(args.base.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if base.get("model", {}).get("identity_sha256") != candidate.get("model", {}).get("identity_sha256"):
        raise ValueError("base and candidate model identities differ")
    result = copy.deepcopy(base)
    result.setdefault("provenance", {})
    result["provenance"]["blend"] = {
        "base": str(args.base), "candidate": str(args.candidate), "alpha": args.alpha,
        "linear": args.linear, "kv": args.kv,
    }
    if args.linear:
        for name, entry in result["linear_groups"].items():
            a = float(base["linear_groups"][name]["scale"])
            b = float(candidate["linear_groups"][name]["scale"])
            entry["scale"] = math.exp((1.0 - args.alpha) * math.log(a) + args.alpha * math.log(b))
    if args.kv:
        for layer, entry in result["kv_cache"]["layers"].items():
            for kind in ("key", "value"):
                values = []
                for a, b in zip(base["kv_cache"]["layers"][layer][kind]["scales"], candidate["kv_cache"]["layers"][layer][kind]["scales"], strict=True):
                    values.append(math.exp((1.0 - args.alpha) * math.log(float(a)) + args.alpha * math.log(float(b))))
                entry[kind]["scales"] = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="ascii")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
