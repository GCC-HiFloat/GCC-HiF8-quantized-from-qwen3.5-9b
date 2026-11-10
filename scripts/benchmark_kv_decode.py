#!/usr/bin/env python3
"""Measure request-local KV sidecar decoding with a selected custom OPP."""

import argparse
import json
import statistics
import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend_hif8 import native
from vllm_ascend_hif8.codec import ASCEND_DECODE_TABLE
from vllm_ascend_hif8.kv_cache import pack_kv_rows, unpack_kv_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[640, 1152, 4096, 4224])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if min(args.lengths + args.batches + [args.repeats]) <= 0:
        parser.error("lengths, batches and repeats must be positive")
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(42)
    lut = torch.tensor(ASCEND_DECODE_TABLE, dtype=torch.float32)
    for length in args.lengths:
        for batch in args.batches:
            codes = torch.randint(0, 256, (batch * length, 4, 256),
                                  dtype=torch.uint8, generator=generator)
            scales = torch.rand((batch * length, 4), generator=generator) * 0.01 + 1e-5
            expected = (lut[codes.long()] * scales[:, :, None]).to(torch.bfloat16)
            packed = pack_kv_rows(codes.npu(), scales.npu())

            def decode():
                data, scale = unpack_kv_rows(packed, 256)
                return native.scaled_decode(data.reshape(-1, 256), scale.reshape(-1))

            actual = decode().reshape_as(expected).cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            for _ in range(3):
                decode()
            torch.npu.synchronize()
            timings = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                decode()
                torch.npu.synchronize()
                timings.append((time.perf_counter() - start) * 1000)
            print(json.dumps({"length": length, "batch": batch,
                              "decode_one_ms": statistics.median(timings),
                              "min_ms": min(timings), "max_ms": max(timings),
                              "repeats": args.repeats}), flush=True)


if __name__ == "__main__":
    main()
