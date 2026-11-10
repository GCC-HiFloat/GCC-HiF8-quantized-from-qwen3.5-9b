#!/usr/bin/env python3
"""Run resumable, serial BF16/HIF8 throughput comparisons on one 910B."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from summarize_vllm_bench import collect, validate_pair


PROFILES = {"interactive": (512, 256), "long-output": (512, 1024),
            "long-input": (4096, 256)}


def read_manifest(root):
    return json.loads((root / "matrix.json").read_text())


def wait_idle(card):
    """Allow worker teardown to release the board before starting another run."""
    for _ in range(30):
        smi = subprocess.check_output(["npu-smi", "info"], text=True, timeout=10)
        if f"No running processes found in NPU {card}" in smi:
            return
        time.sleep(2)
    raise RuntimeError("NPU is occupied; resume after it becomes idle")


def validate_run(record, manifest, profile, batch, mode):
    expected = {
        "random-input-len": manifest["profiles"][profile][0],
        "random-output-len": manifest["profiles"][profile][1],
        "num-prompts": manifest["num_prompts"], "num-warmups": manifest["num_warmups"],
        "max-model-len": manifest["max_model_len"], "max-num-seqs": batch,
        "max-num-batched-tokens": 8192, "kv-cache-memory-bytes": manifest["kv_bytes"],
        "tensor-parallel-size": 1, "seed": 42,
    }
    if record["protocol"] != expected:
        raise ValueError(f"Run differs from matrix: {profile}/{mode}/bs{batch}")
    argv = record["command"]["argv"]
    if Path(argv[argv.index("--model") + 1]).resolve() != Path(manifest["models"][mode]):
        raise ValueError("Model differs from matrix")
    if record["command"]["npu_smi_card"] != manifest["card"]:
        raise ValueError("NPU differs from matrix")
    if manifest.get("require_no_preemptions"):
        summary = record["summary"]
        if not summary.get("preemption_tracking"):
            raise ValueError("Missing scheduler preemption counter")
        if summary.get("preemptions_including_warmup") != 0:
            raise ValueError("Cache preemptions detected; use a larger --kv-bytes in a new matrix")


def jobs(manifest):
    for profile in manifest["profiles"]:
        for batch in (4, 1):
            for trial in range(1, manifest["repetitions"] + 1):
                # Alternate pair order to reduce systematic warm-system bias.
                modes = ("bf16", "hif8") if trial % 2 else ("hif8", "bf16")
                for mode in modes:
                    yield profile, batch, trial, mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--bf16", type=Path, default=Path("/filesdir/qwen3.5-9b"))
    parser.add_argument("--hif8", type=Path, default=Path("/filesdir/qwen3.5-9b-hif8-static"))
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--kv-bytes", type=int, default=2147483648,
                        help="KV cache budget per run (default: 2 GiB)")
    parser.add_argument("--publish-to-git", action="store_true",
                        help="publish complete results through an isolated Git worktree")
    args = parser.parse_args()
    if args.num_prompts < 4 or args.repetitions < 2:
        parser.error("Use at least 4 requests and 2 repetitions")
    if args.kv_bytes <= 0:
        parser.error("--kv-bytes must be positive")
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[1]
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected = {
            "profiles": PROFILES, "num_prompts": args.num_prompts,
            "num_warmups": 4, "repetitions": args.repetitions,
            "max_model_len": 8192, "kv_bytes": args.kv_bytes, "card": args.card,
            "models": {"bf16": str(args.bf16.resolve()), "hif8": str(args.hif8.resolve())},
        }
        expected = json.loads(json.dumps(expected))
        manifest_path = root / "matrix.json"
        if manifest_path.exists():
            manifest = read_manifest(root)
            if any(manifest.get(k) != v for k, v in expected.items()):
                raise ValueError("Existing matrix has different parameters")
        else:
            source_head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            diff = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--binary"])
            manifest = dict(expected, source_head=source_head,
                            tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),
                            require_no_preemptions=True)
            manifest["distribution_versions"] = {}
            for name in ("torch", "torch-npu", "triton", "triton-ascend", "transformers", "vllm", "vllm-ascend"):
                try:
                    manifest["distribution_versions"][name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    manifest["distribution_versions"][name] = None
            manifest["source_checkouts"] = {}
            for variable, fallback in (("HIF8_VLLM_ROOT", repo / ".deps/vllm"),
                                       ("HIF8_VLLM_ASCEND_ROOT", repo / ".deps/vllm-ascend")):
                checkout = str(Path(os.environ.get(variable, fallback)).resolve())
                head = subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip()
                upstream_diff = subprocess.check_output(["git", "-C", checkout, "diff", "HEAD", "--binary"])
                manifest["source_checkouts"][checkout] = {
                    "head": head, "tracked_diff_sha256": hashlib.sha256(upstream_diff).hexdigest()}
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        run_jobs = list(jobs(manifest))
        for index, (profile, batch, trial, mode) in enumerate(run_jobs, 1):
            relative = f"{profile}/repeat-{trial}/{mode}-bs{batch}"
            out = root / relative
            status = {"state": "running", "job": relative, "job_index": index,
                      "total_jobs": len(run_jobs), "pid": os.getpid(), "updated_unix": time.time()}
            (root / "status.json").write_text(json.dumps(status, indent=2) + "\n")
            if out.exists():
                record = collect(out)
                validate_run(record, manifest, profile, batch, mode)
                print(f"[{index}/{len(run_jobs)}] complete: {relative}", flush=True)
                continue
            wait_idle(args.card)
            input_len, output_len = manifest["profiles"][profile]
            env = dict(os.environ, HIF8_SMI_CARD=str(args.card),
                       HIF8_BENCH_INPUT_LEN=str(input_len), HIF8_BENCH_OUTPUT_LEN=str(output_len),
                       HIF8_BENCH_NUM_PROMPTS=str(args.num_prompts), HIF8_BENCH_NUM_WARMUPS="4",
                       HIF8_BENCH_MAX_MODEL_LEN="8192", HIF8_BENCH_KV_BYTES=str(args.kv_bytes))
            print(f"[{index}/{len(run_jobs)}] starting: {relative}", flush=True)
            subprocess.run(["bash", str(repo / "scripts/run_vllm_bench_910b.sh"), mode,
                            str(batch), manifest["models"][mode], str(out)], env=env, check=True)
            record = collect(out)
            validate_run(record, manifest, profile, batch, mode)
            pair = root / profile / f"repeat-{trial}"
            if all((pair / f"{m}-bs{batch}" / "summary.json").is_file() for m in ("bf16", "hif8")):
                validate_pair(collect(pair / f"bf16-bs{batch}"), collect(pair / f"hif8-bs{batch}"))
        status.update(state="measured", updated_unix=time.time())
        (root / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        if args.publish_to_git:
            subprocess.run([sys.executable, str(repo / "scripts/publish_vllm_bench_matrix.py"),
                            str(root), "--push"], check=True)
            status.update(state="published", updated_unix=time.time())
            (root / "status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Matrix stopped: {error}", file=sys.stderr, flush=True)
        if len(sys.argv) > 1:
            status_path = Path(sys.argv[1]) / "status.json"
            if status_path.is_file():
                status = json.loads(status_path.read_text())
                status.update(state="failed", error=str(error), updated_unix=time.time())
                status_path.write_text(json.dumps(status, indent=2) + "\n")
        raise
