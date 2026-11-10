#!/usr/bin/env python3
"""Run a CLI benchmark unchanged and sample board HBM (not tensor memory)."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time


def parse_hbm(text, card):
    """Parse npu-smi's two-row 910B table, retaining its reported MB unit."""
    selected = False
    for line in text.splitlines():
        header = re.match(r"\|\s*(\d+)\s+910\w+\s*\|", line)
        if header:
            selected = int(header[1]) == card
            continue
        if selected:
            match = re.search(r"(\d+)\s*/\s*(\d+)\s*\|\s*$", line)
            if match:
                return {"used_mb": int(match[1]), "capacity_mb": int(match[2])}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a benchmark command is required after --")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    samples = []
    stop = threading.Event()

    def sample():
        try:
            result = subprocess.run(["npu-smi", "info"], text=True,
                                    capture_output=True, timeout=5)
            raw = result.stdout + result.stderr
            record = {"unix_time": time.time(), "returncode": result.returncode,
                      "hbm": parse_hbm(raw, args.card), "raw": raw}
        except (OSError, subprocess.TimeoutExpired) as error:
            record = {"unix_time": time.time(), "hbm": None, "error": str(error)}
        samples.append(record)
        with (args.out / "memory-samples.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")

    def monitor():
        while not stop.wait(1):
            sample()

    # Only a small allowlist is recorded: never dump credentials/proxy variables.
    env_keys = ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_CUSTOM_OPP_PATH", "HIF8_SOC",
                "HIF8_COMPUTE_BACKEND", "HIF8_KV_CACHE", "HIF8_STATIC_FUSED_GENERATION",
                "HIF8_STABLE_M_TILE", "VLLM_VERSION", "VLLM_PLUGINS", "PYTHONPATH",
                "VLLM_ASCEND_DISABLE_CUSTOM_OP", "VLLM_ASCEND_GDN_CUSTOM_OP")
    (args.out / "command.json").write_text(json.dumps({
        "argv": command, "environment": {k: os.environ.get(k) for k in env_keys},
        "started_unix": started, "npu_smi_card": args.card,
    }, indent=2) + "\n")
    sample()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        with (args.out / "console.log").open("w") as log:
            try:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            except OSError as error:
                log.write(str(error) + "\n")
                returncode = 127
            else:
                try:
                    returncode = child.wait()
                except KeyboardInterrupt:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    returncode = 130
    finally:
        stop.set()
        thread.join(timeout=6)
    sample()
    readings = [s["hbm"]["used_mb"] for s in samples if s.get("hbm")]
    log_text = (args.out / "console.log").read_text()
    token_match = re.search(r"Total num output tokens:\s+(\d+)", log_text)
    result_path = args.out / "throughput.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else None
    record = {
        "returncode": returncode, "wall_seconds_including_load": time.time() - started,
        "official_throughput": result,
        "output_tokens": int(token_match[1]) if token_match else None,
        "hbm_unit": "MB as reported by npu-smi",
        "hbm_scope": "whole board, sampled across load/warmup/measurement/exit",
        "hbm_baseline_mb": samples[0]["hbm"]["used_mb"] if samples[0].get("hbm") else None,
        "hbm_sampled_peak_mb": max(readings) if readings else None,
        "hbm_last_mb": samples[-1]["hbm"]["used_mb"] if samples[-1].get("hbm") else None,
        "memory_sample_count": len(samples), "valid_memory_sample_count": len(readings),
        "preemption_tracking": "HIF8_BENCH_PREEMPTION_TRACKING=1" in log_text,
        "preemptions_including_warmup": (
            log_text.count("HIF8_BENCH_PREEMPTION=1")
            if "HIF8_BENCH_PREEMPTION_TRACKING=1" in log_text else None
        ),
    }
    if result and token_match:
        record["output_tokens_per_second"] = int(token_match[1]) / result["elapsed_time"]
    (args.out / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
