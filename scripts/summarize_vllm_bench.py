#!/usr/bin/env python3
"""Publish compact, hash-linked evidence for the four paired bench runs."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark_protocol(command):
    """Read the fixed-length workload and engine limits from the recorded CLI."""
    argv = command["argv"]
    names = ("random-input-len", "random-output-len", "num-prompts", "num-warmups",
             "max-model-len", "max-num-seqs", "max-num-batched-tokens",
             "kv-cache-memory-bytes", "tensor-parallel-size", "seed")
    result = {}
    for name in names:
        flag = "--" + name
        if argv.count(flag) != 1:
            raise ValueError(f"Expected one {flag}")
        result[name] = int(argv[argv.index(flag) + 1])
    for name in names:
        if result[name] < (0 if name == "seed" else 1):
            raise ValueError(f"Invalid {name}")
    for flag, expected in (("--dataset-name", "random"), ("--random-range-ratio", "0"),
                           ("--dtype", "bfloat16"), ("--backend", "vllm")):
        if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != expected:
            raise ValueError(f"Invalid {flag}")
    for flag in ("--enforce-eager", "--no-async-scheduling", "--no-enable-prefix-caching"):
        if flag not in argv:
            raise ValueError(f"Missing {flag}")
    if result["random-input-len"] + result["random-output-len"] > result["max-model-len"]:
        raise ValueError("Input + output exceeds model context")
    return result


def collect(run):
    summary = json.loads((run / "summary.json").read_text())
    command = json.loads((run / "command.json").read_text())
    protocol = benchmark_protocol(command)
    bench = summary["official_throughput"]
    requests = protocol["num-prompts"]
    expected_output = requests * protocol["random-output-len"]
    expected_total = requests * (protocol["random-input-len"] + protocol["random-output-len"])
    if summary["returncode"] != 0 or not bench or bench["num_requests"] != requests:
        raise ValueError(f"Incomplete benchmark: {run}")
    if (summary["output_tokens"] != expected_output or bench["total_num_tokens"] != expected_total
            or not summary["valid_memory_sample_count"]):
        raise ValueError(f"Missing fixed-length tokens/memory: {run}")
    if bench != json.loads((run / "throughput.json").read_text()):
        raise ValueError(f"Throughput summary differs from CLI output: {run}")
    elapsed = bench["elapsed_time"]
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError(f"Invalid benchmark duration: {run}")
    if not math.isclose(summary["output_tokens_per_second"], expected_output / elapsed, rel_tol=1e-9):
        raise ValueError(f"Invalid output throughput: {run}")
    if summary["hbm_baseline_mb"] is None:
        raise ValueError(f"Missing baseline: {run}")
    samples = [json.loads(line) for line in (run / "memory-samples.jsonl").read_text().splitlines()]
    valid = [s for s in samples if s.get("hbm")]
    if not valid or len(valid) != summary["valid_memory_sample_count"]:
        raise ValueError(f"Invalid memory samples: {run}")
    peak = max(valid, key=lambda s: s["hbm"]["used_mb"])
    if peak["hbm"]["used_mb"] != summary["hbm_sampled_peak_mb"]:
        raise ValueError(f"Invalid memory peak: {run}")
    log = (run / "console.log").read_text()
    if summary.get("preemption_tracking"):
        if ("HIF8_BENCH_PREEMPTION_TRACKING=1" not in log or
                summary.get("preemptions_including_warmup") != log.count("HIF8_BENCH_PREEMPTION=1")):
            raise ValueError(f"Invalid scheduler preemption count: {run}")
    patterns = ("non-default args", "Initializing a V1", "Loading model weights took",
                "Loaded checkpoint-static", "KV quantization path", "static_quant",
                "Throughput:", "Total num", "Warming up")
    excerpts = [line for line in log.splitlines() if any(p in line for p in patterns)]
    weight = re.search(r"Loading model weights took ([0-9.]+) GB", log)
    argv = command["argv"]
    model = Path(argv[argv.index("--model") + 1])
    fingerprints = {name: sha256(model / name) for name in (
        "config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
        "hif8_metadata.json", "quant_model_description.json") if (model / name).is_file()}
    return {
        "summary": summary, "command": command, "protocol": protocol,
        "utc_started": datetime.fromtimestamp(command["started_unix"], timezone.utc).isoformat(),
        "weight_load_log_value_gb": float(weight[1]) if weight else None,
        "source_directory": str(run),
        "model_metadata_sha256": fingerprints,
        "model_fingerprint_scope": "metadata/tokenizer only; weight shards not hashed by this tool",
        "sha256": {name: sha256(run / name) for name in (
            "summary.json", "command.json", "throughput.json", "console.log", "memory-samples.jsonl")},
        "log_excerpts": excerpts,
        "memory_evidence": {"first": samples[0], "sampled_peak": peak, "last": samples[-1]},
    }


def validate_pair(bf16, hif8):
    """Require matching workloads and checkpoint-static HIF8 execution."""
    if bf16["protocol"] != hif8["protocol"]:
        raise ValueError("BF16/HIF8 benchmark protocols differ")
    env = hif8["command"]["environment"]
    if (env.get("HIF8_STATIC_FUSED_GENERATION") != "1"
            or env.get("HIF8_KV_CACHE") != "1"
            or env.get("HIF8_COMPUTE_BACKEND") != "soft_hif8"):
        raise ValueError("Missing static HIF8 runtime settings")
    log = "\n".join(hif8["log_excerpts"])
    if "Loaded checkpoint-static" not in log or "KV writes use static_quant" not in log:
        raise ValueError("Missing static KV runtime evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = {f"{mode}-bs{bs}": collect(args.root / f"{mode}-bs{bs}")
               for bs in (1, 4) for mode in ("bf16", "hif8")}
    for bs in (1, 4):
        validate_pair(records[f"bf16-bs{bs}"], records[f"hif8-bs{bs}"])
    provenance = {"publisher_python": sys.version, "distribution_versions": {}, "source_checkouts": {}}
    for name in ("torch", "torch-npu", "triton", "triton-ascend", "transformers", "vllm", "vllm-ascend"):
        try:
            provenance["distribution_versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            provenance["distribution_versions"][name] = None
    search = records["bf16-bs1"]["command"]["environment"]["PYTHONPATH"].split(":")
    search.append(str(Path(__file__).resolve().parents[1]))
    for entry in search:
        path = Path(entry)
        if not (path / ".git").exists():
            continue
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "-C", str(path), "diff", "HEAD", "--binary"])
        provenance["source_checkouts"][str(path)] = {"head": head, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "evidence.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    lines = ["# vLLM bench throughput", "",
             "BF16 与静态 W8A8KV8 的 BS=1/4 离线吞吐及显存占用。",
             "HBM 是 npu-smi 整卡报告的 MB；采样覆盖加载、预热、测试和退出。",
             "显存峰值取整卡采样最大值。", "",
             "| BS 上限 | 模式 | 计时/s | requests/s | 输入+输出 tokens/s | 输出 tokens/s | 基线 HBM/MB | 采样峰值 HBM/MB |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, record in records.items():
        s = record["summary"]
        b = s["official_throughput"]
        mode, bs = name.split("-bs")
        lines.append(f"| {bs} | {mode.upper()} | {b['elapsed_time']:.4f} | {b['requests_per_second']:.4f} | "
                     f"{b['tokens_per_second']:.4f} | {s['output_tokens_per_second']:.4f} | "
                     f"{s['hbm_baseline_mb']} | {s['hbm_sampled_peak_mb']} |")
    lines += ["", "| BS 上限 | HIF8/BF16 输出吞吐 | HBM 采样峰值差（BF16−HIF8）/MB |",
              "| --- | ---: | ---: |"]
    for bs in (1, 4):
        b = records[f"bf16-bs{bs}"]["summary"]
        h = records[f"hif8-bs{bs}"]["summary"]
        lines.append(f"| {bs} | {h['output_tokens_per_second']/b['output_tokens_per_second']:.4f}x | "
                     f"{b['hbm_sampled_peak_mb']-h['hbm_sampled_peak_mb']} |")
    lines += ["", "原始指标：[evidence.json](evidence.json)。测试配置与复现命令见 [bench 报告](../../vllm-bench-report.md)。",
              "", "测试为单轮离线负载；在线延迟需使用 serving 测试单独测量。"]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
