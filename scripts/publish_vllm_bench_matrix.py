#!/usr/bin/env python3
"""Validate a complete throughput matrix and publish its reports."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

from run_vllm_bench_matrix import read_manifest, validate_run
from summarize_vllm_bench import collect, validate_pair


def aggregate(root):
    manifest = read_manifest(root)
    if manifest["repetitions"] < 2:
        raise ValueError("At least two repetitions are required")
    evidence, rows = {}, []
    for profile, (input_len, output_len) in manifest["profiles"].items():
        for batch in (1, 4):
            groups = {"bf16": [], "hif8": []}
            for trial in range(1, manifest["repetitions"] + 1):
                pair = {}
                for mode in groups:
                    name = f"{profile}/repeat-{trial}/{mode}-bs{batch}"
                    record = collect(root / name)
                    validate_run(record, manifest, profile, batch, mode)
                    evidence[name] = pair[mode] = record
                    groups[mode].append(record)
                validate_pair(pair["bf16"], pair["hif8"])
            row = {"profile": profile, "input_len": input_len, "output_len": output_len, "batch": batch}
            for mode, records in groups.items():
                speeds = [r["summary"]["output_tokens_per_second"] for r in records]
                row[mode] = {
                    "median_output_tokens_per_second": statistics.median(speeds),
                    "min_output_tokens_per_second": min(speeds),
                    "max_output_tokens_per_second": max(speeds),
                    "sample_cv_percent": statistics.stdev(speeds) / statistics.mean(speeds) * 100,
                    "median_requests_per_second": statistics.median(
                        r["summary"]["official_throughput"]["requests_per_second"] for r in records),
                    "peak_hbm_mb": max(r["summary"]["hbm_sampled_peak_mb"] for r in records),
                    "baseline_hbm_mb": [r["summary"]["hbm_baseline_mb"] for r in records],
                }
            row["hif8_bf16_ratio"] = (row["hif8"]["median_output_tokens_per_second"] /
                                      row["bf16"]["median_output_tokens_per_second"])
            rows.append(row)
    return manifest, evidence, rows


def result_table(rows):
    lines = ["| 输入/输出 tokens | BS 上限 | BF16 输出 token/s | HIF8 输出 token/s | HIF8/BF16 | BF16 / HIF8 HBM 峰值（MB） |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        b, h = row["bf16"], row["hif8"]
        lines.append(f"| {row['input_len']}/{row['output_len']} | {row['batch']} | "
                     f"{b['median_output_tokens_per_second']:.4f} | {h['median_output_tokens_per_second']:.4f} | "
                     f"{row['hif8_bf16_ratio']:.4f}x | {b['peak_hbm_mb']:,} / {h['peak_hbm_mb']:,} |")
    return "\n".join(lines)


def write_reports(repo, root, manifest, evidence, rows):
    report_rel = Path("docs/reports/vllm-bench")
    report_dir = repo / report_rel
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    (report_dir / "matrix.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (report_dir / "aggregate.json").write_text(json.dumps(rows, indent=2) + "\n")
    versions = {}
    for name in ("torch", "torch-npu", "triton", "triton-ascend", "transformers", "vllm", "vllm-ascend"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    (report_dir / "provenance.json").write_text(json.dumps({
        "publisher_python": sys.version, "distribution_versions": versions,
        "runner_distribution_versions": manifest.get("distribution_versions", {}),
        "source_checkouts": manifest.get("source_checkouts", {}),
        "source_head": manifest["source_head"], "tracked_diff_sha256": manifest["tracked_diff_sha256"],
    }, indent=2) + "\n")
    table = result_table(rows)
    kv_gib = manifest["kv_bytes"] / 1024**3
    settings = (f"每组预热 {manifest['num_warmups']} 条请求，计时 {manifest['num_prompts']} 条，"
                f"重复 {manifest['repetitions']} 次。吞吐取各轮中位数，HBM 取各轮整卡采样峰值的最大值。")
    if manifest.get("require_no_preemptions"):
        settings += "所有轮次的缓存抢占次数（含预热）均为 0。"
    detailed = ["| 输入/输出 | BS | 模式 | 最低 / 最高输出 token/s | 吞吐样本变异系数 |",
                "| --- | ---: | --- | ---: | ---: |"]
    for row in rows:
        for mode in ("bf16", "hif8"):
            s = row[mode]
            detailed.append(f"| {row['input_len']}/{row['output_len']} | {row['batch']} | {mode.upper()} | "
                            f"{s['min_output_tokens_per_second']:.4f} / {s['max_output_tokens_per_second']:.4f} | "
                            f"{s['sample_cv_percent']:.2f}% |")
    variation = "\n".join(detailed)
    baselines = [value for row in rows for mode in ("bf16", "hif8")
                 for value in row[mode]["baseline_hbm_mb"]]
    readme_section = ("## 性能\n\n"
                      "`vllm bench throughput`，Qwen3.5-9B、Ascend 910B3、TP=1、eager、同步调度。\n"
                      f"静态 W8A8KV8 使用固定 activation/KV scale。KV 预算 {kv_gib:g} GiB，上下文上限 8192。\n\n"
                      + settings + "\n\n" + table + "\n\n"
                      "HBM 包含驱动、模型、KV 和 workspace。结果对应表中的固定长度离线负载，\n"
                      "在线延迟需单独测量。配置和结果见 [性能测试](docs/vllm-bench-report.md)。\n\n")
    readme = (repo / "README.md").read_text()
    begin = readme.index("## 性能\n")
    end = readme.index("## 代码结构\n", begin)
    (repo / "README.md").write_text(readme[:begin] + readme_section + readme[end:])
    report = f"""# 性能与显存验证报告

使用固定源码的 `vllm bench throughput` 比较 Qwen3.5-9B 的 BF16 与静态 W8A8KV8。
硬件为 Ascend 910B3，环境版本见 [README](../README.md#环境要求)。

## 测试配置

| 场景 | 输入 tokens | 输出 tokens |
| --- | ---: | ---: |
| 常规问答 | 512 | 256 |
| 长输出 | 512 | 1024 |
| 长输入 | 4096 | 256 |

{settings}

| 参数 | 值 |
| --- | --- |
| BF16 / HIF8 checkpoint | `{manifest['models']['bf16']}` / `{manifest['models']['hif8']}` |
| vLLM commit | `d02df748bf9efd99022f1a062597dc3cb3808485` |
| vLLM-Ascend commit | `53b489b78560363871ba5402dd72cfd124c4b209` + 仓库补丁 |
| TP / BS 上限 | 1 / 分别设置 1、4 |
| 数据 / seed | random / 42，固定长度，`random-range-ratio=0` |
| 上下文 / token budget | 8192 / 8192 |
| KV 预算 | {kv_gib:g} GiB |
| 调度 | eager、同步、关闭 prefix caching |
| 输入类型 | 文本，image/video limit 均为 0 |
| Sampling | temperature=1、top_p=1、ignore_eos=True |
| HIF8 | `soft_hif8`、`HIF8_STATIC_FUSED_GENERATION=1`、`HIF8_KV_CACHE=1` |

BF16 与 HIF8 串行运行，相邻轮次交替运行顺序。每轮重新加载模型并预热，
计时不包含加载与预热；包含 prefill、decode 和调度。
重复测量使用同一组固定 seed 请求，用于观察时间波动。

## 吞吐与显存

{table}

输出吞吐为实际输出 token 数除以计时时长。HIF8/BF16 为两者吞吐中位数之比。
HBM 为 `npu-smi` 整卡 MB，约每秒采样，覆盖加载、预热、计时及退出；
包含驱动、模型、KV 和 workspace，采样可能漏掉瞬时峰值。
各轮启动前整卡 HBM 为 {min(baselines):,}～{max(baselines):,} MB，未从峰值中扣除。
权重加载日志和逐轮显存基线见 [原始记录](reports/vllm-bench/evidence.json)。

## 重复测量波动

{variation}

变异系数为三轮吞吐的样本标准差除以均值。使用三轮中位数比较，
保留最低值和最高值供判断波动。

{manifest['num_prompts']} 条请求的固定长度测试适合比较这些负载，不能据此推断线上容量或延迟百分位。
在线 TTFT、TPOT 和延迟分布应使用 `vllm bench serve` 测量。

## 复现与原始记录

完整结果：`{root}`。
运行命令、续跑和各组输出目录见 [吞吐测试指南](benchmarking.md)。

| 文件 | 内容 |
| --- | --- |
| [aggregate.json](reports/vllm-bench/aggregate.json) | 三轮吞吐、波动和 HBM 汇总 |
| [evidence.json](reports/vllm-bench/evidence.json) | 每轮命令、指标、日志摘要、显存采样证据和 SHA-256 |
| [matrix.json](reports/vllm-bench/matrix.json) | 模型路径、负载、缓存预算和源码标识 |
| [provenance.json](reports/vllm-bench/provenance.json) | Python、依赖版本和上游源码身份 |

HIF8 使用固定 activation/KV scale，KV 解码采用向量查表和 FP32 scale 乘法。
数据集精度单独见 [ACC 验证报告](validation-status.md)。
"""
    (repo / "docs/vllm-bench-report.md").write_text(report)
    return ["README.md", "docs/vllm-bench-report.md", str(report_rel)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[1]
    manifest, evidence, rows = aggregate(root)
    source = manifest["source_head"]
    # A detached worktree keeps publication separate from ongoing local edits.
    worktree = Path(tempfile.mkdtemp(prefix="hif8-bench-publish-"))
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", str(worktree), source], check=True)
    paths = write_reports(worktree, root, manifest, evidence, rows)
    subprocess.run(["git", "-C", str(worktree), "diff", "--check"], check=True)
    subprocess.run(["git", "-C", str(worktree), "add", "--", *paths], check=True)
    subprocess.run(["git", "-C", str(worktree), "commit", "-m", "docs: report repeated multi-length 910B throughput"], check=True)
    result = {"worktree": str(worktree), "commit": subprocess.check_output(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True).strip(), "pushed": False}
    (root / "publication.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.push:
        # No force push: a concurrently updated remote requires manual integration.
        subprocess.run(["git", "-C", str(worktree), "push", "origin", "HEAD:refs/heads/w8a8kv8"], check=True)
        remote = subprocess.check_output(["git", "-C", str(worktree), "ls-remote", "origin",
                                          "refs/heads/w8a8kv8"], text=True).split()[0]
        if remote != result["commit"]:
            raise RuntimeError("Remote branch changed after publication")
        result["pushed"] = True
        (root / "publication.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
