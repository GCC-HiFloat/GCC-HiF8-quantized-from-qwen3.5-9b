# 吞吐测试

使用真实 BF16 和静态 W8A8KV8 checkpoint，通过固定源码的
`vllm bench throughput` 比较离线吞吐与显存。

## 测试负载

| 场景 | 输入 tokens | 输出 tokens |
| --- | ---: | ---: |
| 常规问答 | 512 | 256 |
| 长输出 | 512 | 1024 |
| 长输入 | 4096 | 256 |

每个场景分别测 BS=1、4 和 BF16/HIF8。每组预热 4 条、计时 16 条请求，
重复 3 次，共 36 组。使用相同 seed，交替运行 BF16/HIF8，报告吞吐中位数、
最小值、最大值、变异系数和 HBM 采样峰值。

测试使用 eager、同步调度，关闭 prefix caching；TP=1，KV 预算默认 2 GiB，
上下文上限和 token budget 均为 8192。HIF8 的 activation 和 KV scale 固定。

## 运行

选择空闲设备，按 [安装说明](../README.md#安装与部署)准备环境。
修改算子后先按 [开发指南](development.md#构建与测试)重新构建，并指定实际 OPP。

```bash
conda activate hifi
export HIF8_VLLM_ROOT="$PWD/.deps/vllm"
export HIF8_VLLM_ASCEND_ROOT="$PWD/.deps/vllm-ascend"
export ASCEND_RT_VISIBLE_DEVICES=0
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export HIF8_OPP_BUILD_DIR="$PWD/csrc/opp/build/ascend910b"
export ASCEND_CUSTOM_OPP_PATH="$HIF8_OPP_BUILD_DIR/package/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors/vllm_ascend_hif8"
python scripts/run_vllm_bench_matrix.py benchmarks/my-length-matrix \
  --card 6 --num-prompts 16 --repetitions 3 --kv-bytes 2147483648
```

`--card` 为 npu-smi 显示的物理卡号。两种模型串行运行，每轮重新加载并预热。
模型加载与预热不计入吞吐时间；计时包含 prefill、decode 和调度。

调度器保持 Ascend 的调度策略，额外记录缓存抢占次数（含预热），结果写入
`summary.json` 的 `preemptions_including_warmup`。新矩阵要求抢占次数为 0；
如发生抢占，增大 `--kv-bytes`，在新目录重测 BF16/HIF8 两组。
4096-token 输入、BS=4 时，1 GiB 预算会触发重复 prefill。

### KV 解码微基准

在空闲 NPU 上，设置待测算子的 `ASCEND_CUSTOM_OPP_PATH` 后运行：

```bash
PYTHONPATH=src python scripts/benchmark_kv_decode.py \
  --batches 1 4 --lengths 640 1152 4096 4224 --repeats 10
```

输入为 `[batch × length, 4, 256]` 的 HiFloat8 KV，附带固定 FP32 scale。
脚本先与 CPU 解码参考结果逐元素比较，再报告单个 K 或 V 张量的解码耗时。
计时包含 sidecar 解包、连续化、解码算子和设备同步，不包含分页 gather 或 attention。

### 单组测试

```bash
HIF8_SMI_CARD=6 HIF8_BENCH_INPUT_LEN=512 HIF8_BENCH_OUTPUT_LEN=256 \
  HIF8_BENCH_NUM_PROMPTS=16 HIF8_BENCH_MAX_MODEL_LEN=8192 \
  HIF8_BENCH_KV_BYTES=2147483648 \
  bash scripts/run_vllm_bench_910b.sh hif8 4 \
    /filesdir/qwen3.5-9b-hif8-static benchmarks/my-single-run
```

| 环境变量 | 默认值 |
| --- | ---: |
| `HIF8_BENCH_INPUT_LEN` | 128 |
| `HIF8_BENCH_OUTPUT_LEN` | 32 |
| `HIF8_BENCH_NUM_PROMPTS` | 8 |
| `HIF8_BENCH_NUM_WARMUPS` | 4 |
| `HIF8_BENCH_MAX_MODEL_LEN` | 512 |
| `HIF8_BENCH_KV_BYTES` | 1073741824 |

单组脚本的默认值用于短负载检查。输出目录必须是新目录。

## 续跑与结果

重新执行相同的 matrix 命令可续跑。脚本校验已完成项的配置、请求数、实际输出
token 数和显存采样后跳过。失败或未完成项会停止队列；将该项目录移出结果树，
排除错误后再执行相同命令。`matrix.json` 固定运行参数，`status.json` 记录进度。

每组结果位于 `场景/repeat-N/模式-bsB/`：

| 文件 | 内容 |
| --- | --- |
| `command.json` | 命令、运行环境和设备编号 |
| `throughput.json` | vLLM bench 原始吞吐结果 |
| `console.log` | 加载、静态 KV 路径、生成与抢占日志 |
| `memory-samples.jsonl` | 每秒整卡 HBM 采样及 npu-smi 原文 |
| `summary.json` | 输出吞吐、显存峰值、抢占次数和退出状态 |

完整矩阵通过校验后可生成报告：

```bash
python scripts/publish_vllm_bench_matrix.py benchmarks/my-length-matrix
```

发布器在独立 Git worktree 中更新 README 性能表、`docs/vllm-bench-report.md`
和 `docs/reports/vllm-bench/` 的 JSON 证据并提交，
不修改当前工作区。加 `--push` 可推送到 `origin/w8a8kv8`；远端出现分叉时停止，
不执行强推。提交和 worktree 路径记录在结果目录的 `publication.json`。

需要测完自动发布时，为 matrix 命令添加 `--publish-to-git`。
运行前先提交脚本改动；报告以 `matrix.json` 中记录的源码提交为基线。
发布后可在干净工作区使用 `git pull --ff-only origin w8a8kv8` 同步。

完整吞吐、显存和三轮波动见 [性能与显存验证报告](vllm-bench-report.md)。

16 条请求用于多长度初步对比。线上容量与 TTFT、TPOT、延迟百分位应结合实际
请求分布使用 `vllm bench serve` 单独测量。
