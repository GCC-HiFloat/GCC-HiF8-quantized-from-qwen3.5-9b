# 性能与显存验证报告

使用固定源码的 `vllm bench throughput` 比较 Qwen3.5-9B 的 BF16 与静态 W8A8KV8。
硬件为 Ascend 910B3，环境版本见 [README](../README.md#环境要求)。

## 测试配置

| 场景 | 输入 tokens | 输出 tokens |
| --- | ---: | ---: |
| 常规问答 | 512 | 256 |
| 长输出 | 512 | 1024 |
| 长输入 | 4096 | 256 |

每组预热 4 条请求，计时 16 条，重复 3 次。吞吐取各轮中位数，HBM 取各轮整卡采样峰值的最大值。所有轮次的缓存抢占次数（含预热）均为 0。

| 参数 | 值 |
| --- | --- |
| BF16 / HIF8 checkpoint | `/filesdir/qwen3.5-9b` / `/filesdir/qwen3.5-9b-hif8-static` |
| vLLM commit | `d02df748bf9efd99022f1a062597dc3cb3808485` |
| vLLM-Ascend commit | `53b489b78560363871ba5402dd72cfd124c4b209` + 仓库补丁 |
| TP / BS 上限 | 1 / 分别设置 1、4 |
| 数据 / seed | random / 42，固定长度，`random-range-ratio=0` |
| 上下文 / token budget | 8192 / 8192 |
| KV 预算 | 2 GiB |
| 调度 | eager、同步、关闭 prefix caching |
| 输入类型 | 文本，image/video limit 均为 0 |
| Sampling | temperature=1、top_p=1、ignore_eos=True |
| HIF8 | `soft_hif8`、`HIF8_STATIC_FUSED_GENERATION=1`、`HIF8_KV_CACHE=1` |

BF16 与 HIF8 串行运行，相邻轮次交替运行顺序。每轮重新加载模型并预热，
计时不包含加载与预热；包含 prefill、decode 和调度。
重复测量使用同一组固定 seed 请求，用于观察时间波动。

## 吞吐与显存

| 输入/输出 tokens | BS 上限 | BF16 输出 token/s | HIF8 输出 token/s | HIF8/BF16 | BF16 / HIF8 HBM 峰值（MB） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/256 | 1 | 8.4305 | 4.4498 | 0.5278x | 23,121 / 18,137 |
| 512/256 | 4 | 27.9306 | 19.1375 | 0.6852x | 23,301 / 18,059 |
| 512/1024 | 1 | 8.8742 | 5.3873 | 0.6071x | 23,122 / 17,959 |
| 512/1024 | 4 | 28.0955 | 18.6130 | 0.6625x | 23,302 / 17,977 |
| 4096/256 | 1 | 9.6256 | 5.5932 | 0.5811x | 23,842 / 18,135 |
| 4096/256 | 4 | 32.5737 | 16.6103 | 0.5099x | 24,723 / 18,158 |

输出吞吐为实际输出 token 数除以计时时长。HIF8/BF16 为两者吞吐中位数之比。
HBM 为 `npu-smi` 整卡 MB，约每秒采样，覆盖加载、预热、计时及退出；
包含驱动、模型、KV 和 workspace，采样可能漏掉瞬时峰值。
各轮启动前整卡 HBM 为 3,405～3,407 MB，未从峰值中扣除。
权重加载日志和逐轮显存基线见 [原始记录](reports/vllm-bench/evidence.json)。

## 重复测量波动

| 输入/输出 | BS | 模式 | 最低 / 最高输出 token/s | 吞吐样本变异系数 |
| --- | ---: | --- | ---: | ---: |
| 512/256 | 1 | BF16 | 6.9655 / 9.2733 | 14.20% |
| 512/256 | 1 | HIF8 | 4.3572 / 5.6825 | 15.32% |
| 512/256 | 4 | BF16 | 24.7678 / 28.5831 | 7.53% |
| 512/256 | 4 | HIF8 | 18.4085 / 20.6546 | 5.91% |
| 512/1024 | 1 | BF16 | 8.7114 / 9.2920 | 3.34% |
| 512/1024 | 1 | HIF8 | 5.2601 / 5.4325 | 1.67% |
| 512/1024 | 4 | BF16 | 26.6657 / 36.5514 | 17.55% |
| 512/1024 | 4 | HIF8 | 17.9764 / 18.9632 | 2.70% |
| 4096/256 | 1 | BF16 | 8.5017 / 9.8603 | 7.78% |
| 4096/256 | 1 | HIF8 | 5.4094 / 5.6861 | 2.53% |
| 4096/256 | 4 | BF16 | 29.3586 / 34.3226 | 7.85% |
| 4096/256 | 4 | HIF8 | 15.1620 / 18.7366 | 10.68% |

变异系数为三轮吞吐的样本标准差除以均值。使用三轮中位数比较，
保留最低值和最高值供判断波动。

16 条请求的固定长度测试适合比较这些负载，不能据此推断线上容量或延迟百分位。
在线 TTFT、TPOT 和延迟分布应使用 `vllm bench serve` 测量。

## 复现与原始记录

完整结果：`/filesdir/vllm-ascend-hif8/benchmarks/vllm-bench-kv-vector-2g-16x3`。
运行命令、续跑和各组输出目录见 [吞吐测试指南](benchmarking.md)。

| 文件 | 内容 |
| --- | --- |
| [aggregate.json](reports/vllm-bench/aggregate.json) | 三轮吞吐、波动和 HBM 汇总 |
| [evidence.json](reports/vllm-bench/evidence.json) | 每轮命令、指标、日志摘要、显存采样证据和 SHA-256 |
| [matrix.json](reports/vllm-bench/matrix.json) | 模型路径、负载、缓存预算和源码标识 |
| [provenance.json](reports/vllm-bench/provenance.json) | Python、依赖版本和上游源码身份 |

HIF8 使用固定 activation/KV scale，KV 解码采用向量查表和 FP32 scale 乘法。
数据集精度单独见 [ACC 验证报告](validation-status.md)。
