# 精度验证报告

使用 lm-evaluation-harness 对 Qwen3.5-9B 的 BF16 和静态 W8A8KV8 模型进行配对测评。
硬件为 Ascend 910B3，运行配置为文本 TP=1、batch=4、eager、同步调度。
软件环境见 [README](../README.md#环境要求)。

## 测评配置

BF16 与 HIF8 使用相同的数据、任务模板、few-shot、seed、batch 和调度参数。
常规任务采用 `acc,none`，GSM8K 采用 `exact_match,strict-match`。
任务配置位于 [`local_tasks/`](../local_tasks/)。

整体精度按样本数加权，验收条件为 HIF8 相对 BF16 的下降不超过 0.5 个百分点：

```text
overall_acc = sum(task_acc * task_samples) / sum(task_samples)
loss_pp = max(0, BF16_overall_acc - HIF8_overall_acc) * 100
```

## 测评结果

| 数据集 | 样本数 | BF16 | HIF8 | HIF8−BF16（百分点） |
| --- | ---: | ---: | ---: | ---: |
| ARC-Easy | 2,376 | 0.799242 | 0.816498 | +1.7256 |
| ARC-Challenge | 1,172 | 0.526451 | 0.545222 | +1.8771 |
| PIQA | 1,838 | 0.792709 | 0.789989 | -0.2720 |
| WinoGrande | 1,267 | 0.509866 | 0.497238 | -1.2628 |
| BoolQ | 3,270 | 0.872783 | 0.874924 | +0.2141 |
| HellaSwag | 10,042 | 0.581060 | 0.581259 | +0.0199 |
| SciQ | 1,000 | 0.962000 | 0.973000 | +1.1000 |
| GSM8K | 1,319 | 0.843821 | 0.863533 | +1.9712 |
| LAMBADA OpenAI | 5,153 | 0.721133 | 0.712401 | -0.8733 |
| MMLU | 14,042 | 0.471300 | 0.474078 | +0.2777 |
| **样本加权整体** | **41,479** | **0.620000** | **0.621977** | **+0.1977** |
| **十项宏平均** | **10 项** | **0.708037** | **0.712814** | **+0.4777** |

样本加权整体精度损失为 0，满足验收条件。WinoGrande 和 LAMBADA 的单项下降
分别为 1.2628 和 0.8733 个百分点。

## 运行测评

准备本地数据和 lm-eval 环境后，依次执行：

```bash
./scripts/run_ten_dataset_bf16_batch4.sh
./scripts/run_ten_dataset_hif8_batch4.sh
python scripts/summarize_ten_hif8_compare.py \
  --bf16-root benchmarks/qwen35-ten-bf16-batch4 \
  --hif8-root benchmarks/qwen35-ten-hif8-batch4 \
  --output benchmarks/ten-hif8-batch4-vs-bf16.md
```

脚本按任务输出结果，并跳过已有结果文件的任务。续跑前检查结果中的
`n-samples`、模型路径和参数，确认跳过项为完整测评。
BF16/HIF8 的响应缓存需按模型、精度、batch、任务隔离。
BS1 或新模型必须建立独立结果目录，并与同配置 BF16 配对。

| 输出 | 位置 |
| --- | --- |
| BF16 结果与逐题输出 | `benchmarks/qwen35-ten-bf16-batch4/` |
| HIF8 结果与逐题输出 | `benchmarks/qwen35-ten-hif8-batch4/` |
| 对比报告 | `benchmarks/ten-hif8-batch4-vs-bf16.md` |

算子与格式测试见 [开发指南](development.md)，吞吐和显存见 [性能测试](vllm-bench-report.md)。
