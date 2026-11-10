# Ascend 910B 静态 W8A8KV8 HiFloat8

vLLM-Ascend 的 HiFloat8 量化插件，用于 Qwen3.5-9B 在 Ascend 910B 上的文本推理。
支持单 NPU、TP=1、静态 activation/KV scaling 和 OpenAI-compatible API。

## 功能

| 部分 | 实现 |
| --- | --- |
| W8 | HiFloat8 `uint8` 权重，固定 FP32 per-output-channel scale |
| A8 | 从 checkpoint 加载固定 FP32 `input_scale` |
| KV8 | 全注意力 K/V 使用固定 per-layer/per-head scale，页面存储 HiFloat8 字节和 FP32 scale |
| 计算 | `soft_hif8` 在线解码局部 tile，使用 BF16 Cube |

推理时 activation 和 KV 的 scale 不随请求或 batch 变化。GDN recurrent state、
norm、LM head 等使用模型的浮点格式；注意力计算前解码当前请求需要的 KV 数据。

部署启用 `HIF8_STATIC_FUSED_GENERATION=1`。模型投影走
`hif8_static_quant → soft_hif8_gemm`，小尺寸矩阵可使用静态融合算子。
推荐使用 batch/microbatch=1 或 4 进行精度复现。

支持 eager V1 和同步调度。KV 适配器不支持异步调度、ACL graph、DCP/PCP、
滑动窗口、attention sinks 或非 dense attention。

## 环境要求

| 组件 | 版本 |
| --- | --- |
| NPU | Ascend 910B3/A2 |
| CANN / npu-smi | 9.0.0 / 25.2.3 |
| Python | 3.11.15 |
| PyTorch / torch-npu | 2.10.0+cpu / 2.10.0 |
| vLLM 源码 | `d02df748bf9efd99022f1a062597dc3cb3808485` |
| vLLM-Ascend 源码 | `53b489b78560363871ba5402dd72cfd124c4b209` |
| Triton 运行时 / triton-ascend distribution | 3.2.0 / 3.2.1 |

vLLM 使用固定源码提交，源码版本为 `0.23.1rc1.dev1451+gd02df748b`。
`SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0` 设置构建元数据，
`VLLM_VERSION=0.26.0` 选择 vLLM-Ascend 的 API 兼容分支。
安装方式为源码安装，版本与补丁说明见 [上游集成](patches/README.md)。

## 安装与部署

准备 CANN、Python、torch、torch-npu 和 Triton Ascend 后运行：

```bash
conda activate hifi
./scripts/install_vllm_ascend_910b.sh
HIF8_MAX_NUM_SEQS=4 ./scripts/deploy_910b_hif8.sh /filesdir/qwen3.5-9b-hif8-static
```

安装器将固定上游源码放入 `.deps/vllm` 和 `.deps/vllm-ascend`，应用仓库补丁，
以 `--no-deps --no-build-isolation` 安装。部署脚本检查 checkpoint、构建 OPP
和 Python 扩展，然后启动服务。

checkpoint 需要包含 `hif8_metadata.json`、`quant_model_description.json`、
`config.json`、`model.safetensors.index.json` 及索引列出的 shards。

默认监听 `0.0.0.0:8000`，上下文上限 16,384，调度并发上限 16，
`max_num_batched_tokens=8192`，KV 预算 8 GiB。示例将并发上限设置为 4。
共享网络部署时使用 `--api-key`；仅本机访问可设置 `--host 127.0.0.1`。

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-9b-hif8-static","messages":[{"role":"user","content":"用一句话介绍你自己。"}],"max_tokens":64,"temperature":0}'
```

### 部署选项

```bash
./scripts/deploy_910b_hif8.sh --detach /filesdir/qwen3.5-9b-hif8-static
./scripts/deploy_910b_hif8.sh --check-only --no-build-opp \
  --no-build-extension --no-install-plugin /filesdir/qwen3.5-9b-hif8-static
```

`--check-only` 可按需构建缺失组件；加上三个 `--no-*` 参数可执行纯检查。
已有源码缓存时可使用 `./scripts/install_vllm_ascend_910b.sh --offline`。
完整参数见各脚本的 `--help`。

| 环境变量 | 用途 |
| --- | --- |
| `HIF8_PYTHON` | Python 可执行文件 |
| `ASCEND_HOME_PATH` | CANN 安装目录 |
| `HIF8_VLLM_ROOT` / `HIF8_VLLM_ASCEND_ROOT` | 固定提交的源码目录 |
| `HIF8_VLLM_REPO_URL` / `HIF8_VLLM_ASCEND_REPO_URL` | Git 镜像地址 |
| `ASCEND_RT_VISIBLE_DEVICES` | 逻辑 NPU 编号，默认 `0` |
| `HIF8_MAX_NUM_SEQS` | 调度并发上限 |
| `HIF8_KV_CACHE_MEMORY_BYTES` | KV cache 预算 |

## 模型导出

使用模型对应的校准文件导出静态 checkpoint：

```bash
PYTHONPATH=src python -m vllm_ascend_hif8.exporter \
  /filesdir/qwen3.5-9b /filesdir/qwen3.5-9b-hif8-static-new \
  --calibration calibration/qwen35-wikitext2/qwen35-9b-wikitext2-static-calibration.json
PYTHONPATH=src python scripts/verify_export.py \
  /filesdir/qwen3.5-9b /filesdir/qwen3.5-9b-hif8-static-new
```

校准方法见 [校准说明](calibration/qwen35-wikitext2/README.md)，
张量、融合层映射及 KV 布局见 [checkpoint 格式](docs/checkpoint-format.md)。

## 精度

Qwen3.5-9B，文本 TP=1，batch=4。BF16/HIF8 使用相同数据、模板、seed 和调度参数。
常规任务采用 ACC，GSM8K 采用严格答案匹配率。

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

整体验收条件为 `max(0, BF16_ACC - HIF8_ACC) * 100 <= 0.5` 个百分点。
样本加权整体精度损失为 0，满足验收条件；WinoGrande 和 LAMBADA 单项下降超过
0.5 个百分点。配置、命令和结果位置见 [精度测评](docs/validation-status.md)。

## 性能

`vllm bench throughput`，Qwen3.5-9B、Ascend 910B3、TP=1、eager、同步调度。
静态 W8A8KV8 使用固定 activation/KV scale。KV 预算 2 GiB，上下文上限 8192。

每组预热 4 条请求，计时 16 条，重复 3 次。吞吐取各轮中位数，HBM 取各轮整卡采样峰值的最大值。所有轮次的缓存抢占次数（含预热）均为 0。

| 输入/输出 tokens | BS 上限 | BF16 输出 token/s | HIF8 输出 token/s | HIF8/BF16 | BF16 / HIF8 HBM 峰值（MB） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/256 | 1 | 8.4305 | 4.4498 | 0.5278x | 23,121 / 18,137 |
| 512/256 | 4 | 27.9306 | 19.1375 | 0.6852x | 23,301 / 18,059 |
| 512/1024 | 1 | 8.8742 | 5.3873 | 0.6071x | 23,122 / 17,959 |
| 512/1024 | 4 | 28.0955 | 18.6130 | 0.6625x | 23,302 / 17,977 |
| 4096/256 | 1 | 9.6256 | 5.5932 | 0.5811x | 23,842 / 18,135 |
| 4096/256 | 4 | 32.5737 | 16.6103 | 0.5099x | 24,723 / 18,158 |

HBM 包含驱动、模型、KV 和 workspace。结果对应表中的固定长度离线负载，
在线延迟需单独测量。配置和结果见 [性能测试](docs/vllm-bench-report.md)。

## 代码结构

```text
src/vllm_ascend_hif8/  # 校准、导出、量化 scheme 和 KV 适配
csrc/extension/       # Python/C++ 与 ACLNN 接口
csrc/opp/             # AscendC 算子、host tiling 和构建
scripts/              # 安装、部署、测评和校准工具
patches/              # vLLM-Ascend 兼容补丁
calibration/          # 校准数据与 artifact
local_tasks/          # lm-eval 任务定义
tests/                # CPU 契约与 NPU 测试
docs/                 # 使用说明、格式和测评报告
benchmarks/           # 测评输出与日志
```

模块职责见 [代码结构](docs/code-structure.md)，构建与测试见 [开发指南](docs/development.md)。
完整文档入口见 [文档索引](docs/README.md)。
