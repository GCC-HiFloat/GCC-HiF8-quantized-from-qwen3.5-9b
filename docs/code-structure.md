# 代码目录与功能

## 目录结构

```text
vllm-ascend-hif8/
├── README.md                    # 项目概述、安装与测评结果
├── pyproject.toml               # 包元数据、插件和命令注册
├── setup.py                     # 原生扩展构建、OPP 打包
├── src/
│   ├── sitecustomize.py         # Triton API 兼容
│   └── vllm_ascend_hif8/        # 量化、模型加载、KV 适配
├── csrc/
│   ├── extension/              # pybind / ACLNN 接口
│   └── opp/
│       ├── op_host/            # shape/type 推导、tiling 与注册
│       ├── op_kernel/          # AscendC 量化、解码、GEMM/GEMV
│       └── build.sh            # 算子构建入口
├── scripts/                    # 安装、部署、校准与测评工具
├── patches/                    # 上游兼容补丁
├── calibration/                # 校准数据、参数及来源
├── local_tasks/                # lm-eval 本地任务定义
├── tests/                      # CPU 与 NPU 测试
├── docs/                       # 使用与开发文档
│   └── reports/vllm-bench/     # 最新性能报告的 JSON 证据
└── benchmarks/                 # 完整结果、样本和日志
```

安装脚本将上游源码放在 `.deps/`。构建产物位于 `csrc/opp/build/`、
`build/` 和 `dist/`；模型 checkpoint 存放在仓库外。

完整测评输出存放在 `benchmarks/`，随仓库发布的性能指标与证据放在
`docs/reports/vllm-bench/`。文档入口为 [docs/README.md](README.md)。

## Python 模块

以下文件均位于 `src/vllm_ascend_hif8/`。

| 文件 | 功能 |
| --- | --- |
| `plugin.py` | 注册 vLLM 插件、量化 scheme 和 worker KV hook |
| `scheme.py` | 创建和加载 Linear/embedding 参数，按固定 input scale 分派算子 |
| `exporter.py` | 导出 Qwen3.5 HiFloat8 权重、scale、索引和 metadata |
| `qwen35_calibrator.py` | 离线采集 BF16 Qwen3.5 的 Linear 与全注意力 KV 统计 |
| `calibration.py` | 管理校准参数、选择 scale、校验源模型身份和层覆盖 |
| `kv_cache.py` | 定义 HiFloat8 KV 页面布局、FP32 scale sidecar 和容量计算 |
| `kv_runtime.py` | 适配 KV 分配、写入、解码和同步 dense attention |
| `native.py` | 加载原生扩展，规范化 Tensor 布局并调用算子 |
| `runtime.py` | 查找包内 OPP，配置 `ASCEND_CUSTOM_OPP_PATH` |
| `codec.py` / `torch_codec.py` | 提供 HiFloat8 编解码参考实现 |
| `reference.py` | 提供算子数值参考实现 |
| `tablegen.py` | 生成编解码查表数据 |
| `fake_tensors.py` | 注册算子 fake/meta 实现，支持形状推导 |

Checkpoint 字段见 [checkpoint 格式](checkpoint-format.md)。

## 原生算子

| 文件或算子 | 功能 |
| --- | --- |
| `csrc/extension/pybind.cpp` | 导出 Python 接口 |
| `csrc/extension/hif8_native.cpp` | 校验 Tensor 和设备，调用 ACLNN |
| `hif8_static_quant` | 按固定 FP32 scale 将 BF16/FP16 输入编码为 HiFloat8 |
| `soft_hif8_gemm` / `soft_hif8_gemv` | 局部 tile 解码、矩阵/向量乘与 scale 应用 |
| `soft_hif8_static_fused_gemm` | 小形状静态融合计算；模型大投影使用静态量化 + GEMM |
| `hif8_scaled_decode` | 向量查表、乘固定 FP32 scale，将 HiFloat8 KV 解码为 BF16 |
| `causal_conv1d` | GDN 路径所需的因果卷积 |

算子的 shape、dtype 和布局约束见 [算子接口](checkpoint-format.md#算子接口)。
`op_host/` 与 `op_kernel/` 共同定义算子接口和执行逻辑。

## 执行流程

```text
BF16 checkpoint + 校准数据
  → qwen35_calibrator
  → 静态校准 JSON
  → exporter + verify_export
  → HiFloat8 checkpoint
  → plugin → scheme / kv_runtime → native → AscendC OPP
  → 生成输出 / lm-eval 候选打分
```

Linear 使用固定 scale，依次执行 `hif8_static_quant` 和 `soft_hif8_gemm`。
KV 使用固定 per-layer/per-head scale 写入 HiFloat8 页面，在 attention 前
解码所需的 KV。GDN recurrent state、norm 和 LM head 使用模型对应的浮点格式。
请求分组和调度不改变 scale。

## 脚本

| 任务 | 入口 |
| --- | --- |
| 安装上游依赖 | `scripts/install_vllm_ascend_910b.sh` |
| 部署服务 | `scripts/deploy_910b_hif8.sh` |
| Qwen3.5 校准 | `scripts/calibrate_qwen35_static_hif8.py` |
| 导出校验 | `scripts/verify_export.py` |
| 十项 ACC 测评 | `scripts/run_ten_dataset_{bf16,hif8}_batch4.sh` |
| ACC 汇总 | `scripts/summarize_ten_hif8_compare.py` |
| 模型生成测速 | `scripts/benchmark_model.py` |
| vLLM bench throughput | `scripts/run_vllm_bench_910b.sh` |
| 多长度重复测试 | `scripts/run_vllm_bench_matrix.py` |
| 多长度报告发布 | `scripts/publish_vllm_bench_matrix.py` |
| bench 输出与显存采集 | `scripts/collect_vllm_bench.py` |
| bench 抢占计数 | `scripts/bench_scheduler.py`，沿用 Ascend 调度策略 |
| bench 报告生成 | `scripts/summarize_vllm_bench.py` |
| 算子微基准 | `scripts/benchmark_soft_hif8_*.py` |
| KV 解码微基准 | `scripts/benchmark_kv_decode.py` |
| 静态 scale 搜索 | `scripts/*static_calibration.py`、`scripts/compose_weight_scale_checkpoint.py` |

测评配置和结果路径见 [精度测评](validation-status.md)与
[vLLM bench 报告](vllm-bench-report.md)。
多长度测试和续跑见 [吞吐测试](benchmarking.md)。

## 测试

| 模块 | 测试内容 |
| --- | --- |
| 编码与 scale | codec、reference、calibration、exporter |
| 模型加载 | plugin、static scheme、quant description、embedding |
| KV cache | 页面布局、runtime 适配、NPU 写入与解码 |
| 原生算子 | shape/dtype、数值结果、同步与模型冒烟 |
| bench 工具 | 输出解析、显存采集、结果汇总 |

CPU 测试默认运行；NPU 测试通过 `HIF8_RUN_NPU_TESTS=1` 开启，
模型测试还需指定 checkpoint。构建和测试命令见 [开发指南](development.md)。
