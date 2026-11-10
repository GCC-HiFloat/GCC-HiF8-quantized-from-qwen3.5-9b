# 文档

Ascend 910B 上的 Qwen3.5-9B 静态 W8A8KV8 文档。安装和一键部署见
[项目 README](../README.md#安装与部署)。

## 测评与验证

| 文档 | 内容 |
| --- | --- |
| [性能与显存验证报告](vllm-bench-report.md) | vLLM bench 的 BS1/4 吞吐、显存峰值、三轮波动、配置与原始记录 |
| [精度验证报告](validation-status.md) | 十项数据集 ACC、整体验收条件、复测命令与结果目录 |
| [吞吐测试指南](benchmarking.md) | 环境准备、单组/多长度测试、断点续跑、显存采样和报告生成 |

## 代码与开发

| 文档 | 内容 |
| --- | --- |
| [代码目录与功能](code-structure.md) | 目录、Python 模块、原生算子、执行流程和脚本入口 |
| [开发与模型适配](development.md) | 固定上游版本、补丁、构建、回归测试和新模型静态量化流程 |
| [Checkpoint 与算子接口](checkpoint-format.md) | 权重及 scale、融合层映射、KV 布局、算子形状和运行约束 |

`reports/vllm-bench/` 保存性能报告对应的 JSON 证据，完整测评日志与逐题结果
位于各报告注明的 `benchmarks/` 目录。
