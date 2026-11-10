# 测评报告

Qwen3.5-9B 在 Ascend 910B 上的静态 W8A8KV8 与 BF16 对照。

- [精度测评](../docs/validation-status.md)：十项数据集、41,479 条样本，batch=4。
- [性能与显存](../docs/vllm-bench-report.md)：BS=1/4 的 vLLM bench 吞吐、HBM 峰值和三轮波动。
- [吞吐测试指南](../docs/benchmarking.md)：运行命令、断点续跑和输出文件。
- [开发指南](../docs/development.md)：构建、测试与模型适配。
