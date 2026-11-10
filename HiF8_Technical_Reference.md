# 910B 静态 HiFloat8 技术说明

静态 W8A8KV8 的字节格式、scaling 与执行路径。环境和部署方法见 [README](README.md)。

## 字节格式

权重、量化 activation 和全注意力 KV 使用 Ascend-layout HiFloat8 字节。
参考实现是 `src/vllm_ascend_hif8/codec.py`，算子 LUT 由
`scripts/generate_tables.py` 生成，字节编码与舍入规则由 codec 测试覆盖。

| 常量 | 本项目参考编码 |
| --- | --- |
| 零 | `0x00` |
| NaN | `0x80` |
| 正无穷 / 负无穷 | `0x6f` / `0xef` |
| 最小正有限值 | `2^-22` |
| 最大有限值 | `2^15` |

## 固定 scaling

权重离线编码并保存 FP32 per-output-channel scale。Linear activation
使用校准得到的 FP32 `input_scale`；K/V 使用 checkpoint 记录的固定
per-layer/per-head scale。推理时 scale 不依赖当前 batch 或当前 token
的 amax，不进行在线校准。

KV 页面每行包含 `head_size` 个 HiFloat8 字节与 4 字节 FP32 scale
sidecar。sidecar 随 token 行重复保存，并不代表每个 token 动态搜索 scale。
具体字段和融合层映射见 `docs/checkpoint-format.md`。

## 910B 执行路径

实际模型投影经过 `hif8_static_quant` 和在线 tile 解码 GEMM/GEMV；
Cube 输入为局部 BF16 tile，不持久化完整 BF16 权重副本。
`HIF8_STATIC_FUSED_GENERATION=1` 启用静态模式，小形状可以使用
静态融合算子。全注意力前仅解码当前请求所需 KV，GDN state 保持浮点。
算子签名及路由边界见 [算子接口](docs/checkpoint-format.md#算子接口)。

## 验收

在相同数据、模板、seed、batch 和调度参数下比较 BF16/HIF8 的数据集 ACC。
整体精度按样本数加权，验收条件为 `max(0, BF16_ACC - HIF8_ACC) * 100 <= 0.5`
个百分点。结果见 [精度测评](docs/validation-status.md)和[性能报告](docs/vllm-bench-report.md)。
