# Checkpoint 与算子接口

## 权重与 activation

导出时使用 `--calibration`。目标 Linear 的 `quant_model_description.json`
条目为 `W8A8HIF8_STATIC`，其余浮点参数为 `FLOAT`。

```text
model.language_model.layers.0.mlp.down_proj.weight        U8   [4096, 12288]
model.language_model.layers.0.mlp.down_proj.weight_scale  F32  [4096, 1]
model.language_model.layers.0.mlp.down_proj.input_scale   F32  [1]
```

`weight_scale` 为固定 per-output-channel scale；`input_scale` 为正、有限的
FP32 标量，跨请求和 batch 不变。融合 Linear 的组件共享规范化后的
activation scale 名称，导出器避免重复写入。

Qwen3.5-9B 的目标包括 32 个 full-attention 投影、72 个 GDN 投影、
96 个 MLP 投影以及 token embedding，共 201 个权重。text-only 模型可省略
路径中的 `language_model`。embedding 使用 `W8A8HIF8_SOFT` 加载接口，
按 token ID gather 后按固定 weight scale 解码，不执行 activation 量化。

GDN `in_proj_b` / `in_proj_a`、卷积与 recurrent state、LM head、norm、
visual tower 和 MTP 不属于这组 W8A8 Linear。

## vLLM 加载映射

导出器使用原始组件权重名，通过 `quant_model_description.json` 交给
vLLM-Ascend ModelSlim 加载。此格式要求 `config.json.quantization_config` 为空，
服务使用 `--quantization ascend`。

| 融合 Linear | 原始组件 | 量化类型 |
| --- | --- | --- |
| `self_attn.qkv_proj` | `q_proj`, `k_proj`, `v_proj` | `W8A8HIF8_STATIC` |
| `mlp.gate_up_proj` | `gate_proj`, `up_proj` | `W8A8HIF8_STATIC` |
| `linear_attn.in_proj_qkvz` | `in_proj_qkv`, `in_proj_z` | `W8A8HIF8_STATIC` |
| `linear_attn.in_proj_ba` | `in_proj_b`, `in_proj_a` | `FLOAT` |

## KV8

`hif8_metadata.json` 的 `kv_cache.mode` 必须为 `static_per_layer_per_head`；
`kv_cache.layers` 为每层分别保存 `key_scales` 和 `value_scales`。
适配器从 checkpoint 获取固定 scale，不根据当前 token 重新校准。

每个 dense K/V cache 行按以下布局存储：

```text
[head_size 个 HiFloat8 字节][4 字节 little-endian FP32 scale]
```

scale sidecar 随 token 行存储，但数值来自同一层同一 head 的固定 scale。
逻辑 head size 为 256 时物理宽度为 260 字节。注意力前仅 gather/decode
当前请求需要的 KV；GDN state 仍使用原生浮点 Mamba cache。

metadata 的 `calibration` 字段记录校准来源。校验入口为
`scripts/verify_export.py`，CPU 契约测试为 `tests/test_static_scheme.py`、
`tests/test_quant_description_contract.py` 和 `tests/test_hif8_kv_cache.py`。

## 算子接口

扩展模块为 `vllm_ascend_hif8._C`，构建目标为 `HIF8_SOC=ascend910b`。
原生操作数需要连续且 storage offset 为零，Python wrapper 负责规范化视图。

```text
hif8_static_quant(x: BF16|FP16 [..., K], fixed_scale: F32 [S])
  -> (data: U8 [..., K], row_scale: F32 [M])

soft_hif8_gemm / soft_hif8_gemv(
  activation: U8 [M, K], weight: U8 [N, K],
  row_scale: F32 [M], channel_scale: F32 [N],
  bias: BF16|FP16 [N] | None, output_dtype
) -> output [M, N]

hif8_scaled_decode(data: U8 [rows, dim], scale: F32 [rows])
  -> BF16 [rows, dim]
```

Linear 使用 `S=1`；KV 将 `[tokens, heads, dim]` 展平后使用 `S=heads`，
按 head 顺序重复固定 scale。量化不执行运行时 amax/ReduceMax。
Linear 在线解码局部 tile，使用 BF16 Cube 或 vector reduction，
并应用 `row_scale[m] * channel_scale[n]`。

普通 Linear 在 `M=1`、`N<=1536`、`K<=8192` 且未设置
`HIF8_DISABLE_GEMV=1` 时使用 GEMV，其余使用 GEMM。
`HIF8_STATIC_FUSED_GENERATION=1` 下，实际模型投影使用独立静态量化和 GEMM/GEMV；
`M<=4` 且 `K<1024` 的小形状允许静态融合算子，默认使用单 AIC。

KV 解码通过向量查表得到 FP32 值，乘固定 scale 后舍入为 BF16 临时张量；
非对齐尾部使用标量路径。全注意力 KV 与 GDN recurrent state 分区存储，
请求调度及 batch 变化不改变 scale。
