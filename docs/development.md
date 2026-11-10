# 开发与模型适配

代码入口见 [代码目录与功能](code-structure.md)，数据格式和算子签名见
[Checkpoint 与算子接口](checkpoint-format.md)。当前验证对象是 910B 上的
Qwen3.5-9B 文本推理；其他模型需要独立适配、校准和 ACC 验收。

## 上游版本与补丁

实际运行代码由两个固定提交确定：

| 组件 | 提交 |
| --- | --- |
| vLLM | `d02df748bf9efd99022f1a062597dc3cb3808485` |
| vLLM-Ascend | `53b489b78560363871ba5402dd72cfd124c4b209` |

vLLM 源码版本为 `0.23.1rc1.dev1451+gd02df748b`，Ascend 适配基于
`releases/v0.26.0rc` 的接口。`VLLM_VERSION=0.26.0` 选择匹配的 Ascend API 分支；
`SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0` 设置源码构建的版本标签。
这里的 0.26.0 是兼容配置，不是要求安装同名 PyPI wheel，也不是性能开关。

安装器 [`install_vllm_ascend_910b.sh`](../scripts/install_vllm_ascend_910b.sh)
检出固定源码、应用 [上游补丁](../patches/README.md)，以
`--no-deps --no-build-isolation` 安装，保留已有 torch、torch-npu 和 Triton。
补丁覆盖 910B 的 GDN/causal-convolution 算子集合、Python 构建解释器及运行接口。
更换上游提交时需要重新检查补丁、构建扩展并验证生成与 ACC。

排查导入问题先检查源码路径和实际导入位置：

```bash
export HIF8_VLLM_ROOT="$PWD/.deps/vllm"
export HIF8_VLLM_ASCEND_ROOT="$PWD/.deps/vllm-ascend"
export VLLM_VERSION=0.26.0 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$PWD/src:$HIF8_VLLM_ROOT:$HIF8_VLLM_ASCEND_ROOT"
git -C "$HIF8_VLLM_ROOT" rev-parse HEAD
git -C "$HIF8_VLLM_ASCEND_ROOT" rev-parse HEAD
python -c 'import vllm, vllm_ascend; print(vllm.__file__); print(vllm_ascend.__file__)'
```

环境版本表见 [README](../README.md#环境要求)。`pip show` 的 distribution
版本不能替代源码提交和导入路径检查。

## 构建与测试

使用运行环境中的 Python 和 CANN 构建。修改 AscendC 源码后应重新构建 OPP，
不能只更新 Python 文件。

```bash
conda activate hifi
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export HIF8_OPP_BUILD_DIR="$PWD/csrc/opp/build/ascend910b"
MAX_JOBS=4 bash csrc/opp/build.sh ascend910b
HIF8_SOC=ascend910b python setup.py build_ext --inplace
export ASCEND_CUSTOM_OPP_PATH="$HIF8_OPP_BUILD_DIR/package/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors/vllm_ascend_hif8"
```

部署 wheel 使用 `HIF8_SOC=ascend910b python setup.py bdist_wheel` 构建，
wheel 中包含对应 OPP。源码运行通过 `ASCEND_CUSTOM_OPP_PATH` 选择 OPP。

在上面的固定源码环境中运行 CPU 测试：

```bash
python -m unittest discover -s tests -t tests -v
```

空闲 NPU 上运行算子和 KV 测试：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 HIF8_RUN_NPU_TESTS=1 \
  python -m unittest discover -s tests -p test_npu_ops.py -v
ASCEND_RT_VISIBLE_DEVICES=0 HIF8_RUN_NPU_TESTS=1 \
  python -m unittest discover -s tests -p test_hif8_kv_npu.py -v
```

真实权重测试还需设置 `HIF8_EXPORTED_MODEL`。算子改动应覆盖非对齐尺寸、
连续异步调用、真实 KV 形状和模型生成；KV 向量解码需检查缓冲区复用的流水线同步。
FIA 测试使用布尔因果 mask。

## 新模型适配

1. **建立 BF16 基线。** 检查 architecture、tokenizer、GQA/MQA head 数、
   head size、attention 类型与融合投影，在同一设备上验证 BS1/4 加载和生成。
2. **定义显式映射。** 为 exporter、calibrator 和 verifier 增加模型专属分支，
   列明量化 Linear、融合组件、embedding、保持浮点的参数，以及全注意力 KV 层。
   Qwen3.5 的映射不能直接用于新架构。
3. **离线校准。** 在目标 BF16 模型上采集 Linear 输入和 KV 统计，生成固定
   FP32 input scale、per-layer/per-head K/V scale，记录模型、tokenizer 和数据 hash。
   校准不更新模型权重；更改 scale 算法需要新建 artifact 和 checkpoint。
4. **导出和校验。** 保存 HiFloat8 权重及 scale，检查张量集合、融合映射、
   metadata、scale 正有限性、源模型身份和权重 round-trip。
5. **验证静态运行路径。** 保持 `HIF8_STATIC_FUSED_GENERATION=1`、`soft_hif8`、
   eager 和同步调度。确认 Linear 与 KV 使用 checkpoint scale，KV 页面为字节存储，
   不常驻完整解码权重；BS1/4 输出应非空且正常。
6. **配对 ACC 与性能。** BF16/HIF8 使用相同任务、数据、seed、模板、TP 和 batch，
   先筛查小样本，再运行全量测评。达到 [ACC 验收条件](validation-status.md)后，
   按 [吞吐测试指南](benchmarking.md)比较性能和显存。

现有 Qwen3.5 导出入口如下；新模型先完成上述适配，再使用模型对应校准文件：

```bash
python -m vllm_ascend_hif8.exporter /path/to/model-bf16 /path/to/model-hif8-static \
  --calibration /path/to/model-static-calibration.json
python scripts/verify_export.py /path/to/model-bf16 /path/to/model-hif8-static
```

每个模型单独保存环境、源码/补丁身份、权重与校准 hash、BF16/HIF8 的 BS1/4
结果和逐题输出。每种精度各有 2 个 BS × 10 个任务，两种精度共 40 个测评组合。
响应缓存按模型、精度、BS、任务隔离；核对实际样本数与 `limit` 后再认定任务完成。
新增模型的结果不能覆盖现有 Qwen3.5 报告。
