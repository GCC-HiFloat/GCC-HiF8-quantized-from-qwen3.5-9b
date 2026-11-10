# Upstream integration

`vllm-ascend-910b-hif8.patch` is the reproducible patch applied by
`scripts/install_vllm_ascend_910b.sh` to the pinned vLLM-Ascend checkout.

| Component | Repository state | Purpose |
| --- | --- | --- |
| vLLM | `d02df748bf9efd99022f1a062597dc3cb3808485` | Main-branch snapshot used by the validation |
| vLLM-Ascend | `53b489b78560363871ba5402dd72cfd124c4b209` | `releases/v0.26.0rc` source state |
| Compatibility selector | `VLLM_VERSION=0.26.0` | Selects the v0.26.0 API patch branches in vLLM-Ascend |

The installer uses pinned source commits. `SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0`
sets the build metadata, and `VLLM_VERSION=0.26.0` selects the matching
vLLM-Ascend API paths.

The patch configures the 910B custom-op set, GDN/causal-convolution fallbacks,
Python interpreter selection, and runtime compatibility for static HiFloat8.

The patch is ordinary unified diff text. The installer checks it with
`git apply --check`, applies it once, and detects an already-applied patch on
subsequent runs. It refuses to change a dirty upstream checkout unless
`--allow-dirty` is explicitly supplied.
