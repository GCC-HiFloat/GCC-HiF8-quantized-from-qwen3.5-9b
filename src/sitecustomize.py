"""Runtime compatibility hooks for the torch-npu 2.10 hifi environment."""
try:
    from vllm.triton_utils import triton
    if not hasattr(triton, "next_power_of_2"):
        def _next_power_of_2(x):
            x = int(x)
            return 1 if x <= 1 else 1 << (x - 1).bit_length()
        triton.next_power_of_2 = _next_power_of_2
except Exception:
    pass
