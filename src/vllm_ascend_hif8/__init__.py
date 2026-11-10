"""Software HiFloat8 W8A8 support for vLLM on Ascend."""

# Configure OPP path at import time to avoid torch.compile issues
from .runtime import configure_opp_path as _configure_opp_path

try:
    _configure_opp_path(required=False)
except Exception:
    pass  # Will be raised later if actually needed

from .codec import (
    ASCEND_NAN,
    ASCEND_NINF,
    ASCEND_PINF,
    ASCEND_ZERO,
    decode_ascend,
    encode_ascend,
    quantize_rows,
)

__all__ = [
    "ASCEND_NAN",
    "ASCEND_NINF",
    "ASCEND_PINF",
    "ASCEND_ZERO",
    "decode_ascend",
    "encode_ascend",
    "quantize_rows",
]
