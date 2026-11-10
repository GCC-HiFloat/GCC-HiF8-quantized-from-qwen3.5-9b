"""Configure package-local Ascend custom OPP discovery."""

from __future__ import annotations

import os
from pathlib import Path

_OPP_ENVIRONMENT = "ASCEND_CUSTOM_OPP_PATH"
_PACKAGE_ROOT = Path(__file__).resolve().parent


class PackagedOppUnavailable(RuntimeError):
    """Raised when the wheel has no usable package-local OPP payload."""


def packaged_vendor_roots() -> tuple[Path, ...]:
    """Return the package-local Ascend 910B vendor payload, if present."""
    vendor_root = _PACKAGE_ROOT / "vendor" / "ascend910b" / "vendors" / "vllm_ascend_hif8"
    return (vendor_root,) if vendor_root.is_dir() else ()


def configure_opp_path(*, required: bool = True) -> tuple[Path, ...]:
    """Prepend packaged vendor roots while preserving existing OPP paths."""
    packaged = packaged_vendor_roots()
    existing = [entry for entry in os.environ.get(_OPP_ENVIRONMENT, "").split(os.pathsep) if entry]
    if required and not packaged and not existing:
        raise PackagedOppUnavailable(
            "the package-local Ascend HiFloat8 OPP payload is missing and "
            "ASCEND_CUSTOM_OPP_PATH is empty; build a wheel after building "
            "csrc/opp, or configure an external vendor path for a source-tree run"
        )

    entries: list[str] = []
    seen: set[str] = set()
    for entry in (*map(str, packaged), *existing):
        normalized = os.path.normpath(entry)
        if normalized not in seen:
            entries.append(entry)
            seen.add(normalized)
    if entries:
        os.environ[_OPP_ENVIRONMENT] = os.pathsep.join(entries)
    return packaged
